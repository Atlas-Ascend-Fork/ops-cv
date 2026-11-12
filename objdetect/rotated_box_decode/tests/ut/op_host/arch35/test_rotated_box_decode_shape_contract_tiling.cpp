/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Shape contract tests for RotatedBoxDecode tiling:
 *   - shape[0] (B) and shape[2] (N) must be non-negative and materialized (not -1) at tiling time.
 *   - Unknown dims (-1) or negative values are rejected with GRAPH_FAILED.
 *   - B*N products feeding totalCount / core split / buffer sizing are guarded against
 *     int64 signed-multiply overflow (B*N > INT64_MAX/160 rejected); in-range large
 *     shapes must keep succeeding.
 *   - Empty tensors (B = 0 or N = 0) remain a valid short-circuit success.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/rotated_box_decode_tiling_data.h"

namespace {

using TensorDescription = gert::TilingContextPara::TensorDescription;

struct RotatedBoxDecodeCompileInfoForTest {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

RotatedBoxDecodeCompileInfoForTest g_compileInfo{64, 262144};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeContext(const std::vector<int64_t>& anchorDims, const std::vector<int64_t>& deltasDims,
                                    const std::vector<int64_t>& outputDims)
{
    const auto anchorShape = MakeShape(anchorDims);
    const auto deltasShape = MakeShape(deltasDims);
    const auto outputShape = MakeShape(outputDims);
    const std::vector<TensorDescription> inputs = {
        {anchorShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {deltasShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<TensorDescription> outputs = {
        {outputShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr(
            "weight", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f})),
    };
    return gert::TilingContextPara("RotatedBoxDecode", inputs, outputs, attrs, &g_compileInfo, "Ascend950", 64, 262144,
                                   4096);
}

void PrintResult(const char* label, bool success, const TilingInfo& info)
{
    std::cout << "case=" << label << " status=" << (success ? "GRAPH_SUCCESS" : "GRAPH_FAILED")
              << " block_num=" << info.blockNum;
    if (info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData)) {
        const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
        std::cout << " td.B=" << td->B << " td.N=" << td->N << " td.totalCount=" << td->totalCount
                  << " td.perCoreCount=" << td->perCoreCount;
    }
    std::cout << std::endl;
}

} // namespace

class RotatedBoxDecodeShapeContractTiling : public testing::Test {};

// Positive control: fully materialized positive shape succeeds.
TEST_F(RotatedBoxDecodeShapeContractTiling, valid_control_succeeds)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, 8}, {2, 5, 8}, {2, 5, 8}), info);
    PrintResult("valid_control", success, info);
    EXPECT_TRUE(success);
    ASSERT_TRUE(info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData));
    const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->B, 2);
    EXPECT_EQ(td->N, 8);
    EXPECT_EQ(td->totalCount, 16);
}

// Unconcretized unknown dim B = -1 (dynamic shape not materialized before tiling) must be rejected.
TEST_F(RotatedBoxDecodeShapeContractTiling, unknown_b_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({-1, 5, 100}, {-1, 5, 100}, {-1, 5, 100}), info);
    PrintResult("unknown_b", success, info);
    EXPECT_FALSE(success);
}

// Unconcretized unknown dim N = -1 must be rejected.
TEST_F(RotatedBoxDecodeShapeContractTiling, unknown_n_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, -1}, {2, 5, -1}, {2, 5, -1}), info);
    PrintResult("unknown_n", success, info);
    EXPECT_FALSE(success);
}

// Explicit negative B must be rejected.
TEST_F(RotatedBoxDecodeShapeContractTiling, negative_b_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({-3, 5, 100}, {-3, 5, 100}, {-3, 5, 100}), info);
    PrintResult("negative_b", success, info);
    EXPECT_FALSE(success);
}

// Explicit negative N must be rejected.
TEST_F(RotatedBoxDecodeShapeContractTiling, negative_n_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, -7}, {2, 5, -7}, {2, 5, -7}), info);
    PrintResult("negative_n", success, info);
    EXPECT_FALSE(success);
}

// Cross-input shape mismatch (positive dims) still fails, confirming the value check does not
// shadow the shape-consistency check.
TEST_F(RotatedBoxDecodeShapeContractTiling, cross_input_shape_mismatch_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, 8}, {3, 5, 8}, {2, 5, 8}), info);
    PrintResult("cross_input_mismatch", success, info);
    EXPECT_FALSE(success);
}

// deltas with channel count != 5 (e.g. [B, 6, N]) used to bypass validation because only B/N
// were compared across inputs; the kernel would then mis-read the fixed 5-channel layout
// (b*5+c)*N+n and silently produce wrong values. Must be rejected by the deltas-side check.
TEST_F(RotatedBoxDecodeShapeContractTiling, deltas_channel_not_five_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, 8}, {2, 6, 8}, {2, 5, 8}), info);
    PrintResult("deltas_channel_not_five", success, info);
    EXPECT_FALSE(success);
}

// deltas with rank > 3 (e.g. [B, 5, N, 2]) also passes the old B/N-only comparison but breaks
// the 3D layout contract; must be rejected.
TEST_F(RotatedBoxDecodeShapeContractTiling, deltas_rank4_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, 8}, {2, 5, 8, 2}, {2, 5, 8}), info);
    PrintResult("deltas_rank4", success, info);
    EXPECT_FALSE(success);
}

// deltas with rank < 3 (e.g. [B, N]) is rejected as well; previously it was only caught
// indirectly via the missing dim-2 value comparison, now it fails the explicit rank check.
TEST_F(RotatedBoxDecodeShapeContractTiling, deltas_rank2_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({2, 5, 8}, {2, 8}, {2, 5, 8}), info);
    PrintResult("deltas_rank2", success, info);
    EXPECT_FALSE(success);
}

// B*N itself overflows int64 (B = INT64_MAX, N = 2): totalCount, SelectUbAxis products
// and kernel totalElems=(B*5)*N would all be UB; must be rejected before any multiply.
TEST_F(RotatedBoxDecodeShapeContractTiling, b_times_n_overflow_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({INT64_MAX, 5, 2}, {INT64_MAX, 5, 2}, {INT64_MAX, 5, 2}), info);
    PrintResult("b_times_n_overflow", success, info);
    EXPECT_FALSE(success);
}

// B*N fits in int64 but the derived product B*N*160 (SelectUbAxis minDtypeBits for fp32,
// the widest dtype) overflows: B=1e17 > INT64_MAX/160 ~= 5.76e16. Must be rejected too.
TEST_F(RotatedBoxDecodeShapeContractTiling, scaled_bits_overflow_should_fail)
{
    TilingInfo info;
    const bool success = ExecuteTiling(
        MakeContext({100000000000000000LL, 5, 1}, {100000000000000000LL, 5, 1}, {100000000000000000LL, 5, 1}), info);
    PrintResult("scaled_bits_overflow", success, info);
    EXPECT_FALSE(success);
}

// Large but in-range product (B*N = 5e16 <= INT64_MAX/160) must keep succeeding: the guard
// is pure overflow protection and must not clamp/reject materializable-in-principle shapes.
TEST_F(RotatedBoxDecodeShapeContractTiling, large_but_in_range_succeeds)
{
    TilingInfo info;
    const bool success = ExecuteTiling(
        MakeContext({50000000000000000LL, 5, 1}, {50000000000000000LL, 5, 1}, {50000000000000000LL, 5, 1}), info);
    PrintResult("large_but_in_range", success, info);
    EXPECT_TRUE(success);
    ASSERT_TRUE(info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData));
    const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->B, 50000000000000000LL);
    EXPECT_EQ(td->N, 1);
    EXPECT_EQ(td->totalCount, 50000000000000000LL);
}

// N==1 shapes route to the SIMT branch (tiling key 512, RBD_UB_AXIS_I=2):
// no UB tiles (perCoreCount/ubFactor = 0), AIV count scales with the work —
// blockNum = clamp(CeilDiv(B, RBD_SIMT_THREAD_NUM=256), 1, 64).
TEST_F(RotatedBoxDecodeShapeContractTiling, n1_routes_to_simt_branch_large_b)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({783634, 5, 1}, {783634, 5, 1}, {783634, 5, 1}), info);
    PrintResult("n1_simt_large", success, info);
    EXPECT_TRUE(success);
    ASSERT_TRUE(info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData));
    const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 512); // (NDDMA=0, UB_AXIS_SEL_I=2) -> 2<<8
    EXPECT_EQ(td->ubAxis, 2);       // RBD_UB_AXIS_I
    EXPECT_EQ(td->perCoreCount, 0); // unused by the SIMT branch
    EXPECT_EQ(td->ubFactor, 0);     // no UB tiles
    EXPECT_EQ(info.blockNum, 64);   // min(CeilDiv(783634, 256)=3060, 64)
}

TEST_F(RotatedBoxDecodeShapeContractTiling, n1_routes_to_simt_branch_small_b)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({318, 5, 1}, {318, 5, 1}, {318, 5, 1}), info);
    PrintResult("n1_simt_small", success, info);
    EXPECT_TRUE(success);
    ASSERT_TRUE(info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData));
    const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 512);
    EXPECT_EQ(td->ubAxis, 2);
    EXPECT_EQ(info.blockNum, 2); // CeilDiv(318, 256) = 2
}

// Empty tensor B = 0 stays a valid short-circuit success (perCoreCount = 0, single block).
TEST_F(RotatedBoxDecodeShapeContractTiling, empty_b_zero_succeeds)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({0, 5, 100}, {0, 5, 100}, {0, 5, 100}), info);
    PrintResult("empty_b_zero", success, info);
    EXPECT_TRUE(success);
    ASSERT_TRUE(info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData));
    const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->B, 0);
    EXPECT_EQ(td->N, 100);
    EXPECT_EQ(td->totalCount, 0);
    EXPECT_EQ(td->perCoreCount, 0);
    EXPECT_EQ(info.blockNum, 1);
}

// Empty tensor N = 0 stays a valid short-circuit success.
TEST_F(RotatedBoxDecodeShapeContractTiling, empty_n_zero_succeeds)
{
    TilingInfo info;
    const bool success = ExecuteTiling(MakeContext({8, 5, 0}, {8, 5, 0}, {8, 5, 0}), info);
    PrintResult("empty_n_zero", success, info);
    EXPECT_TRUE(success);
    ASSERT_TRUE(info.tilingData != nullptr && info.tilingDataSize >= sizeof(RotatedBoxDecodeTilingData));
    const auto* td = reinterpret_cast<const RotatedBoxDecodeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->B, 8);
    EXPECT_EQ(td->N, 0);
    EXPECT_EQ(td->totalCount, 0);
    EXPECT_EQ(td->perCoreCount, 0);
    EXPECT_EQ(info.blockNum, 1);
}
