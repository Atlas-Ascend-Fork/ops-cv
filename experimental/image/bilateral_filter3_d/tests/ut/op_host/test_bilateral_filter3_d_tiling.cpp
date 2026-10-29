/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/bilateral_filter3_d_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"

namespace {
using Attr = gert::TilingContextPara::OpAttr;

// Attribute order mirrors BilateralFilter3DTilingFunc: kernel_size(0) -> sigma_space(1) -> sigma_color(2).
std::vector<Attr> MakeAttrs(int64_t kernelSize, float sigmaSpace = 1.0F, float sigmaColor = 1.0F)
{
    return {Attr("kernel_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(kernelSize)),
            Attr("sigma_space", Ops::Cv::AnyValue::CreateFrom<float>(sigmaSpace)),
            Attr("sigma_color", Ops::Cv::AnyValue::CreateFrom<float>(sigmaColor))};
}

// The tiling function reads the core count and the UB size from PlatformInfo, not
// from CompileInfo, so these two values only have to be self-consistent for the
// TilingParse stage -- they do not steer the result. The faker's platform currently
// reports 64 AIV cores, which is why core-bound cases below do not pin an exact
// block count.
constexpr int64_t AIV_CORE_NUM = 48;
constexpr uint64_t UB_SIZE = 196608;

// Passed as expectBlockNum when the launch is core-bound rather than row-bound: the
// exact value is whatever the faker's platform reports, so only the invariant is
// asserted. The row-bound cases pin exact numbers instead, and those are what
// actually lock down the usedCoreNum = min(totalRows, coreNum) clamp.
constexpr uint32_t BLOCK_NUM_CORE_BOUND = 0;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

// Accepted configurations are checked through ExecuteTiling so the tiling key and
// the launch block count can be asserted directly, rather than by matching a
// stringified tiling-data dump.
void ExpectTilingSuccess(const std::vector<int64_t>& shape, ge::DataType dtype, int64_t kernelSize, float sigmaSpace,
                         float sigmaColor, uint64_t expectTilingKey, uint32_t expectBlockNum)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    const gert::TilingContextPara::TensorDescription desc(MakeStorageShape(shape), dtype, ge::FORMAT_ND);
    gert::TilingContextPara para("BilateralFilter3D", {desc}, {desc}, MakeAttrs(kernelSize, sigmaSpace, sigmaColor),
                                 &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);

    // One row per block, so the launch can never ask for more blocks than there are
    // rows -- otherwise the surplus cores would re-walk rows already owned by others.
    int64_t totalRows = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        totalRows *= shape[i];
    }
    const uint32_t blockNum = static_cast<uint32_t>(tilingInfo.blockNum);
    EXPECT_GT(blockNum, 0U);
    EXPECT_LE(static_cast<int64_t>(blockNum), totalRows);
    if (expectBlockNum != BLOCK_NUM_CORE_BOUND) {
        EXPECT_EQ(blockNum, expectBlockNum);
    }
}
} // namespace

class BilateralFilter3DTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BilateralFilter3DTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BilateralFilter3DTiling TearDown" << std::endl; }
};

// ---------------------------------------------------------------------------
// Rank constraints: only 3/4/5-D ND tensors are accepted.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DTiling, RankTwoFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{17, 19}, {17, 19}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{17, 19}, {17, 19}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, RankSixFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para(
        "BilateralFilter3D", {{{{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// Shape limits: D <= 1000, H/W <= 4096, no empty tensors, prod(leading) <= 12800.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DTiling, DepthExceedsLimitFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{1001, 4, 4}, {1001, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{1001, 4, 4}, {1001, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3),
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, HeightExceedsLimitFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{4, 4097, 4}, {4, 4097, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{4, 4097, 4}, {4, 4097, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3),
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, WidthExceedsLimitFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{4, 4, 4097}, {4, 4, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{4, 4, 4097}, {4, 4, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3),
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, EmptyTensorFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{0, 4, 4}, {0, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{0, 4, 4}, {0, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, LeadingBatchExceedsLimitFails)
{
    // prod(N, C) = 201 * 64 = 12864 > 200 * 64.
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para(
        "BilateralFilter3D", {{{{201, 64, 4, 4, 4}, {201, 64, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{201, 64, 4, 4, 4}, {201, 64, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// kernel_size must be one of 3/5/7/9.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DTiling, KernelSizeOneFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(1), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, KernelSizeEvenFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(4), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, KernelSizeTooLargeFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(11), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, KernelSizeNegativeFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(-3), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// sigma_space / sigma_color must be positive and finite.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DTiling, SigmaSpaceZeroFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 0.0F, 1.0F),
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, SigmaSpaceNegativeFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, -1.0F, 1.0F),
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, SigmaSpaceInfiniteFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 MakeAttrs(3, std::numeric_limits<float>::infinity(), 1.0F), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, SigmaColorZeroFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, 0.0F),
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DTiling, SigmaColorNanFails)
{
    optiling::BilateralFilter3DCompileInfo compileInfo{AIV_CORE_NUM, UB_SIZE};
    gert::TilingContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 MakeAttrs(3, 1.0F, std::numeric_limits<float>::quiet_NaN()), &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------
// Accepted configurations.
// ---------------------------------------------------------------------------

// fp32 selects tiling key 1; totalRows = 32 * 32 = 1024 exceeds any plausible core
// count, so the launch is core-bound.
TEST_F(BilateralFilter3DTiling, Fp32Rank3UsesFp32Key)
{
    ExpectTilingSuccess({32, 32, 32}, ge::DT_FLOAT, 5, 1.0F, 1.0F, optiling::BF3D_TILING_KEY_FP32,
                        BLOCK_NUM_CORE_BOUND);
}

// fp16 selects tiling key 0; leading dims collapse into nVolume = 2 * 3 = 6, so
// totalRows = 6 * 16 * 20 = 1920, still above the core count.
TEST_F(BilateralFilter3DTiling, Fp16Rank5UsesFp16Key)
{
    ExpectTilingSuccess({2, 3, 16, 20, 24}, ge::DT_FLOAT16, 9, 1.2F, 2.0F, optiling::BF3D_TILING_KEY_FP16,
                        BLOCK_NUM_CORE_BOUND);
}

// totalRows = 2 * 3 = 6, fewer rows than cores: usedCoreNum (and so the block dim) must
// be clamped down to the row count, otherwise idle cores would re-walk the rows.
TEST_F(BilateralFilter3DTiling, FewerRowsThanCoresClampsBlockDim)
{
    ExpectTilingSuccess({2, 3, 8}, ge::DT_FLOAT, 3, 1.0F, 1.0F, optiling::BF3D_TILING_KEY_FP32, 6U);
}

// 0.5/sigma^2 overflows fp32 for sigma = 1e-20; the tiling function saturates the
// reciprocal at 1e30 rather than rejecting the attribute.
TEST_F(BilateralFilter3DTiling, TinySigmaClampedStillSucceeds)
{
    ExpectTilingSuccess({8, 8, 8}, ge::DT_FLOAT, 5, 1e-20F, 1e-20F, optiling::BF3D_TILING_KEY_FP32,
                        BLOCK_NUM_CORE_BOUND);
}

// W smaller than one aligned chunk: chunkLen must clamp to 8 rather than to W, and
// the UB derivation must still accept the shape.
TEST_F(BilateralFilter3DTiling, WidthBelowOneChunkSucceeds)
{
    // totalRows = 4 * 5 = 20, well under any 910B core count, so the block dim is
    // row-bound and can be pinned exactly.
    ExpectTilingSuccess({4, 5, 3}, ge::DT_FLOAT, 9, 1.0F, 1.0F, optiling::BF3D_TILING_KEY_FP32, 20U);
}

// W = 4096 is the documented maximum and must stay inside the UB budget.
// totalRows = 2 * 2 = 4 here, so the block dim is row-bound rather than core-bound.
TEST_F(BilateralFilter3DTiling, MaxWidthSucceeds)
{
    ExpectTilingSuccess({2, 2, 4096}, ge::DT_FLOAT, 9, 1.0F, 1.0F, optiling::BF3D_TILING_KEY_FP32, 4U);
}
