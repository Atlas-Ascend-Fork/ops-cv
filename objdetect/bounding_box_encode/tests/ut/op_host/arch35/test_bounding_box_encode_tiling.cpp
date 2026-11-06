/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/bounding_box_encode_tiling_data.h"

namespace {

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct BoundingBoxEncodeCompileInfoForTest {};

BoundingBoxEncodeCompileInfoForTest g_compileInfo;

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

std::vector<OpAttr> MakeAttrs(const std::vector<float>& means, const std::vector<float>& stds)
{
    return {
        OpAttr("means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>(means)),
        OpAttr("stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>(stds)),
    };
}

std::vector<OpAttr> DefaultAttrs() { return MakeAttrs({0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}); }

gert::TilingContextPara MakeContext(const std::vector<int64_t>& anchorShape, const std::vector<int64_t>& gtShape,
                                    ge::DataType anchorDtype, ge::DataType gtDtype, const std::vector<OpAttr>& attrs,
                                    uint64_t coreNum = 64, uint64_t ubSize = 262144)
{
    std::vector<TensorDesc> inputs = {
        {MakeShape(anchorShape), anchorDtype, ge::FORMAT_ND},
        {MakeShape(gtShape), gtDtype, ge::FORMAT_ND},
    };
    std::vector<TensorDesc> outputs = {
        {MakeShape(anchorShape), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    return gert::TilingContextPara("BoundingBoxEncode", inputs, outputs, attrs, &g_compileInfo, "Ascend950", coreNum,
                                   ubSize);
}

bool RunTiling(const gert::TilingContextPara& context, TilingInfo& info, BoundingBoxEncodeTilingData& tilingData)
{
    if (!ExecuteTiling(context, info) || info.tilingDataSize < sizeof(tilingData)) {
        return false;
    }
    std::memcpy(&tilingData, info.tilingData.get(), sizeof(tilingData));
    return true;
}

void ExpectTilingFailed(const gert::TilingContextPara& context)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

} // namespace

class BoundingBoxEncodeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BoundingBoxEncodeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BoundingBoxEncodeTiling TearDown" << std::endl; }
};

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp32_small)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 8);
    EXPECT_EQ(td.blockBoxFactor, 64);
    EXPECT_EQ(td.ubBoxFactor, 64);
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp16_small)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 8);
    EXPECT_EQ(td.blockBoxFactor, 128);
    EXPECT_EQ(td.ubBoxFactor, 128);
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp32_nonaligned)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({33, 4}, {33, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 33);
    EXPECT_EQ(td.blockBoxFactor, 64);
    EXPECT_EQ(td.ubBoxFactor, 64);
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp16_nonaligned)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({33, 4}, {33, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 33);
    EXPECT_EQ(td.blockBoxFactor, 128);
    EXPECT_EQ(td.ubBoxFactor, 128);
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp32_multi_core)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({100000, 4}, {100000, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 100000);
    EXPECT_EQ(td.blockBoxFactor, 1600);
    EXPECT_EQ(td.ubBoxFactor, 1600);
    EXPECT_EQ(info.blockNum, 63U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp16_multi_core)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(
        RunTiling(MakeContext({100000, 4}, {100000, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 100000);
    EXPECT_EQ(td.blockBoxFactor, 1664);
    EXPECT_EQ(td.ubBoxFactor, 1664);
    EXPECT_EQ(info.blockNum, 61U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_fp32_ub_limited)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(
        RunTiling(MakeContext({1000000, 4}, {1000000, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 1000000);
    EXPECT_EQ(td.blockBoxFactor, 15680);
    EXPECT_EQ(td.ubBoxFactor, 2688);
    EXPECT_EQ(info.blockNum, 64U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_small_ub_success)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs(), 64, 8192), info, td));
    EXPECT_EQ(td.boxNum, 8);
    EXPECT_EQ(td.blockBoxFactor, 64);
    EXPECT_EQ(td.ubBoxFactor, 64);
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_empty)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({0, 4}, {0, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()), info, td));
    EXPECT_EQ(td.boxNum, 0);
    EXPECT_EQ(td.blockBoxFactor, 0);
    EXPECT_EQ(td.ubBoxFactor, 0);
    EXPECT_EQ(info.blockNum, 1U);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_custom_attrs)
{
    TilingInfo info;
    BoundingBoxEncodeTilingData td{};
    ASSERT_TRUE(RunTiling(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT,
                                      MakeAttrs({1.5f, -2.0f, 0.25f, 4.0f}, {2.0f, 4.0f, 0.5f, 8.0f})),
                          info, td));
    EXPECT_FLOAT_EQ(td.means0, 1.5f);
    EXPECT_FLOAT_EQ(td.means1, -2.0f);
    EXPECT_FLOAT_EQ(td.means2, 0.25f);
    EXPECT_FLOAT_EQ(td.means3, 4.0f);
    EXPECT_FLOAT_EQ(td.invStds0, 0.5f);
    EXPECT_FLOAT_EQ(td.invStds1, 0.25f);
    EXPECT_FLOAT_EQ(td.invStds2, 2.0f);
    EXPECT_FLOAT_EQ(td.invStds3, 0.125f);
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_invalid_dtype)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_INT32, ge::DT_INT32, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_dtype_mismatch)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT16, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_anchor_rank)
{
    ExpectTilingFailed(MakeContext({32}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_gt_rank)
{
    ExpectTilingFailed(MakeContext({8, 4}, {32}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_anchor_last_dim)
{
    ExpectTilingFailed(MakeContext({8, 3}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_gt_last_dim)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 3}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_shape_mismatch)
{
    ExpectTilingFailed(MakeContext({8, 4}, {4, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs()));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_missing_attrs)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, {}));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_means_length)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT,
                                   MakeAttrs({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f})));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_stds_length)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT,
                                   MakeAttrs({0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f})));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_zero_stds)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT,
                                   MakeAttrs({0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f})));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_zero_core_num)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs(), 0, 262144));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_zero_ub)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs(), 64, 0));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_ub_reserved)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs(), 64, 256));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_ub_one_box)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs(), 64, 300));
}

TEST_F(BoundingBoxEncodeTiling, bbe_tiling_ub_aligned_tile)
{
    ExpectTilingFailed(MakeContext({8, 4}, {8, 4}, ge::DT_FLOAT, ge::DT_FLOAT, DefaultAttrs(), 64, 400));
}
