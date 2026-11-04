/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bilinear2d_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class UpsampleBilinear2dTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpsampleBilinear2dTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpsampleBilinear2dTiling TearDown" << std::endl; }
};

TEST_F(UpsampleBilinear2dTiling, upsample_bilinear2d_tiling_001)
{
    optiling::UpsampleBilinear2dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2d",
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2, 3}))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "0 64 64 1 4611686019505324032 0 274877915136 0 0 0 0 1 64 0 0 0 0 1 64 0 1 1 1 128 1 1 "
                              "0 0 4294967297 549755813888 4294967424 549755813952 274877906960 4294967424 4294967297 "
                              "1 0 87960930222080 4096 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33570816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dTiling, upsample_bilinear2d_scales_empty_uses_output_size)
{
    optiling::UpsampleBilinear2dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2d",
        {
            {{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1);
    EXPECT_GT(tilingInfo.blockNum, 0);
}

TEST_F(UpsampleBilinear2dTiling, upsample_bilinear2d_scales_one_element_rejected)
{
    optiling::UpsampleBilinear2dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2d",
        {
            {{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(UpsampleBilinear2dTiling, upsample_bilinear2d_tiling_002)
{
    optiling::UpsampleBilinear2dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2d",
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1, 1}))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "0 128 128 2 4575657222473777152 512 549755813888 4294967296 0 0 0 0 0 0 1 1 1 1 128 0 1 "
                              "1 1 128 1 1 1 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 "
                              "549755814016 549755814016 4294967424 549755814016 4294967304 4294967297 1 0 "
                              "35184372088832 65536 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33555456};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBilinear2dTiling, upsample_bilinear2d_tiling_003)
{
    optiling::UpsampleBilinear2dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2d",
        {
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{3}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 3, 256}, {1, 1, 3, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({3, 2}))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 128 128 2 1056898815 1024 549755823232 4294967297 2 2 2 1 128 0 1 1 1 1 128 0 1 1 1 "
                              "128 1 1 3 256 4294967297 549755814144 4294967424 313532612736 549755813904 8589934656 "
                              "4294967298 1 0 316659348799488 8192 4294967297 4294967297 8589934594 0 8589934594 1 0 0 "
                              "0 0 0 0 0 0 12884901889 549755814144 549755814016 4294967552 1099511627904 4294967304 "
                              "4294967297 1 0 52776558133248 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 "
                              "0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33592832};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Explicit positive scales must remain effective for same-size axes, including values close to one.
// The L0 wrapper converts user scales to the coordinate scale before this tiling entry point.
TEST_F(UpsampleBilinear2dTiling, upsample_bilinear2d_same_size_keeps_explicit_near_one_scale)
{
    optiling::UpsampleBilinear2dCompileInfo compileInfo = {1};
    gert::TilingContextPara tilingContextPara(
        "UpsampleBilinear2d",
        {
            {{{1, 1, 2, 2}, {1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 2, 2}, {1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.001f, 1.001f}))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    constexpr size_t kScaleOffset = sizeof(uint64_t) * 4;
    const float expectedScale = 1.001f;
    uint32_t expectedScaleBits = 0;
    std::memcpy(&expectedScaleBits, &expectedScale, sizeof(expectedScaleBits));
    uint32_t scaleWBits = 0;
    uint32_t scaleHBits = 0;
    std::memcpy(&scaleWBits, tilingInfo.tilingData.get() + kScaleOffset, sizeof(scaleWBits));
    std::memcpy(&scaleHBits, tilingInfo.tilingData.get() + kScaleOffset + sizeof(scaleWBits), sizeof(scaleHBits));
    EXPECT_EQ(scaleWBits, expectedScaleBits);
    EXPECT_EQ(scaleHBits, expectedScaleBits);
}
