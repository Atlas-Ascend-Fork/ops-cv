/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rotated_feature_align_grad_tiling.cpp
 * \brief UT for rotated_feature_align_grad tiling
 */
#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

class TilingForRotatedFeatureAlignGrad : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TilingForRotatedFeatureAlignGrad SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TilingForRotatedFeatureAlignGrad TearDown" << std::endl; }
};

TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_pts1)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    std::vector<size_t> expectWorkspaces = {16777216};
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaces.size());
    for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
        EXPECT_EQ(tilingInfo.workspaceSizes[i], expectWorkspaces[i]);
    }
}

TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_pts5)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{1, 8, 8, 16}, {1, 8, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 8, 5}, {1, 8, 8, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 8, 8, 16}, {1, 8, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(5))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), 1);
}

TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_empty_shape)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{0, 4, 4, 3}, {0, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 4, 4, 5}, {0, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{0, 4, 4, 3}, {0, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
}

// dy dtype 非 float32：触发 OP_LOGE_FOR_INVALID_DTYPE，tiling 返回失败
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_dtype)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// dy 非 4D：触发 OP_LOGE_FOR_INVALID_SHAPEDIM，tiling 返回失败
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_dy_shapedim)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// bboxes 末维非 5：触发 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON，tiling 返回失败
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_bboxes_last_dim)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 6}, {2, 4, 4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// dy/bboxes 的 N 维不一致：触发 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON，tiling 返回失败
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_dim_mismatch)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3, 4, 4, 5}, {3, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// points 非 1/5：触发 OP_LOGE_FOR_INVALID_VALUE，tiling 返回失败
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_points)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4, 4, 5}, {2, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(3))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// 单维超过 INT32_MAX：触发 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON，tiling 返回失败
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_dim_exceed_int32)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{2147483648LL, 1, 1, 3}, {2147483648LL, 1, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2147483648LL, 1, 1, 5}, {2147483648LL, 1, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2147483648LL, 1, 1, 3}, {2147483648LL, 1, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// 每一维都可表示为 int32，但 N*H*W 不可表示为 int64 时必须拒绝。
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_nhw_overflow)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    constexpr int64_t kMaxDim = 2147483647LL;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{kMaxDim, kMaxDim, 3, 1}, {kMaxDim, kMaxDim, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{kMaxDim, kMaxDim, 3, 5}, {kMaxDim, kMaxDim, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{kMaxDim, kMaxDim, 3, 1}, {kMaxDim, kMaxDim, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// N*H*W 可表示为 int64，但元素总数 N*H*W*C / N*H*W*5 超过 int64：Kernel 侧
// int64 地址偏移（inIdx*C、inIdx*5、dxBase）会溢出，tiling 必须拒绝。
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_total_elements_overflow)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    constexpr int64_t kMaxDim = 2147483647LL;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{kMaxDim, kMaxDim, 1, 3}, {kMaxDim, kMaxDim, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{kMaxDim, kMaxDim, 1, 5}, {kMaxDim, kMaxDim, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{kMaxDim, kMaxDim, 1, 3}, {kMaxDim, kMaxDim, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// 维度为负（如未知维哨兵 -1）：不是合法的具体形状，参与分块算术前 tiling 必须拒绝。
TEST_F(TilingForRotatedFeatureAlignGrad, rotated_feature_align_grad_tiling_fail_negative_dim)
{
    struct RotatedFeatureAlignGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "RotatedFeatureAlignGrad",
        {{{{-1, 4, 4, 3}, {-1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{-1, 4, 4, 5}, {-1, 4, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{-1, 4, 4, 3}, {-1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25f)),
         gert::TilingContextPara::OpAttr("points", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        {1, 1}, {1}, &compileInfo, "Ascend950", 64, 262144, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
