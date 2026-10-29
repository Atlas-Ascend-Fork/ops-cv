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
 * \file test_pixel_shuffle_infershape.cpp
 * \brief PixelShuffle InferShape UT：校验 4D/5D 输出 shape 与非法输入报错
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class PixelShuffleInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PixelShuffleInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PixelShuffleInfershape TearDown" << std::endl; }
};

// 4D fp32 r=2: [2, 16, 8, 8] -> [2, 4, 16, 16]
TEST_F(PixelShuffleInfershape, infershape_4d_r2_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 4, 16, 16}});
}

// 4D fp16 r=3: [1, 18, 5, 7] -> [1, 2, 15, 21]
TEST_F(PixelShuffleInfershape, infershape_4d_r3_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{1, 18, 5, 7}, {1, 18, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(3))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{1, 2, 15, 21}});
}

// 4D fp16 r=4: [2, 32, 3, 3] -> [2, 2, 12, 12]
TEST_F(PixelShuffleInfershape, infershape_4d_r4_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 32, 3, 3}, {2, 32, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(4))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 2, 12, 12}});
}

// 5D fp32 r=2: [2, 32, 3, 4, 5] -> [2, 4, 6, 8, 10]
TEST_F(PixelShuffleInfershape, infershape_5d_r2_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 32, 3, 4, 5}, {2, 32, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 4, 6, 8, 10}});
}

// 非法 rank=3 -> GRAPH_FAILED
TEST_F(PixelShuffleInfershape, infershape_invalid_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 非法 scale_factor=1 -> GRAPH_FAILED
TEST_F(PixelShuffleInfershape, infershape_invalid_scale)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 非法通道不可被 r^2 整除 (C=15, r=2) -> GRAPH_FAILED
TEST_F(PixelShuffleInfershape, infershape_channel_not_divisible)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 15, 8, 8}, {2, 15, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 非法 scale_factor=0 -> GRAPH_FAILED（下界校验）
TEST_F(PixelShuffleInfershape, infershape_invalid_scale_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 非法 scale_factor 超上界 -> GRAPH_FAILED（上界校验）
// r=2^22 走 5D 时 r^3 = 2^66，修复前 int64 累乘溢出为 0，第 69 行 "C_in % rPow" 整数除零崩溃
TEST_F(PixelShuffleInfershape, infershape_scale_overflow_bound)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{1, 32, 2, 4, 4}, {1, 32, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(4194304LL))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 大宽度 4K: [1, 4, 1080, 3840] r=2 -> [1, 1, 2160, 7680]
TEST_F(PixelShuffleInfershape, infershape_large_width_4k)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{1, 4, 1080, 3840}, {1, 4, 1080, 3840}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{1, 1, 2160, 7680}});
}

// 动态 shape 4D：C=-1 未知。修复前 "-1 % 4 != 0" 会误判为非法
// [4, -1, 8, 8] r=2 -> [4, -1, 16, 16]（未知通道保持 -1，空间维正常放大）
TEST_F(PixelShuffleInfershape, infershape_4d_dynamic_channel)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{4, -1, 8, 8}, {4, -1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{4, -1, 16, 16}});
}

// 动态 shape 4D：N/C/H/W 全未知 [ -1, -1, -1, -1] r=4 -> 全部保持 -1
TEST_F(PixelShuffleInfershape, infershape_4d_dynamic_all_dims)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{-1, -1, -1, -1}, {-1, -1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(4))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{-1, -1, -1, -1}});
}

// 动态 shape 5D：末两维未知 [2, 32, 3, -1, -1] r=2 -> [2, 4, 6, -1, -1]
// 修复前 H=-1 会算成 -2（-1 * 2），输出维度写错
TEST_F(PixelShuffleInfershape, infershape_5d_dynamic_spatial)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{2, 32, 3, -1, -1}, {2, 32, 3, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 4, 6, -1, -1}});
}

// 非法畸形 shape：dim < -1。C=-4、H=-3 相乘为正，可穿透整除与长度校验（(1,-4,-3,2) r=2）
// => 必须由“仅允许 -1”的维度校验拒绝
TEST_F(PixelShuffleInfershape, infershape_invalid_negative_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PixelShuffle", {{{{1, -4, -3, 2}, {1, -4, -3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
