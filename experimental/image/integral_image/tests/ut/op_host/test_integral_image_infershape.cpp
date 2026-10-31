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
 * \file test_integral_image_infershape.cpp
 * \brief IntegralImage Infershape UT
 *
 * 覆盖点：
 *   - rank2 (H,W) -> (H+1,W+1)
 *   - rank3 HWC (H,W,C) -> (H+1,W+1,C)，C 维保持不变
 *   - 动态维度 (-1) 保持未知，不执行 +1
 *   - 可选输出 sqsum/tilted 传 nullptr（形态 1/2）时跳过其 shape 设置且不报错
 *   - rank 非法（1/4）返回失败
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class IntegralImageInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IntegralImageInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IntegralImageInfershape TearDown" << std::endl; }
};

// rank2 (8,64) -> (9,65)，UINT8 输入对应 INT32 sat
TEST_F(IntegralImageInfershape, integral_image_infershape_rank2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IntegralImage",
        {
            {{{8, 64}, {8, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(-1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{9, 65}, {9, 65}, {9, 65}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank3 HWC (4,6,3) -> (5,7,3)，通道维保持不变
TEST_F(IntegralImageInfershape, integral_image_infershape_rank3_hwc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IntegralImage",
        {
            {{{4, 6, 3}, {4, 6, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(-1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 7, 3}, {5, 7, 3}, {5, 7, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank3 HWC + sdepth=1：Shape 推导保持 H/W 各加 1、C 不变
TEST_F(IntegralImageInfershape, integral_image_infershape_rank3_sdepth_f32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IntegralImage",
        {
            {{{1, 1, 2}, {1, 1, 2}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 Shape：(-1,-1) 保持未知，不执行 +1
TEST_F(IntegralImageInfershape, integral_image_infershape_dynamic_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IntegralImage",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(-1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank=1 非法
TEST_F(IntegralImageInfershape, integral_image_infershape_rank1_failed)
{
    gert::InfershapeContextPara infershapeContextPara("IntegralImage",
                                                      {
                                                          {{{64}, {64}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// rank=4 非法
TEST_F(IntegralImageInfershape, integral_image_infershape_rank4_failed)
{
    gert::InfershapeContextPara infershapeContextPara("IntegralImage",
                                                      {
                                                          {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 形态 1：只声明 sat，sqsum/tilted 不实例化（GetOutputShape(1)/(2) 返回 nullptr）。
// 仅推导已提供输出的 Shape。
TEST_F(IntegralImageInfershape, integral_image_infershape_optional_sat_only)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IntegralImage",
        {
            {{{8, 64}, {8, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(-1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{9, 65}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 形态 2：声明 sat + sqsum，tilted 不实例化（GetOutputShape(2) 返回 nullptr）。
// 推导 sat 与 sqsum，跳过缺省的 tilted。
TEST_F(IntegralImageInfershape, integral_image_infershape_optional_no_tilted)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IntegralImage",
        {
            {{{4, 6, 3}, {4, 6, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(-1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 7, 3}, {5, 7, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
