/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scharr_infershape.cpp
 * \brief Scharr InferShape UT：输出 shape 与输入一致（支持 ND 任意前置批次维）。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace Ops::Cv;

class ScharrInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScharrInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScharrInfershape TearDown" << std::endl; }
};

TEST_F(ScharrInfershape, scharr_infershape_2d)
{
    gert::InfershapeContextPara para(
        "Scharr", {{{{65, 64}, {65, 64}}, ge::DT_UINT8, ge::FORMAT_ND}}, {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("direction", Ops::Cv::AnyValue::CreateFrom<std::string>("X")),
         gert::InfershapeContextPara::OpAttr("border_type",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("REFLECT_101"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{65, 64}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScharrInfershape, scharr_infershape_4d_batch)
{
    gert::InfershapeContextPara para(
        "Scharr", {{{{2, 3, 63, 65}, {2, 3, 63, 65}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("direction", Ops::Cv::AnyValue::CreateFrom<std::string>("Y")),
         gert::InfershapeContextPara::OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<std::string>("REPLICATE"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 63, 65}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScharrInfershape, scharr_infershape_8d)
{
    gert::InfershapeContextPara para(
        "Scharr", {{{{1, 2, 1, 3, 1, 1, 33, 45}, {1, 2, 1, 3, 1, 1, 33, 45}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("direction", Ops::Cv::AnyValue::CreateFrom<std::string>("X")),
         gert::InfershapeContextPara::OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<std::string>("CONSTANT"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 1, 3, 1, 1, 33, 45}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScharrInfershape, scharr_infershape_w1)
{
    gert::InfershapeContextPara para(
        "Scharr", {{{{16, 1}, {16, 1}}, ge::DT_UINT8, ge::FORMAT_ND}}, {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("direction", Ops::Cv::AnyValue::CreateFrom<std::string>("X")),
         gert::InfershapeContextPara::OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<std::string>("CONSTANT"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
