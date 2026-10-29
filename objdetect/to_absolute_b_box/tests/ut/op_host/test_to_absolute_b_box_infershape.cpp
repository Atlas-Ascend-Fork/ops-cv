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
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ToAbsoluteBBoxInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ToAbsoluteBBoxInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ToAbsoluteBBoxInfershape TearDown" << std::endl; }
};

TEST_F(ToAbsoluteBBoxInfershape, tabb_fp32_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ToAbsoluteBBox",
        {
            {{{2, 8, 4}, {2, 8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ToAbsoluteBBoxInfershape, tabb_fp16_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ToAbsoluteBBox",
        {
            {{{1, 16, 4}, {1, 16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 16, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ToAbsoluteBBoxInfershape, tabb_fp32_reversed_box)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ToAbsoluteBBox",
        {
            {{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ToAbsoluteBBoxInfershape, tabb_fp32_empty_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ToAbsoluteBBox",
        {
            {{{0, 8, 4}, {0, 8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("reversed_box", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
