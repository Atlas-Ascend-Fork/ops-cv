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

class BoundingBoxEncodeInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BoundingBoxEncodeInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BoundingBoxEncodeInfershape TearDown" << std::endl; }
};

TEST_F(BoundingBoxEncodeInfershape, bbe_fp32_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BoundingBoxEncode",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f, 0.0f, 0.0f})),
            gert::InfershapeContextPara::OpAttr(
                "stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BoundingBoxEncodeInfershape, bbe_fp16_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BoundingBoxEncode",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f, 0.0f, 0.0f})),
            gert::InfershapeContextPara::OpAttr(
                "stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BoundingBoxEncodeInfershape, bbe_fp32_empty)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BoundingBoxEncode",
        {
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f, 0.0f, 0.0f})),
            gert::InfershapeContextPara::OpAttr(
                "stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BoundingBoxEncodeInfershape, bbe_rejects_mismatched_shapes)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BoundingBoxEncode",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f, 0.0f, 0.0f})),
            gert::InfershapeContextPara::OpAttr(
                "stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(BoundingBoxEncodeInfershape, bbe_rejects_invalid_last_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BoundingBoxEncode",
        {
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f, 0.0f, 0.0f})),
            gert::InfershapeContextPara::OpAttr(
                "stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(BoundingBoxEncodeInfershape, bbe_rejects_mismatched_dtype)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BoundingBoxEncode",
        {
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr(
                "means", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f, 0.0f, 0.0f})),
            gert::InfershapeContextPara::OpAttr(
                "stds", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f, 1.0f, 1.0f, 1.0f})),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
