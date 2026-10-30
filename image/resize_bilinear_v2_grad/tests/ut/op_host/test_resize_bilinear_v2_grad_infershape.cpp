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
 * \file test_resize_bilinear_v2_grad_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ResizeBilinearV2GradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeBilinearV2GradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeBilinearV2GradInfershape TearDown" << std::endl; }
};

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_infer_test_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{3, 9, 9, 5}, {3, 9, 9, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 9, 9, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_infer_test_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, -1, -1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_infer_test_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{3, 5, 8, 6}, {3, 5, 8, 6}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{3, 5, 16, 12}, {3, 5, 16, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, 16, 12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_rank_three_rejected)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{3, 9, 5}, {3, 9, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{3, 9, 5}, {3, 9, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_rank_three_grad_with_unknown_rank_image_rejected)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{3, 9, 5}, {3, 9, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_unknown_rank_grad_with_rank_three_image_rejected)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{3, 9, 5}, {3, 9, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ResizeBilinearV2GradInfershape, resize_bilinear_v2_grad_unknown_rank_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f, 0.0f})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
