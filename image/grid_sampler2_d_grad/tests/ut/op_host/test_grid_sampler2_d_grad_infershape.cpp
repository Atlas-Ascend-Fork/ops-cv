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
 * \file test_grid_sampler2d_grad.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class grid_sampler2d_grad : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "grid_sampler2d_grad SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "grid_sampler2d_grad TearDown" << std::endl; }
};

class GridSampler2DGradMissingShapes : public testing::TestWithParam<int> {};

TEST_P(GridSampler2DGradMissingShapes, rejects_missing_context_or_shape)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("GridSampler2DGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    const int scenario = GetParam();
    SCOPED_TRACE(scenario);
    if (scenario == 0) {
        EXPECT_EQ(impl->infer_shape(nullptr), ge::GRAPH_FAILED);
        return;
    }

    gert::Tensor grad({{2, 3, 8, 8}, {2, 3, 8, 8}},
                      gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                      gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor x({{2, 3, 16, 16}, {2, 3, 16, 16}},
                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                   gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor grid({{2, 8, 8, 2}, {2, 8, 8, 2}},
                      gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                      gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GridSampler2DGrad")
                      .NodeIoNum(3, 2)
                      .InputTensors({&grad, &x, &grid})
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(0), nullptr);
    ASSERT_NE(context->GetInputShape(1), nullptr);
    ASSERT_NE(context->GetInputShape(2), nullptr);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    ASSERT_NE(context->GetOutputShape(1), nullptr);

    // Clear one framework-owned slot after constructing an otherwise valid context.
    auto kernelContext = reinterpret_cast<gert::KernelContext*>(context);
    if (scenario <= 3) {
        auto input = kernelContext->MutableInput(scenario - 1);
        ASSERT_NE(input, nullptr);
        input->Set(nullptr, nullptr);
        ASSERT_EQ(context->GetInputShape(scenario - 1), nullptr);
    } else {
        auto output = kernelContext->GetOutput(scenario - 4);
        ASSERT_NE(output, nullptr);
        output->Set(nullptr, nullptr);
        ASSERT_EQ(context->GetOutputShape(scenario - 4), nullptr);
    }
    EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(MissingFrameworkInputs, GridSampler2DGradMissingShapes, testing::Range(0, 6));

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fp16)
{
    int n = 3;
    int c = 4;
    int h_in = 4;
    int w_in = 5;
    int h_out = 5;
    int w_out = 6;

    gert::InfershapeContextPara infershapeContextPara(
        "GridSampler2DGrad",
        {
            {{{n, c, h_out, w_out}, {n, c, h_out, w_out}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, c, h_in, w_in}, {n, c, h_in, w_in}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, h_out, w_out, 2}, {n, h_out, w_out, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("interpolation_mode",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
         gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
         gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {n, c, h_in, w_in},
        {n, h_out, w_out, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fp32)
{
    int n = 13;
    int c = 24;
    int h_in = 14;
    int w_in = 25;
    int h_out = 15;
    int w_out = 26;

    gert::InfershapeContextPara infershapeContextPara(
        "GridSampler2DGrad",
        {
            {{{n, c, h_out, w_out}, {n, c, h_out, w_out}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, c, h_in, w_in}, {n, c, h_in, w_in}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, h_out, w_out, 2}, {n, h_out, w_out, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("interpolation_mode",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
         gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
         gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {n, c, h_in, w_in},
        {n, h_out, w_out, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_double)
{
    int n = 13;
    int c = 24;
    int h_in = 14;
    int w_in = 25;
    int h_out = 15;
    int w_out = 26;

    gert::InfershapeContextPara infershapeContextPara(
        "GridSampler2DGrad",
        {
            {{{n, c, h_out, w_out}, {n, c, h_out, w_out}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{n, c, h_in, w_in}, {n, c, h_in, w_in}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{n, h_out, w_out, 2}, {n, h_out, w_out, 2}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("interpolation_mode",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
         gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
         gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {n, c, h_in, w_in},
        {n, h_out, w_out, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fail_x)
{
    int n = 1;
    int c = 2;
    int h_in = 4;
    int w_in = 25;
    int h_out = 15;
    int w_out = 16;

    gert::InfershapeContextPara infershapeContextPara(
        "GridSampler2DGrad",
        {
            {{{n, c, h_out, w_out}, {n, c, h_out, w_out}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, c, h_in}, {n, c, h_in}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{n, h_out, w_out, 2}, {n, h_out, w_out, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("interpolation_mode",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
         gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
         gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fail_grad)
{
    int n = 13;
    int c = 24;
    int h_in = 54;
    int w_in = 25;
    int h_out = 15;
    int w_out = 16;

    gert::InfershapeContextPara infershapeContextPara(
        "GridSampler2DGrad",
        {
            {{{n, h_out, w_out}, {n, h_out, w_out}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{n, c, h_in, w_in}, {n, c, h_in, w_in}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{n, h_out, w_out, 2}, {n, h_out, w_out, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("interpolation_mode",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
         gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
         gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(grid_sampler2d_grad, grid_sampler2d_grad_infershape_fail_grid)
{
    int n = 13;
    int c = 24;
    int h_in = 54;
    int w_in = 25;
    int h_out = 15;
    int w_out = 16;

    gert::InfershapeContextPara infershapeContextPara(
        "GridSampler2DGrad",
        {
            {{{n, c, h_out, w_out}, {n, c, h_out, w_out}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{n, c, h_in, w_in}, {n, c, h_in, w_in}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{n, h_out, w_out}, {n, h_out, w_out}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("interpolation_mode",
                                             Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
         gert::InfershapeContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("zeros")),
         gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
