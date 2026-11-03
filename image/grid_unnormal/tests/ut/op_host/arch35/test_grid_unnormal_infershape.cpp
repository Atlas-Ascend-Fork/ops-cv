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
#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "op_infer_shape_range_context_builder.h"
#include "any_value.h"

class GridUnnormalArch35Infershape : public testing::Test {};

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_infershape_rejects_assist_rank_when_grid_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},           // grid
            {{{1, 6, 5}, {1, 6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        attrs);
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_infershape_rejects_scalar_assist_when_grid_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara("GridUnnormal",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // assist
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      attrs);
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_infershape_rejects_rank1_assist_when_grid_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara("GridUnnormal",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
                                                          {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      attrs);
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_infershape_allows_4d_assist_when_grid_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // grid
            {{{-1, 6, 5, 2}, {-1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_infershape_does_not_restrict_assist_last_dim_when_grid_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // grid
            {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_inferdatatype_supported)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferDataType = registry->GetOpImpl("GridUnnormal")->infer_datatype;
    ASSERT_NE(inferDataType, nullptr);

    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("GridUnnormal")
                      .OpName("GridUnnormal")
                      .IONum(2, 2)
                      .InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataType(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_inferdatatype_rejects_unsupported_dtype)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferDataType = registry->GetOpImpl("GridUnnormal")->infer_datatype;
    ASSERT_NE(inferDataType, nullptr);

    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("GridUnnormal")
                      .OpName("GridUnnormal")
                      .IONum(2, 2)
                      .InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataType(context), ge::GRAPH_FAILED);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_inferdatatype_rejects_mismatched_dtype)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferDataType = registry->GetOpImpl("GridUnnormal")->infer_datatype;
    ASSERT_NE(inferDataType, nullptr);

    auto holder = gert::OpInferDataTypeContextBuilder()
                      .OpType("GridUnnormal")
                      .OpName("GridUnnormal")
                      .IONum(2, 2)
                      .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataType(context), ge::GRAPH_FAILED);
}

TEST_F(GridUnnormalArch35Infershape, grid_unnormal_infershape_range)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferShapeRange = registry->GetOpImpl("GridUnnormal")->infer_shape_range;
    ASSERT_NE(inferShapeRange, nullptr);

    gert::Tensor gridMin({{1, 1, 1, 2}, {1, 1, 1, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_FLOAT,
                         nullptr);
    gert::Tensor gridMax({{8, 32, 64, 2}, {8, 32, 64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost,
                         ge::DT_FLOAT, nullptr);
    gert::Tensor assistMin({{1, 1, 1, 2}, {1, 1, 1, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost,
                           ge::DT_FLOAT, nullptr);
    gert::Tensor assistMax({{8, 32, 64, 2}, {8, 32, 64, 2}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost,
                           ge::DT_FLOAT, nullptr);
    gert::Range<gert::Tensor> gridRange(&gridMin, &gridMax);
    gert::Range<gert::Tensor> assistRange(&assistMin, &assistMax);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("GridUnnormal").OpName("GridUnnormal").IONum(2, 2);
    builder.InputTensorsRange({&gridRange, &assistRange});
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeRange(context), ge::GRAPH_SUCCESS);

    for (size_t outputIndex = 0; outputIndex < 2; ++outputIndex) {
        const auto* outputRange = context->GetOutputShapeRange(outputIndex);
        ASSERT_NE(outputRange, nullptr);
        EXPECT_EQ(*outputRange->GetMin(), gridMin.GetOriginShape());
        EXPECT_EQ(*outputRange->GetMax(), gridMax.GetOriginShape());
    }
}
