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
 * \file test_non_max_suppression_v3_infershape.cpp
 * \brief
 */

// --------------- NonMaxSuppressionV3 Op start-------------------

#include <gtest/gtest.h>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"
#include "op_infer_shape_range_context_builder.h"

namespace {
constexpr size_t kInputNum = 5U;
constexpr size_t kOutputNum = 1U;
constexpr size_t kMaxOutputSizeIndex = 2U;
constexpr size_t kIouThresholdIndex = 3U;
constexpr size_t kScoreThresholdIndex = 4U;
constexpr size_t kSelectedIndicesIndex = 0U;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

void ExpectShape(const gert::StorageShape& boxesShape, const gert::StorageShape& scoresShape,
                 ge::graphStatus expectedStatus, const std::vector<std::vector<int64_t>>& expectedShapes = {},
                 ge::DataType boxesDtype = ge::DT_FLOAT, ge::DataType scoresDtype = ge::DT_FLOAT)
{
    gert::InfershapeContextPara context("NonMaxSuppressionV3",
                                        {{boxesShape, boxesDtype, ge::FORMAT_ND},
                                         {scoresShape, scoresDtype, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    ExecuteTestCase(context, expectedStatus, expectedShapes);
}

void ExpectMissingInput(size_t missingIndex)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    inputs.erase(inputs.begin() + missingIndex, inputs.end());
    gert::InfershapeContextPara context("NonMaxSuppressionV3", inputs, {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

void ExpectControlShapes(const std::vector<int64_t>& maxOutputSizeShape, const std::vector<int64_t>& iouThresholdShape,
                         const std::vector<int64_t>& scoreThresholdShape, ge::graphStatus expectedStatus)
{
    const auto boxesShape = MakeStorageShape({8, 4});
    const auto scoresShape = MakeStorageShape({8});
    const auto maxShape = MakeStorageShape(maxOutputSizeShape);
    const auto iouShape = MakeStorageShape(iouThresholdShape);
    const auto scoreShape = MakeStorageShape(scoreThresholdShape);
    gert::InfershapeContextPara context("NonMaxSuppressionV3",
                                        {{boxesShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {scoresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {maxShape, ge::DT_INT32, ge::FORMAT_ND},
                                         {iouShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {scoreShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    ExecuteTestCase(context, expectedStatus);
}
} // namespace

TEST(NonMaxSuppressionV3Infershape, infers_static_shape)
{
    ExpectShape({{8, 4}, {8, 4}}, {{8}, {8}}, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(NonMaxSuppressionV3Infershape, rejects_mixed_boxes_scores_dtypes)
{
    ExpectShape({{8, 4}, {8, 4}}, {{8}, {8}}, ge::GRAPH_FAILED, {}, ge::DT_FLOAT, ge::DT_FLOAT16);
    ExpectShape({{8, 4}, {8, 4}}, {{8}, {8}}, ge::GRAPH_FAILED, {}, ge::DT_FLOAT16, ge::DT_FLOAT);
}

TEST(NonMaxSuppressionV3Infershape, accepts_compatible_dynamic_dimensions)
{
    ExpectShape({{-1, 4}, {-1, 4}}, {{8}, {8}}, ge::GRAPH_SUCCESS, {{-1}});
    ExpectShape({{8, -1}, {8, -1}}, {{-1}, {-1}}, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(NonMaxSuppressionV3Infershape, accepts_unknown_rank_boxes)
{
    ExpectShape({{-2}, {-2}}, {{8}, {8}}, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(NonMaxSuppressionV3Infershape, accepts_unknown_rank_scores)
{
    ExpectShape({{8, 4}, {8, 4}}, {{-2}, {-2}}, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(NonMaxSuppressionV3Infershape, still_validates_scores_when_boxes_rank_is_unknown)
{
    ExpectShape({{-2}, {-2}}, {{8, 1}, {8, 1}}, ge::GRAPH_FAILED);
}

TEST(NonMaxSuppressionV3Infershape, rejects_boxes_rank)
{
    ExpectShape({{8, 4, 1}, {8, 4, 1}}, {{8}, {8}}, ge::GRAPH_FAILED);
}

TEST(NonMaxSuppressionV3Infershape, rejects_scores_rank)
{
    ExpectShape({{8, 4}, {8, 4}}, {{8, 1}, {8, 1}}, ge::GRAPH_FAILED);
}

TEST(NonMaxSuppressionV3Infershape, rejects_box_count_mismatch)
{
    ExpectShape({{8, 4}, {8, 4}}, {{7}, {7}}, ge::GRAPH_FAILED);
}

TEST(NonMaxSuppressionV3Infershape, rejects_box_coordinate_dimension)
{
    ExpectShape({{8, 5}, {8, 5}}, {{8}, {8}}, ge::GRAPH_FAILED);
}

TEST(NonMaxSuppressionV3Infershape, rejects_missing_scalar_inputs)
{
    for (const size_t missingIndex : {kMaxOutputSizeIndex, kIouThresholdIndex, kScoreThresholdIndex}) {
        SCOPED_TRACE(missingIndex);
        ExpectMissingInput(missingIndex);
    }
}

TEST(NonMaxSuppressionV3Infershape, accepts_single_element_control_inputs)
{
    ExpectControlShapes({1}, {1}, {1}, ge::GRAPH_SUCCESS);
}

TEST(NonMaxSuppressionV3Infershape, rejects_multi_element_control_inputs)
{
    ExpectControlShapes({2}, {}, {}, ge::GRAPH_FAILED);
    ExpectControlShapes({}, {2}, {}, ge::GRAPH_FAILED);
    ExpectControlShapes({}, {}, {2}, ge::GRAPH_FAILED);
}

TEST(NonMaxSuppressionV3Infershape, registers_selected_indices_shape_dependency)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("NonMaxSuppressionV3");
    ASSERT_NE(opImpl, nullptr);
    EXPECT_TRUE(opImpl->IsOutputShapeDependOnCompute());
    EXPECT_TRUE(opImpl->IsOutputShapeDependOnCompute(kSelectedIndicesIndex));
}

TEST(NonMaxSuppressionV3Infershape, sets_selected_indices_shape_range)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferShapeRange = registry->GetOpImpl("NonMaxSuppressionV3")->infer_shape_range;
    ASSERT_NE(inferShapeRange, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("NonMaxSuppressionV3").OpName("NonMaxSuppressionV3");
    builder.IONum(kInputNum, kOutputNum);
    builder.OutputTensorDesc(kSelectedIndicesIndex, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeRange(context), ge::GRAPH_SUCCESS);

    const auto* selectedIndicesRange = context->GetOutputShapeRange(kSelectedIndicesIndex);
    ASSERT_NE(selectedIndicesRange, nullptr);
    EXPECT_EQ(selectedIndicesRange->GetMin()->GetDimNum(), 1U);
    EXPECT_EQ(selectedIndicesRange->GetMin()->GetDim(0U), 0);
    EXPECT_EQ(selectedIndicesRange->GetMax()->GetDimNum(), 1U);
    EXPECT_EQ(selectedIndicesRange->GetMax()->GetDim(0U), ge::UNKNOWN_DIM);
}
