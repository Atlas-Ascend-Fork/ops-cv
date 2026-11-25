/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {

// Helper: x only (no theta)
void ExpectShape(const gert::StorageShape& xShape, ge::Format xFormat, ge::graphStatus expectedStatus,
                 const std::vector<std::vector<int64_t>>& expectedShapes = {})
{
    gert::InfershapeContextPara context("SpatialTransformer", {{xShape, ge::DT_FLOAT, xFormat}},
                                        {{{{}, {}}, ge::DT_FLOAT, xFormat}});
    ExecuteTestCase(context, expectedStatus, expectedShapes);
}

// Helper: x + theta
void ExpectShapeWithTheta(const gert::StorageShape& xShape, const gert::StorageShape& thetaShape, ge::Format xFormat,
                          ge::graphStatus expectedStatus, const std::vector<std::vector<int64_t>>& expectedShapes = {})
{
    gert::InfershapeContextPara context("SpatialTransformer",
                                        {{xShape, ge::DT_FLOAT, xFormat}, {thetaShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, xFormat}});
    ExecuteTestCase(context, expectedStatus, expectedShapes);
}

} // namespace

// ===== NCHW 正常推导 =====

TEST(SpatialTransformerInfershape, infers_nchw_with_theta)
{
    ExpectShapeWithTheta({{2, 3, 8, 9}, {2, 3, 8, 9}}, {{2, 3}, {2, 3}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS,
                         {{2, 3, 8, 9}});
}

TEST(SpatialTransformerInfershape, infers_nchw_batch_theta)
{
    ExpectShapeWithTheta({{2, 3, 8, 9}, {2, 3, 8, 9}}, {{2, 2, 3}, {2, 2, 3}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS,
                         {{2, 3, 8, 9}});
}

TEST(SpatialTransformerInfershape, infers_nchw_without_theta)
{
    ExpectShape({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{2, 3, 8, 9}});
}

// ===== NC1HWC0 正常推导 =====

TEST(SpatialTransformerInfershape, infers_nc1hwc0)
{
    ExpectShape({{2, 1, 8, 9, 16}, {2, 1, 8, 9, 16}}, ge::FORMAT_NC1HWC0, ge::GRAPH_SUCCESS, {{2, 1, 8, 9, 16}});
}

// ===== 动态维度透传 =====

TEST(SpatialTransformerInfershape, preserves_dynamic_dimensions_nchw)
{
    ExpectShape({{-1, 3, -1, -1}, {-1, 3, -1, -1}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{-1, 3, -1, -1}});
}

TEST(SpatialTransformerInfershape, images_unknown_rank_passes_through)
{
    ExpectShape({{-2}, {-2}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{-2}});
}

// ===== theta 校验 =====

TEST(SpatialTransformerInfershape, rejects_theta_rank_1)
{
    ExpectShapeWithTheta({{2, 3, 8, 9}, {2, 3, 8, 9}}, {{3}, {3}}, ge::FORMAT_NCHW, ge::GRAPH_FAILED);
}

TEST(SpatialTransformerInfershape, rejects_theta_wrong_dims_2d)
{
    // [3, 2] instead of [2, 3]
    ExpectShapeWithTheta({{2, 3, 8, 9}, {2, 3, 8, 9}}, {{3, 2}, {3, 2}}, ge::FORMAT_NCHW, ge::GRAPH_FAILED);
}

TEST(SpatialTransformerInfershape, rejects_theta_batch_mismatch_3d)
{
    // theta batch=3, x batch=2
    ExpectShapeWithTheta({{2, 3, 8, 9}, {2, 3, 8, 9}}, {{3, 2, 3}, {3, 2, 3}}, ge::FORMAT_NCHW, ge::GRAPH_FAILED);
}

// ===== x rank 校验 =====

TEST(SpatialTransformerInfershape, rejects_nchw_rank_3)
{
    ExpectShape({{2, 3, 8}, {2, 3, 8}}, ge::FORMAT_NCHW, ge::GRAPH_FAILED);
}

TEST(SpatialTransformerInfershape, rejects_nc1hwc0_rank_4)
{
    ExpectShape({{2, 1, 8, 9}, {2, 1, 8, 9}}, ge::FORMAT_NC1HWC0, ge::GRAPH_FAILED);
}

TEST(SpatialTransformerInfershape, rejects_unsupported_format)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, ge::FORMAT_NHWC, ge::GRAPH_FAILED);
}

// ===== output_size 属性分支（带属性构造的辅助函数）=====

namespace {

void ExpectShapeWithAttr(const gert::StorageShape& xShape, ge::Format xFormat, ge::graphStatus expectedStatus,
                         const std::vector<std::vector<int64_t>>& expectedShapes,
                         const std::vector<int64_t>& outputSize)
{
    gert::InfershapeContextPara context(
        "SpatialTransformer", {{xShape, ge::DT_FLOAT, xFormat}}, {{{{}, {}}, ge::DT_FLOAT, xFormat}},
        {gert::InfershapeContextPara::OpAttr("output_size",
                                             Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize))});
    ExecuteTestCase(context, expectedStatus, expectedShapes);
}

} // namespace

TEST(SpatialTransformerInfershape, output_size_default_uses_input_hw)
{
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{2, 3, 8, 9}}, {-1, -1});
}

TEST(SpatialTransformerInfershape, output_size_explicit_values)
{
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{2, 3, 5, 6}}, {5, 6});
}

TEST(SpatialTransformerInfershape, output_size_partial_passthrough)
{
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{2, 3, 8, 6}}, {-1, 6});
}

TEST(SpatialTransformerInfershape, output_size_explicit_nc1hwc0)
{
    ExpectShapeWithAttr({{2, 1, 8, 9, 16}, {2, 1, 8, 9, 16}}, ge::FORMAT_NC1HWC0, ge::GRAPH_SUCCESS, {{2, 1, 5, 6, 16}},
                        {5, 6});
}

TEST(SpatialTransformerInfershape, rejects_output_size_below_minus_one)
{
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_FAILED, {}, {-2, 6});
}

TEST(SpatialTransformerInfershape, output_size_empty_list_uses_input_hw)
{
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{2, 3, 8, 9}}, {});
}

TEST(SpatialTransformerInfershape, rejects_output_size_single_element)
{
    // RT1.0 rejects a single-element output_size instead of silently using the input H/W.
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_FAILED, {}, {5});
}

TEST(SpatialTransformerInfershape, output_size_three_elements_uses_first_two)
{
    ExpectShapeWithAttr({{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS, {{2, 3, 5, 6}}, {5, 6, 7});
}

TEST(SpatialTransformerInfershape, dynamic_batch_theta_passes)
{
    // x batch=-1 (dynamic), theta batch=2 → should succeed (skip check)
    gert::InfershapeContextPara context("SpatialTransformer",
                                        {{{{-1, 3, 8, 9}, {-1, 3, 8, 9}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
                                         {{{2, 2, 3}, {2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 3, 8, 9}});
}

// ===== theta 矩阵维度部分动态 =====

TEST(SpatialTransformerInfershape, theta_dynamic_matrix_dim_passes)
{
    // theta = [2, -1] (partial dynamic) → succeed
    gert::InfershapeContextPara context("SpatialTransformer",
                                        {{{{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
                                         {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 8, 9}});
}

TEST(SpatialTransformerInfershape, theta_dynamic_matrix_dim_3d_passes)
{
    // theta = [2, -1, -1] (partial dynamic 3D) → succeed
    gert::InfershapeContextPara context("SpatialTransformer",
                                        {{{{2, 3, 8, 9}, {2, 3, 8, 9}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
                                         {{{2, -1, -1}, {2, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 8, 9}});
}

// ===== origin NCHW + storage NC1HWC0 =====

// NOTE: origin=NCHW(4D)+storage=NC1HWC0(5D) needs separate origin/storage format which InfershapeContextPara doesn't
// support. GetOriginFormat fix validated by real GE GEIR e2e.

// ===== InferDataType 推导 =====

TEST(SpatialTransformerInfershape, inferdatatype_rejects_null_context)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("SpatialTransformer");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    EXPECT_EQ(opImpl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}

TEST(SpatialTransformerInfershape, inferdatatype_preserves_x_dtype)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("SpatialTransformer");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    for (const ge::DataType xDtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_INT32, ge::DT_UINT8}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("SpatialTransformer").OpName("SpatialTransformer").IONum(2, 1);
        builder.InputTensorDesc(0, xDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
        builder.InputTensorDesc(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        context->SetOutputDataType(0, ge::DT_UNDEFINED);

        EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), xDtype);
    }
}
