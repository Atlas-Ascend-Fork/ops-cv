/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"
#include "op_infer_shape_range_context_builder.h"

namespace {

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

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& boxesShape, ge::DataType dtype = ge::DT_FLOAT)
{
    const gert::StorageShape boxes = MakeStorageShape(boxesShape);
    const gert::StorageShape output = MakeStorageShape({});
    const std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {boxes, dtype, ge::FORMAT_ND},
    };
    const std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {output, ge::DT_INT32, ge::FORMAT_ND},
    };
    return gert::InfershapeContextPara("SortedNMS", inputs, outputs);
}
} // namespace

TEST(SortedNMSInfershape, selected_indices_follows_unknown_first_dimension)
{
    auto context = MakeContext({16, 4});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(SortedNMSInfershape, accepts_unknown_boxes_dimension)
{
    auto context = MakeContext({-1, 4});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(SortedNMSInfershape, rank_independent_inference)
{
    // Shape inference does not require a known input rank; concrete input
    // constraints belong to tiling.
    auto context = MakeContext({8});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1}});
}

TEST(SortedNMSInfershape, sets_selected_indices_shape_range)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferShapeRange = registry->GetOpImpl("SortedNMS")->infer_shape_range;
    ASSERT_NE(inferShapeRange, nullptr);

    const gert::StorageShape minShape = MakeStorageShape({0, 4});
    const gert::StorageShape maxShape = MakeStorageShape({16, 4});
    gert::Tensor minTensor(minShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                           gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor maxTensor(maxShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                           gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Range<gert::Tensor> boxesRange(&minTensor, &maxTensor);
    const std::vector<gert::Range<gert::Tensor>*> inputRanges = {&boxesRange};

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("SortedNMS").OpName("SortedNMS");
    builder.IONum(1, 1);
    builder.InputTensorsRange(inputRanges);
    builder.OutputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeRange(context), ge::GRAPH_SUCCESS);

    const auto* selectedRange = context->GetOutputShapeRange(0);
    ASSERT_NE(selectedRange, nullptr);
    EXPECT_EQ(selectedRange->GetMin()->GetDimNum(), 1U);
    EXPECT_EQ(selectedRange->GetMin()->GetDim(0U), 0);
    EXPECT_EQ(selectedRange->GetMax()->GetDimNum(), 1U);
    EXPECT_EQ(selectedRange->GetMax()->GetDim(0U), 16);
}

TEST(SortedNMSInfershape, rejects_empty_boxes_shape_range)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto inferShapeRange = registry->GetOpImpl("SortedNMS")->infer_shape_range;
    ASSERT_NE(inferShapeRange, nullptr);

    const gert::StorageShape minShape = MakeStorageShape({});
    const gert::StorageShape maxShape = MakeStorageShape({});
    gert::Tensor minTensor(minShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                           gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor maxTensor(maxShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                           gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Range<gert::Tensor> boxesRange(&minTensor, &maxTensor);
    const std::vector<gert::Range<gert::Tensor>*> inputRanges = {&boxesRange};

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("SortedNMS").OpName("SortedNMS");
    builder.IONum(1, 1);
    builder.InputTensorsRange(inputRanges);
    builder.OutputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeRange(context), ge::GRAPH_FAILED);
}
