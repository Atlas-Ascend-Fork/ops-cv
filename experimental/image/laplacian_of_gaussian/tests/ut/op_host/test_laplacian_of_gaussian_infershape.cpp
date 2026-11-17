/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
TEST(LoGInfer, Shapes)
{
    for (auto dt : {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT})
        for (auto shape : {std::vector<int64_t>{3, 5},
                           {2, 3, 5},
                           {2, 1, 3, 5},
                           {-1, 5},
                           {-2},
                           {2, -1, 5},
                           {-1, 1, -1, 5},
                           {2, 1, 3, -1}}) {
            gert::Shape s;
            for (auto d : shape)
                s.AppendDim(d);
            gert::StorageShape storage;
            storage.MutableOriginShape() = s;
            storage.MutableStorageShape() = s;
            gert::InfershapeContextPara p("LaplacianOfGaussian", {{storage, dt, ge::FORMAT_ND}},
                                          {{storage, ge::DT_FLOAT, ge::FORMAT_ND}});
            ExecuteTestCase(p, ge::GRAPH_SUCCESS, {shape});
        }
}

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
TEST(LoGInfer, DataTypes)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto dt : {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT}) {
        auto holder = gert::OpInferDataTypeContextBuilder()
                          .OpType("LaplacianOfGaussian")
                          .OpName("LOG")
                          .IONum(1, 1)
                          .InputTensorDesc(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                          .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
        auto c = holder.GetContext();
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(impl->infer_datatype(c), ge::GRAPH_SUCCESS);
        EXPECT_EQ(c->GetOutputDataType(0), ge::DT_FLOAT);
    }
}

TEST(LoGInfer, NullContexts)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    EXPECT_EQ(impl->infer_shape(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(impl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}
#include "base/context_builder/op_infer_shape_context_builder.h"
TEST(LoGInfer, MissingShapes)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("LaplacianOfGaussian");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);

    gert::StorageShape shape;
    shape.MutableStorageShape().AppendDim(3);
    shape.MutableStorageShape().AppendDim(5);
    shape.MutableOriginShape() = shape.GetStorageShape();
    gert::Tensor input(shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                       gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    for (int missing : {0, 1}) {
        gert::OpInferShapeContextBuilder b;
        b.OpType("LaplacianOfGaussian").OpName("LOG").IONum(1, 1);
        b.InputTensors({&input}).OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = b.Build();
        auto c = holder.GetContext();
        ASSERT_NE(c, nullptr);
        if (missing == 0)
            const_cast<gert::Chain*>(reinterpret_cast<gert::KernelContext*>(c)->GetInput(0))->Set(nullptr, nullptr);
        else
            reinterpret_cast<gert::KernelContext*>(c)->GetOutput(0)->Set(nullptr, nullptr);
        EXPECT_EQ(impl->infer_shape(c), ge::GRAPH_FAILED);
    }
}
