/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nms_with_mask_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class NMSWithMaskInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NMSWithMaskInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NMSWithMaskInfershape TearDown" << std::endl; }
};

TEST_F(NMSWithMaskInfershape, nms_with_mask_infershape_test_normal_shape)
{
    gert::InfershapeContextPara infershapeContextPara("NMSWithMask",
                                                      {{{{16, 5}, {16, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                      {{"iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5f)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 5}, {16}, {16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NMSWithMaskInfershape, nms_with_mask_infershape_test_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("NMSWithMask",
                                                      {{{{-2, 5}, {-2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                      {{"iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5f)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2, 5}, {-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

class NMSWithMaskInferDataType : public NMSWithMaskInfershape, public testing::WithParamInterface<ge::DataType> {};

TEST_P(NMSWithMaskInferDataType, nms_with_mask_inferdatatype_test)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("NMSWithMask");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    const ge::DataType inputDtype = GetParam();
    auto contextHolder = gert::OpInferDataTypeContextBuilder()
                             .OpType("NMSWithMask")
                             .OpName("NMSWithMask")
                             .IONum(1, 3)
                             .InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .OutputTensorDesc(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inputDtype);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_UINT8);
}

INSTANTIATE_TEST_SUITE_P(NMSWithMaskInferDataTypeSuite, NMSWithMaskInferDataType,
                         testing::Values(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16));
