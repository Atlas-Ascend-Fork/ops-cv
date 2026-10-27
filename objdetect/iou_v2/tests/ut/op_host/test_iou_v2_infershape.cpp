/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"

class IouV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IouV2 Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IouV2 Proto Test TearDown" << std::endl; }
};

TEST_F(IouV2, IouV2_infershape_iou_false_case_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2",
        {{{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1024, 1024},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IouV2, IouV2_infershape_iou_true_case_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2",
        {{{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1024, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IouV2, IouV2_infershape_rejects_invalid_coordinate_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2", {{{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 3}, {1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(IouV2, IouV2_infershape_rejects_invalid_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2", {{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(IouV2, IouV2_infershape_rejects_aligned_count_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2", {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(true))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(IouV2, IouV2_infershape_unknown_rank_is_propagated)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2", {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(IouV2, IouV2_infershape_unknown_dim_is_accepted)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IouV2", {{{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("iou")),
         gert::InfershapeContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, -1}});
}

TEST_F(IouV2, IouV2_inferdatatype_rejects_mixed_input_dtype)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("IouV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("IouV2").OpName("IouV2").IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_FAILED);
}

TEST_F(IouV2, IouV2_inferdatatype_accepts_matching_input_dtype)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("IouV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("IouV2").OpName("IouV2").IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// TEST_F(IouV2, IouV2_infershape_iof_false_case_2)
// {
//     ge::op::IouV2 op;
//     float32_t eps_value = 1;
//     op.UpdateInputDesc("bboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.UpdateInputDesc("gtboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.SetAttr("mode", "iof");
//     op.SetAttr("eps", eps_value);
//     op.SetAttr("aligned", false);
//     Runtime2TestParam param{{"mode", "eps", "aligned"}, {}, {}};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

//     auto output_desc = op.GetOutputDesc("overlap");
//     std::vector<int64_t> expected_output_shape = {1020, 1020};
//     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(IouV2, IouV2_infershape_iou_true_case_3)
// {
//     ge::op::IouV2 op;
//     float32_t eps_value = 1;
//     op.UpdateInputDesc("bboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.UpdateInputDesc("gtboxes", create_desc({1020, 4}, ge::DT_FLOAT));
//     op.SetAttr("mode", "iou");
//     op.SetAttr("eps", eps_value);
//     op.SetAttr("aligned", true);
//     Runtime2TestParam param{{"mode", "eps", "aligned"}, {}, {}};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

//     auto output_desc = op.GetOutputDesc("overlap");
//     std::vector<int64_t> expected_output_shape = {4, 1};
//     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
// }

// TEST_F(IouV2, IouV2_InferDtype_case_0)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("IouV2"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("IouV2")->infer_datatype;
//     if (data_type_func != nullptr)
//     {
//         ge::DataType bboxes = ge::DT_FLOAT;
//         ge::DataType gtboxes = ge::DT_FLOAT;
//         ge::DataType output_ref = ge::DT_FLOAT;
//         auto context_holder = gert::InferDataTypeContextFaker()
//                                   .IrInputNum(2)
//                                   .NodeIoNum(2, 1)
//                                   .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .InputDataTypes({&bboxes, &gtboxes})
//                                   .OutputDataTypes({&output_ref})
//                                   .Build();
//         auto context = context_holder.GetContext<gert::InferDataTypeContext>();
//         EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
//         ASSERT_NE(context, nullptr);

//         EXPECT_EQ(context->GetInputDataType(0), bboxes);
//         EXPECT_EQ(context->GetInputDataType(1), gtboxes);
//         EXPECT_EQ(context->GetOutputDataType(0), output_ref);
//     }
// }
