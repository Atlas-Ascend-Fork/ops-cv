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
#include <numeric>
#include <cstdint>
#include <string>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class UpsampleNearest3dTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpsampleNearest3dTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpsampleNearest3dTest TearDown" << std::endl; }
};

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test1_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "UpsampleNearest3d", {{{{1, 1, 10, 10, 10}, {1, 1, 10, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("input_size",
                                             Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 5, 5})),
         gert::InfershapeContextPara::OpAttr("scales",
                                             Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0, 2.0}))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 5, 5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test2_failed)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc(
//         "x", create_desc_with_ori({1, 1, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5}, ge::FORMAT_ND));
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_FAILED);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
// }

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test3_failed)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc(
//         "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
//     std::vector<int64_t> output_size_vec = {10, 10};
//     op.SetAttr("output_size", output_size_vec);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_FAILED);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
// }

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test4_failed)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc(
//         "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
//     std::vector<float> scales_vec = {2.0, 2.0};
//     op.SetAttr("scales", scales_vec);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_FAILED);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
// }

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test5_failed)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc(
//         "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
//     std::vector<int64_t> output_size_vec = {10, 10, 10};
//     op.SetAttr("output_size", output_size_vec);
//     std::vector<float> scales_vec = {2.0, 2.0, 2.0};
//     op.SetAttr("scales", scales_vec);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_FAILED);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
// }

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test1_success)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc(
//         "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
//     std::vector<int64_t> output_size_vec = {10, 10, 10};
//     op.SetAttr("output_size", output_size_vec);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//     auto out_var_desc = op.GetOutputDescByName("y");
//     EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
//     std::vector<int64_t> expected_var_output_shape = {1, 1, 10, 10, 10};
//     EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
//     out_var_desc = op.GetOutputDescByName("y");
//     EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
// }

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test2_success)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc(
//         "x", create_desc_with_ori({1, 1, 5, 5, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 5, 5, 5}, ge::FORMAT_ND));
//     std::vector<float> scales_vec = {2.0, 2.0, 2.0};
//     op.SetAttr("scales", scales_vec);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//     auto out_var_desc = op.GetOutputDescByName("y");
//     EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
//     std::vector<int64_t> expected_var_output_shape = {1, 1, 10, 10, 10};
//     EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
//     out_var_desc = op.GetOutputDescByName("y");
//     EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
// }

// TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_infer_test3_success)
// {
//     ge::op::UpsampleNearest3d op;
//     op.UpdateInputDesc("x",
//         create_desc_with_ori({16, 3, 5, 512, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {16, 3, 5, 512, 512},
//         ge::FORMAT_ND));
//     std::vector<float> scales_vec = {2.5, 2.0, 3.8};
//     op.SetAttr("scales", scales_vec);
//     auto ret = op.InferShapeAndType();
//     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//     auto out_var_desc = op.GetOutputDescByName("y");
//     EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
//     std::vector<int64_t> expected_var_output_shape = {16, 3, 12, 1024, 1945};
//     EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

//     Runtime2TestParam param;
//     param.attrs = {"output_size", "scales"};
//     EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
//     out_var_desc = op.GetOutputDescByName("y");
//     EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
//     auto status = op.VerifyAllAttr(true);
//     EXPECT_EQ(status, ge::GRAPH_SUCCESS);
// }

// Dynamic-shape regression tests for the shared UpsampleNearest3d infer-shape callback.
static gert::StorageShape MakeUpsampleNearest3dTestShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

static gert::InfershapeContextPara MakeUpsampleNearest3dTestContext(const std::string& opName,
                                                                    const std::vector<int64_t>& inputDims,
                                                                    const std::vector<int64_t>& outputSize,
                                                                    const std::vector<float>& scales)
{
    using OpAttr = gert::InfershapeContextPara::OpAttr;
    using TensorDesc = gert::InfershapeContextPara::TensorDescription;
    return gert::InfershapeContextPara(
        opName, {TensorDesc(MakeUpsampleNearest3dTestShape(inputDims), ge::DT_FLOAT, ge::FORMAT_ND)},
        {TensorDesc(MakeUpsampleNearest3dTestShape({}), ge::DT_FLOAT, ge::FORMAT_ND)},
        {OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)),
         OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>(scales))});
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_scales_preserves_unknown_dim)
{
    auto context = MakeUpsampleNearest3dTestContext("UpsampleNearest3d", {1, 1, ge::UNKNOWN_DIM, 3, 4}, {},
                                                    {2.0f, 1.0f, 1.0f});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 1, ge::UNKNOWN_DIM, 3, 4}});
}

TEST_F(UpsampleNearest3dTest, UpsampleNearestExact3d_scales_preserves_unknown_dim)
{
    auto context = MakeUpsampleNearest3dTestContext("UpsampleNearestExact3d", {1, 1, ge::UNKNOWN_DIM, 3, 4}, {},
                                                    {2.0f, 1.0f, 1.0f});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 1, ge::UNKNOWN_DIM, 3, 4}});
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_scales_keeps_known_dim_calculation)
{
    auto context = MakeUpsampleNearest3dTestContext("UpsampleNearest3d", {1, 1, 2, 3, 4}, {}, {2.0f, 1.0f, 1.0f});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 1, 4, 3, 4}});
}

TEST_F(UpsampleNearest3dTest, UpsampleNearest3d_output_size_without_scales)
{
    auto context = MakeUpsampleNearest3dTestContext("UpsampleNearest3d", {1, 1, ge::UNKNOWN_DIM, 3, 4}, {5, 6, 7}, {});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{1, 1, 5, 6, 7}});
}
