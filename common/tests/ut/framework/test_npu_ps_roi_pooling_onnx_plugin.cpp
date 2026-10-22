/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>

#include "../../../src/framework/npu_ps_roi_pooling_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuPsRoiPoolingPluginTest, ParseRequiredAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1,"f":"0.125"},{"name":"output_dim","type":2,"i":8},)"
        R"({"name":"group_size","type":2,"i":4}]})");
    ge::Operator op_dest = CreateOperator("ps_roi");
    float spatial_scale = 0.0f;
    int output_dim = 0;
    int group_size = 0;

    EXPECT_EQ(domi::ParseParamsNpuPsRoiPooling(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.125f);
    EXPECT_EQ(op_dest.GetAttr("output_dim", output_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_dim, 8);
    EXPECT_EQ(op_dest.GetAttr("group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 4);
}

TEST(OnnxNpuPsRoiPoolingPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1},{"name":"output_dim","type":2},)"
        R"({"name":"group_size","type":2}]})");
    ge::Operator op_dest = CreateOperator("ps_roi");
    float spatial_scale = 1.0f;
    int output_dim = 1;
    int group_size = 1;

    EXPECT_EQ(domi::ParseParamsNpuPsRoiPooling(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("output_dim", output_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_dim, 0);
    EXPECT_EQ(op_dest.GetAttr("group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 0);
}

TEST(OnnxNpuPsRoiPoolingPluginTest, ReturnFailedWhenRequiredAttrsMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("ps_roi");
    EXPECT_EQ(domi::ParseParamsNpuPsRoiPooling(op_src, op_dest), domi::FAILED);
}

TEST(OnnxNpuPsRoiPoolingPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("ps_roi");
    EXPECT_EQ(domi::ParseParamsNpuPsRoiPooling(op_src, op_dest), domi::FAILED);
}
