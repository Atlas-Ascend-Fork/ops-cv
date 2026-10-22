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

#include "../../../src/framework/col2imv2_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxCol2ImV2PluginTest, ParseDilationsPadsStrides)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"dilations","type":7,"ints":[1,2]},{"name":"pads","type":7,"ints":[3,4]},)"
        R"({"name":"strides","type":7,"ints":[5,6]}]})",
        "col2im_src");
    ge::Operator op_dest = CreateOperator("col2im");
    std::vector<int32_t> dilation;
    std::vector<int32_t> padding;
    std::vector<int32_t> stride;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseOnnxParamsCol2ImV2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("dilation", dilation), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dilation.size(), 2U);
    EXPECT_EQ(dilation[0], 1);
    EXPECT_EQ(dilation[1], 2);
    EXPECT_EQ(op_dest.GetAttr("padding", padding), ge::GRAPH_SUCCESS);
    ASSERT_EQ(padding.size(), 2U);
    EXPECT_EQ(padding[0], 3);
    EXPECT_EQ(padding[1], 4);
    EXPECT_EQ(op_dest.GetAttr("stride", stride), ge::GRAPH_SUCCESS);
    ASSERT_EQ(stride.size(), 2U);
    EXPECT_EQ(stride[0], 5);
    EXPECT_EQ(stride[1], 6);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "col2im_src");
}

TEST(OnnxCol2ImV2PluginTest, SuccessWhenAttributeEmpty)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("col2im");
    EXPECT_EQ(domi::ParseOnnxParamsCol2ImV2(op_src, op_dest), domi::SUCCESS);
}

TEST(OnnxCol2ImV2PluginTest, ReturnFailedWhenDilationsLengthInvalid)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"dilations","type":7,"ints":[1]}]})");
    ge::Operator op_dest = CreateOperator("col2im");
    EXPECT_EQ(domi::ParseOnnxParamsCol2ImV2(op_src, op_dest), domi::FAILED);
}

TEST(OnnxCol2ImV2PluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("col2im");
    EXPECT_EQ(domi::ParseOnnxParamsCol2ImV2(op_src, op_dest), domi::FAILED);
}
