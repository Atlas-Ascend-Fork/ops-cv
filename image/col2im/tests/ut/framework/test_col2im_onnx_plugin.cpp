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

#include <string>
#include <vector>

#include "../../../framework/col2im_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxCol2imPluginTest, ParseAllRequiredAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"kernel_size","type":7,"ints":[3,3]},{"name":"dilation","type":7,"ints":[1,1]},)"
        R"({"name":"padding","type":7,"ints":[0,0]},{"name":"stride","type":7,"ints":[1,1]}]})");
    ge::Operator op_dest = CreateOperator("col2im");
    // ChangeFormatFromOnnx needs existing input/output tensor descs
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    ge::TensorDesc tensor_desc(ge::Shape({1, 1, 1, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
    op_dest.UpdateInputDesc("x", tensor_desc);
    op_dest.UpdateOutputDesc("y", tensor_desc);
    std::vector<int32_t> kernel_size;
    std::vector<int32_t> dilation;
    std::vector<int32_t> padding;
    std::vector<int32_t> stride;

    EXPECT_EQ(domi::ParseOnnxParamsCol2im(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("kernel_size", kernel_size), ge::GRAPH_SUCCESS);
    ASSERT_EQ(kernel_size.size(), 2U);
    EXPECT_EQ(kernel_size[0], 3);
    EXPECT_EQ(kernel_size[1], 3);
    EXPECT_EQ(op_dest.GetAttr("dilation", dilation), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dilation.size(), 2U);
    EXPECT_EQ(dilation[0], 1);
    EXPECT_EQ(dilation[1], 1);
    EXPECT_EQ(op_dest.GetAttr("padding", padding), ge::GRAPH_SUCCESS);
    ASSERT_EQ(padding.size(), 2U);
    EXPECT_EQ(padding[0], 0);
    EXPECT_EQ(padding[1], 0);
    EXPECT_EQ(op_dest.GetAttr("stride", stride), ge::GRAPH_SUCCESS);
    ASSERT_EQ(stride.size(), 2U);
    EXPECT_EQ(stride[0], 1);
    EXPECT_EQ(stride[1], 1);
}

TEST(OnnxCol2imPluginTest, ReturnFailedWhenRequiredAttrsMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"kernel_size","type":7,"ints":[3,3]}]})");
    ge::Operator op_dest = CreateOperator("col2im");
    EXPECT_EQ(domi::ParseOnnxParamsCol2im(op_src, op_dest), domi::FAILED);
}

TEST(OnnxCol2imPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("col2im");
    EXPECT_EQ(domi::ParseOnnxParamsCol2im(op_src, op_dest), domi::FAILED);
}
