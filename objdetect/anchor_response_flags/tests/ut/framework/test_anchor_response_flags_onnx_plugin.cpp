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

#include <string>
#include <vector>

#include "../../../framework/anchor_response_flags_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxAnchorResponseFlagsPluginTest, ParseRequiredAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"featmap_sizes","type":7,"ints":[8,8]},{"name":"strides","type":7,"ints":[4,4]},)"
        R"({"name":"num_base_anchors","type":2,"i":3}]})");
    ge::Operator op_dest = CreateOperator("anchor_response_flags");
    std::vector<int32_t> featmap_size;
    std::vector<int32_t> strides;
    int64_t num_base_anchors = 0;

    EXPECT_EQ(domi::ParseParamsNpuAnchorResponseFlags(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("featmap_size", featmap_size), ge::GRAPH_SUCCESS);
    ASSERT_EQ(featmap_size.size(), 2U);
    EXPECT_EQ(featmap_size[0], 8);
    EXPECT_EQ(featmap_size[1], 8);
    EXPECT_EQ(op_dest.GetAttr("strides", strides), ge::GRAPH_SUCCESS);
    ASSERT_EQ(strides.size(), 2U);
    EXPECT_EQ(strides[0], 4);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(op_dest.GetAttr("num_base_anchors", num_base_anchors), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_base_anchors, 3);
}

TEST(OnnxAnchorResponseFlagsPluginTest, ParseZeroNumBaseAnchorsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"featmap_sizes","type":7,"ints":[8,8]},{"name":"strides","type":7,"ints":[4,4]},)"
        R"({"name":"num_base_anchors","type":2}]})");
    ge::Operator op_dest = CreateOperator("anchor_response_flags");
    int64_t num_base_anchors = 3;

    EXPECT_EQ(domi::ParseParamsNpuAnchorResponseFlags(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("num_base_anchors", num_base_anchors), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_base_anchors, 0);
}

TEST(OnnxAnchorResponseFlagsPluginTest, ReturnFailedWhenRequiredAttrsMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("anchor_response_flags");
    EXPECT_EQ(domi::ParseParamsNpuAnchorResponseFlags(op_src, op_dest), domi::FAILED);
}

TEST(OnnxAnchorResponseFlagsPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("anchor_response_flags");
    EXPECT_EQ(domi::ParseParamsNpuAnchorResponseFlags(op_src, op_dest), domi::FAILED);
}
