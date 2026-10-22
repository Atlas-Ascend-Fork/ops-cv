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

#include "../../../framework/non_max_suppression_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNonMaxSuppressionPluginTest, ParseCenterPointBox)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"center_point_box","type":2,"i":1}]})",
                                               "nms_src");
    ge::Operator op_dest = CreateOperator("nms");
    int64_t center_point_box = 0;
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsNonMaxSuppression(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("center_point_box", center_point_box), ge::GRAPH_SUCCESS);
    EXPECT_EQ(center_point_box, 1);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "nms_src");
}

TEST(OnnxNonMaxSuppressionPluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("nms");
    int64_t center_point_box = 9;

    EXPECT_EQ(domi::ParseParamsNonMaxSuppression(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("center_point_box", center_point_box), ge::GRAPH_SUCCESS);
    EXPECT_EQ(center_point_box, 0);
}

TEST(OnnxNonMaxSuppressionPluginTest, ParseZeroCenterPointBoxWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"center_point_box","type":2}]})");
    ge::Operator op_dest = CreateOperator("nms");
    int64_t center_point_box = 9;

    EXPECT_EQ(domi::ParseParamsNonMaxSuppression(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("center_point_box", center_point_box), ge::GRAPH_SUCCESS);
    EXPECT_EQ(center_point_box, 0);
}

TEST(OnnxNonMaxSuppressionPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("nms");
    EXPECT_EQ(domi::ParseParamsNonMaxSuppression(op_src, op_dest), domi::FAILED);
}
