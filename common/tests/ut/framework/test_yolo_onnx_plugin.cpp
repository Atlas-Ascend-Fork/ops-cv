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

#include "../../../src/framework/yolo_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxYoloPluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2,"i":5},{"name":"coords","type":2,"i":6},)"
        R"({"name":"classes","type":2,"i":20},{"name":"yolo_version","type":3,"s":"V4"}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    int boxes = 0;
    int coords = 0;
    int classes = 0;
    ge::AscendString yolo_version;
    bool softmax = true;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 5);
    EXPECT_EQ(op_dest.GetAttr("coords", coords), ge::GRAPH_SUCCESS);
    EXPECT_EQ(coords, 6);
    EXPECT_EQ(op_dest.GetAttr("classes", classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(classes, 20);
    EXPECT_EQ(op_dest.GetAttr("yolo_version", yolo_version), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(yolo_version.GetString(), "V4");
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmax);
}

TEST(OnnxYoloPluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("yolo");
    int boxes = 0;
    int classes = 0;
    ge::AscendString yolo_version;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 3);
    EXPECT_EQ(op_dest.GetAttr("classes", classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(classes, 80);
    EXPECT_EQ(op_dest.GetAttr("yolo_version", yolo_version), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(yolo_version.GetString(), "V3");
}

TEST(OnnxYoloPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2},{"name":"coords","type":2},{"name":"classes","type":2}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    int boxes = 3;
    int coords = 4;
    int classes = 80;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 0);
    EXPECT_EQ(op_dest.GetAttr("coords", coords), ge::GRAPH_SUCCESS);
    EXPECT_EQ(coords, 0);
    EXPECT_EQ(op_dest.GetAttr("classes", classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(classes, 0);
}

TEST(OnnxYoloPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("yolo");
    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::FAILED);
}
