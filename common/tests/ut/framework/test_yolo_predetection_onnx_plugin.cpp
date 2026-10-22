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

#include "../../../src/framework/yolo_predetection_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxYoloPreDetectionPluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2,"i":4},{"name":"coords","type":2,"i":5},)"
        R"({"name":"classes","type":2,"i":10},{"name":"yolo_version","type":3,"s":"V3"},)"
        R"({"name":"softmax","type":2,"i":1},{"name":"background","type":2,"i":1},)"
        R"({"name":"softmaxtree","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("yolo_pred");
    int boxes = 0;
    ge::AscendString yolo_version;
    bool softmax = false;
    bool background = false;
    bool softmaxtree = false;

    EXPECT_EQ(domi::ParseParamsYoloPreDetection(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 4);
    EXPECT_EQ(op_dest.GetAttr("yolo_version", yolo_version), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(yolo_version.GetString(), "V3");
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(softmax);
    EXPECT_EQ(op_dest.GetAttr("background", background), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(background);
    EXPECT_EQ(op_dest.GetAttr("softmaxtree", softmaxtree), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(softmaxtree);
}

TEST(OnnxYoloPreDetectionPluginTest, UseDefaultsWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("yolo_pred");
    int boxes = 0;
    ge::AscendString yolo_version;
    bool softmax = true;

    EXPECT_EQ(domi::ParseParamsYoloPreDetection(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 3);
    EXPECT_EQ(op_dest.GetAttr("yolo_version", yolo_version), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(yolo_version.GetString(), "V5");
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmax);
}

TEST(OnnxYoloPreDetectionPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2},{"name":"coords","type":2},{"name":"classes","type":2},)"
        R"({"name":"softmax","type":2},{"name":"background","type":2},{"name":"softmaxtree","type":2}]})");
    ge::Operator op_dest = CreateOperator("yolo_pred");
    int boxes = 3;
    int classes = 80;
    bool softmax = true;

    EXPECT_EQ(domi::ParseParamsYoloPreDetection(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 0);
    EXPECT_EQ(op_dest.GetAttr("classes", classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(classes, 0);
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmax);
}

TEST(OnnxYoloPreDetectionPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("yolo_pred");
    EXPECT_EQ(domi::ParseParamsYoloPreDetection(op_src, op_dest), domi::FAILED);
}
