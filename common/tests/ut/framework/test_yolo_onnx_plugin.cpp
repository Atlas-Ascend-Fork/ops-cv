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

TEST(OnnxYoloPluginTest, ParseSoftmaxAndBackgroundAttributes)
{
    // Regression for V2 models through the ONNX path: softmax/background used to be
    // hardcoded to false, silently degrading the model to MODE_1. Both must pass through.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2,"i":5},{"name":"coords","type":2,"i":4},)"
        R"({"name":"classes","type":2,"i":20},{"name":"yolo_version","type":3,"s":"V2"},)"
        R"({"name":"softmax","type":2,"i":1},{"name":"background","type":2,"i":1},)"
        R"({"name":"softmaxtree","type":2,"i":0}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    bool softmax = false;
    bool background = false;
    bool softmaxtree = true;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(softmax);
    EXPECT_EQ(op_dest.GetAttr("background", background), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(background);
    EXPECT_EQ(op_dest.GetAttr("softmaxtree", softmaxtree), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmaxtree);
}

TEST(OnnxYoloPluginTest, ParseDistinctSoftmaxAndBackgroundValues)
{
    // Explicit zero must keep false while background stays true: the two attrs must
    // not be mixed up with each other.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"softmax","type":2,"i":0},{"name":"background","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    bool softmax = true;
    bool background = false;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmax);
    EXPECT_EQ(op_dest.GetAttr("background", background), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(background);
}

TEST(OnnxYoloPluginTest, ParseNonZeroIntAsTrueForBoolAttrs)
{
    // Bool attrs follow ONNX int semantics: any non-zero value (including negative)
    // parses as true, not only literal 1.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"softmax","type":2,"i":-1},{"name":"background","type":2,"i":2},)"
        R"({"name":"softmaxtree","type":2,"i":0}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    bool softmax = false;
    bool background = false;
    bool softmaxtree = true;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(softmax);
    EXPECT_EQ(op_dest.GetAttr("background", background), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(background);
    EXPECT_EQ(op_dest.GetAttr("softmaxtree", softmaxtree), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmaxtree);
}

TEST(OnnxYoloPluginTest, ParseZeroValueBoolAttrsWhenScalarFieldOmitted)
{
    // GE omits "i" when the int is zero: the bool attrs must fall back to false.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"softmax","type":2},{"name":"background","type":2},)"
        R"({"name":"softmaxtree","type":2}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    bool softmax = true;
    bool background = true;
    bool softmaxtree = true;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmax);
    EXPECT_EQ(op_dest.GetAttr("background", background), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(background);
    EXPECT_EQ(op_dest.GetAttr("softmaxtree", softmaxtree), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmaxtree);
}

TEST(OnnxYoloPluginTest, RejectSoftmaxtreeQuickly)
{
    // softmaxtree is a Lite interface leftover with no kernel semantics: a true value
    // must fail fast instead of being silently swallowed.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"softmax","type":2,"i":1},{"name":"background","type":2,"i":0},)"
        R"({"name":"softmaxtree","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    bool softmaxtree = false;
    bool softmax = false;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::FAILED);
    EXPECT_NE(op_dest.GetAttr("softmaxtree", softmaxtree), ge::GRAPH_SUCCESS);
    EXPECT_NE(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
}

TEST(OnnxYoloPluginTest, IgnoreBoolAttrsWhenTypeMismatch)
{
    // String-typed entries with the same names must not be consumed by the int-typed
    // branches: the attrs keep their false defaults.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"softmax","type":3,"s":"1"},{"name":"background","type":3,"s":"1"},)"
        R"({"name":"softmaxtree","type":3,"s":"1"}]})");
    ge::Operator op_dest = CreateOperator("yolo");
    bool softmax = true;
    bool background = true;
    bool softmaxtree = true;

    EXPECT_EQ(domi::ParseParamsYolo(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("softmax", softmax), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmax);
    EXPECT_EQ(op_dest.GetAttr("background", background), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(background);
    EXPECT_EQ(op_dest.GetAttr("softmaxtree", softmaxtree), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(softmaxtree);
}
