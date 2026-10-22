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

#include "../../../src/framework/yolov5_detection_output_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxYoloV5DetectionOutputPluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2,"i":6},{"name":"classes","type":2,"i":10},)"
        R"({"name":"alpha","type":1,"f":"3.5"},{"name":"relative","type":2,"i":0},)"
        R"({"name":"resize_origin_img_to_net","type":2,"i":1},)"
        R"({"name":"biases","type":6,"floats":["4.0","5.0"]}]})");
    ge::Operator op_dest = CreateOperator("yolov5");
    int boxes = 0;
    float alpha = 0.0f;
    bool relative = true;
    bool resize_origin_img_to_net = false;
    std::vector<float> biases;

    EXPECT_EQ(domi::ParseParamsYoloV5DetectionOutput(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 6);
    EXPECT_EQ(op_dest.GetAttr("alpha", alpha), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(alpha, 3.5f);
    EXPECT_EQ(op_dest.GetAttr("relative", relative), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(relative);
    EXPECT_EQ(op_dest.GetAttr("resize_origin_img_to_net", resize_origin_img_to_net), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(resize_origin_img_to_net);
    EXPECT_EQ(op_dest.GetAttr("biases", biases), ge::GRAPH_SUCCESS);
    ASSERT_EQ(biases.size(), 2U);
}

TEST(OnnxYoloV5DetectionOutputPluginTest, UseDefaultsWithRequiredBiases)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"biases","type":6,"floats":["1.0"]}]})");
    ge::Operator op_dest = CreateOperator("yolov5");
    float alpha = 0.0f;
    bool relative = false;

    EXPECT_EQ(domi::ParseParamsYoloV5DetectionOutput(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("alpha", alpha), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(alpha, 2.0f);
    EXPECT_EQ(op_dest.GetAttr("relative", relative), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(relative);
}

TEST(OnnxYoloV5DetectionOutputPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2},{"name":"alpha","type":1},{"name":"relative","type":2},)"
        R"({"name":"biases","type":6,"floats":["1.0"]}]})");
    ge::Operator op_dest = CreateOperator("yolov5");
    int boxes = 3;
    float alpha = 2.0f;
    bool relative = true;

    EXPECT_EQ(domi::ParseParamsYoloV5DetectionOutput(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 0);
    EXPECT_EQ(op_dest.GetAttr("alpha", alpha), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(alpha, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("relative", relative), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(relative);
}

TEST(OnnxYoloV5DetectionOutputPluginTest, ReturnFailedWhenBiasesMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("yolov5");
    EXPECT_EQ(domi::ParseParamsYoloV5DetectionOutput(op_src, op_dest), domi::FAILED);
}

TEST(OnnxYoloV5DetectionOutputPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("yolov5");
    EXPECT_EQ(domi::ParseParamsYoloV5DetectionOutput(op_src, op_dest), domi::FAILED);
}
