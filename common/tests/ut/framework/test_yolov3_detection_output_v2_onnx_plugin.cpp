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

#include "../../../src/framework/yolov3_detection_output_v2_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxYolov3DetectionOutputV2PluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2,"i":5},{"name":"classes","type":2,"i":20},)"
        R"({"name":"N","type":2,"i":8},{"name":"obj_threshold","type":1,"f":"0.6"},)"
        R"({"name":"score_threshold","type":1,"f":"0.7"},{"name":"iou_threshold","type":1,"f":"0.3"},)"
        R"({"name":"biases","type":6,"floats":["10.0","20.0","30.0"]}]})");
    ge::Operator op_dest = CreateOperator("yolov3");
    int boxes = 0;
    int classes = 0;
    int N = 0;
    float obj_threshold = 0.0f;
    float iou_threshold = 0.0f;
    std::vector<float> biases;

    EXPECT_EQ(domi::ParseParamsYolov3detectionoutputv2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 5);
    EXPECT_EQ(op_dest.GetAttr("classes", classes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(classes, 20);
    EXPECT_EQ(op_dest.GetAttr("N", N), ge::GRAPH_SUCCESS);
    EXPECT_EQ(N, 8);
    EXPECT_EQ(op_dest.GetAttr("obj_threshold", obj_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(obj_threshold, 0.6f);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.3f);
    EXPECT_EQ(op_dest.GetAttr("biases", biases), ge::GRAPH_SUCCESS);
    ASSERT_EQ(biases.size(), 3U);
    EXPECT_FLOAT_EQ(biases[0], 10.0f);
}

TEST(OnnxYolov3DetectionOutputV2PluginTest, UseDefaultsWithRequiredBiases)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"biases","type":6,"floats":["1.0","2.0"]}]})");
    ge::Operator op_dest = CreateOperator("yolov3");
    int boxes = 0;
    float obj_threshold = 0.0f;
    bool relative = false;

    EXPECT_EQ(domi::ParseParamsYolov3detectionoutputv2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 3);
    EXPECT_EQ(op_dest.GetAttr("obj_threshold", obj_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(obj_threshold, 0.5f);
    EXPECT_EQ(op_dest.GetAttr("relative", relative), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(relative);
}

TEST(OnnxYolov3DetectionOutputV2PluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"boxes","type":2},{"name":"N","type":2},{"name":"obj_threshold","type":1},)"
        R"({"name":"biases","type":6,"floats":["1.0"]}]})");
    ge::Operator op_dest = CreateOperator("yolov3");
    int boxes = 3;
    int N = 10;
    float obj_threshold = 0.5f;

    EXPECT_EQ(domi::ParseParamsYolov3detectionoutputv2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("boxes", boxes), ge::GRAPH_SUCCESS);
    EXPECT_EQ(boxes, 0);
    EXPECT_EQ(op_dest.GetAttr("N", N), ge::GRAPH_SUCCESS);
    EXPECT_EQ(N, 0);
    EXPECT_EQ(op_dest.GetAttr("obj_threshold", obj_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(obj_threshold, 0.0f);
}

TEST(OnnxYolov3DetectionOutputV2PluginTest, ReturnFailedWhenBiasesMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("yolov3");
    EXPECT_EQ(domi::ParseParamsYolov3detectionoutputv2(op_src, op_dest), domi::FAILED);
}

TEST(OnnxYolov3DetectionOutputV2PluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("yolov3");
    EXPECT_EQ(domi::ParseParamsYolov3detectionoutputv2(op_src, op_dest), domi::FAILED);
}
