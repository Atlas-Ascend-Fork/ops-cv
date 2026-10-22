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

#include "../../../src/framework/batch_multi_class_non_max_suppression_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxBatchMultiClassNMSPluginTest, ParseAllAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"iou_threshold","type":1,"f":"0.6"},{"name":"score_threshold","type":1,"f":"0.4"},)"
        R"({"name":"max_size_per_class","type":2,"i":50},{"name":"max_total_size","type":2,"i":100},)"
        R"({"name":"image_size","type":7,"ints":[640,480]}]})");
    ge::Operator op_dest = CreateOperator("bmc_nms");
    float iou_threshold = 0.0f;
    float score_threshold = 0.0f;
    int max_size_per_class = 0;
    int max_total_size = 0;
    std::vector<int64_t> image_size;

    EXPECT_EQ(domi::ParseParamsBatchMultiClassNMS(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.6f);
    EXPECT_EQ(op_dest.GetAttr("score_threshold", score_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(score_threshold, 0.4f);
    EXPECT_EQ(op_dest.GetAttr("max_size_per_class", max_size_per_class), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_size_per_class, 50);
    EXPECT_EQ(op_dest.GetAttr("max_total_size", max_total_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_total_size, 100);
    EXPECT_EQ(op_dest.GetAttr("image_size", image_size), ge::GRAPH_SUCCESS);
    ASSERT_EQ(image_size.size(), 2U);
    EXPECT_EQ(image_size[0], 640);
    EXPECT_EQ(image_size[1], 480);
}

TEST(OnnxBatchMultiClassNMSPluginTest, SuccessWhenAttributeEmpty)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("bmc_nms");
    EXPECT_EQ(domi::ParseParamsBatchMultiClassNMS(op_src, op_dest), domi::SUCCESS);
}

TEST(OnnxBatchMultiClassNMSPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"iou_threshold","type":1},{"name":"score_threshold","type":1},)"
        R"({"name":"max_size_per_class","type":2},{"name":"max_total_size","type":2}]})");
    ge::Operator op_dest = CreateOperator("bmc_nms");
    float iou_threshold = 1.0f;
    float score_threshold = 1.0f;
    int max_size_per_class = 1;
    int max_total_size = 1;

    EXPECT_EQ(domi::ParseParamsBatchMultiClassNMS(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("score_threshold", score_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(score_threshold, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("max_size_per_class", max_size_per_class), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_size_per_class, 0);
    EXPECT_EQ(op_dest.GetAttr("max_total_size", max_total_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_total_size, 0);
}

TEST(OnnxBatchMultiClassNMSPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("bmc_nms");
    EXPECT_EQ(domi::ParseParamsBatchMultiClassNMS(op_src, op_dest), domi::FAILED);
}
