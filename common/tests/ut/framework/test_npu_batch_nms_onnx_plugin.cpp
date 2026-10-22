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

#include "../../../src/framework/npu_batch_nms_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuBatchNmsPluginTest, ParseAllAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"iou_threshold","type":1,"f":"0.55"},{"name":"score_threshold","type":1,"f":"0.3"},)"
        R"({"name":"max_size_per_class","type":2,"i":20},{"name":"max_total_size","type":2,"i":40},)"
        R"({"name":"change_coordinate_frame","type":2,"i":1},{"name":"transpose_box","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("batch_nms");
    float iou_threshold = 0.0f;
    bool change_coordinate_frame = false;
    bool transpose_box = false;

    EXPECT_EQ(domi::ParseParamsBatchNMS(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.55f);
    EXPECT_EQ(op_dest.GetAttr("change_coordinate_frame", change_coordinate_frame), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(change_coordinate_frame);
    EXPECT_EQ(op_dest.GetAttr("transpose_box", transpose_box), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(transpose_box);
}

TEST(OnnxNpuBatchNmsPluginTest, SuccessWhenAttributeEmpty)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("batch_nms");
    EXPECT_EQ(domi::ParseParamsBatchNMS(op_src, op_dest), domi::SUCCESS);
}

TEST(OnnxNpuBatchNmsPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"iou_threshold","type":1},{"name":"score_threshold","type":1},)"
        R"({"name":"max_size_per_class","type":2},{"name":"max_total_size","type":2},)"
        R"({"name":"change_coordinate_frame","type":2},{"name":"transpose_box","type":2}]})");
    ge::Operator op_dest = CreateOperator("batch_nms");
    float iou_threshold = 1.0f;
    int max_size_per_class = 1;
    bool change_coordinate_frame = true;
    bool transpose_box = true;

    EXPECT_EQ(domi::ParseParamsBatchNMS(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("iou_threshold", iou_threshold), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(iou_threshold, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("max_size_per_class", max_size_per_class), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_size_per_class, 0);
    EXPECT_EQ(op_dest.GetAttr("change_coordinate_frame", change_coordinate_frame), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(change_coordinate_frame);
    EXPECT_EQ(op_dest.GetAttr("transpose_box", transpose_box), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(transpose_box);
}

TEST(OnnxNpuBatchNmsPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("batch_nms");
    EXPECT_EQ(domi::ParseParamsBatchNMS(op_src, op_dest), domi::FAILED);
}
