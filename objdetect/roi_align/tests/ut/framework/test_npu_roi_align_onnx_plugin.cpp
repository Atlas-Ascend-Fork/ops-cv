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

#include "../../../framework/npu_roi_align_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuRoiAlignPluginTest, ParseRequiredAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1,"f":"0.5"},{"name":"pooled_height","type":2,"i":7},)"
        R"({"name":"pooled_width","type":2,"i":7},{"name":"sample_num","type":2,"i":4},)"
        R"({"name":"roi_end_mode","type":2,"i":0}]})");
    ge::Operator op_dest = CreateOperator("npu_roi_align");
    float spatial_scale = 0.0f;
    int64_t pooled_height = 0;
    int64_t pooled_width = 0;
    int64_t sample_num = 0;
    int64_t roi_end_mode = 0;

    EXPECT_EQ(domi::ParseParamsNpuRoiAlign(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.5f);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 7);
    EXPECT_EQ(op_dest.GetAttr("pooled_width", pooled_width), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_width, 7);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 4);
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 0);
}

TEST(OnnxNpuRoiAlignPluginTest, KeepOptionalDefaultsWhenAttrsAbsent)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1,"f":"0.5"},{"name":"pooled_height","type":2,"i":7},)"
        R"({"name":"pooled_width","type":2,"i":7}]})");
    ge::Operator op_dest = CreateOperator("npu_roi_align");
    int64_t sample_num = 0;
    int64_t roi_end_mode = 0;

    EXPECT_EQ(domi::ParseParamsNpuRoiAlign(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 2);
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 1);
}

TEST(OnnxNpuRoiAlignPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"spatial_scale","type":1},{"name":"pooled_height","type":2},)"
        R"({"name":"pooled_width","type":2},{"name":"sample_num","type":2},{"name":"roi_end_mode","type":2}]})");
    ge::Operator op_dest = CreateOperator("npu_roi_align");
    float spatial_scale = 1.0f;
    int64_t pooled_height = 1;
    int64_t sample_num = 2;
    int64_t roi_end_mode = 1;

    EXPECT_EQ(domi::ParseParamsNpuRoiAlign(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 0);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 0);
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 0);
}

TEST(OnnxNpuRoiAlignPluginTest, ReturnFailedWhenRequiredAttrsMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("npu_roi_align");
    EXPECT_EQ(domi::ParseParamsNpuRoiAlign(op_src, op_dest), domi::FAILED);
}

TEST(OnnxNpuRoiAlignPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("npu_roi_align");
    EXPECT_EQ(domi::ParseParamsNpuRoiAlign(op_src, op_dest), domi::FAILED);
}
