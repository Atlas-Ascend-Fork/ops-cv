/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>

#include "../../../framework/roi_align_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignAllAttributes)
{
    // Include mode=max: this is the regression for silently dropping ONNX mode.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"output_height","type":2,"i":7},{"name":"output_width","type":2,"i":8},)"
        R"({"name":"sampling_ratio","type":2,"i":2},{"name":"spatial_scale","type":1,"f":"0.25"},)"
        R"({"name":"mode","type":3,"s":"max"}]})",
        "roi_align_src");
    ge::Operator op_dest = CreateOperator("roi_align");
    int64_t pooled_height = 0;
    int64_t pooled_width = 0;
    int64_t sample_num = 0;
    float spatial_scale = 0.0f;
    int64_t roi_end_mode = 9;
    std::string pool_mode = "avg";
    ge::AscendString name;

    EXPECT_EQ(domi::ParseParamsRoiAlign(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 7);
    EXPECT_EQ(op_dest.GetAttr("pooled_width", pooled_width), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_width, 8);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 2);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.25f);
    EXPECT_EQ(op_dest.GetAttr("pool_mode", pool_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pool_mode, "max");
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 0);
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(name.GetString(), "roi_align_src");
}

TEST(OnnxRoiAlignPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"output_height","type":2},{"name":"output_width","type":2},)"
        R"({"name":"spatial_scale","type":1}]})");
    ge::Operator op_dest = CreateOperator("roi_align");
    int64_t pooled_height = 9;
    int64_t pooled_width = 9;
    int64_t sample_num = 9;
    float spatial_scale = 1.0f;

    EXPECT_EQ(domi::ParseParamsRoiAlign(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 0);
    EXPECT_EQ(op_dest.GetAttr("pooled_width", pooled_width), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_width, 0);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 0);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.0f);
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignV16ZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"output_height","type":2},{"name":"output_width","type":2},)"
        R"({"name":"spatial_scale","type":1}]})");
    ge::Operator op_dest = CreateOperator("roi_align_v16");
    int64_t pooled_height = 9;
    int64_t pooled_width = 9;
    float spatial_scale = 1.0f;
    int64_t roi_end_mode = 0;

    EXPECT_EQ(domi::ParseParamsRoiAlignV16(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 0);
    EXPECT_EQ(op_dest.GetAttr("pooled_width", pooled_width), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_width, 0);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 2);
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignUseDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("roi_align");
    int64_t pooled_height = 0;
    int64_t pooled_width = 0;
    int64_t sample_num = 9;
    float spatial_scale = 0.0f;
    int64_t roi_end_mode = 9;
    std::string pool_mode = "max";

    EXPECT_EQ(domi::ParseParamsRoiAlign(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 1);
    EXPECT_EQ(op_dest.GetAttr("pooled_width", pooled_width), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_width, 1);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 0);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale, 1.0f);
    // mode absent → ONNX/GE default avg (initializer), not an empty string.
    EXPECT_EQ(op_dest.GetAttr("pool_mode", pool_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pool_mode, "avg");
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 0);
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignInvalidJson)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("roi_align");
    EXPECT_EQ(domi::ParseParamsRoiAlign(op_src, op_dest), domi::FAILED);
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignV16DefaultRoiEndMode)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("roi_align_v16");
    int64_t roi_end_mode = 0;
    std::string pool_mode = "max";

    EXPECT_EQ(domi::ParseParamsRoiAlignV16(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 2);
    EXPECT_EQ(op_dest.GetAttr("pool_mode", pool_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pool_mode, "avg");
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignV16OutputHalfPixel)
{
    // V16-only attr + shared mode parse: UpdateInfo must keep pool_mode after V16 post-pass.
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"coordinate_transformation_mode","type":3,"s":"output_half_pixel"},)"
        R"({"name":"mode","type":3,"s":"max"}]})");
    ge::Operator op_dest = CreateOperator("roi_align_v16");
    int64_t roi_end_mode = 9;
    std::string pool_mode = "avg";

    EXPECT_EQ(domi::ParseParamsRoiAlignV16(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("roi_end_mode", roi_end_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(roi_end_mode, 0);
    EXPECT_EQ(op_dest.GetAttr("pool_mode", pool_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pool_mode, "max");
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignV16InvalidJson)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("roi_align_v16");
    EXPECT_EQ(domi::ParseParamsRoiAlignV16(op_src, op_dest), domi::FAILED);
}

TEST(OnnxRoiAlignPluginTest, ParseParamsRoiAlignRejectsInvalidOrEmptyMode)
{
    // Illegal ONNX mode, and attr present without "s" (not the absent-attr default path).
    ge::Operator invalid_src = CreateSourceOperator(R"({"attribute":[{"name":"mode","type":3,"s":"bilinear"}]})");
    ge::Operator invalid_dest = CreateOperator("roi_align");
    ge::Operator empty_src = CreateSourceOperator(R"({"attribute":[{"name":"mode","type":3}]})");
    ge::Operator empty_dest = CreateOperator("roi_align");

    EXPECT_EQ(domi::ParseParamsRoiAlign(invalid_src, invalid_dest), domi::FAILED);
    EXPECT_EQ(domi::ParseParamsRoiAlign(empty_src, empty_dest), domi::FAILED);
}
