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

#include "../../../src/framework/roi_extractor_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}

ge::Operator CreateSourceOperatorWithInputs(const std::string& attrs, uint32_t input_num)
{
    ge::Operator op_src = CreateSourceOperator(attrs);
    op_src.DynamicInputRegister("x", input_num);
    return op_src;
}

} // namespace

TEST(OnnxRoiExtractorPluginTest, ParseAttributes)
{
    ge::Operator op_src = CreateSourceOperatorWithInputs(
        R"({"attribute":[{"name":"finest_scale","type":2,"i":16},{"name":"roi_scale_factor","type":1,"f":"1.5"},)"
        R"({"name":"spatial_scale","type":6,"floats":["0.25","0.125"]},{"name":"pooled_height","type":2,"i":7},)"
        R"({"name":"pooled_width","type":2,"i":7},{"name":"sample_num","type":2,"i":2},)"
        R"({"name":"pool_mode","type":3,"s":"avg"},{"name":"aligned","type":2,"i":1}]})",
        3);
    ge::Operator op_dest = CreateOperator("roi_extractor");
    int64_t finest_scale = 0;
    float roi_scale_factor = 0.0f;
    std::vector<float> spatial_scale;
    int64_t pooled_height = 0;
    ge::AscendString pool_mode;
    bool aligned = false;

    EXPECT_EQ(domi::ParseParamsRoiExtractor(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("finest_scale", finest_scale), ge::GRAPH_SUCCESS);
    EXPECT_EQ(finest_scale, 16);
    EXPECT_EQ(op_dest.GetAttr("roi_scale_factor", roi_scale_factor), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(roi_scale_factor, 1.5f);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale", spatial_scale), ge::GRAPH_SUCCESS);
    ASSERT_EQ(spatial_scale.size(), 2U);
    EXPECT_FLOAT_EQ(spatial_scale[0], 0.25f);
    EXPECT_FLOAT_EQ(spatial_scale[1], 0.125f);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 7);
    EXPECT_EQ(op_dest.GetAttr("pool_mode", pool_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(pool_mode.GetString(), "avg");
    EXPECT_EQ(op_dest.GetAttr("aligned", aligned), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(aligned);
}

TEST(OnnxRoiExtractorPluginTest, SuccessWhenAttributeEmptyWithEnoughInputs)
{
    ge::Operator op_src = CreateSourceOperatorWithInputs(R"({"attribute":[]})", 2);
    ge::Operator op_dest = CreateOperator("roi_extractor");
    EXPECT_EQ(domi::ParseParamsRoiExtractor(op_src, op_dest), domi::SUCCESS);
}

TEST(OnnxRoiExtractorPluginTest, ParseZeroValueAttrsWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperatorWithInputs(
        R"({"attribute":[{"name":"finest_scale","type":2},{"name":"roi_scale_factor","type":1},)"
        R"({"name":"pooled_height","type":2},{"name":"pooled_width","type":2},)"
        R"({"name":"sample_num","type":2},{"name":"aligned","type":2}]})",
        2);
    ge::Operator op_dest = CreateOperator("roi_extractor");
    int64_t finest_scale = 1;
    float roi_scale_factor = 1.0f;
    int64_t pooled_height = 1;
    int64_t sample_num = 1;
    bool aligned = true;

    EXPECT_EQ(domi::ParseParamsRoiExtractor(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("finest_scale", finest_scale), ge::GRAPH_SUCCESS);
    EXPECT_EQ(finest_scale, 0);
    EXPECT_EQ(op_dest.GetAttr("roi_scale_factor", roi_scale_factor), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(roi_scale_factor, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("pooled_height", pooled_height), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_height, 0);
    EXPECT_EQ(op_dest.GetAttr("sample_num", sample_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sample_num, 0);
    EXPECT_EQ(op_dest.GetAttr("aligned", aligned), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(aligned);
}

TEST(OnnxRoiExtractorPluginTest, ReturnFailedWhenInputsInsufficient)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("roi_extractor");
    EXPECT_EQ(domi::ParseParamsRoiExtractor(op_src, op_dest), domi::FAILED);
}

TEST(OnnxRoiExtractorPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperatorWithInputs("{", 2);
    ge::Operator op_dest = CreateOperator("roi_extractor");
    EXPECT_EQ(domi::ParseParamsRoiExtractor(op_src, op_dest), domi::FAILED);
}
