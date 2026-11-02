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

#include "../../../framework/grid_sample_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

class GridSampleDescriptorOperator : public ge::Operator {
public:
    explicit GridSampleDescriptorOperator(const char* name) : ge::Operator(name, "TestOp")
    {
        InputRegister("x");
        OutputRegister("y");
    }
};

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleAllAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"align_corners","type":2,"i":1},{"name":"mode","type":3,"s":"bilinear"},)"
        R"({"name":"padding_mode","type":3,"s":"zeros"}]})");
    ge::Operator op_dest = CreateOperator("grid_sample");
    bool align_corners = false;
    ge::AscendString mode;
    ge::AscendString padding_mode;

    EXPECT_EQ(domi::ParseParamsGridSample(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(align_corners);
    EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "bilinear");
    EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(padding_mode.GetString(), "zeros");
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleUseDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("grid_sample");
    bool align_corners = true;
    ge::AscendString mode;
    ge::AscendString padding_mode;

    EXPECT_EQ(domi::ParseParamsGridSample(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(align_corners);
    EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "bilinear");
    EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(padding_mode.GetString(), "zeros");
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleZeroWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"align_corners","type":2}]})");
    ge::Operator op_dest = CreateOperator("grid_sample");
    bool align_corners = true;

    EXPECT_EQ(domi::ParseParamsGridSample(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(align_corners);
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleInvalidJson)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("grid_sample");
    EXPECT_EQ(domi::ParseParamsGridSample(op_src, op_dest), domi::FAILED);
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleV2AllAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"align_corners","type":2,"i":1},{"name":"mode","type":3,"s":"nearest"},)"
        R"({"name":"padding_mode","type":3,"s":"border"},{"name":"channel_last","type":2,"i":1},)"
        R"({"name":"scheduler_mode","type":2,"i":2}]})");
    ge::Operator op_dest = CreateOperator("grid_sample_v2");
    bool align_corners = false;
    bool channel_last = false;
    int scheduler_mode = 0;
    ge::AscendString mode;
    ge::AscendString padding_mode;

    EXPECT_EQ(domi::ParseParamsGridSampleV2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(align_corners);
    EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "nearest");
    EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(padding_mode.GetString(), "border");
    EXPECT_EQ(op_dest.GetAttr("channel_last", channel_last), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(channel_last);
    EXPECT_EQ(op_dest.GetAttr("scheduler_mode", scheduler_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(scheduler_mode, 2);
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleV2UseDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("grid_sample_v2");
    bool align_corners = true;
    bool channel_last = true;
    int scheduler_mode = 9;
    ge::AscendString mode;
    ge::AscendString padding_mode;

    EXPECT_EQ(domi::ParseParamsGridSampleV2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(align_corners);
    EXPECT_EQ(op_dest.GetAttr("channel_last", channel_last), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(channel_last);
    EXPECT_EQ(op_dest.GetAttr("scheduler_mode", scheduler_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(scheduler_mode, 0);
    EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(mode.GetString(), "bilinear");
    EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(padding_mode.GetString(), "zeros");
}

TEST(OnnxGridSamplePluginTest, MissingAttributeMetadataUsesDefaults)
{
    for (auto parser : {domi::ParseParamsGridSample, domi::ParseParamsGridSampleV2}) {
        ge::Operator op_src = CreateOperator("src");
        ge::Operator op_dest = CreateOperator("grid_sample");
        ge::AscendString mode;
        ge::AscendString padding_mode;

        EXPECT_EQ(parser(op_src, op_dest), domi::SUCCESS);
        EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(mode.GetString(), "bilinear");
        EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(padding_mode.GetString(), "zeros");
    }
}

TEST(OnnxGridSamplePluginTest, ExplicitModePreservesDefaultPadding)
{
    for (auto parser : {domi::ParseParamsGridSample, domi::ParseParamsGridSampleV2}) {
        ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"mode","type":3,"s":"nearest"}]})");
        ge::Operator op_dest = CreateOperator("grid_sample");
        ge::AscendString mode;
        ge::AscendString padding_mode;

        EXPECT_EQ(parser(op_src, op_dest), domi::SUCCESS);
        EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(mode.GetString(), "nearest");
        EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(padding_mode.GetString(), "zeros");
    }
}

TEST(OnnxGridSamplePluginTest, ExplicitPaddingPreservesDefaultMode)
{
    for (auto parser : {domi::ParseParamsGridSample, domi::ParseParamsGridSampleV2}) {
        ge::Operator op_src = CreateSourceOperator(
            R"({"attribute":[{"name":"padding_mode","type":3,"s":"reflection"}]})");
        ge::Operator op_dest = CreateOperator("grid_sample");
        ge::AscendString mode;
        ge::AscendString padding_mode;

        EXPECT_EQ(parser(op_src, op_dest), domi::SUCCESS);
        EXPECT_EQ(op_dest.GetAttr("interpolation_mode", mode), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(mode.GetString(), "bilinear");
        EXPECT_EQ(op_dest.GetAttr("padding_mode", padding_mode), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(padding_mode.GetString(), "reflection");
    }
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleV2ZeroWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"align_corners","type":2},{"name":"channel_last","type":2},)"
        R"({"name":"scheduler_mode","type":2}]})");
    ge::Operator op_dest = CreateOperator("grid_sample_v2");
    bool align_corners = true;
    bool channel_last = true;
    int scheduler_mode = 9;

    EXPECT_EQ(domi::ParseParamsGridSampleV2(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("align_corners", align_corners), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(align_corners);
    EXPECT_EQ(op_dest.GetAttr("channel_last", channel_last), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(channel_last);
    EXPECT_EQ(op_dest.GetAttr("scheduler_mode", scheduler_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(scheduler_mode, 0);
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleV2InvalidJson)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("grid_sample_v2");
    EXPECT_EQ(domi::ParseParamsGridSampleV2(op_src, op_dest), domi::FAILED);
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSamplePreservesFloat16OutputDescriptor)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    GridSampleDescriptorOperator op_dest("grid_sample");
    const ge::TensorDesc input_desc(ge::Shape({1, 3, 8, 8}), ge::FORMAT_ND, ge::DT_FLOAT16);
    const ge::TensorDesc output_desc(ge::Shape({1, 3, 4, 5}), ge::FORMAT_ND, ge::DT_FLOAT16);
    ASSERT_EQ(op_dest.UpdateInputDesc("x", input_desc), ge::GRAPH_SUCCESS);
    ASSERT_EQ(op_dest.UpdateOutputDesc("y", output_desc), ge::GRAPH_SUCCESS);

    ASSERT_EQ(domi::ParseParamsGridSample(op_src, op_dest), domi::SUCCESS);
    const auto actual = op_dest.GetOutputDescByName("y");
    EXPECT_EQ(actual.GetShape().GetDims(), output_desc.GetShape().GetDims());
    EXPECT_EQ(actual.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(actual.GetFormat(), ge::FORMAT_NCHW);
    EXPECT_EQ(actual.GetOriginFormat(), ge::FORMAT_NCHW);
}

TEST(OnnxGridSamplePluginTest, ParseParamsGridSampleKeepsDistinctInputAndOutputShapes)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    GridSampleDescriptorOperator op_dest("grid_sample");
    const ge::TensorDesc input_desc(ge::Shape({2, 7, 8, 9}), ge::FORMAT_ND, ge::DT_FLOAT);
    const ge::TensorDesc output_desc(ge::Shape({2, 7, 3, 6}), ge::FORMAT_ND, ge::DT_FLOAT);
    ASSERT_EQ(op_dest.UpdateInputDesc("x", input_desc), ge::GRAPH_SUCCESS);
    ASSERT_EQ(op_dest.UpdateOutputDesc("y", output_desc), ge::GRAPH_SUCCESS);

    ASSERT_EQ(domi::ParseParamsGridSample(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetInputDescByName("x").GetShape().GetDims(), input_desc.GetShape().GetDims());
    EXPECT_EQ(op_dest.GetOutputDescByName("y").GetShape().GetDims(), output_desc.GetShape().GetDims());
}
