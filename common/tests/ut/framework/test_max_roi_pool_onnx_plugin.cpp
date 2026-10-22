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

#include "../../../src/framework/max_roi_pool_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name.c_str(), "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs, const std::string& name = "src")
{
    ge::Operator op_src = CreateOperator(name);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxMaxRoiPoolPluginTest, ParsePooledShapeAndSpatialScale)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"pooled_shape","type":7,"ints":[7,8]},{"name":"spatial_scale","type":1,"f":"0.25"}]})");
    ge::Operator op_dest = CreateOperator("max_roi_pool");
    int pooled_h = 0;
    int pooled_w = 0;
    float spatial_scale_h = 0.0f;
    float spatial_scale_w = 0.0f;

    EXPECT_EQ(domi::ParseParamsMaxRoiPool(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pooled_h", pooled_h), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_h, 7);
    EXPECT_EQ(op_dest.GetAttr("pooled_w", pooled_w), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pooled_w, 8);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale_h", spatial_scale_h), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale_h, 0.25f);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale_w", spatial_scale_w), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale_w, 0.25f);
}

TEST(OnnxMaxRoiPoolPluginTest, UseDefaultSpatialScaleWhenMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"pooled_shape","type":7,"ints":[2,3]}]})");
    ge::Operator op_dest = CreateOperator("max_roi_pool");
    float spatial_scale_h = 0.0f;

    EXPECT_EQ(domi::ParseParamsMaxRoiPool(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale_h", spatial_scale_h), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale_h, 1.0f);
}

TEST(OnnxMaxRoiPoolPluginTest, ParseZeroSpatialScaleWhenScalarFieldOmitted)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"pooled_shape","type":7,"ints":[2,3]},{"name":"spatial_scale","type":1}]})");
    ge::Operator op_dest = CreateOperator("max_roi_pool");
    float spatial_scale_h = 1.0f;

    EXPECT_EQ(domi::ParseParamsMaxRoiPool(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("spatial_scale_h", spatial_scale_h), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(spatial_scale_h, 0.0f);
}

TEST(OnnxMaxRoiPoolPluginTest, ReturnFailedWhenPooledShapeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("max_roi_pool");
    EXPECT_EQ(domi::ParseParamsMaxRoiPool(op_src, op_dest), domi::FAILED);
}

TEST(OnnxMaxRoiPoolPluginTest, ReturnFailedWhenAttributeJsonInvalid)
{
    ge::Operator op_src = CreateSourceOperator("{");
    ge::Operator op_dest = CreateOperator("max_roi_pool");
    EXPECT_EQ(domi::ParseParamsMaxRoiPool(op_src, op_dest), domi::FAILED);
}
