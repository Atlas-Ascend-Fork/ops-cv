/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_host/arch35/resize_linear_grad_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeLinearGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeLinearGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeLinearGradTilingTest TearDown" << std::endl; }
};

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_01)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 0 96 1 1 25088 96 96 100352 32 4575657222473777152 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_02)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape outputShape = {{1, 3, 16}, {1, 3, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 16843008;
    string expectTilingData = "48 0 1 0 1 0 0 0 16 32 4612886978239118831 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_03)
{
    gert::StorageShape inputGradsShape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape inputOriImageShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape outputShape = {{1, 3, 16}, {1, 3, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 260;
    string expectTilingData = "3 48 1 0 1 1 0 1 16 1 4575657222473777152 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_04)
{
    gert::StorageShape inputGradsShape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape inputOriImageShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape outputShape = {{1, 3, 16}, {1, 3, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 259;
    string expectTilingData = "3 48 1 0 1 1 0 1 16 1 1098907648 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 05: upsampling with integer scale, align_corners=false
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_05)
{
    gert::StorageShape inputGradsShape = {{2, 5, 32}, {2, 5, 32}};
    gert::StorageShape inputOriImageShape = {{2, 5, 16}, {2, 5, 16}};
    gert::StorageShape outputShape = {{2, 5, 16}, {2, 5, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(2.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: large tensor upsampling with scales
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_06)
{
    gert::StorageShape inputGradsShape = {{200, 8, 64}, {200, 8, 64}};
    gert::StorageShape inputOriImageShape = {{200, 8, 32}, {200, 8, 32}};
    gert::StorageShape outputShape = {{200, 8, 32}, {200, 8, 32}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 07: downsampling with align_corners=true
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_07)
{
    gert::StorageShape inputGradsShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}
