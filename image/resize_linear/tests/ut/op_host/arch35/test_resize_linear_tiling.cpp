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
#include <limits>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_host/arch35/resize_linear_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeLinearTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeLinearTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeLinearTilingTest TearDown" << std::endl; }
};

TEST_F(ResizeLinearTilingTest, resize_linear_tiling_01)
{
    gert::StorageShape inputXShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};
    int size_value[1] = {32};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 65793;
    string expectTilingData = "64 1 32 32 32 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearTilingTest, resize_linear_tiling_02)
{
    gert::StorageShape inputXShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};
    int size_value[1] = {32};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(2.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 257;
    string expectTilingData = "64 1 32 32 32 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearTilingTest, resize_linear_tiling_03)
{
    gert::StorageShape inputXShape = {{1024, 2048, 4096}, {1024, 2048, 4096}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{1024, 2048, 4096}, {1024, 2048, 4096}};
    int size_value[1] = {4096};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(2.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 134217728 0 4096 4096 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 04: upsampling with integer scale + align_corners (DIM_4)
TEST_F(ResizeLinearTilingTest, resize_linear_tiling_04)
{
    gert::StorageShape inputXShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{1, 3, 64}, {1, 3, 64}};
    int size_value[1] = {64};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(4.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 05: input size = 1 with align_corners=false (DIM_2)
TEST_F(ResizeLinearTilingTest, resize_linear_tiling_05)
{
    gert::StorageShape inputXShape = {{2, 5, 1}, {2, 5, 1}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{2, 5, 12}, {2, 5, 12}};
    int size_value[1] = {12};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: output size = 1 with align_corners=true (DIM_3)
TEST_F(ResizeLinearTilingTest, resize_linear_tiling_06)
{
    gert::StorageShape inputXShape = {{1, 3, 64}, {1, 3, 64}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{1, 3, 1}, {1, 3, 1}};
    int size_value[1] = {1};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 07: non-integer scale upsampling (DIM_0 fallback)
TEST_F(ResizeLinearTilingTest, resize_linear_tiling_07)
{
    gert::StorageShape inputXShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputSizeShape = {{
                                             1,
                                         },
                                         {
                                             1,
                                         }};
    gert::StorageShape outputShape = {{1, 3, 50}, {1, 3, 50}};
    int size_value[1] = {50};

    ResizeLinearCompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinear",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.5f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ResizeLinearTilingTest, resize_linear_tiling_nonfinite_scale)
{
    for (const float scale : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        gert::StorageShape inputXShape = {{1, 3, 2}, {1, 3, 2}};
        gert::StorageShape inputSizeShape = {{1}, {1}};
        gert::StorageShape outputShape = {{1, 3, 4}, {1, 3, 4}};
        int size_value[1] = {4};
        ResizeLinearCompileInfo compileInfo = {64, 200704};
        gert::TilingContextPara tilingContextPara(
            "ResizeLinear",
            {{inputXShape, ge::DT_FLOAT, ge::FORMAT_ND},
             {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
            {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
             gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(scale))},
            &compileInfo);
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << scale;
    }
}
