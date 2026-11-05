/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "../../../op_host/grid_sampler_2d_grad_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class GridSampler2DGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GridSampler2DGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GridSampler2DGradTiling TearDown" << std::endl; }
};

struct GridSampler2DGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool regBase{false};
};

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT, ge::FORMAT_ND}, {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "8589934594 12884901920 68719476752 34359738376 3573412790320 0 4294967297 206158430209 "
                              "32 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case2)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT, ge::FORMAT_ND}, {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "8589934594 12884901920 68719476752 34359738376 3573412790320 8589934592 4294967297 "
                              "206158430209 32 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case3)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT, ge::FORMAT_ND}, {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "8589934594 12884901920 68719476752 34359738376 7421703487536 1 137438953473 "
                              "206158430210 32 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case4)
{
    gert::StorageShape grad = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape x = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape grid = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    gert::StorageShape output1 = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape output2 = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT, ge::FORMAT_ND}, {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "4294967297 4294967296 4294967297 4294967297 7421703487489 4294967297 137438953473 "
                              "4294967298 1 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case5)
{
    gert::StorageShape grad = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape x = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape grid = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    gert::StorageShape output1 = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape output2 = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT, ge::FORMAT_ND}, {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "1 4294967296 4294967297 4294967297 7696581394433 4294967297 137438953473 4294967298 1 "
                              "28864 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case5)
// {
//     gert::StorageShape grad = {{65, 1, 1, 1}, {65, 1, 1, 1}};
//     gert::StorageShape x = {{65, 1, 1, 1}, {65, 1, 1, 1}};
//     gert::StorageShape grid = {{65, 1, 1, 2}, {65, 1, 1, 2}};
//     gert::StorageShape output1 = {{65, 1, 1, 1}, {65, 1, 1, 1}};
//     gert::StorageShape output2 = {{65, 1, 1, 2}, {65, 1, 1, 2}};
//     GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
//     gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
//                                                 {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {x, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
//                                                 {{output1, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
//                                                 {gert::TilingContextPara::OpAttr("interpolation_mode",
//                                                 Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
//                                                 gert::TilingContextPara::OpAttr("padding_mode",
//                                                 Ops::Cv::AnyValue::CreateFrom<string>("border")),
//                                                 gert::TilingContextPara::OpAttr("align_corners",
//                                                 Ops::Cv::AnyValue::CreateFrom<bool>(true))}, &compileInfo);
//     uint64_t expectTilingKey = 2;
//     string expectTilingData = "4294967361 4294967313 4294967297 4294967297 7421703487536 4294967297 137438953473
//     206158430210 73014444033 28864 0 "; std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float16_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_ND},
         {x, ge::DT_FLOAT16, ge::FORMAT_ND},
         {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT16, ge::FORMAT_ND}, {output2, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "549755813890 12884901888 68719476752 34359738376 3573412790273 0 4294967297 4294967299 "
                              "1536 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16783360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float16_case2)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_ND},
         {x, ge::DT_FLOAT16, ge::FORMAT_ND},
         {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT16, ge::FORMAT_ND}, {output2, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "549755813890 12884901888 68719476752 34359738376 3573412790273 8589934592 4294967297 "
                              "4294967299 1536 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16783360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float16_case3)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_ND},
         {x, ge::DT_FLOAT16, ge::FORMAT_ND},
         {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT16, ge::FORMAT_ND}, {output2, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "549755813890 12884901888 68719476752 34359738376 7421703487489 1 137438953473 "
                              "4294967300 1536 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16783360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float16_case4)
{
    gert::StorageShape grad = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape x = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape grid = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    gert::StorageShape output1 = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape output2 = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_ND},
         {x, ge::DT_FLOAT16, ge::FORMAT_ND},
         {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{output1, ge::DT_FLOAT16, ge::FORMAT_ND}, {output2, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "4294967297 4294967296 4294967297 4294967297 7421703487489 4294967297 137438953473 "
                              "4294967300 1 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16777220};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_bfloat16_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_BF16, ge::FORMAT_ND}, {x, ge::DT_BF16, ge::FORMAT_ND}, {grid, ge::DT_BF16, ge::FORMAT_ND}},
        {{output1, ge::DT_BF16, ge::FORMAT_ND}, {output2, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 5;
    string expectTilingData = "549755813890 12884901888 68719476752 34359738376 3573412790273 0 4294967297 4294967301 "
                              "1536 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16783360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_bfloat16_case2)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_BF16, ge::FORMAT_ND}, {x, ge::DT_BF16, ge::FORMAT_ND}, {grid, ge::DT_BF16, ge::FORMAT_ND}},
        {{output1, ge::DT_BF16, ge::FORMAT_ND}, {output2, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 5;
    string expectTilingData = "549755813890 12884901888 68719476752 34359738376 3573412790273 8589934592 4294967297 "
                              "4294967301 1536 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16783360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_bfloat16_case3)
{
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_BF16, ge::FORMAT_ND}, {x, ge::DT_BF16, ge::FORMAT_ND}, {grid, ge::DT_BF16, ge::FORMAT_ND}},
        {{output1, ge::DT_BF16, ge::FORMAT_ND}, {output2, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 6;
    string expectTilingData = "549755813890 12884901888 68719476752 34359738376 7421703487489 1 137438953473 "
                              "4294967302 1536 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16783360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_bfloat16_case4)
{
    gert::StorageShape grad = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape x = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape grid = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    gert::StorageShape output1 = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape output2 = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_BF16, ge::FORMAT_ND}, {x, ge::DT_BF16, ge::FORMAT_ND}, {grid, ge::DT_BF16, ge::FORMAT_ND}},
        {{output1, ge::DT_BF16, ge::FORMAT_ND}, {output2, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 6;
    string expectTilingData = "4294967297 4294967296 4294967297 4294967297 7421703487489 4294967297 137438953473 "
                              "4294967302 1 28864 0 ";
    std::vector<size_t> expectWorkspaces = {16777220};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_dtype_mismatch_case5)
{
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
    gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::StorageShape dx = {{2, 16, 16, 3}, {2, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TilingData 字段序：batch, pNumPerCore, tailPNum, channel, height, width, gridH, gridW, ...
static constexpr size_t TILING_DATA_CHANNEL_INDEX = 3;

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_regbase_nd_input_parsed_as_nchw)
{
    // regbase(950) 平台，ND 标签输入按 NCHW 语义解析：channel 取 x.dim1；
    // padding=border + 大通道使其余 SIMD 条件全部满足，验证 ND 标签仍不被派发到 SIMD kernel(19-21)
    gert::StorageShape grad = {{2, 64, 4, 4}, {2, 64, 4, 4}};
    gert::StorageShape x = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    gert::StorageShape dx = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape dgrid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // ND 标签走 NCHW 解析，channel=dim1=64；若误走 NHWC 解析 channel=dim3=16
    const auto* tilingData = reinterpret_cast<const uint32_t*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData[TILING_DATA_CHANNEL_INDEX], 64U);
    // ND 标签即便满足其余 SIMD 条件（border + 大通道 + float + 非确定），也不派发 SIMD kernel(19-21)
    EXPECT_LE(tilingInfo.tilingKey, 18);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_regbase_nhwc_simd_selected)
{
    // regbase(950) 平台，NHWC 标签 + FLOAT + 大通道 border + 非确定：选 SIMD tilingKey(19-21)
    gert::StorageShape grad = {{2, 4, 4, 64}, {2, 4, 4, 64}};
    gert::StorageShape x = {{2, 16, 16, 64}, {2, 16, 16, 64}};
    gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    gert::StorageShape dx = {{2, 16, 16, 64}, {2, 16, 16, 64}};
    gert::StorageShape dgrid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {x, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_NHWC}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 19);
    // NHWC 标签走 NHWC 解析，channel=dim3=64
    const auto* tilingData = reinterpret_cast<const uint32_t*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData[TILING_DATA_CHANNEL_INDEX], 64U);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_regbase_nhwc_channel_boundary_2048)
{
    // regbase(950) 平台，NHWC 标签 channel=2048 边界：SIMD 条件仍成立（<=2048 取边界值）
    gert::StorageShape grad = {{1, 4, 4, 2048}, {1, 4, 4, 2048}};
    gert::StorageShape x = {{1, 16, 16, 2048}, {1, 16, 16, 2048}};
    gert::StorageShape grid = {{1, 4, 4, 2}, {1, 4, 4, 2}};
    gert::StorageShape dx = {{1, 16, 16, 2048}, {1, 16, 16, 2048}};
    gert::StorageShape dgrid = {{1, 4, 4, 2}, {1, 4, 4, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {x, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_NHWC}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 20);
    // channel 超过 2048：SIMD 不成立且 isNhwc=true，落入 NHWC 拒绝分支
    gert::StorageShape gradOver = {{1, 4, 4, 2064}, {1, 4, 4, 2064}};
    gert::StorageShape xOver = {{1, 16, 16, 2064}, {1, 16, 16, 2064}};
    gert::StorageShape dxOver = {{1, 16, 16, 2064}, {1, 16, 16, 2064}};
    gert::TilingContextPara paraOver(
        "GridSampler2DGrad",
        {{gradOver, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {xOver, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dxOver, ge::DT_FLOAT, ge::FORMAT_NHWC}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(paraOver, ge::GRAPH_FAILED);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_regbase_mixed_format_rejected)
{
    // regbase(950) 平台，grad 与 x 混合标签（grad=NHWC、x=ND）：无法同时按两种布局解析 x，拒绝
    gert::StorageShape grad = {{2, 4, 4, 64}, {2, 4, 4, 64}};
    gert::StorageShape x = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    gert::StorageShape dx = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape dgrid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_NHWC}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_regbase_nhwc_without_simd_rejected)
{
    // regbase(950) 平台，NHWC 标签但不满足 SIMD 条件（float16）：拒绝，避免落 NCHW 寻址 kernel 静默算错
    gert::StorageShape grad = {{2, 4, 4, 64}, {2, 4, 4, 64}};
    gert::StorageShape x = {{2, 16, 16, 64}, {2, 16, 16, 64}};
    gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    gert::StorageShape dx = {{2, 16, 16, 64}, {2, 16, 16, 64}};
    gert::StorageShape dgrid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {x, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT16, ge::FORMAT_NHWC}, {dgrid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_regbase_nhwc_label_with_nchw_shape_rejected)
{
    // NHWC 语义 shape 下仅让 grad.dim1(4) 与 grid.dim1(5) 一处不一致（其余约束自洽）：
    // border+大通道使 SIMD 条件成立（排除 NHWC 拒绝分支），失败只能来自空间维一致性检查翻转
    gert::StorageShape grad = {{2, 4, 4, 64}, {2, 4, 4, 64}};
    gert::StorageShape x = {{2, 4, 16, 64}, {2, 4, 16, 64}};
    gert::StorageShape grid = {{2, 5, 4, 2}, {2, 5, 4, 2}};
    gert::StorageShape dx = {{2, 4, 16, 64}, {2, 4, 16, 64}};
    gert::StorageShape dgrid = {{2, 5, 4, 2}, {2, 5, 4, 2}};
    GridSampler2DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler2DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {x, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_NHWC}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case5)
// {
//     gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     GridSampler2DGradCompileInfo compileInfo = {48, 196608, false};
//     gert::TilingContextPara tilingContextPara("GridSampler2DGrad",
//                                                 {{grad, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {x, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
//                                                 {{output1, ge::DT_FLOAT, ge::FORMAT_ND},
//                                                 {output2, ge::DT_FLOAT, ge::FORMAT_ND}},
//                                                 {gert::TilingContextPara::OpAttr("interpolation_mode",
//                                                 Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
//                                                 gert::TilingContextPara::OpAttr("padding_mode",
//                                                 Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
//                                                 gert::TilingContextPara::OpAttr("align_corners",
//                                                 Ops::Cv::AnyValue::CreateFrom<bool>(true))}, &compileInfo);
//     uint64_t expectTilingKey = 2;
//     string expectTilingData = "8589934594 12884901920 68719476752 34359738376 7421703487536 1 137438953473
//     206158430210 32 28864 0 "; std::vector<size_t> expectWorkspaces = {16777216}; ExecuteTestCase(tilingContextPara,
//     ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }
// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float32_case2)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 3}, {2, 4, 4, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 1);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_failed_case0)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{1, 2, 3, 4, 4}, {1, 2, 3, 4, 4}};
//     gert::StorageShape x = {{1, 2, 3, 16, 16}, {1, 2, 3, 16, 16}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{1, 2, 3, 16, 16}, {1, 2, 3, 16, 16}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_failed_case1)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 3, 5, 4}, {2, 3, 5, 4}};
//     gert::StorageShape x = {{2, 3, 16, 16}, {2, 3, 16, 16}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 3, 16, 16}, {2, 3, 16, 16}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_failed_case2)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 3}, {2, 4, 4, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("reflection")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_failed_case3)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 3}, {2, 4, 4, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bicubic")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_nearest_case0)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 3}, {2, 4, 4, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics); EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS); auto tiling_key =
//     tiling_context->GetTilingKey(); ASSERT_EQ(tiling_key, 2);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_nearest_case1)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 125}, {2, 4, 4, 125}};
//     gert::StorageShape x = {{2, 16, 16, 125}, {2, 16, 16, 125}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 125}, {2, 16, 16, 125}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics); EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS); auto tiling_key =
//     tiling_context->GetTilingKey(); ASSERT_EQ(tiling_key, 2);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_nearest_case2)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 525}, {2, 4, 4, 525}};
//     gert::StorageShape x = {{2, 16, 16, 525}, {2, 16, 16, 525}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 525}, {2, 16, 16, 525}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics); EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS); auto tiling_key =
//     tiling_context->GetTilingKey(); ASSERT_EQ(tiling_key, 2);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_nearest_case3)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 4, 4, 1025}, {2, 4, 4, 1025}};
//     gert::StorageShape x = {{2, 16, 16, 1025}, {2, 16, 16, 1025}};
//     gert::StorageShape grid = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 1025}, {2, 16, 16, 1025}};
//     gert::StorageShape output2 = {{2, 4, 4, 2}, {2, 4, 4, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics); EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS); auto tiling_key =
//     tiling_context->GetTilingKey(); ASSERT_EQ(tiling_key, 2);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float16_case1)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 3);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_float16_1_case1)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 4);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_bfloat16_case1)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("bilinear")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 5);
// }

// TEST_F(GridSampler2DGradTiling, grid_sampler_grad_tiling_test_bfloat16_1_case1)
// {
//     std::string op_type("GridSampler2DGrad");
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//     auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//     auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//     string compile_info_string = R"({
//                                         "hardware_info": {
//                                             "BT_SIZE": 0,
//                                             "load3d_constraints": "1",
//                                             "Intrinsic_fix_pipe_l0c2out": false,
//                                             "Intrinsic_data_move_l12ub": true,
//                                             "Intrinsic_data_move_l0c2ub": true,
//                                             "Intrinsic_data_move_out2l1_nd2nz": false,
//                                             "UB_SIZE": 196608,
//                                             "L2_SIZE": 33554432,
//                                             "L1_SIZE": 524288,
//                                             "L0A_SIZE": 65536,
//                                             "L0B_SIZE": 65536,
//                                             "L0C_SIZE": 131072,
//                                             "CORE_NUM": 48
//                                         }
//                                     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//     GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
//     // platform info
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     // compile info
//     optiling::Tiling4GridSampler2DGradCompileInfo compile_info;

//     // tilingParseFunc simulate
//     auto kernel_holder =
//         gert::KernelRunContextFaker()
//             .KernelIONum(2, 1)
//             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//             .Outputs({&compile_info})
//             .Build();
//     ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
//     aicore_spec);
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
//         "AICoreintrinsicDtypeMap", intrinsics);
//     ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
//     // tilingFunc simulate
//     auto param = gert::TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//     auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//     gert::StorageShape grad = {{2, 8, 8, 3}, {2, 8, 8, 3}};
//     gert::StorageShape x = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape grid = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     gert::StorageShape output1 = {{2, 16, 16, 3}, {2, 16, 16, 3}};
//     gert::StorageShape output2 = {{2, 8, 8, 2}, {2, 8, 8, 2}};
//     auto holder = gert::TilingContextFaker()
//                       .NodeIoNum(3, 2)
//                       .IrInstanceNum({1, 1, 1})
//                       .InputShapes({&grad, &x, &grid})
//                       .OutputShapes({&output1, &output2})
//                       .CompileInfo(&compile_info)
//                       .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                       .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
//                       .NodeAttrs(
//                           {{"interpolation_mode", ge::AnyValue::CreateFrom<std::string>("nearest")},
//                            {"padding_mode", ge::AnyValue::CreateFrom<std::string>("zeros")},
//                            {"align_corners", ge::AnyValue::CreateFrom<bool>(true)}})
//                       .TilingData(param.get())
//                       .Workspace(ws_size)
//                       .Build();
//     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//     intrinsics);
//     // workspaces nullptr return failed
//     EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//     auto tiling_key = tiling_context->GetTilingKey();
//     ASSERT_EQ(tiling_key, 6);
// }
