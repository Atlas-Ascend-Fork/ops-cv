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
#include <cstring>
#include <iostream>
#include <vector>
#include "../../../op_host/grid_sampler3_d_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class GridSampler3DGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GridSampler3DGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GridSampler3DGradTiling TearDown" << std::endl; }
};

struct GridSampler3DGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool regBase{false};
};

struct GridSampler3DGradUsedCoreBoundaryCase {
    const char* name;
    uint32_t batch;
    uint32_t gridD;
    uint32_t gridH;
    uint32_t gridW;
    uint32_t expectedPNumPerCore;
    uint32_t expectedTailPNum;
};

static gert::TilingContextPara MakeGridSampler3DGradUsedCoreBoundaryContext(
    const GridSampler3DGradUsedCoreBoundaryCase& testCase, GridSampler3DGradCompileInfo* compileInfo)
{
    gert::StorageShape grad = {{testCase.batch, testCase.gridD, testCase.gridH, testCase.gridW, 1},
                               {testCase.batch, testCase.gridD, testCase.gridH, testCase.gridW, 1}};
    gert::StorageShape x = {{testCase.batch, 1, 1, 1, 1}, {testCase.batch, 1, 1, 1, 1}};
    gert::StorageShape grid = {{testCase.batch, testCase.gridD, testCase.gridH, testCase.gridW, 3},
                               {testCase.batch, testCase.gridD, testCase.gridH, testCase.gridW, 3}};
    gert::StorageShape dx = {{testCase.batch, 1, 1, 1, 1}, {testCase.batch, 1, 1, 1, 1}};
    gert::StorageShape dgrid = {{testCase.batch, testCase.gridD, testCase.gridH, testCase.gridW, 3},
                                {testCase.batch, testCase.gridD, testCase.gridH, testCase.gridW, 3}};
    return gert::TilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        compileInfo);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 0 206158430209 68719476757 "
                              "1099511627777 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case2)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("reflection")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 8589934592 206158430209 68719476757 "
                              "1099511627777 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case3)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 1 206158430209 68719476757 "
                              "6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case4)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 300}, {2, 8, 8, 8, 300}};
    gert::StorageShape x = {{2, 16, 16, 16, 300}, {2, 300, 16, 16, 300}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 300}, {2, 300, 16, 16, 300}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1288490188802 68719477036 34359738384 34359738376 4294967297 206158430209 68719476757 "
                              "5222680231952 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case5)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 600}, {2, 8, 8, 8, 600}};
    gert::StorageShape x = {{2, 16, 16, 16, 600}, {2, 16, 16, 16, 600}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 600}, {2, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2576980377602 68719476752 34359738384 34359738376 1 206158430209 68719476757 "
                              "4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case6)
{
    gert::StorageShape grad = {{1, 2, 2, 2, 600}, {1, 2, 2, 2, 600}};
    gert::StorageShape x = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2576980377601 68719476752 8589934608 8589934594 1 34359738369 1 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case7)
{
    gert::StorageShape grad = {{1, 2, 2, 2, 600}, {1, 2, 2, 2, 600}};
    gert::StorageShape x = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bicubic")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2576980377601 68719476752 8589934608 8589934594 1 34359738369 1 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case8)
{
    gert::StorageShape grad = {{1, 2, 2, 2, 600}, {1, 2, 2, 2, 600}};
    gert::StorageShape x = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 600}, {1, 16, 16, 16, 600}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("xxx")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2576980377601 68719476752 8589934608 8589934594 1 34359738369 1 4947802325000 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case9)
{
    gert::StorageShape grad = {{1, 3, 3, 3, 3}, {1, 3, 3, 3, 3}};
    gert::StorageShape x = {{1, 16, 16, 16, 3}, {1, 16, 16, 16, 3}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 3}, {1, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901889 68719476752 8589934608 8589934594 1 34359738369 1 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case10)
{
    gert::StorageShape grad = {{1, 3, 3, 3, 3}, {1, 3, 3, 3, 3}};
    gert::StorageShape x = {{1, 16, 16, 3}, {1, 16, 16, 3}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape dx = {{1, 16, 16, 16, 3}, {1, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901889 68719476752 8589934595 8589934594 1 34359738369 1 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case11)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 1600}, {2, 8, 8, 8, 1600}};
    gert::StorageShape x = {{2, 16, 16, 16, 1600}, {2, 16, 16, 16, 1600}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 1600}, {2, 16, 16, 16, 1600}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "6871947673602 68719476752 34359738384 34359738376 1 206158430209 68719476757 "
                              "5772436045825 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float32_case12)
{
    // regbase 下 ND 与 NCDHW 同族，均按 (N, C, D, H, W) 语义解析，此处按 NCDHW 排布构造
    gert::StorageShape grad = {{2, 3, 8, 8, 8}, {2, 3, 8, 8, 8}};
    gert::StorageShape x = {{2, 3, 16, 16, 16}, {2, 3, 16, 16, 16}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 3, 16, 16, 16}, {2, 3, 16, 16, 16}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 1 51539607553 0 6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// regbase 下 ND 与 NCDHW 同族，须按同一 (N, C, D, H, W) 语义取维。用非对称 shape（C=3、gridD=2）构造：
// 若 ND 被误当作 NDHWC 解读会得到 outD=dim1=3 与 gridD=2 不符而失败，修复后两者必须产出相同的 tiling 数据
static gert::TilingContextPara MakeRegBaseFormatPara(ge::Format inputFormat, const gert::StorageShape& grad,
                                                     const gert::StorageShape& x, const gert::StorageShape& grid,
                                                     GridSampler3DGradCompileInfo* compileInfo)
{
    auto dx = x;
    auto dgrid = grid;
    return gert::TilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, inputFormat}, {x, ge::DT_FLOAT, inputFormat}, {grid, ge::DT_FLOAT, inputFormat}},
        {{dx, ge::DT_FLOAT, inputFormat}, {dgrid, ge::DT_FLOAT, inputFormat}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        compileInfo);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_regbase_nd_equals_ncdhw)
{
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, true};
    // (N, C, D, H, W) = (1, 3, 2, 2, 2)，grid 为 (1, 2, 2, 2, 3)：C 与 D/H/W 均不对称
    gert::StorageShape grad = {{1, 3, 2, 2, 2}, {1, 3, 2, 2, 2}};
    gert::StorageShape x = {{1, 3, 2, 2, 2}, {1, 3, 2, 2, 2}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};

    TilingInfo ndInfo;
    ASSERT_TRUE(ExecuteTiling(MakeRegBaseFormatPara(ge::FORMAT_ND, grad, x, grid, &compileInfo), ndInfo));
    TilingInfo ncdhwInfo;
    ASSERT_TRUE(ExecuteTiling(MakeRegBaseFormatPara(ge::FORMAT_NCDHW, grad, x, grid, &compileInfo), ncdhwInfo));

    EXPECT_EQ(ndInfo.tilingKey, ncdhwInfo.tilingKey);
    EXPECT_EQ(ndInfo.blockNum, ncdhwInfo.blockNum);
    ASSERT_EQ(ndInfo.tilingDataSize, ncdhwInfo.tilingDataSize);
    EXPECT_EQ(std::memcmp(ndInfo.tilingData.get(), ncdhwInfo.tilingData.get(), ndInfo.tilingDataSize), 0);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_regbase_ndhwc_still_channel_last)
{
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, true};
    // NDHWC 排布 (N, D, H, W, C) = (1, 2, 2, 2, 3)，C 维度必须仍按末维解析
    gert::StorageShape grad = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape x = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    gert::StorageShape grid = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};

    TilingInfo nhwcInfo;
    ASSERT_TRUE(ExecuteTiling(MakeRegBaseFormatPara(ge::FORMAT_NDHWC, grad, x, grid, &compileInfo), nhwcInfo));

    const auto* tilingData = reinterpret_cast<const uint32_t*>(nhwcInfo.tilingData.get());
    EXPECT_EQ(tilingData[1], 3U); // channel 取末维
    EXPECT_EQ(tilingData[2], 2U); // xD
    EXPECT_EQ(tilingData[3], 2U); // xH
    EXPECT_EQ(tilingData[4], 2U); // xW
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_float16_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT16, ge::FORMAT_ND},
         {x, ge::DT_FLOAT16, ge::FORMAT_ND},
         {grid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT16, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 1 206158430209 68719476757 "
                              "6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_bfloat16_case1)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_BF16, ge::FORMAT_ND}, {x, ge::DT_BF16, ge::FORMAT_ND}, {grid, ge::DT_BF16, ge::FORMAT_ND}},
        {{dx, ge::DT_BF16, ge::FORMAT_ND}, {dgrid, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "12884901890 68719476752 34359738384 34359738376 1 206158430209 68719476757 "
                              "6597069766688 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_test_used_core_uint32_boundaries)
{
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    const std::vector<GridSampler3DGradUsedCoreBoundaryCase> testCases = {
        {"uint32_max", 1, 1, 65535, 65537, 89478485, 15},
        {"exact_uint32_wrap", 2, 1, 32768, 65536, 89478485, 16},
        {"just_over_uint32", 1, 1, 65536, 65537, 89479850, 32},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testCase.name);
        auto tilingContextPara = MakeGridSampler3DGradUsedCoreBoundaryContext(testCase, &compileInfo);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        ASSERT_EQ(tilingInfo.blockNum, compileInfo.coreNum);
        ASSERT_GE(tilingInfo.tilingDataSize, 14U * sizeof(uint32_t));
        const auto* tilingData = reinterpret_cast<const uint32_t*>(tilingInfo.tilingData.get());
        EXPECT_EQ(tilingData[11], compileInfo.coreNum);
        EXPECT_EQ(tilingData[12], testCase.expectedPNumPerCore);
        EXPECT_EQ(tilingData[13], testCase.expectedTailPNum);
    }
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_n_mismatch_should_fail)
{
    gert::StorageShape grad = {{3, 8, 8, 8, 3}, {3, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_c_mismatch_should_fail)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 4}, {2, 8, 8, 8, 4}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridSampler3DGradTiling, grid_sampler3_d_grad_tiling_grid_last_dim_not3_should_fail)
{
    gert::StorageShape grad = {{2, 8, 8, 8, 3}, {2, 8, 8, 8, 3}};
    gert::StorageShape x = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape grid = {{2, 8, 8, 8, 2}, {2, 8, 8, 8, 2}};
    gert::StorageShape dx = {{2, 16, 16, 16, 3}, {2, 16, 16, 16, 3}};
    gert::StorageShape dgrid = {{2, 8, 8, 8, 2}, {2, 8, 8, 8, 2}};
    GridSampler3DGradCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "GridSampler3DGrad",
        {{grad, ge::DT_FLOAT, ge::FORMAT_ND}, {x, ge::DT_FLOAT, ge::FORMAT_ND}, {grid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{dx, ge::DT_FLOAT, ge::FORMAT_ND}, {dgrid, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<string>("bilinear")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<string>("zeros")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
