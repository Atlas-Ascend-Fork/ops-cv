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
#include "../../../../op_host/rasterizer_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class RasterizerTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RasterizerTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RasterizerTiling TearDown" << std::endl; }
};

struct RasterizerCompileInfo {};

TEST_F(RasterizerTiling, rasterizer_tiling_001)
{
    gert::StorageShape v_shape = {{3, 4}, {3, 4}};
    gert::StorageShape f_shape = {{1, 3}, {1, 3}};
    gert::StorageShape d_shape = {{10, 10}, {10, 10}};
    gert::StorageShape findices_shape = {{10, 10}, {10, 10}};
    gert::StorageShape bary_shape = {{10, 10, 3}, {10, 10, 3}};
    RasterizerCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Rasterizer",
        {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {f_shape, ge::DT_INT32, ge::FORMAT_ND},
         {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{findices_shape, ge::DT_INT32, ge::FORMAT_ND}, {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData = "12884901889 42949672970 0 ";
    std::vector<size_t> expectWorkspaces = {33629184};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RasterizerTiling, rasterizer_tiling_width_out_of_range_rejected)
{
    for (const int64_t width : {4294967297LL, -4294967295LL}) {
        gert::StorageShape v_shape = {{3, 4}, {3, 4}};
        gert::StorageShape f_shape = {{1, 3}, {1, 3}};
        gert::StorageShape d_shape = {{10, 1}, {10, 1}};
        gert::StorageShape findices_shape = {{10, 1}, {10, 1}};
        gert::StorageShape bary_shape = {{10, 1, 3}, {10, 1, 3}};
        RasterizerCompileInfo compileInfo = {};
        gert::TilingContextPara tilingContextPara(
            "Rasterizer",
            {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
             {f_shape, ge::DT_INT32, ge::FORMAT_ND},
             {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
            {{findices_shape, ge::DT_INT32, ge::FORMAT_ND}, {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(width)),
             gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
             gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
             gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
            &compileInfo);
        uint64_t expectTilingKey = 0;
        string expectTilingData;
        std::vector<size_t> expectWorkspaces = {0};
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
    }
}

TEST_F(RasterizerTiling, rasterizer_tiling_zero_vertices_rejected)
{
    gert::StorageShape v_shape = {{0, 4}, {0, 4}};
    gert::StorageShape f_shape = {{1, 3}, {1, 3}};
    gert::StorageShape d_shape = {{10, 10}, {10, 10}};
    gert::StorageShape findices_shape = {{10, 10}, {10, 10}};
    gert::StorageShape bary_shape = {{10, 10, 3}, {10, 10, 3}};
    RasterizerCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Rasterizer",
        {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {f_shape, ge::DT_INT32, ge::FORMAT_ND},
         {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{findices_shape, ge::DT_INT32, ge::FORMAT_ND}, {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RasterizerTiling, rasterizer_tiling_zero_faces_rejected)
{
    gert::StorageShape v_shape = {{3, 4}, {3, 4}};
    gert::StorageShape f_shape = {{0, 3}, {0, 3}};
    gert::StorageShape d_shape = {{10, 10}, {10, 10}};
    gert::StorageShape findices_shape = {{10, 10}, {10, 10}};
    gert::StorageShape bary_shape = {{10, 10, 3}, {10, 10, 3}};
    RasterizerCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Rasterizer",
        {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {f_shape, ge::DT_INT32, ge::FORMAT_ND},
         {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{findices_shape, ge::DT_INT32, ge::FORMAT_ND}, {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RasterizerTiling, rasterizer_tiling_nonzero_occlusion_truncation_rejected)
{
    gert::StorageShape v_shape = {{3, 4}, {3, 4}};
    gert::StorageShape f_shape = {{1, 3}, {1, 3}};
    gert::StorageShape d_shape = {{10, 10}, {10, 10}};
    gert::StorageShape findices_shape = {{10, 10}, {10, 10}};
    gert::StorageShape bary_shape = {{10, 10, 3}, {10, 10, 3}};
    RasterizerCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Rasterizer",
        {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {f_shape, ge::DT_INT32, ge::FORMAT_ND},
         {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{findices_shape, ge::DT_INT32, ge::FORMAT_ND}, {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.1)),
         gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
