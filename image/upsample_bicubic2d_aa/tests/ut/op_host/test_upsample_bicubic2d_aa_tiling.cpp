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
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bicubic2d_aa_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

class UpsampleBicubic2dAATiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpsampleBicubic2dAATiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpsampleBicubic2dAATiling TearDown" << std::endl; }
};

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_001)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 0;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256}, {1, 1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({256, 256})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "4539628425446424576 4575657222473777152 4611686019501129728 21474836485 39582418609152 0 64 131072 4294967297 "
        "549755814016 4294967297 1099511628032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 "
        "549755814144 549755814016 154618822720 274877907072 4294967336 4294967297 1 0 131941395333120 32768 "
        "4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 274877906945 549755814144 274877907072 "
        "154618822912 1099511627840 8589934624 4294967298 1 0 351843720888320 65536 4294967297 4294967297 8589934594 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33685504};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_002)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 0;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 0, 0}, {1, 1, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({0, 0})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_003)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 0;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 200, 200}, {1, 1, 200, 200}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 36, 36}, {1, 1, 36, 36}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({36, 36})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "4661726013463578396 4483423504145338860 4697754810490930972 107374182425 30786325584896 0 16 29184 4294967297 "
        "858993459400 4294967297 154618822692 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 858993459201 "
        "858993459236 858993459400 481036337168 68719476944 17179869216 4294967300 1 0 492581209243648 13312 "
        "4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 68719476737 858993459236 68719476936 "
        "481036337188 206158430224 4294967408 4294967297 1 0 123145302310912 3072 4294967297 4294967297 4294967297 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33583616};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_004)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 0;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 200, 200}, {1, 1, 200, 200}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 20, 20}, {1, 1, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({20, 20})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "4692750812812673024 4453159313439771853 4728779609840025600 176093659177 54975581401600 0 16 16384 4294967297 "
        "858993459400 4294967297 85899345940 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 858993459201 "
        "858993459220 858993459400 858993459216 68719476944 30064771104 4294967303 1 0 862017116176384 13312 "
        "4294967297 4294967297 30064771079 0 8589934594 1 0 0 0 0 0 0 0 0 68719476737 858993459220 68719476936 "
        "858993459220 137438953488 4294967496 4294967297 1 0 164926744166400 2048 4294967297 4294967297 4294967297 0 "
        "8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33570816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_005)
{
    optiling::UpsampleBicubic2dAACompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{2, 2, 4, 5}, {2, 2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 4, 5}, {2, 2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "40 2 2 4 5 4 5 0 32760 4575657222473777152 4575657222473777152 4611686019501129728 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_006)
{
    optiling::UpsampleBicubic2dAACompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{2, 2, 4, 4}, {2, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 8, 8}, {2, 2, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 257;
    string expectTilingData = "4 2 2 4 4 8 8 0 0 4539628425446424576 4575657222473777152 4611686019501129728 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_007)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 0;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 200, 200}, {1, 1, 200, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 1, 36, 36}, {1, 1, 36, 36}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({36, 36})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "4661726013463578396 4483423504145338860 4697754810490930972 107374182425 15393162792448 0 16 14848 4294967297 "
        "858993459400 4294967297 154618822692 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 858993459201 "
        "858993459236 858993459400 481036337168 68719476944 8589934656 4294967298 1 0 246290604621824 13312 4294967297 "
        "4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 68719476737 858993459236 68719476936 481036337188 "
        "206158430224 4294967408 4294967297 1 0 61572651155456 3072 4294967297 4294967297 4294967297 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33569280};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_008)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 0;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 200, 200}, {1, 1, 200, 200}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 1, 36, 36}, {1, 1, 36, 36}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({36, 36})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    uint64_t expectTilingKey = 3;
    string expectTilingData =
        "4661726013463578396 4483423504145338860 4697754810490930972 107374182425 15393162792448 0 16 14848 4294967297 "
        "858993459400 4294967297 154618822692 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 858993459201 "
        "858993459236 858993459400 481036337168 68719476944 8589934656 4294967298 1 0 246290604621824 13312 4294967297 "
        "4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 68719476737 858993459236 68719476936 481036337188 "
        "206158430224 4294967408 4294967297 1 0 61572651155456 3072 4294967297 4294967297 4294967297 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33569280};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_tiling_009)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 40;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 200, 200}, {1, 1, 200, 200}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 20, 20}, {1, 1, 20, 20}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({20, 20})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "4692750812812673024 4453159313439771853 4728779609840025600 176093659177 54975581401600 8589934632 16 16384 "
        "4294967297 858993459400 4294967297 85899345940 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 68719476752 68719476752 68719476752 68719476752 "
        "68719476752 68719476752 68719476752 68719476752 68719476752 68719476752 0 0 0 0 0 68719476752 68719476752 "
        "68719476752 68719476752 68719476752 68719476752 68719476752 68719476752 68719476752 68719476752 85899345940 "
        "85899345940 85899345940 85899345940 85899345940 85899345940 85899345940 85899345940 85899345940 85899345940 0 "
        "0 0 0 0 42949672960 128849018900 214748364840 300647710780 386547056720 472446402660 558345748600 "
        "644245094540 730144440480 816043786420 42949672960 128849018900 214748364840 300647710780 386547056720 "
        "472446402660 558345748600 644245094540 730144440480 816043786420 0 0 0 0 0 85899345930 171798691870 "
        "257698037810 343597383750 429496729690 515396075630 601295421570 687194767510 773094113450 858993459390 "
        "85899345930 171798691870 257698037810 343597383750 429496729690 515396075630 601295421570 687194767510 "
        "773094113450 858993459390 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 68719476736 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 85899345936 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 858993459201 858993459220 858993459400 858993459216 68719476944 "
        "30064771104 4294967303 1 0 862017116176384 13312 4294967297 4294967297 30064771079 0 8589934594 1 0 0 0 0 0 0 "
        "0 0 68719476737 858993459220 68719476936 858993459220 137438953488 4294967496 4294967297 1 0 164926744166400 "
        "2048 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {34594816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_output_size_valid_control)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 1;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_output_size_extra_element_should_fail)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 1;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8, 99})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_output_size_short_should_fail)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 1;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(UpsampleBicubic2dAATiling, upsample_bicubic2d_aa_output_size_empty_should_fail)
{
    struct UpsampleBicubic2dAACompileInfo {
        uint32_t totalCoreNum = 1;
    } compile_info;

    gert::TilingContextPara tilingContextPara(
        "UpsampleBicubic2dAA", {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compile_info);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
