/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/flood_fill_tiling_data.h"

using namespace std;
using namespace ge;

class FloodFillTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FloodFillTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FloodFillTiling TearDown" << std::endl; }
};

namespace {
// Tiling key: dtype in bits [1:0], input-mask presence in bit 2.
constexpr uint64_t kKeyUint8 = 0;
constexpr uint64_t kKeyInt32 = 1;
constexpr uint64_t kKeyFloat32 = 2;

constexpr uint32_t kDiffModeNeighbor = 0;
constexpr uint32_t kDiffModeFixedRange = 1;

constexpr size_t kWsSysSize = 16U * 1024U * 1024U; // WS_SYS_SIZE
constexpr int32_t kInt32Max = 2147483647;

// 按 int64 序列化 tiling data，与 ExecuteTestCase 内部的 to_string<int64_t>
// 保持一致。整数 dtypes 走 I 字段，float32 走 F 字段（另一组为 0）。
std::string MakeExpectedTilingData(uint32_t height, uint32_t width, uint32_t seedRow, uint32_t seedCol,
                                   int32_t newValueI, int32_t loDiffI, int32_t upDiffI, float newValueF, float loDiffF,
                                   float upDiffF, uint32_t connectivity, uint32_t diffMode)
{
    FloodFillTilingData data = {};
    data.height = height;
    data.width = width;
    data.seedRow = seedRow;
    data.seedCol = seedCol;
    data.newValueI = newValueI;
    data.loDiffI = loDiffI;
    data.upDiffI = upDiffI;
    data.newValueF = newValueF;
    data.loDiffF = loDiffF;
    data.upDiffF = upDiffF;
    data.connectivity = connectivity;
    data.diffMode = diffMode;

    const int64_t* words = reinterpret_cast<const int64_t*>(&data);
    const size_t len = sizeof(FloodFillTilingData) / sizeof(int64_t);
    std::string result;
    for (size_t i = 0; i < len; ++i) {
        result += std::to_string(words[i]) + " ";
    }
    return result;
}

// workspace = 16MB 系统区 + 2112B 并行控制记录
//             + AlignUp32(H*W) visited + AlignUp32(H) 行标志
//             + AlignUp32(4*H) 单区间端点 + AlignUp32(span 栈)；
//             span 栈容量 = (2*max(H,W) + 256) * 8B。
size_t ExpectedWorkspace(size_t height, size_t width, ge::DataType dtype = ge::DT_UINT8)
{
    constexpr size_t kAlign = 32U;
    const auto align = [kAlign](size_t bytes) { return (bytes + kAlign - 1U) / kAlign * kAlign; };
    const size_t visited = align(height * width);
    const size_t rowFlags = align(height);
    const size_t maxDim = height > width ? height : width;
    const size_t stack = align((2U * maxDim + 256U) * 8U);
    const size_t comparison = dtype != ge::DT_INT32 && height * width >= 262144U && width >= 512U ?
                                  2U * (((height * width + 7U) / 8U + 63U) / 64U * 64U + 64U) :
                                  0U;
    return kWsSysSize + FLOOD_FILL_CONTROL_BYTES + comparison + visited + rowFlags + align(height * 4U) + stack;
}
} // namespace

TEST_F(FloodFillTiling, u8_basic_neighbor)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(3, 4, 1, 2, 200, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(3, 4)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_fixed_range_8conn)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{5, 7}, {5, 7}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{5, 7}, {5, 7}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{7, 9}, {7, 9}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({100.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({10.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({20.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("FIXED_RANGE")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(5, 7, 0, 0, 100, 10, 20, 0.0f, 0.0f, 0.0f, 8,
                                                           kDiffModeFixedRange);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(5, 7)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_quantize_clamp_255)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{4, 4}, {4, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4, 4}, {4, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{6, 6}, {6, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({300.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({260.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({250.4f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    // 300 -> 255 (clamp), 260 -> 255 (clamp), 250.4 -> 250 (round, no clamp)
    const string expectTilingData = MakeExpectedTilingData(4, 4, 2, 2, 255, 255, 250, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(4, 4)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_large_workspace_1024)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1026, 1026}, {1026, 1026}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(1024, 1024, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // 1048576 bytes already 32-aligned.
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(1024, 1024)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_odd_size_align_33)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{33, 33}, {33, 33}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{33, 33}, {33, 33}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{35, 35}, {35, 35}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({128.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({5.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({5.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(33, 33, 16, 16, 128, 5, 5, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // 1089 bytes -> align32 -> 1120
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(33, 33)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_seed_corner)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({77.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(3, 4, 2, 3, 77, 1, 1, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(3, 4)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, empty_height_zero)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{0, 4}, {0, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{0, 4}, {0, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 6}, {2, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    // Degenerate branch: attrs are not parsed, tiling data keeps zero
    // defaults with connectivity=4 / diffMode=NEIGHBOR.
    const string expectTilingData = MakeExpectedTilingData(0, 4, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 4, kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(0, 4)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, empty_width_zero)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 0}, {3, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 0}, {3, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 2}, {5, 2}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(3, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 4, kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(3, 0)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, i32_negative_new_value)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({-7.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({10.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({10.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(3, 4, 1, 1, -7, 10, 10, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(3, 4)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyInt32, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, i32_clamp_int32_max)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10, 10}, {10, 10}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({3.0e9f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({3.0e9f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({3.0e9f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("FIXED_RANGE")),
        },
        &compileInfo, "Ascend910B");

    // 3e9 > INT32_MAX: value clamps to INT32_MAX, diffs clamp to INT32_MAX.
    const string expectTilingData = MakeExpectedTilingData(8, 8, 4, 4, kInt32Max, kInt32Max, kInt32Max, 0.0f, 0.0f,
                                                           0.0f, 8, kDiffModeFixedRange);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(8, 8)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyInt32, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, f32_passthrough)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({42.5f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.25f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(3, 4, 1, 2, 0, 0, 0, 42.5f, 0.5f, 0.25f, 4,
                                                           kDiffModeNeighbor);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(3, 4)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyFloat32, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, f32_fixed_range_8conn)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{7, 7}, {7, 7}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({1.5f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.125f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.75f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("FIXED_RANGE")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(5, 5, 2, 3, 0, 0, 0, 1.5f, 0.125f, 2.75f, 8,
                                                           kDiffModeFixedRange);
    const std::vector<size_t> expectWorkspaces = {ExpectedWorkspace(5, 5)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyFloat32, expectTilingData, expectWorkspaces);
}

// ---------------------------------------------------------------------------
// Invalid cases
// ---------------------------------------------------------------------------

TEST_F(FloodFillTiling, invalid_dtype_int64)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_rank3)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 5, 6}, {2, 5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_rank1)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{4}, {4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, negative_height)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{-1, 4}, {-1, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{-1, 4}, {-1, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 6}, {1, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, pixels_over_limit)
{
    struct CompileInfo {
    } compileInfo;
    // 0x80000000 * 1 > MAX_TOTAL_PIXELS (0x7FFFFFFF).
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{static_cast<int64_t>(0x80000000), 1}, {static_cast<int64_t>(0x80000000), 1}},
             ge::DT_UINT8,
             ge::FORMAT_ND},
        },
        {
            {{{static_cast<int64_t>(0x80000000), 1}, {static_cast<int64_t>(0x80000000), 1}},
             ge::DT_UINT8,
             ge::FORMAT_ND},
            {{{static_cast<int64_t>(0x80000000) + 2, 3}, {static_cast<int64_t>(0x80000000) + 2, 3}},
             ge::DT_UINT8,
             ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_seed_row_out_of_range)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({5, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_seed_col_negative)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, -1})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_seed_size_3)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_new_value_size_2)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value",
                                            Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f, 100.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_lo_diff_negative)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({-0.5f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_new_value_nan)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>(
                                                             {std::numeric_limits<float>::quiet_NaN()})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_up_diff_infinite)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr(
                "up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::infinity()})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_connectivity_6)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, invalid_diff_mode)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({200.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("MEDIAN")),
        },
        &compileInfo, "Ascend910B");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FloodFillTiling, u8_comparison_one_below)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{511, 513}, {511, 513}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{511, 513}, {511, 513}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{513, 515}, {513, 515}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(511, 513, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17054304U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_comparison_exact_threshold)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{514, 514}, {514, 514}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(512, 512, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17119936U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_comparison_above_threshold)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{513, 512}, {513, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{513, 512}, {513, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{515, 514}, {515, 514}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(513, 512, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17120672U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_comparison_width_excluded)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{514, 511}, {514, 511}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{514, 511}, {514, 511}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{516, 513}, {516, 513}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(514, 511, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17054880U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, u8_comparison_unaligned_above)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{511, 514}, {511, 514}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{511, 514}, {511, 514}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{513, 516}, {513, 516}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(511, 514, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17120608U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyUint8, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, f32_comparison_exact_threshold)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{514, 514}, {514, 514}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(512, 512, 0, 0, 0, 0, 0, 255.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17119936U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyFloat32, expectTilingData, expectWorkspaces);
}

TEST_F(FloodFillTiling, i32_comparison_stays_excluded)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "FloodFill",
        {
            {{{512, 512}, {512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{514, 514}, {514, 514}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})),
            gert::TilingContextPara::OpAttr("new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({255.0f})),
            gert::TilingContextPara::OpAttr("lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})),
            gert::TilingContextPara::OpAttr("connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")),
        },
        &compileInfo, "Ascend910B");

    const string expectTilingData = MakeExpectedTilingData(512, 512, 0, 0, 255, 0, 0, 0.0f, 0.0f, 0.0f, 4,
                                                           kDiffModeNeighbor);
    // Literal ABI size: 16 MiB sys + control, predicates, visited and span metadata.
    const std::vector<size_t> expectWorkspaces = {17054272U};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kKeyInt32, expectTilingData, expectWorkspaces);
}

namespace {
void CheckInputMask(const gert::StorageShape& maskShape, ge::DataType maskType, bool valid)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara para("FloodFill",
                                 {
                                     {{{5, 7}, {5, 7}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     {maskShape, maskType, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{5, 7}, {5, 7}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     {{{7, 9}, {7, 9}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {"seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
                                     {"new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({42.0f})},
                                     {"lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})},
                                     {"up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.0f})},
                                     {"connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)},
                                     {"diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")},
                                 },
                                 &compileInfo, "Ascend910B");
    ExecuteTestCase(para, valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, kKeyUint8 | 4U,
                    valid ? MakeExpectedTilingData(5, 7, 1, 2, 42, 0, 0, 0, 0, 0, 4, kDiffModeNeighbor) : "",
                    valid ? std::vector<size_t>{ExpectedWorkspace(5, 7)} : std::vector<size_t>{});
}
} // namespace
TEST_F(FloodFillTiling, mask_input_valid) { CheckInputMask({{7, 9}, {7, 9}}, ge::DT_UINT8, true); }
TEST_F(FloodFillTiling, mask_input_wrong_height) { CheckInputMask({{5, 9}, {5, 9}}, ge::DT_UINT8, false); }
TEST_F(FloodFillTiling, mask_input_wrong_width) { CheckInputMask({{7, 7}, {7, 7}}, ge::DT_UINT8, false); }
TEST_F(FloodFillTiling, mask_input_rank1) { CheckInputMask({{63}, {63}}, ge::DT_UINT8, false); }
TEST_F(FloodFillTiling, mask_input_rank3) { CheckInputMask({{1, 7, 9}, {1, 7, 9}}, ge::DT_UINT8, false); }
TEST_F(FloodFillTiling, mask_input_int32_rejected) { CheckInputMask({{7, 9}, {7, 9}}, ge::DT_INT32, false); }
TEST_F(FloodFillTiling, mask_input_float_rejected) { CheckInputMask({{7, 9}, {7, 9}}, ge::DT_FLOAT, false); }

namespace {
void CheckIntegerQuantization(ge::DataType dtype, float newValue, float loDiff, float upDiff, int32_t expectedNewValue,
                              int32_t expectedLoDiff, int32_t expectedUpDiff)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara para("FloodFill",
                                 {
                                     {{{2, 2}, {2, 2}}, dtype, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 2}, {2, 2}}, dtype, ge::FORMAT_ND},
                                     {{{4, 4}, {4, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {"seed", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                                     {"new_value", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({newValue})},
                                     {"lo_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({loDiff})},
                                     {"up_diff", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({upDiff})},
                                     {"connectivity", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)},
                                     {"diff_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("NEIGHBOR")},
                                 },
                                 &compileInfo, "Ascend910B");
    const auto expected = MakeExpectedTilingData(2, 2, 0, 0, expectedNewValue, expectedLoDiff, expectedUpDiff, 0.0f,
                                                 0.0f, 0.0f, 4, kDiffModeNeighbor);
    const uint64_t key = dtype == ge::DT_UINT8 ? kKeyUint8 : kKeyInt32;
    const std::vector<size_t> workspaces = {ExpectedWorkspace(2, 2, dtype)};

    // ARM can return saturated integers even when llround raises FE_INVALID.
    // Check the actual tiling fields and the exception flag independently.
    std::fenv_t savedEnvironment;
    ASSERT_EQ(std::fegetenv(&savedEnvironment), 0);
    const int clearStatus = std::feclearexcept(FE_ALL_EXCEPT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, key, expected, workspaces);
    const int invalid = std::fetestexcept(FE_INVALID);
    const int restoreStatus = std::fesetenv(&savedEnvironment);
    EXPECT_EQ(clearStatus, 0);
    EXPECT_EQ(restoreStatus, 0);
    EXPECT_EQ(invalid, 0);
}
} // namespace

TEST_F(FloodFillTiling, u8_finite_max_positive_attributes)
{
    const float largest = std::numeric_limits<float>::max();
    CheckIntegerQuantization(ge::DT_UINT8, largest, largest, largest, 255, 255, 255);
}

TEST_F(FloodFillTiling, u8_finite_max_negative_new_value)
{
    const float largest = std::numeric_limits<float>::max();
    CheckIntegerQuantization(ge::DT_UINT8, -largest, largest, largest, 0, 255, 255);
}

TEST_F(FloodFillTiling, i32_finite_max_positive_attributes)
{
    const float largest = std::numeric_limits<float>::max();
    CheckIntegerQuantization(ge::DT_INT32, largest, largest, largest, kInt32Max, kInt32Max, kInt32Max);
}

TEST_F(FloodFillTiling, i32_finite_max_negative_new_value)
{
    const float largest = std::numeric_limits<float>::max();
    CheckIntegerQuantization(ge::DT_INT32, -largest, largest, largest, std::numeric_limits<int32_t>::min(), kInt32Max,
                             kInt32Max);
}

TEST_F(FloodFillTiling, u8_half_integer_rounding)
{
    CheckIntegerQuantization(ge::DT_UINT8, 10.5f, 0.5f, 1.5f, 11, 1, 2);
}

TEST_F(FloodFillTiling, i32_negative_half_integer_rounding)
{
    CheckIntegerQuantization(ge::DT_INT32, -0.5f, 0.5f, 1.5f, -1, 1, 2);
}
