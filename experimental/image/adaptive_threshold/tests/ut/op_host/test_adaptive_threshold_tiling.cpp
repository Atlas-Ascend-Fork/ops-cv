/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/adaptive_threshold_tiling_data.h"
#include "../gaussian_test_data.h"

using namespace std;
using namespace ge;

class AdaptiveThresholdTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveThresholdTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveThresholdTiling TearDown" << std::endl; }
};

namespace {
constexpr uint32_t kMean = 0;
constexpr uint32_t kGaussian = 1;
constexpr uint32_t kBinary = 0;
constexpr uint32_t kBinaryInv = 1;
constexpr size_t kWorkspaceSize = 16U * 1024U * 1024U; // WS_SYS_SIZE

// 按 int64 序列化 tiling data，与 ExecuteTestCase 内部的 to_string<int64_t> 保持一致。
std::string MakeExpectedTilingData(uint32_t imageCount, uint32_t height, uint32_t width, uint32_t totalRows,
                                   uint32_t baseRows, uint32_t extraRows, uint32_t tileWidth, uint32_t blockSize,
                                   uint32_t radius, uint32_t thresholdType, float outputValue, float thresholdDelta,
                                   const std::vector<uint32_t>& gaussianWeights)
{
    AdaptiveThresholdTilingData data = {};
    data.imageCount = imageCount;
    data.height = height;
    data.width = width;
    data.totalRows = totalRows;
    data.baseRows = baseRows;
    data.extraRows = extraRows;
    data.tileWidth = tileWidth;
    data.blockSize = blockSize;
    data.radius = radius;
    data.thresholdType = thresholdType;
    data.outputValue = outputValue;
    data.thresholdDelta = thresholdDelta;
    for (size_t i = 0; i < gaussianWeights.size() && i < ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE; ++i) {
        data.gaussianWeights[i] = gaussianWeights[i];
    }

    const int64_t* words = reinterpret_cast<const int64_t*>(&data);
    const size_t len = sizeof(AdaptiveThresholdTilingData) / sizeof(int64_t);
    std::string result;
    for (size_t i = 0; i < len; ++i) {
        result += std::to_string(words[i]) + " ";
    }
    return result;
}
} // namespace

TEST_F(AdaptiveThresholdTiling, mean_binary_rank2)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 4, 3, 1, 0, 4, 3, 1, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, gaussian_binary_rank2)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("GAUSSIAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kGaussian;
    // block_size=3 的高斯权重：[0.25, 0.5, 0.25] -> 量化 [16384, 32768, 16384]
    const string expectTilingData = MakeExpectedTilingData(1, 3, 4, 3, 1, 0, 4, 3, 1, kBinary, 255.0f, 2.0f,
                                                           {16384, 32768, 16384});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, gaussian_quantized_weights)
{
    // Compare all complete serialized int64 words containing frozen coefficients,
    // including the corrected center and unused zeros. In particular, independently
    // rounding the center gives the wrong total for block 31 and block 255.
    for (uint32_t blockSize : {5U, 7U, 9U, 11U, 31U, 255U}) {
        SCOPED_TRACE(::testing::Message() << "block=" << blockSize);
        struct CompileInfo {
        } compileInfo;
        gert::TilingContextPara tilingContextPara(
            "AdaptiveThreshold", {{{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND}},
            {{{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND}},
            {
                gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0F)),
                gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("GAUSSIAN")),
                gert::TilingContextPara::OpAttr("threshold_type",
                                                Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
                gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(blockSize)),
                gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0F)),
            },
            &compileInfo);
        const auto weights = AdaptiveThresholdTest::GaussianWeights(blockSize);
        ASSERT_EQ(weights.size(), blockSize);
        const auto expected = MakeExpectedTilingData(1, 3, 4, 3, 1, 0, 4, blockSize, blockSize / 2, kBinary, 255.0F,
                                                     2.0F, weights);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kGaussian, expected, {kWorkspaceSize});
    }
}

TEST_F(AdaptiveThresholdTiling, mean_inverse_rank3)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(127.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type",
                                            Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY_INV")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(5)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(-1.2f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(2, 3, 4, 6, 1, 0, 4, 5, 2, kBinaryInv, 127.0f, -2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, mean_binary_rank4)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{2, 1, 3, 4}, {2, 1, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 3, 4}, {2, 1, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(200.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(7)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(5.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(2, 3, 4, 6, 1, 0, 4, 7, 3, kBinary, 200.0f, 5.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, invalid_adaptive_method)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEDIAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_threshold_type)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_TRUNC")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_block_size_even)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_block_size_too_small)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_block_size_too_large)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(257)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_rank)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{4}, {4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_channel)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{2, 2, 3, 4}, {2, 2, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 3, 4}, {2, 2, 3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_max_value_nan)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr(
                "max_value", Ops::Cv::AnyValue::CreateFrom<float>(std::numeric_limits<float>::quiet_NaN())),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, invalid_c_nan)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr(
                "C", Ops::Cv::AnyValue::CreateFrom<float>(std::numeric_limits<float>::quiet_NaN())),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, empty_height_zero)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{0, 4}, {0, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{0, 4}, {0, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 0, 4, 0, 0, 0, 1, 3, 1, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, empty_width_zero)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 0}, {3, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 0}, {3, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 0, 3, 0, 0, 1, 3, 1, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, empty_both_zero)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{0, 0}, {0, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{0, 0}, {0, 0}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 0, 0, 0, 0, 0, 1, 3, 1, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, negative_height)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{-1, 4}, {-1, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{-1, 4}, {-1, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, negative_width)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{4, -1}, {4, -1}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4, -1}, {4, -1}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveThresholdTiling, mean_binary_rank2_float16)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(200.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    // tiling key = (dtype << 1) | adaptiveMethod, float16 = 1, MEAN = 0
    const uint64_t expectTilingKey = 2;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 4, 3, 1, 0, 4, 3, 1, kBinary, 200.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, mean_binary_decimal_c_round_up)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold", {{{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0F)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(1.2F)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 4, 3, 1, 0, 4, 3, 1, kBinary, 255.0F, 2.0F, {});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {kWorkspaceSize});
}

TEST_F(AdaptiveThresholdTiling, gaussian_binary_rank2_float32)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(1.5f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("GAUSSIAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(-0.5f)),
        },
        &compileInfo);

    // tiling key = (dtype << 1) | adaptiveMethod, float32 = 2, GAUSSIAN = 1
    const uint64_t expectTilingKey = 5;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 4, 3, 1, 0, 4, 3, 1, kBinary, 1.5f, -0.5f,
                                                           {16384, 32768, 16384});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, empty_height_float32_max_block)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(255)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = 4;
    const string expectTilingData = MakeExpectedTilingData(1, 0, 4, 0, 0, 0, 1, 255, 127, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, empty_width_float32_max_block)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 0}, {3, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 0}, {3, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(255)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = 4;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 0, 3, 0, 0, 1, 255, 127, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, empty_both_float32_max_block)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{0, 0}, {0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 0}, {0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(255)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = 4;
    const string expectTilingData = MakeExpectedTilingData(1, 0, 0, 0, 0, 0, 1, 255, 127, kBinary, 255.0f, 2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AdaptiveThresholdTiling, nonempty_float32_max_block_rejected)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(255)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
TEST_F(AdaptiveThresholdTiling, mean_streaming_max_window_wide)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{3, 4097}, {3, 4097}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4097}, {3, 4097}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(255)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 3, 4097, 3, 1, 0, 2048, 255, 127, kBinary, 255.0f, 2.0f,
                                                           {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(AdaptiveThresholdTiling, mean_vector_tall_max_window_wide)
{
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AdaptiveThreshold",
        {
            {{{129, 4097}, {129, 4097}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{129, 4097}, {129, 4097}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("max_value", Ops::Cv::AnyValue::CreateFrom<float>(255.0f)),
            gert::TilingContextPara::OpAttr("adaptive_method", Ops::Cv::AnyValue::CreateFrom<string>("MEAN")),
            gert::TilingContextPara::OpAttr("threshold_type", Ops::Cv::AnyValue::CreateFrom<string>("THRESH_BINARY")),
            gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(255)),
            gert::TilingContextPara::OpAttr("C", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);

    const uint64_t expectTilingKey = kMean;
    const string expectTilingData = MakeExpectedTilingData(1, 129, 4097, 129, 2, 1, 2048, 255, 127, kBinary, 255.0f,
                                                           2.0f, {});
    const std::vector<size_t> expectWorkspaces = {kWorkspaceSize};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
