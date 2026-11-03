/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/retinex_enhance.h"
#include "retinex_enhance_golden.h"

namespace {
__global__ __aicore__ void RetinexEnhanceUt(GM_ADDR image, GM_ADDR enhanced, GM_ADDR workspace, GM_ADDR tiling)
{
    const auto* tilingData = reinterpret_cast<const RetinexEnhanceTilingData*>(tiling);
    AscendC::TPipe pipe;
    NsRetinexEnhance::KernelRetinexEnhance op;
    op.Init(image, enhanced, workspace, tilingData, &pipe);
    op.Process();
}

struct GmDeleter {
    void operator()(uint8_t* address) const
    {
        if (address != nullptr) {
            AscendC::GmFree(address);
        }
    }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

uint32_t AlignElements(uint32_t elements) { return (elements + 7U) / 8U * 8U; }

// Match the host's buffer layout and anchor representation, never the golden's
// convolution: golden values are generated independently by NumPy float64.
RetinexEnhanceTilingData MakeTiling(uint32_t height, uint32_t width, uint32_t channels)
{
    RetinexEnhanceTilingData tiling{};
    tiling.height = height;
    tiling.width = width;
    tiling.channels = channels;
    tiling.tilePixels = std::min(32U, width);
    tiling.stripeCount = (width + tiling.tilePixels - 1U) / tiling.tilePixels;
    tiling.rowElements = width * channels;
    tiling.rowElementsAligned = std::max(AlignElements(tiling.rowElements),
                                         AlignElements((tiling.tilePixels + 1500U) * channels));
    tiling.tileElementsAligned = std::max(64U, AlignElements(tiling.tilePixels * channels));
    tiling.verticalTileRows = std::min(32U, height);
    tiling.verticalTileElementsAligned = tiling.verticalTileRows * tiling.tileElementsAligned;
    tiling.maxHaloElementsAligned = AlignElements((tiling.tilePixels + 1500U) * channels);
    tiling.verticalIndexElementsAligned = AlignElements(height + 1500U);
    constexpr uint32_t radii[] = {45, 240, 750};
    constexpr uint32_t offsets[] = {0, 2, 10};
    constexpr uint32_t counts[] = {2, 8, 24};
    constexpr double sigmas[] = {15.0, 80.0, 250.0};
    for (uint32_t scale = 0; scale < RETINEX_SCALE_COUNT; ++scale) {
        const double sigmaSquared = sigmas[scale] * sigmas[scale];
        double normalization = 1.0;
        for (uint32_t tap = 1; tap <= radii[scale]; ++tap) {
            normalization += 2.0 * std::exp(-static_cast<double>(tap * tap) / (2.0 * sigmaSquared));
        }
        tiling.radii[scale] = radii[scale];
        tiling.groupOffsets[scale] = offsets[scale];
        tiling.groupCounts[scale] = counts[scale];
        tiling.ratioSteps[scale] = static_cast<float>(std::exp(-1.0 / sigmaSquared));
        for (uint32_t group = 0; group < counts[scale]; ++group) {
            const uint32_t tap = group * RETINEX_WEIGHT_GROUP_SIZE;
            const uint32_t index = offsets[scale] + group;
            tiling.anchorWeights[index] = static_cast<float>(
                std::exp(-static_cast<double>(tap * tap) / (2.0 * sigmaSquared)) / normalization);
            tiling.anchorRatios[index] = static_cast<float>(
                std::exp(-static_cast<double>(2U * tap + 1U) / (2.0 * sigmaSquared)));
        }
    }
    return tiling;
}

void RunGolden(uint32_t height, uint32_t width, uint32_t channels, const float* input, const float* golden)
{
    const size_t elements = static_cast<size_t>(height) * width * channels;
    const size_t bytes = elements * sizeof(float);
    GmBuffer image(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
    GmBuffer output(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
    GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
    GmBuffer tilingBuffer(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(RetinexEnhanceTilingData))));
    ASSERT_NE(image.get(), nullptr);
    ASSERT_NE(output.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tilingBuffer.get(), nullptr);
    std::memcpy(image.get(), input, bytes);
    const auto tiling = MakeTiling(height, width, channels);
    std::memcpy(tilingBuffer.get(), &tiling, sizeof(tiling));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    std::vector<float> firstRun(elements);
    for (uint32_t round = 0; round < 2; ++round) {
        std::memset(output.get(), 0xFF, bytes);
        std::memset(workspace.get(), 0xFF, bytes);
        ICPU_RUN_KF(RetinexEnhanceUt, tiling.stripeCount, image.get(), output.get(), workspace.get(),
                    tilingBuffer.get());
        const auto* actual = reinterpret_cast<const float*>(output.get());
        std::vector<double> errors;
        double sum = 0.0;
        for (size_t index = 0; index < elements; ++index) {
            ASSERT_TRUE(std::isfinite(actual[index])) << index;
            EXPECT_GE(actual[index], 0.0f);
            EXPECT_LE(actual[index], 255.0f);
            const double error = std::abs(static_cast<double>(actual[index]) - golden[index]);
            errors.push_back(error);
            sum += error;
            if (channels == 4 && index % channels == 3) {
                EXPECT_FLOAT_EQ(actual[index], std::clamp(input[index], 0.0f, 255.0f));
            }
        }
        std::sort(errors.begin(), errors.end());
        EXPECT_LE(errors.back(), 0.5);
        // Upper order statistic bounds the interpolated 99th percentile.
        EXPECT_LE(errors[static_cast<size_t>(std::ceil(0.99 * (elements - 1)))], 0.1);
        EXPECT_LE(sum / elements, 0.02);
        if (round == 0) {
            std::memcpy(firstRun.data(), actual, bytes);
        } else {
            EXPECT_EQ(std::memcmp(firstRun.data(), actual, bytes), 0);
        }
    }
}

TEST(RetinexEnhanceKernel, GrayscaleReflectAndOutputMapping)
{
    RunGolden(2, 3, 1, RetinexGolden::input1, RetinexGolden::output1);
}

TEST(RetinexEnhanceKernel, RgbColorRestoration) { RunGolden(3, 5, 3, RetinexGolden::input3, RetinexGolden::output3); }

TEST(RetinexEnhanceKernel, RgbaTailStripesAndAlpha)
{
    RunGolden(2, 33, 4, RetinexGolden::input4, RetinexGolden::output4);
}
} // namespace
