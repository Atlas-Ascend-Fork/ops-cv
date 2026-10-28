/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/clahe.h"
#include "clahe_golden.h"

namespace {
constexpr uint32_t GUARD_BYTES = 64;
constexpr uint8_t GUARD_VALUE = 0xa5;

struct TileConfig {
    uint32_t height;
    uint32_t width;
    uint32_t tilesX;
    uint32_t tilesY;
    uint32_t tileWidth;
    uint32_t tileHeight;
    uint32_t clipCount;
};

struct GmDeleter {
    void operator()(uint8_t* ptr) const
    {
        if (ptr != nullptr) {
            AscendC::GmFree(ptr);
        }
    }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

template <typename T>
__global__ __aicore__ void ClaheKernelUt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    const auto data = *reinterpret_cast<const ClaheTilingData*>(tiling);
    AscendC::TPipe pipe;
    NsClahe::KernelClahe<T> op;
    op.Init(x, y, workspace, &data, &pipe);
    op.Process();
}

template <typename T, size_t N>
void RunGolden(const std::array<T, N>& input, const std::array<T, N>& golden, const TileConfig& config)
{
    ASSERT_EQ(N, static_cast<size_t>(config.height) * config.width);
    ClaheTilingData data{};
    data.imageElements = N;
    data.batch = 1;
    data.height = config.height;
    data.width = config.width;
    data.tilesX = config.tilesX;
    data.tilesY = config.tilesY;
    data.tileWidth = config.tileWidth;
    data.tileHeight = config.tileHeight;
    data.bins = sizeof(T) == 1 ? 256 : 65536;
    data.clipCount = config.clipCount;
    data.coreNum = 1;
    data.lutElements = static_cast<uint64_t>(data.tilesX) * data.tilesY * data.bins;
    data.regionCount = static_cast<uint64_t>(data.tilesX + 1) * (data.tilesY + 1);
    data.lutScale = static_cast<float>(data.bins - 1) / static_cast<float>(data.tileWidth * data.tileHeight);
    data.invTileWidth = 1.0f / static_cast<float>(data.tileWidth);
    data.invTileHeight = 1.0f / static_cast<float>(data.tileHeight);

    constexpr size_t BYTES = N * sizeof(T);
    GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(BYTES + GUARD_BYTES)));
    GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(BYTES + GUARD_BYTES)));
    GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(data.lutElements * sizeof(uint16_t))));
    GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(data))));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memset(x.get(), GUARD_VALUE, BYTES + GUARD_BYTES);
    std::memset(y.get(), GUARD_VALUE, BYTES + GUARD_BYTES);
    std::memcpy(x.get(), input.data(), BYTES);
    std::memcpy(tiling.get(), &data, sizeof(data));

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(ClaheKernelUt<T>, data.coreNum, x.get(), y.get(), workspace.get(), tiling.get());

    const auto* actual = reinterpret_cast<const T*>(y.get());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(actual[i], golden[i]) << "pixel=" << i;
    }
    EXPECT_EQ(std::memcmp(x.get(), input.data(), BYTES), 0);
    for (size_t i = 0; i < GUARD_BYTES; ++i) {
        EXPECT_EQ(x.get()[BYTES + i], GUARD_VALUE) << "input guard=" << i;
        EXPECT_EQ(y.get()[BYTES + i], GUARD_VALUE) << "output guard=" << i;
    }
}

TEST(ClaheKernel, Uint8ClippingReflectAndInterpolation)
{
    // clip_limit=2, grid=[3,2]; the divisible width also extends in the padding branch.
    RunGolden(ClaheGolden::UINT8_REFLECT_INPUT, ClaheGolden::UINT8_REFLECT_OUTPUT, {7, 9, 3, 2, 4, 4, 1});
}

TEST(ClaheKernel, Uint8UnclippedNearestEven)
{
    // Pixel (0,2): LUT values 128 and 85 interpolate to 106.5, which rounds to even 106.
    RunGolden(ClaheGolden::UINT8_HALF_INPUT, ClaheGolden::UINT8_HALF_OUTPUT, {3, 4, 2, 1, 2, 3, 0});
}

TEST(ClaheKernel, Uint16FullRangeSegmentedCdf)
{
    // clip_limit=40, grid=[1,1]; covers segmented prefix sums, unsigned conversion, and the maximum gray value.
    RunGolden(ClaheGolden::UINT16_REFLECT_INPUT, ClaheGolden::UINT16_REFLECT_OUTPUT, {5, 7, 1, 1, 7, 5, 1});
}

TEST(ClaheKernel, Uint16UnclippedNearestEven)
{
    // Pixel (0,2): LUT values 32768 and 21845 interpolate to 27306.5, which rounds to even 27306.
    RunGolden(ClaheGolden::UINT16_HALF_INPUT, ClaheGolden::UINT16_HALF_OUTPUT, {3, 4, 2, 1, 2, 3, 0});
}
} // namespace
