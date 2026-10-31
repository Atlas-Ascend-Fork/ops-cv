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
#include <cstring>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/adaptive_threshold.h"
#include "../gaussian_test_data.h"

namespace {

template <uint32_t method, typename T>
__global__ __aicore__ void AdaptiveThresholdKernelUt(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, GM_ADDR completed)
{
    AscendC::TPipe pipe;
    const auto ubStart = pipe.GetQueueEndAddress<AscendC::TPosition::VECCALC>();
    NsAdaptiveThreshold::KernelAdaptiveThreshold<method, T> op;
    op.Init(x, y, reinterpret_cast<const AdaptiveThresholdTilingData*>(tiling), &pipe);
    op.Process();
    auto* status = reinterpret_cast<uint64_t*>(completed);
    status[0] = 1U;
    status[1] = pipe.GetQueueEndAddress<AscendC::TPosition::VECCALC>() - ubStart;
}

template <uint32_t method>
void CheckEmptyKernel(uint32_t imageCount, uint32_t height, uint32_t width)
{
    AdaptiveThresholdTilingData data{};
    data.imageCount = imageCount;
    data.height = height;
    data.width = width;
    data.totalRows = imageCount * height;
    data.tileWidth = 1;
    data.blockSize = 255;
    data.radius = 127;

    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(float)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(float)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(data)));
    auto* completed = static_cast<uint8_t*>(AscendC::GmAlloc(2 * sizeof(uint64_t)));
    ASSERT_NE(completed, nullptr);
    std::memset(completed, 0, 2 * sizeof(uint64_t));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memset(y, 0x5A, sizeof(float));
    std::memcpy(tiling, &data, sizeof(data));

    // Bypass ACLNN's empty-tensor handling: Init must not reserve a 262208-byte UB window.
    auto kernel = AdaptiveThresholdKernelUt<method, float>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1U, x, y, tiling, completed);
    EXPECT_EQ(reinterpret_cast<uint64_t*>(completed)[0], 1U) << "Kernel did not return normally";
    EXPECT_EQ(reinterpret_cast<uint64_t*>(completed)[1], 0U) << "Empty kernel allocated UB";
    for (size_t index = 0; index < sizeof(float); ++index) {
        EXPECT_EQ(y[index], 0x5A);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
    AscendC::GmFree(completed);
}

TEST(AdaptiveThresholdKernelTest, empty_float32_max_block)
{
    for (const auto& shape : std::vector<std::vector<uint32_t>>{{1, 0, 4}, {1, 3, 0}, {1, 0, 0}, {0, 3, 4}}) {
        SCOPED_TRACE(::testing::Message() << "N=" << shape[0] << " H=" << shape[1] << " W=" << shape[2]);
        CheckEmptyKernel<ADAPTIVE_THRESHOLD_METHOD_MEAN>(shape[0], shape[1], shape[2]);
        CheckEmptyKernel<ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(shape[0], shape[1], shape[2]);
    }
}

template <uint32_t method, typename T>
double ReferenceMean(const T* input, const AdaptiveThresholdTilingData& data, uint32_t n, uint32_t row, uint32_t col)
{
    // Full 2D convolution: no separable pass, symmetric folding, ring buffer or tile indexing.
    double sum = 0.0;
    uint64_t integerSum = 0;
    for (uint32_t ky = 0; ky < data.blockSize; ++ky) {
        const auto r = std::clamp(static_cast<int32_t>(row + ky) - static_cast<int32_t>(data.radius), 0,
                                  static_cast<int32_t>(data.height) - 1);
        for (uint32_t kx = 0; kx < data.blockSize; ++kx) {
            const auto c = std::clamp(static_cast<int32_t>(col + kx) - static_cast<int32_t>(data.radius), 0,
                                      static_cast<int32_t>(data.width) - 1);
            const T value = input[(n * data.height + r) * data.width + c];
            if constexpr (method == ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN) {
                const uint64_t weight = static_cast<uint64_t>(data.gaussianWeights[ky]) * data.gaussianWeights[kx];
                if constexpr (std::is_same<T, uint8_t>::value) {
                    integerSum += static_cast<uint64_t>(value) * weight;
                } else {
                    sum += static_cast<double>(static_cast<float>(value)) * static_cast<double>(weight);
                }
            } else {
                sum += static_cast<float>(value);
            }
        }
    }
    if constexpr (method == ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN) {
        constexpr uint64_t denominator = 65536ULL * 65536ULL;
        if constexpr (std::is_same<T, uint8_t>::value) {
            return static_cast<double>((integerSum + denominator / 2) / denominator);
        } else {
            return sum / static_cast<double>(denominator);
        }
    } else {
        const double mean = sum / (data.blockSize * data.blockSize);
        if constexpr (std::is_same<T, uint8_t>::value) {
            return static_cast<uint32_t>(mean + 0.5);
        } else {
            return mean;
        }
    }
}

template <typename T, uint32_t method = ADAPTIVE_THRESHOLD_METHOD_MEAN>
void CheckTiledThreshold(uint32_t blockSize, uint32_t thresholdType, uint32_t imageHeight = 5, uint32_t imageWidth = 17,
                         uint32_t tileWidth = 4, float thresholdDelta = 2.0F, float outputValue = 255.0F,
                         const std::vector<T>& input = {}, uint32_t blockDim = 1)
{
    SCOPED_TRACE(::testing::Message() << "method=" << method << " block=" << blockSize << " type=" << thresholdType
                                      << " H=" << imageHeight << " W=" << imageWidth << " tile=" << tileWidth
                                      << " C=" << thresholdDelta);
    AdaptiveThresholdTilingData data{};
    data.imageCount = 2;
    data.height = imageHeight;
    data.width = imageWidth;
    data.totalRows = data.imageCount * data.height;
    ASSERT_GT(blockDim, 0U);
    ASSERT_LE(blockDim, data.totalRows);
    data.baseRows = data.totalRows / blockDim;
    data.extraRows = data.totalRows % blockDim;
    data.tileWidth = tileWidth; // Include nonaligned rows, internal halos and a short tail.
    data.blockSize = blockSize;
    data.radius = blockSize / 2;
    data.thresholdType = thresholdType;
    data.outputValue = outputValue;
    data.thresholdDelta = thresholdDelta;
    if constexpr (method == ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN) {
        const auto weights = AdaptiveThresholdTest::GaussianWeights(blockSize);
        ASSERT_EQ(weights.size(), blockSize);
        ASSERT_EQ(std::accumulate(weights.begin(), weights.end(), uint64_t{0}), 65536U);
        std::copy(weights.begin(), weights.end(), data.gaussianWeights);
    }
    const size_t count = data.totalRows * data.width;
    ASSERT_TRUE(input.empty() || input.size() == count);

    auto* x = static_cast<T*>(AscendC::GmAlloc(count * sizeof(T)));
    auto* y = static_cast<T*>(AscendC::GmAlloc(count * sizeof(T)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(data)));
    auto* completed = static_cast<uint8_t*>(AscendC::GmAlloc(2 * sizeof(uint64_t)));
    ASSERT_NE(completed, nullptr);
    std::memset(completed, 0, 2 * sizeof(uint64_t));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);
    for (size_t index = 0; index < count; ++index) {
        const auto value = (index * 73U + index / data.width * 19U) % 256U;
        if (!input.empty()) {
            x[index] = input[index];
        } else if constexpr (method == ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN && !std::is_same<T, uint8_t>::value) {
            x[index] = static_cast<T>((static_cast<float>(value) - 128.0F) * 0.25F);
        } else {
            x[index] = static_cast<T>(static_cast<float>(value));
        }
    }
    std::memset(y, 0x5A, count * sizeof(T)); // Neither valid output: also catch skipped writes.
    std::memcpy(tiling, &data, sizeof(data));

    auto kernel = AdaptiveThresholdKernelUt<method, T>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, blockDim, reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), tiling, completed);
    EXPECT_EQ(reinterpret_cast<uint64_t*>(completed)[0], 1U) << "Kernel did not return normally";

    for (uint32_t n = 0; n < data.imageCount; ++n) {
        for (uint32_t row = 0; row < data.height; ++row) {
            for (uint32_t col = 0; col < data.width; ++col) {
                const double mean = ReferenceMean<method>(x, data, n, row, col);
                const size_t index = (n * data.height + row) * data.width + col;
                const bool above = static_cast<float>(x[index]) > mean - data.thresholdDelta;
                const T expected = static_cast<T>((above != (thresholdType != 0)) ? data.outputValue : 0.0F);
                EXPECT_EQ(y[index], expected) << "n=" << n << " row=" << row << " col=" << col;
            }
        }
    }
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
    AscendC::GmFree(completed);
}

TEST(AdaptiveThresholdKernelTest, mean_tiles_uint8)
{
    CheckTiledThreshold<uint8_t>(3, 0);
    CheckTiledThreshold<uint8_t>(9, 1); // Radius reaches beyond the tile; image is smaller than the window.
}

TEST(AdaptiveThresholdKernelTest, mean_streaming_uint8)
{
    CheckTiledThreshold<uint8_t>(61, 0); // Last ring-buffer window below the streaming boundary.
    CheckTiledThreshold<uint8_t>(63, 0);
    CheckTiledThreshold<uint8_t>(63, 1);
    CheckTiledThreshold<uint8_t>(255, 0);
    CheckTiledThreshold<uint8_t>(255, 1);
    CheckTiledThreshold<uint8_t>(63, 0, 32); // Every window covers all image rows.
    CheckTiledThreshold<uint8_t>(63, 1, 33); // First height using the ring-buffer fallback.
}

TEST(AdaptiveThresholdKernelTest, mean_vector_uint8)
{
    CheckTiledThreshold<uint8_t>(3, 0, 5, 32, 32);   // Vector dispatch boundary.
    CheckTiledThreshold<uint8_t>(31, 1, 5, 33, 32);  // One-column tail, unaligned GM rows.
    CheckTiledThreshold<uint8_t>(63, 0, 33, 97, 32); // Vertical updates, image reset, halo > tail.
    CheckTiledThreshold<uint8_t>(255, 1, 2, 97, 32); // Radius spans several tiles.
    CheckTiledThreshold<uint8_t>(255, 0, 1, 65, 32); // Both vertical borders coincide.
}

TEST(AdaptiveThresholdKernelTest, mean_tiles_float32)
{
    CheckTiledThreshold<float>(3, 1);
    CheckTiledThreshold<float>(9, 0);
}

TEST(AdaptiveThresholdKernelTest, mean_tiles_float16)
{
    CheckTiledThreshold<half>(3, 0);
    CheckTiledThreshold<half>(9, 1);
}

void CheckMeanRoundingBoundary(uint32_t height, uint32_t width, uint32_t thresholdType, bool belowHalf)
{
    SCOPED_TRACE(::testing::Message() << "H=" << height << " W=" << width << " type=" << thresholdType
                                      << " belowHalf=" << belowHalf);
    AdaptiveThresholdTilingData data{};
    data.imageCount = 1;
    data.height = height;
    data.width = width;
    data.totalRows = height;
    // Run the first core's two rows, retaining all source rows needed by its halo.
    data.baseRows = 2;
    data.tileWidth = width >= 32 ? 32 : 2;
    data.blockSize = 255;
    data.radius = 127;
    data.thresholdType = thresholdType;
    data.outputValue = 255.0F;
    data.thresholdDelta = 0.0F;

    const size_t count = static_cast<size_t>(height) * width;
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(count));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(count));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(data)));
    auto* completed = static_cast<uint8_t*>(AscendC::GmAlloc(2 * sizeof(uint64_t)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);
    ASSERT_NE(completed, nullptr);
    std::memset(x, 129, count);
    std::memset(x, 130, width);
    if (belowHalf) {
        x[width == 255 ? 0 : 1] = 129;
    }
    std::memset(y, 0x5A, count);
    std::memset(completed, 0, 2 * sizeof(uint64_t));
    std::memcpy(tiling, &data, sizeof(data));

    // At [0,127] (wide) or [0,0] (narrow), BORDER_REPLICATE gives the same sums:
    // 8420737 / 65025 is just below 129.5; 8420865 / 65025 is just above it.
    // Float addition of half the area can incorrectly round the first case to 130.
    const uint32_t boundaryColumn = width == 255 ? 127 : 0;
    ASSERT_EQ((ReferenceMean<ADAPTIVE_THRESHOLD_METHOD_MEAN>(x, data, 0, 0, boundaryColumn)),
              belowHalf ? 129.0 : 130.0);
    auto kernel = AdaptiveThresholdKernelUt<ADAPTIVE_THRESHOLD_METHOD_MEAN, uint8_t>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1U, x, y, tiling, completed);
    EXPECT_EQ(reinterpret_cast<uint64_t*>(completed)[0], 1U);
    EXPECT_EQ(y[boundaryColumn], (belowHalf != (thresholdType != 0U)) ? 255U : 0U);
    for (uint32_t row = 0; row < data.baseRows; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            const double mean = ReferenceMean<ADAPTIVE_THRESHOLD_METHOD_MEAN>(x, data, 0, row, col);
            const bool above = x[row * width + col] > mean;
            EXPECT_EQ(y[row * width + col], (above != (thresholdType != 0U)) ? 255U : 0U)
                << "row=" << row << " col=" << col;
        }
    }
    for (size_t index = data.baseRows * width; index < count; ++index) {
        EXPECT_EQ(y[index], 0x5A) << "Unassigned row written at index=" << index;
    }
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
    AscendC::GmFree(completed);
}

TEST(AdaptiveThresholdKernelTest, mean_multi_core_start_row)
{
    // Two cores split the rows evenly; core 1 starts at a nonzero global row.
    CheckTiledThreshold<uint8_t>(3, 0, 4, 17, 4, 2.0F, 255.0F, {}, 2U);
    CheckTiledThreshold<half>(3, 1, 4, 17, 4, 0.5F, 3.5F, {}, 2U);
}

TEST(AdaptiveThresholdKernelTest, mean_uint8_rounding_boundary)
{
    for (uint32_t thresholdType : {0U, 1U}) {
        for (bool belowHalf : {true, false}) {
            CheckMeanRoundingBoundary(128, 255, thresholdType, belowHalf); // Reported vector case, tiled with tail.
            CheckMeanRoundingBoundary(128, 3, thresholdType, belowHalf);   // Narrow streaming path.
            CheckMeanRoundingBoundary(129, 3, thresholdType, belowHalf);   // Narrow ring-buffer path.
        }
    }
}

template <typename T>
void CheckGaussianTiles()
{
    for (uint32_t thresholdType : {0U, 1U}) {
        const float delta = std::is_same<T, uint8_t>::value ? (thresholdType == 0 ? 2.0F : -2.0F) :
                                                              (thresholdType == 0 ? 0.125F : -0.125F);
        const float output = std::is_same<T, uint8_t>::value ? 201.0F : 3.5F;
        for (uint32_t blockSize : {3U, 5U, 7U, 9U, 11U, 31U}) {
            // Small windows wrap the ring twice before resetting at the next image.
            // Odd/even radii exercise both the unrolled pair loop and its scalar tail.
            const uint32_t height = blockSize <= 11 ? 2 * blockSize + 1 : 5;
            CheckTiledThreshold<T, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(blockSize, thresholdType, height, 17, 4, delta,
                                                                       output);
        }
        CheckTiledThreshold<T, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(31, thresholdType, 1, 9, 4, delta, output);
        CheckTiledThreshold<T, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(9, thresholdType, 7, 1, 1, delta, output);
        CheckTiledThreshold<T, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(11, thresholdType, 2, 3, 4, delta, output);

        // Exactly equal to the Gaussian mean: the comparison must remain strictly greater.
        CheckTiledThreshold<T, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(31, thresholdType, 2, 5, 2, 0.0F, output,
                                                                   std::vector<T>(20, static_cast<T>(1.0F)));
    }
}

TEST(AdaptiveThresholdKernelTest, gaussian_tiles_uint8) { CheckGaussianTiles<uint8_t>(); }

TEST(AdaptiveThresholdKernelTest, gaussian_tiles_float16) { CheckGaussianTiles<half>(); }

TEST(AdaptiveThresholdKernelTest, gaussian_tiles_float32) { CheckGaussianTiles<float>(); }

TEST(AdaptiveThresholdKernelTest, gaussian_uint8_round_half_up)
{
    // In the first image, the center's exact weighted mean is 0.5. It rounds to 1.
    for (uint32_t thresholdType : {0U, 1U}) {
        CheckTiledThreshold<uint8_t, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(3, thresholdType, 1, 3, 2, 0.0F, 255.0F,
                                                                         {0, 1, 0, 1, 0, 1});
    }
}

TEST(AdaptiveThresholdKernelTest, gaussian_max_block)
{
    // Nonempty FLOAT32 + block 255 is rejected by Host UB checks. Test the supported types.
    for (uint32_t thresholdType : {0U, 1U}) {
        CheckTiledThreshold<uint8_t, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(255, thresholdType, 2, 5, 2);
        CheckTiledThreshold<half, ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN>(255, thresholdType, 2, 5, 2, -0.125F, 3.5F);
    }
}

} // namespace
