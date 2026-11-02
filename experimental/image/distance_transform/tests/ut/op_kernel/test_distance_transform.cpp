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
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "../../../op_kernel/distance_transform.cpp"

namespace {
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16U * 1024U * 1024U;

float ChamferDistance(uint32_t dr, uint32_t dc, uint32_t maskSize)
{
    const float axial = maskSize == 3U ? 0.955F : 1.0F;
    const float diagonal = maskSize == 3U ? 1.3693F : 1.4F;
    if (maskSize == 3U) {
        const uint32_t diagonalSteps = std::min(dr, dc);
        return diagonalSteps * diagonal + (std::max(dr, dc) - diagonalSteps) * axial;
    }

    const uint32_t columns = dc + 1U;
    std::vector<float> cost((dr + 1U) * columns, std::numeric_limits<float>::infinity());
    cost[0] = 0.0F;
    for (uint32_t row = 0; row <= dr; ++row) {
        for (uint32_t col = 0; col <= dc; ++col) {
            if (row == 0U && col == 0U) {
                continue;
            }
            float best = std::numeric_limits<float>::infinity();
            if (row > 0U) {
                best = std::min(best, cost[(row - 1U) * columns + col] + axial);
            }
            if (col > 0U) {
                best = std::min(best, cost[row * columns + col - 1U] + axial);
            }
            if (row > 0U && col > 0U) {
                best = std::min(best, cost[(row - 1U) * columns + col - 1U] + diagonal);
            }
            if (row > 1U && col > 0U) {
                best = std::min(best, cost[(row - 2U) * columns + col - 1U] + 2.1969F);
            }
            if (row > 0U && col > 1U) {
                best = std::min(best, cost[(row - 1U) * columns + col - 2U] + 2.1969F);
            }
            cost[row * columns + col] = best;
        }
    }
    return cost[dr * columns + dc];
}

float ReferenceAt(const std::vector<uint8_t>& input, uint32_t h, uint32_t w, uint32_t row, uint32_t col, uint32_t mode,
                  uint32_t maskSize)
{
    if (input[row * w + col] == 0U) {
        return 0.0f;
    }
    float best = std::numeric_limits<float>::infinity();
    bool found = false;
    for (uint32_t br = 0; br < h; ++br) {
        for (uint32_t bc = 0; bc < w; ++bc) {
            if (input[br * w + bc] != 0U) {
                continue;
            }
            found = true;
            const uint32_t drValue = br > row ? br - row : row - br;
            const uint32_t dcValue = bc > col ? bc - col : col - bc;
            const float dr = static_cast<float>(drValue);
            const float dc = static_cast<float>(dcValue);
            float distance = dr + dc;
            if (mode == NsDistanceTransform::DISTANCE_L2) {
                distance = maskSize == 0U ? std::sqrt(dr * dr + dc * dc) : ChamferDistance(drValue, dcValue, maskSize);
            } else if (mode == NsDistanceTransform::DISTANCE_CHESSBOARD) {
                distance = std::max(dr, dc);
            }
            best = std::min(best, distance);
        }
    }
    if (found) {
        return best;
    }
    if (mode == NsDistanceTransform::DISTANCE_L1) {
        return static_cast<float>(h) + static_cast<float>(w);
    }
    if (mode == NsDistanceTransform::DISTANCE_CHESSBOARD) {
        return static_cast<float>(std::max(h, w));
    }
    return maskSize == 0U ? std::sqrt(static_cast<float>(h) * h + static_cast<float>(w) * w) :
                            static_cast<float>(h) + static_cast<float>(w);
}

void RunCase(const std::vector<uint8_t>& input, uint32_t h, uint32_t w, uint32_t mode, uint32_t maskSize = 0U,
             uint32_t coreNum = 1U, uint32_t rowBufferOverride = std::numeric_limits<uint32_t>::max(),
             uint32_t exactTileColumns = 0U)
{
    const uint32_t elems = h * w;
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(elems));
    auto* y = static_cast<float*>(AscendC::GmAlloc(elems * sizeof(float)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(SYSTEM_WORKSPACE_BYTES));
    auto* tiling = static_cast<DistanceTransformTilingData*>(AscendC::GmAlloc(sizeof(DistanceTransformTilingData)));
    std::copy(input.begin(), input.end(), x);
    std::fill(y, y + elems, -1.0f);
    const uint32_t defaultRowBuffer = mode == NsDistanceTransform::DISTANCE_L2 && maskSize == 0U ? 0U : 1U;
    const uint32_t useRowBuffer = rowBufferOverride == std::numeric_limits<uint32_t>::max() ? defaultRowBuffer :
                                                                                              rowBufferOverride;
    *tiling = DistanceTransformTilingData{
        1U, h, w, elems, mode, maskSize, coreNum, 3U, 64U, useRowBuffer, exactTileColumns};

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(distance_transform<0>, coreNum, x, reinterpret_cast<uint8_t*>(y), workspace,
                reinterpret_cast<uint8_t*>(tiling));

    // OpenCV mask-3 accumulates the 0.955F chamfer weight step by step, while
    // ChamferDistance computes the equivalent path with multiplication. Their
    // float32 association differs by several ULPs on long axial runs.
    const float tolerance = mode == NsDistanceTransform::DISTANCE_L2 && maskSize == 3U ? 1e-4F : 1e-5F;
    for (uint32_t row = 0; row < h; ++row) {
        for (uint32_t col = 0; col < w; ++col) {
            EXPECT_NEAR(y[row * w + col], ReferenceAt(input, h, w, row, col, mode, maskSize), tolerance)
                << "row=" << row << " col=" << col << " mode=" << mode;
        }
    }
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST(DistanceTransformKernel, StructuredNonAlignedAllModes)
{
    const uint32_t h = 5U;
    const uint32_t w = 7U;
    std::vector<uint8_t> input(h * w, 1U);
    input[0] = 0U;
    input[2U * w + 3U] = 0U;
    input[4U * w + 6U] = 0U;
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L1);
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L2);
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_CHESSBOARD);
}

TEST(DistanceTransformKernel, UnifiedGmFallbackAllModes)
{
    constexpr uint32_t height = 5U;
    constexpr uint32_t width = 7U;
    std::vector<uint8_t> input(height * width, 1U);
    input[0U] = 0U;
    input[2U * width + 3U] = 0U;
    input[4U * width + 6U] = 0U;
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 0U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_CHESSBOARD, 0U, 1U, 0U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L2, 3U, 1U, 0U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L2, 5U, 1U, 0U);
}

TEST(DistanceTransformKernel, L2ReturnsEuclideanDistanceInsteadOfSquaredDistance)
{
    const std::vector<uint8_t> input = {
        0U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 0U,
    };
    RunCase(input, 3U, 5U, NsDistanceTransform::DISTANCE_L2);
}

TEST(DistanceTransformKernel, L2ChamferMask3And5)
{
    const uint32_t h = 7U;
    const uint32_t w = 9U;
    std::vector<uint8_t> input(h * w, 1U);
    input[0U] = 0U;
    input[3U * w + 4U] = 0U;
    input[6U * w + 8U] = 0U;
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L2, 3U);
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L2, 5U);
}

TEST(DistanceTransformKernel, L2EnvelopeSiteValueIsNotOverwritten)
{
    const uint32_t h = 101U;
    const uint32_t w = 31U;
    std::vector<uint8_t> input(h * w, 1U);
    input[0U * w + 15U] = 0U;
    input[10U * w + 1U] = 0U;
    input[100U * w + 15U] = 0U;
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L2);
}

TEST(DistanceTransformKernel, PreciseL2DistributesColumnTilesAcrossCores)
{
    const uint32_t h = 17U;
    const uint32_t w = 34U;
    std::vector<uint8_t> input(h * w, 1U);
    input[0U] = 0U;
    input[8U * w + 17U] = 0U;
    input[16U * w + 33U] = 0U;
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L2, 0U, 5U, 0U, 8U);
}

TEST(DistanceTransformKernel, PreciseL2UbColumnTileHandlesNonAlignedTail)
{
    constexpr uint32_t h = 37U;
    constexpr uint32_t w = 53U;
    std::vector<uint8_t> input(h * w, 1U);
    for (uint32_t row = 0U; row < h; ++row) {
        input[row * w + (row * 11U + 3U) % w] = 0U;
        if ((row % 5U) == 0U) {
            input[row * w + (row * 17U + 29U) % w] = 0U;
        }
    }
    RunCase(input, h, w, NsDistanceTransform::DISTANCE_L2, 0U, 4U, 0U, 16U);
}

TEST(DistanceTransformKernel, AllBackgroundAndAllForeground)
{
    RunCase(std::vector<uint8_t>(3U * 11U, 0U), 3U, 11U, NsDistanceTransform::DISTANCE_L2);
    RunCase(std::vector<uint8_t>(3U * 11U, 1U), 3U, 11U, NsDistanceTransform::DISTANCE_L1);
    RunCase(std::vector<uint8_t>(3U * 11U, 1U), 3U, 11U, NsDistanceTransform::DISTANCE_L2);
    RunCase(std::vector<uint8_t>(3U * 11U, 1U), 3U, 11U, NsDistanceTransform::DISTANCE_CHESSBOARD);
}

TEST(DistanceTransformKernel, L1GroupAndTailBoundaries)
{
    constexpr uint32_t width = 13U;
    for (uint32_t height = 1U; height <= 8U; ++height) {
        std::vector<uint8_t> input(height * width, 1U);
        for (uint32_t row = 0U; row < height; ++row) {
            input[row * width + (row * 5U + 1U) % width] = 0U;
            if ((row & 1U) != 0U) {
                input[row * width + (row * 7U + 3U) % width] = 0U;
            }
        }
        RunCase(input, height, width, NsDistanceTransform::DISTANCE_L1);
    }
}

TEST(DistanceTransformKernel, L1VectorJumpPropagationNonAligned)
{
    constexpr uint32_t height = 7U;
    constexpr uint32_t width = 79U;
    std::vector<uint8_t> input(height * width, 1U);
    for (uint32_t row = 0U; row < height; ++row) {
        input[row * width + (row * 17U + 3U) % width] = 0U;
        input[row * width + (row * 29U + 41U) % width] = 0U;
    }
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, NsDistanceTransform::VECTOR_ROW_BUFFER);
}

TEST(DistanceTransformKernel, ChessboardVectorJumpPropagationNonAligned)
{
    constexpr uint32_t height = 7U;
    constexpr uint32_t width = 79U;
    std::vector<uint8_t> input(height * width, 1U);
    for (uint32_t row = 0U; row < height; ++row) {
        input[row * width + (row * 17U + 3U) % width] = 0U;
        input[row * width + (row * 29U + 41U) % width] = 0U;
    }
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_CHESSBOARD, 0U, 1U,
            NsDistanceTransform::VECTOR_ROW_BUFFER);
}

TEST(DistanceTransformKernel, L2ChamferVectorPropagationNonAligned)
{
    constexpr uint32_t height = 13U;
    constexpr uint32_t width = 79U;
    std::vector<uint8_t> input(height * width, 1U);
    for (uint32_t row = 0U; row < height; ++row) {
        input[row * width + (row * 17U + 3U) % width] = 0U;
        input[row * width + (row * 29U + 41U) % width] = 0U;
    }
    for (uint32_t maskSize : {3U, 5U}) {
        RunCase(input, height, width, NsDistanceTransform::DISTANCE_L2, maskSize, 1U,
                NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
    }
    RunCase(std::vector<uint8_t>(height * width, 0U), height, width, NsDistanceTransform::DISTANCE_L2, 5U, 1U,
            NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
    RunCase(std::vector<uint8_t>(height * width, 1U), height, width, NsDistanceTransform::DISTANCE_L2, 3U, 1U,
            NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
    RunCase(std::vector<uint8_t>(height * width, 1U), height, width, NsDistanceTransform::DISTANCE_L2, 5U, 1U,
            NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
}

TEST(DistanceTransformKernel, L2Mask3AdaptiveHorizontalPropagation)
{
    constexpr uint32_t height = 7U;
    constexpr uint32_t width = 79U;
    std::vector<uint8_t> vectorInput(height * width, 1U);
    std::vector<uint8_t> scalarFallbackInput(height * width, 1U);
    std::vector<uint8_t> mixedInput(height * width, 1U);
    for (uint32_t row = 0U; row < height; ++row) {
        for (uint32_t col = 0U; col < width; col += 16U) {
            vectorInput[row * width + col] = 0U;
        }
        scalarFallbackInput[row * width] = 0U;
    }
    for (uint32_t col = 0U; col < width; col += 16U) {
        mixedInput[(height / 2U) * width + col] = 0U;
    }
    RunCase(vectorInput, height, width, NsDistanceTransform::DISTANCE_L2, 3U, 1U,
            NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
    RunCase(scalarFallbackInput, height, width, NsDistanceTransform::DISTANCE_L2, 3U, 1U,
            NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
    RunCase(mixedInput, height, width, NsDistanceTransform::DISTANCE_L2, 3U, 1U,
            NsDistanceTransform::CHAMFER_VECTOR_ROW_BUFFER);
}

TEST(DistanceTransformKernel, CompactShapesAllModes)
{
    constexpr uint32_t height = 16U;
    constexpr uint32_t width = 16U;
    std::vector<uint8_t> input(height * width, 1U);
    input[0U] = 0U;
    input[3U * width + 11U] = 0U;
    input[8U * width + 7U] = 0U;
    input[15U * width + 15U] = 0U;

    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L2, 0U, 1U, 0U, 8U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L2, 3U, 1U, 1U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_L2, 5U, 1U, 1U);
    RunCase(input, height, width, NsDistanceTransform::DISTANCE_CHESSBOARD, 0U, 1U, 1U);
}

TEST(DistanceTransformKernel, NonAlignedAndConstantInputs)
{
    constexpr uint32_t height = 7U;
    constexpr uint32_t width = 13U;
    std::vector<uint8_t> structured(height * width, 1U);
    structured[1U] = 0U;
    structured[4U * width + 6U] = 0U;
    structured[6U * width + 12U] = 0U;
    RunCase(structured, height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);
    RunCase(structured, height, width, NsDistanceTransform::DISTANCE_L2, 0U, 1U, 0U, 8U);
    RunCase(structured, height, width, NsDistanceTransform::DISTANCE_CHESSBOARD, 0U, 1U, 1U);
    RunCase(std::vector<uint8_t>(height * width, 0U), height, width, NsDistanceTransform::DISTANCE_L2, 0U, 1U, 0U, 8U);
    RunCase(std::vector<uint8_t>(height * width, 1U), height, width, NsDistanceTransform::DISTANCE_L2, 0U, 1U, 0U, 8U);
    RunCase(std::vector<uint8_t>(height * width, 1U), height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);
}

TEST(DistanceTransformKernel, L1AlignedRowsUseVectorVerticalPass)
{
    for (uint32_t width : {8U, 16U, 24U, 32U}) {
        for (uint32_t height : {1U, 7U, 16U, 32U}) {
            std::vector<uint8_t> input(height * width, 1U);
            for (uint32_t row = 0U; row < height; ++row) {
                input[row * width + (row * 5U + 1U) % width] = 0U;
                if ((row & 1U) != 0U) {
                    input[row * width + (row * 7U + 3U) % width] = 0U;
                }
            }
            RunCase(input, height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);
        }
    }
}

TEST(DistanceTransformKernel, L1HandlesConstantAndCornerInputs)
{
    constexpr uint32_t height = 16U;
    constexpr uint32_t width = 16U;
    RunCase(std::vector<uint8_t>(height * width, 0U), height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);
    RunCase(std::vector<uint8_t>(height * width, 1U), height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);

    std::vector<uint8_t> corners(height * width, 1U);
    corners[0U] = 0U;
    corners[width - 1U] = 0U;
    corners[(height - 1U) * width] = 0U;
    corners[height * width - 1U] = 0U;
    RunCase(corners, height, width, NsDistanceTransform::DISTANCE_L1, 0U, 1U, 1U);
}
