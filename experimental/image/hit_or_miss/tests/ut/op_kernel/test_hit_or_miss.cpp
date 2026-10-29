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
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "hit_or_miss_tiling.h"
#include "../../../op_kernel/hit_or_miss.h"

namespace {
struct KernelCase {
    const char* name;
    std::vector<int64_t> shape;
    int64_t kernelH;
    int64_t kernelW;
    std::vector<int8_t> values;
    std::array<int64_t, 2> axes;
    std::array<int64_t, 2> anchor;
    bool borderValue;
    uint32_t cores;
    uint32_t pattern;
};

__global__ __aicore__ void hit_or_miss_kernel_ut(GM_ADDR input, GM_ADDR kernel, GM_ADDR output, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA_WITH_STRUCT(HitOrMissTilingData, tilingData, tiling);
    NsHitOrMiss::HitOrMissKernel op;
    op.Init(input, kernel, output, &tilingData);
    op.Process();
}

std::vector<KernelCase> MakeKernelCases()
{
    const std::vector<int8_t> cross = {0, -1, 0, -1, 1, -1, 0, -1, 0};
    std::vector<int8_t> large(63, 0);
    large.front() = 1;
    large.back() = -1;
    return {
        {"AllZeroKernel", {17, 19}, 3, 3, std::vector<int8_t>(9, 0), {-2, -1}, {-1, -1}, true, 3, 0},
        {"Positive1x1", {5, 7}, 1, 1, {1}, {-2, -1}, {-1, -1}, true, 2, 2},
        {"Negative1x1", {5, 7}, 1, 1, {-1}, {-2, -1}, {-1, -1}, true, 2, 0},
        {"MixedBorderTrue", {17, 19}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 3, 0},
        {"MixedBorderFalse", {17, 19}, 3, 3, cross, {-2, -1}, {-1, -1}, false, 3, 0},
        {"NegativeOutsideTrue", {1, 1}, 1, 3, {-1, 0, 0}, {-2, -1}, {0, 1}, true, 1, 1},
        {"NegativeOutsideFalse", {1, 1}, 1, 3, {-1, 0, 0}, {-2, -1}, {0, 1}, false, 1, 1},
        {"Width128True", {9, 128}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 1, 0},
        {"Width128False", {9, 128}, 3, 3, cross, {-2, -1}, {-1, -1}, false, 1, 0},
        {"Wide256", {17, 256}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 1, 0},
        {"Wide384", {17, 384}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 3, 0},
        {"Interior512", {17, 512}, 3, 3, cross, {-2, -1}, {-1, -1}, false, 8, 0},
        {"Tail129", {9, 129}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 3, 0},
        {"MultiPlane", {2, 9, 129}, 3, 3, cross, {-2, -1}, {-1, -1}, false, 3, 0},
        {"GenericAxes", {5, 3, 7}, 3, 3, cross, {0, 2}, {-1, -1}, true, 3, 0},
        {"ReversedAxes", {7, 9}, 3, 3, cross, {1, 0}, {-1, -1}, false, 2, 0},
        {"HeightOne", {1, 257}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 3, 0},
        {"WidthOne", {257, 1}, 3, 3, cross, {-2, -1}, {-1, -1}, true, 3, 0},
        {"EvenKernel", {13, 65}, 2, 2, {1, 0, -1, 0}, {-2, -1}, {-1, -1}, true, 3, 0},
        {"ExplicitAnchor", {13, 65}, 3, 3, cross, {-2, -1}, {0, 2}, false, 3, 0},
        {"Dense25", {9, 128}, 5, 5, std::vector<int8_t>(25, -1), {-2, -1}, {-1, -1}, false, 3, 3},
        {"Dense49", {11, 129}, 7, 7, std::vector<int8_t>(49, -1), {-2, -1}, {-1, -1}, true, 3, 3},
        {"LargePadding", {5, 65}, 1, 63, large, {-2, -1}, {0, 0}, true, 3, 0},
    };
}

size_t ElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

int64_t NormalizeAxis(int64_t axis, size_t rank) { return axis < 0 ? axis + static_cast<int64_t>(rank) : axis; }

std::vector<uint8_t> MakeInput(const KernelCase& testCase)
{
    std::vector<uint8_t> input(ElementCount(testCase.shape));
    uint32_t state = 0x52A41B97U;
    for (size_t index = 0; index < input.size(); ++index) {
        state = state * 1664525U + 1013904223U;
        const uint32_t divisor = testCase.pattern == 3 ? 31U : 7U;
        bool foreground = ((state >> 24U) % divisor) == 0U;
        if (testCase.pattern == 1) {
            foreground = false;
        } else if (testCase.pattern == 2) {
            foreground = true;
        }
        input[index] = foreground ? (index % 2 == 0 ? 2U : 255U) : 0U;
    }
    return input;
}

// 直接按多维坐标和约束定义计算，不依赖被测 Kernel 的辅助函数。
std::vector<uint8_t> Reference(const KernelCase& testCase, const std::vector<uint8_t>& input)
{
    const int64_t axis0 = NormalizeAxis(testCase.axes[0], testCase.shape.size());
    const int64_t axis1 = NormalizeAxis(testCase.axes[1], testCase.shape.size());
    const int64_t anchorH = testCase.anchor[0] == -1 ? testCase.kernelH / 2 : testCase.anchor[0];
    const int64_t anchorW = testCase.anchor[1] == -1 ? testCase.kernelW / 2 : testCase.anchor[1];
    const bool hasActive = std::any_of(testCase.values.begin(), testCase.values.end(),
                                       [](int8_t value) { return value == 1 || value == -1; });

    std::vector<uint8_t> output(input.size());
    std::vector<int64_t> coordinate(testCase.shape.size());
    for (size_t flat = 0; flat < input.size(); ++flat) {
        if (!hasActive) {
            output[flat] = input[flat] != 0;
            continue;
        }
        size_t remaining = flat;
        for (size_t index = testCase.shape.size(); index > 0; --index) {
            coordinate[index - 1] = remaining % static_cast<size_t>(testCase.shape[index - 1]);
            remaining /= static_cast<size_t>(testCase.shape[index - 1]);
        }
        bool matches = true;
        for (int64_t row = 0; row < testCase.kernelH && matches; ++row) {
            for (int64_t col = 0; col < testCase.kernelW; ++col) {
                const int8_t constraint = testCase.values[row * testCase.kernelW + col];
                if (constraint != 1 && constraint != -1) {
                    continue;
                }
                const int64_t sourceRow = coordinate[axis0] + row - anchorH;
                const int64_t sourceCol = coordinate[axis1] + col - anchorW;
                bool currentMatch = testCase.borderValue;
                if (sourceRow >= 0 && sourceRow < testCase.shape[axis0] && sourceCol >= 0 &&
                    sourceCol < testCase.shape[axis1]) {
                    size_t sourceOffset = 0;
                    for (size_t dim = 0; dim < testCase.shape.size(); ++dim) {
                        int64_t value = coordinate[dim];
                        if (static_cast<int64_t>(dim) == axis0) {
                            value = sourceRow;
                        } else if (static_cast<int64_t>(dim) == axis1) {
                            value = sourceCol;
                        }
                        sourceOffset = sourceOffset * static_cast<size_t>(testCase.shape[dim]) +
                                       static_cast<size_t>(value);
                    }
                    const bool foreground = input[sourceOffset] != 0;
                    currentMatch = constraint == 1 ? foreground : !foreground;
                }
                if (!currentMatch) {
                    matches = false;
                    break;
                }
            }
        }
        output[flat] = matches;
    }
    return output;
}

HitOrMissTilingData MakeTiling(const KernelCase& testCase)
{
    HitOrMissTilingData data{};
    data.totalLength = static_cast<int64_t>(ElementCount(testCase.shape));
    const int64_t alignedBlocks = (data.totalLength + 31) / 32;
    data.usedCoreNum = static_cast<int32_t>(std::min<int64_t>(testCase.cores, alignedBlocks));
    data.blockLength = ((alignedBlocks + data.usedCoreNum - 1) / data.usedCoreNum) * 32;
    const int64_t axis0 = NormalizeAxis(testCase.axes[0], testCase.shape.size());
    const int64_t axis1 = NormalizeAxis(testCase.axes[1], testCase.shape.size());
    int64_t stride = 1;
    for (size_t index = testCase.shape.size(); index > 0; --index) {
        const int64_t axis = static_cast<int64_t>(index - 1);
        if (axis == axis0) {
            data.axis0Stride = stride;
            data.axis0Dim = testCase.shape[index - 1];
        }
        if (axis == axis1) {
            data.axis1Stride = stride;
            data.axis1Dim = testCase.shape[index - 1];
        }
        stride *= testCase.shape[index - 1];
    }
    data.kernelH = static_cast<int32_t>(testCase.kernelH);
    data.kernelW = static_cast<int32_t>(testCase.kernelW);
    data.anchorH = static_cast<int32_t>(testCase.anchor[0] == -1 ? testCase.kernelH / 2 : testCase.anchor[0]);
    data.anchorW = static_cast<int32_t>(testCase.anchor[1] == -1 ? testCase.kernelW / 2 : testCase.anchor[1]);
    data.borderValue = testCase.borderValue ? 1 : 0;
    return data;
}

struct GmDeleter {
    void operator()(uint8_t* pointer) const { AscendC::GmFree(pointer); }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

size_t AlignBytes(size_t bytes) { return (bytes + 31U) / 32U * 32U; }

class HitOrMissKernelGolden : public testing::TestWithParam<KernelCase> {};

TEST_P(HitOrMissKernelGolden, MatchesCpuReference)
{
    const auto& testCase = GetParam();
    ASSERT_EQ(testCase.values.size(), static_cast<size_t>(testCase.kernelH * testCase.kernelW));
    const auto input = MakeInput(testCase);
    const auto golden = Reference(testCase, input);
    const auto data = MakeTiling(testCase);

    constexpr size_t GUARD_BYTES = 32;
    constexpr uint8_t SENTINEL = 0xA5;
    const size_t outputBytes = AlignBytes(input.size()) + 2 * GUARD_BYTES;
    GmBuffer inputGm(static_cast<uint8_t*>(AscendC::GmAlloc(AlignBytes(input.size()))));
    GmBuffer kernelGm(static_cast<uint8_t*>(AscendC::GmAlloc(AlignBytes(testCase.values.size()))));
    GmBuffer outputGm(static_cast<uint8_t*>(AscendC::GmAlloc(outputBytes)));
    GmBuffer tilingGm(static_cast<uint8_t*>(AscendC::GmAlloc(AlignBytes(sizeof(data)))));
    ASSERT_NE(inputGm.get(), nullptr);
    ASSERT_NE(kernelGm.get(), nullptr);
    ASSERT_NE(outputGm.get(), nullptr);
    ASSERT_NE(tilingGm.get(), nullptr);

    std::memset(inputGm.get(), 0x6D, AlignBytes(input.size()));
    std::memcpy(inputGm.get(), input.data(), input.size());
    std::memset(kernelGm.get(), 0, AlignBytes(testCase.values.size()));
    std::memcpy(kernelGm.get(), testCase.values.data(), testCase.values.size());
    std::memset(outputGm.get(), SENTINEL, outputBytes);
    std::memcpy(tilingGm.get(), &data, sizeof(data));
    auto* output = outputGm.get() + GUARD_BYTES;

    ICPU_RUN_KF(hit_or_miss_kernel_ut, static_cast<uint32_t>(data.usedCoreNum), inputGm.get(), kernelGm.get(), output,
                nullptr, tilingGm.get());

    for (size_t index = 0; index < golden.size(); ++index) {
        ASSERT_EQ(output[index], golden[index]) << "output index=" << index;
    }
    for (size_t index = 0; index < GUARD_BYTES; ++index) {
        ASSERT_EQ(outputGm.get()[index], SENTINEL) << "prefix guard index=" << index;
    }
    for (size_t index = GUARD_BYTES + input.size(); index < outputBytes; ++index) {
        ASSERT_EQ(outputGm.get()[index], SENTINEL) << "suffix guard index=" << index;
    }
}

INSTANTIATE_TEST_SUITE_P(NumericCases, HitOrMissKernelGolden, testing::ValuesIn(MakeKernelCases()),
                         [](const testing::TestParamInfo<KernelCase>& info) { return std::string(info.param.name); });
} // namespace
