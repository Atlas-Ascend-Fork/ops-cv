/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "../../flood_fill_cases.h"
// Use the existing diagnostic fields to assert that fixtures actually enter
// the requested branches. No capacity override and no mock kernel are used.
#define FLOOD_FILL_PROFILE 1
#include "../../../op_kernel/flood_fill.h"

using namespace FloodFillTest;

template <typename T, bool MASK_INPUT>
void FloodFillUnderTest(GM_ADDR x, GM_ADDR maskIn, GM_ADDR y, GM_ADDR maskOut, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    NsFloodFill::KernelFloodFill<T, MASK_INPUT> op;
    op.Init(x, maskIn, y, maskOut, workspace, reinterpret_cast<FloodFillTilingData*>(tiling), &pipe);
    op.Process();
}

namespace {
struct GmBuffer {
    uint8_t* data;
    explicit GmBuffer(size_t bytes) : data(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)))
    {
        std::memset(data, 0xa5, bytes);
    }
    ~GmBuffer() { AscendC::GmFree(data); }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
};

enum class Path { OVERFLOW, ROW_SEEDS, COMPARISON_WINDOWS, MASK };

template <typename T, bool MASK_INPUT = false>
void RunKernelCase(Case c, uint32_t blocks, Path path)
{
    c.maskInput = MASK_INPUT;
    const auto f = MakeFixture<T>(c);
    const size_t bytes = f.input.size() * sizeof(T);
    const auto align32 = [](size_t n) { return (n + 31U) / 32U * 32U; };
    const size_t capacity = 2U * std::max(c.height, c.width) + 256U;
    const size_t comparison = !std::is_same<T, int32_t>::value && f.input.size() >= 262144U && c.width >= 512U ?
                                  2U * (((f.input.size() + 7U) / 8U + 63U) / 64U * 64U + 64U) :
                                  0U;
    const size_t workspaceBytes = FLOOD_FILL_CONTROL_BYTES + comparison + align32(f.input.size()) + align32(c.height) +
                                  align32(4U * c.height) + align32(8U * capacity);
    GmBuffer input(bytes), output(bytes), maskIn(f.mask.size()), maskOut(f.mask.size());
    GmBuffer workspace(workspaceBytes), tiling(sizeof(FloodFillTilingData));
    std::memcpy(input.data, f.input.data(), bytes);
    std::memcpy(maskIn.data, f.mask.data(), f.mask.size());
    FloodFillTilingData t{};
    t.height = c.height;
    t.width = c.width;
    t.seedRow = c.SeedRow();
    t.seedCol = c.SeedCol();
    t.newValueI = static_cast<int32_t>(c.newValue);
    t.newValueF = c.newValue;
    t.loDiffI = static_cast<int32_t>(c.Lo());
    t.upDiffI = static_cast<int32_t>(c.Up());
    t.loDiffF = c.Lo();
    t.upDiffF = c.Up();
    t.connectivity = c.connectivity;
    t.diffMode = c.fixed ? FLOODFILL_DIFF_MODE_FIXED_RANGE : FLOODFILL_DIFF_MODE_NEIGHBOR;
    std::memcpy(tiling.data, &t, sizeof(t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kernel = FloodFillUnderTest<T, MASK_INPUT>;
    ICPU_RUN_KF(kernel, blocks, input.data, MASK_INPUT ? maskIn.data : nullptr, output.data,
                c.maskOutput ? maskOut.data : nullptr, workspace.data, tiling.data);
    ASSERT_EQ(std::memcmp(output.data, f.expected.data(), bytes), 0) << "image differs from pixel BFS";
    ASSERT_EQ(std::memcmp(input.data, f.input.data(), bytes), 0) << "input image modified";
    ASSERT_EQ(std::memcmp(maskIn.data, f.mask.data(), f.mask.size()), 0) << "input mask modified";
    if (c.maskOutput)
        ASSERT_EQ(std::memcmp(maskOut.data, f.expectedMask.data(), f.mask.size()), 0) << "mask labels/border differ";
    const auto* profile = reinterpret_cast<const uint64_t*>(workspace.data);
    ASSERT_EQ(profile[0], 0x464650524f463131ULL);
    std::printf("COVERAGE: blocks=%u spans=%llu overflow=%llu comparison_passes=%llu row_seed_rows=%llu\n", blocks,
                static_cast<unsigned long long>(profile[6]), static_cast<unsigned long long>(profile[7]),
                static_cast<unsigned long long>(profile[29]), static_cast<unsigned long long>(profile[34]));
    EXPECT_EQ(profile[8], 0U) << "fixture unexpectedly used whole-image shortcut";
    if (path == Path::OVERFLOW) {
        EXPECT_EQ(profile[7], 1U) << "natural span capacity was not exceeded";
        EXPECT_EQ(profile[6], capacity);
    } else if (path == Path::ROW_SEEDS) {
        EXPECT_GT(profile[29], 0U) << "no comparison/pause/resume pass";
        EXPECT_EQ(profile[33], 1U) << "row seeds not attempted";
        EXPECT_GT(profile[34], 0U) << "row seeds not adopted";
    } else if (path == Path::COMPARISON_WINDOWS) {
        EXPECT_GT(profile[29], 1U) << "expected multiple comparison windows and resumes";
        EXPECT_EQ(profile[33], 0U) << "directed edges must not use undirected row seeds";
    }
}
} // namespace

TEST(FloodFillKernel, NaturalOverflowUint8MaskOutput)
{
    RunKernelCase<uint8_t>({Pattern::COMB, 65, 65}, 1, Path::OVERFLOW);
}

TEST(FloodFillKernel, NaturalOverflowInt32WorkspaceVisited)
{
    RunKernelCase<int32_t>({Pattern::COMB, 65, 65, 8, false, false, false}, 1, Path::OVERFLOW);
}

TEST(FloodFillKernel, NaturalOverflowFloatInputMask)
{
    RunKernelCase<float, true>({Pattern::COMB, 65, 65, 8, false}, 1, Path::OVERFLOW);
}

TEST(FloodFillKernel, ParallelComparisonAndRowSeeds)
{
    RunKernelCase<uint8_t>({Pattern::BAND, 512, 513}, 4, Path::ROW_SEEDS);
}

TEST(FloodFillKernel, ParallelDirectedComparisonWindows)
{
    RunKernelCase<float>({Pattern::DIRECTED_BAND, 512, 513, 4, false, false, false}, 4, Path::COMPARISON_WINDOWS);
}

TEST(FloodFillKernel, ParallelOverflowInputAndOutputMask)
{
    RunKernelCase<uint8_t, true>({Pattern::COMB, 512, 513}, 4, Path::OVERFLOW);
}

TEST(FloodFillKernel, ParallelMaskBarrier)
{
    RunKernelCase<int32_t, true>({Pattern::MASK_BARRIER, 512, 513}, 4, Path::MASK);
}

TEST(FloodFillKernel, ParallelBlockedSeed)
{
    RunKernelCase<float, true>({Pattern::BLOCKED_SEED, 512, 513, 8, false}, 4, Path::MASK);
}

TEST(FloodFillKernel, RestoredOverflowComb)
{
    RunKernelCase<uint8_t>({Pattern::LEGACY_COMB, 97, 129, 4, false}, 1, Path::OVERFLOW);
}

TEST(FloodFillKernel, RestoredOverflowDisconnectedIsland)
{
    RunKernelCase<int32_t>({Pattern::LEGACY_ISLANDS, 97, 129, 4, true, false, false, -42.0F}, 1, Path::OVERFLOW);
}
