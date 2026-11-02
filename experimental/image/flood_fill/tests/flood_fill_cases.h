/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOOD_FILL_TEST_CASES_H_
#define FLOOD_FILL_TEST_CASES_H_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace FloodFillTest {
enum class Pattern { COMB, BAND, DIRECTED_BAND, MASK_BARRIER, BLOCKED_SEED, LEGACY_COMB, LEGACY_ISLANDS };

struct Case {
    Pattern pattern;
    uint32_t height;
    uint32_t width;
    uint32_t connectivity = 4;
    bool fixed = true;
    bool maskInput = false;
    bool maskOutput = true;
    float newValue = 200.0F;
    const char* name = "matrix";

    bool Legacy() const { return pattern == Pattern::LEGACY_COMB || pattern == Pattern::LEGACY_ISLANDS; }
    uint32_t SeedRow() const { return Legacy() ? height / 2U : 0U; }
    uint32_t SeedCol() const { return Legacy() ? width / 2U : (pattern == Pattern::COMB ? 0U : width / 3U); }
    float Lo() const { return Legacy() || pattern == Pattern::DIRECTED_BAND ? 0.0F : 1.0F; }
    float Up() const { return Legacy() ? 0.0F : 1.0F; }
};

template <typename T>
struct Fixture {
    std::vector<T> input;
    std::vector<uint8_t> mask;
    std::vector<T> expected;
    std::vector<uint8_t> expectedMask;
    size_t filled = 0;
};

// The oracle is a pixel queue over ORIGINAL values, independent of the
// kernel's spans, deferred rows, comparison planes and overflow recovery.
template <typename T>
Fixture<T> MakeFixture(const Case& c)
{
    Fixture<T> f;
    const size_t pixels = static_cast<size_t>(c.height) * c.width;
    const size_t pitch = c.width + 2U;
    f.input.resize(pixels);
    f.mask.assign(static_cast<size_t>(c.height + 2U) * pitch, 0U);
    for (uint32_t r = 0; r < c.height; ++r) {
        for (uint32_t col = 0; col < c.width; ++col) {
            const bool eligible = c.pattern == Pattern::COMB ? (r % 2U == 0U || col % 2U == 0U) : col + 13U < c.width;
            f.input[static_cast<size_t>(r) * c.width + col] = static_cast<T>(eligible ? (col < c.width / 2U ? 10 : 11) :
                                                                                        100);
            // Restore the original 97x129, spacing=4 Comb/CombIslands
            // fixtures removed by 1a3ed752, including the disconnected tail.
            if (c.Legacy()) {
                const bool islandTail = c.pattern == Pattern::LEGACY_ISLANDS && r >= c.height - 4U;
                const bool originalEligible = islandTail ? r > c.height - 4U : (r % 2U == 0U || col % 4U == 0U);
                f.input[static_cast<size_t>(r) * c.width + col] = static_cast<T>(originalEligible ? 100 : 0);
            }
            if (c.maskInput && c.pattern == Pattern::MASK_BARRIER && r == c.height / 2U) {
                f.mask[(r + 1U) * pitch + col + 1U] = col % 2U == 0U ? 7U : 9U;
            }
        }
    }
    if (c.maskInput) {
        // Input borders are deliberately not all one; they are ignored and
        // output borders must be normalized without changing this input.
        f.mask[0] = 23U;
        f.mask.back() = 31U;
        if (c.pattern == Pattern::BLOCKED_SEED) {
            f.mask[pitch + c.SeedCol() + 1U] = 9U;
        } else {
            f.mask[(c.height - 1U) * pitch + c.width] = 7U;
        }
    }
    f.expected = f.input;
    f.expectedMask = f.mask;
    std::vector<uint8_t> visited(pixels, 0U);
    std::vector<size_t> queue;
    const size_t seed = static_cast<size_t>(c.SeedRow()) * c.width + c.SeedCol();
    const auto blocked = [&](uint32_t r, uint32_t col) {
        return c.maskInput && f.mask[(r + 1U) * pitch + col + 1U] != 0U;
    };
    if (!blocked(c.SeedRow(), c.SeedCol())) {
        queue.push_back(seed);
        visited[seed] = 1U;
    }
    for (size_t head = 0; head < queue.size(); ++head) {
        const size_t p = queue[head];
        const int row = static_cast<int>(p / c.width);
        const int col = static_cast<int>(p % c.width);
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if ((dr == 0 && dc == 0) || (c.connectivity == 4U && dr != 0 && dc != 0))
                    continue;
                const int nr = row + dr;
                const int nc = col + dc;
                if (nr < 0 || nr >= static_cast<int>(c.height) || nc < 0 || nc >= static_cast<int>(c.width))
                    continue;
                const size_t n = static_cast<size_t>(nr) * c.width + nc;
                if (visited[n] || blocked(nr, nc))
                    continue;
                // Pixel values and tolerances are small exact integers in
                // every dtype, so this float subtraction is also exact.
                const float delta = static_cast<float>(f.input[n]) - static_cast<float>(f.input[c.fixed ? seed : p]);
                if (delta >= -c.Lo() && delta <= c.Up()) {
                    visited[n] = 1U;
                    queue.push_back(n);
                }
            }
        }
    }
    f.filled = queue.size();
    for (size_t p : queue) {
        f.expected[p] = static_cast<T>(c.newValue);
        f.expectedMask[(p / c.width + 1U) * pitch + p % c.width + 1U] = 1U;
    }
    for (size_t p = 0; p < f.expectedMask.size(); ++p) {
        if (p < pitch || p >= f.expectedMask.size() - pitch || p % pitch == 0U || p % pitch + 1U == pitch)
            f.expectedMask[p] = 1U;
    }
    return f;
}

inline std::vector<Case> RegressionCases()
{
    std::vector<Case> cases;
    for (uint32_t connectivity : {4U, 8U}) {
        for (bool fixed : {false, true}) {
            for (bool maskOutput : {false, true}) {
                for (bool maskInput : {false, true}) {
                    cases.push_back({Pattern::COMB, 65, 65, connectivity, fixed, maskInput, maskOutput});
                    cases.push_back({Pattern::COMB, 512, 513, connectivity, fixed, maskInput, maskOutput});
                }
                cases.push_back({Pattern::BAND, 512, 513, connectivity, fixed, false, maskOutput});
                cases.push_back({Pattern::DIRECTED_BAND, 512, 513, connectivity, fixed, false, maskOutput});
                cases.push_back({Pattern::MASK_BARRIER, 512, 513, connectivity, fixed, true, maskOutput});
                cases.push_back({Pattern::BLOCKED_SEED, 512, 513, connectivity, fixed, true, maskOutput});
            }
        }
    }
    return cases;
}
} // namespace FloodFillTest

#endif // FLOOD_FILL_TEST_CASES_H_
