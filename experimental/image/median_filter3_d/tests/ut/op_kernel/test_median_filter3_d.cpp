/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>
#include "tikicpulib.h"
#include "../../../op_kernel/median_filter3_d.h"
namespace {
template <typename T>
__global__ __aicore__ void MedianUt(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    const auto data = *reinterpret_cast<const MedianFilter3DTilingData*>(tiling);
    AscendC::TPipe pipe;
    NsMedianFilter3D::KernelMedianFilter3D<T> op;
    op.Init(x, y, &data, &pipe);
    op.Process();
}
int Ref(int p, int n)
{
    while (p < 0 || p >= n) {
        p = p < 0 ? -p - 1 : 2 * n - p - 1;
    }
    return p;
}
template <typename T>
void RunGoldenCase(uint32_t d, uint32_t h, uint32_t w, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t mode,
                   uint32_t cores = 1)
{
    uint32_t n = 2 * d * h * w;
    std::vector<T> in(n), expected(n);
    for (uint32_t i = 0; i < n; ++i) {
        in[i] = static_cast<T>((i * 17 + 3) % 101);
    }
    for (uint32_t i = 0; i < n; ++i) {
        int vol = i / (d * h * w), z = (i / (h * w)) % d, y = (i / w) % h, x = i % w;
        std::vector<float> values;
        for (int a = 0; a < kd; ++a)
            for (int b = 0; b < kh; ++b)
                for (int c = 0; c < kw; ++c) {
                    int zz = z + a - kd / 2, yy = y + b - kh / 2, xx = x + c - kw / 2;
                    bool valid = zz >= 0 && zz < d && yy >= 0 && yy < h && xx >= 0 && xx < w;
                    if (!mode) {
                        zz = Ref(zz, d);
                        yy = Ref(yy, h);
                        xx = Ref(xx, w);
                    }
                    values.push_back(valid || !mode ? static_cast<float>(in[((vol * d + zz) * h + yy) * w + xx]) :
                                                      7.5f);
                }
        std::sort(values.begin(), values.end());
        expected[i] = static_cast<T>(values[values.size() / 2]);
    }
    MedianFilter3DTilingData t{};
    t.rows = 2 * d * h;
    t.d = d;
    t.h = h;
    t.w = w;
    t.kd = kd;
    t.kh = kh;
    t.kw = kw;
    t.count = kd * kh * kw;
    t.padded = 1;
    while (t.padded < t.count)
        t.padded *= 2;
    t.tile = 32;
    t.tilesPerRow = (w + 31) / 32;
    t.tasks = t.rows * t.tilesPerRow;
    t.cores = cores;
    t.mode = mode;
    t.cval = 7.5f;
    using GmBuffer = std::unique_ptr<uint8_t[], decltype(&AscendC::GmFree)>;
    GmBuffer gx(static_cast<uint8_t*>(AscendC::GmAlloc(n * sizeof(T) + 64)), AscendC::GmFree);
    ASSERT_NE(gx.get(), nullptr);
    GmBuffer gy(static_cast<uint8_t*>(AscendC::GmAlloc(n * sizeof(T) + 64)), AscendC::GmFree);
    ASSERT_NE(gy.get(), nullptr);
    GmBuffer gt(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(t))), AscendC::GmFree);
    ASSERT_NE(gt.get(), nullptr);
    std::memcpy(gx.get(), in.data(), n * sizeof(T));
    std::memset(gy.get(), 0xa5, n * sizeof(T) + 64);
    std::memcpy(gt.get(), &t, sizeof(t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(MedianUt<T>, cores, gx.get(), gy.get(), gt.get());
    const T* got = reinterpret_cast<const T*>(gy.get());
    for (uint32_t i = 0; i < n; ++i)
        EXPECT_EQ(static_cast<float>(got[i]), static_cast<float>(expected[i])) << i;
    EXPECT_EQ(std::memcmp(gx.get(), in.data(), n * sizeof(T)), 0);
    for (uint32_t i = 0; i < 64; ++i)
        EXPECT_EQ(gy[n * sizeof(T) + i], 0xa5);
}
} // namespace
TEST(MedianFilter3DKernel, Uint8ReflectEven) { RunGoldenCase<uint8_t>(1, 2, 5, 2, 4, 2, 0); }
TEST(MedianFilter3DKernel, FloatConstantOdd) { RunGoldenCase<float>(2, 3, 5, 3, 3, 3, 1); }
TEST(MedianFilter3DKernel, HalfReflectLargerWindow) { RunGoldenCase<half>(1, 2, 3, 3, 3, 5, 0); }
TEST(MedianFilter3DKernel, FloatInteriorTailMulticore) { RunGoldenCase<float>(1, 1, 35, 1, 1, 3, 0, 2); }
