/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bilateral_filter3_d_kernel_golden.cpp
 * \brief CPU-twin kernel UT for BilateralFilter3D.
 *
 * The reference is an independent double-precision implementation of the spec
 * (valid-neighbourhood truncation with renormalisation), so the test does not
 * reuse any of the kernel's own tiling arithmetic. Shapes are deliberately tiny
 * but chosen to exercise the structural edge cases of the vector kernel:
 *   - W a multiple of 8 vs. a ragged tail (the per-tile cnt mask);
 *   - W < 8 (the headCnt = min(W, 8) path);
 *   - the a == 0 tile with negative dx (the scalar nbHead/nbFix assembly);
 *   - multi-core grid-stride row assignment;
 *   - several leading volumes (the row % (D*H) decomposition);
 *   - sigma reciprocals clamped at 1e30 (weights collapse onto the centre voxel).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "bilateral_filter3_d_tiling_def.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void bilateral_filter3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes)))
    {
        if (ptr_ != nullptr) {
            std::memset(ptr_, 0, bytes);
        }
    }
    ~GmBuffer()
    {
        if (ptr_ != nullptr) {
            AscendC::GmFree(ptr_);
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    uint8_t* Get() const { return ptr_; }

private:
    uint8_t* ptr_ = nullptr;
};

template <typename T>
float ToFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
T FromFloat(float value)
{
    return static_cast<T>(value);
}

template <typename T>
float Tolerance()
{
    if constexpr (std::is_same_v<T, float>) {
        return 2.0e-5F;
    }
    return 3.0e-3F; // fp16 output quantisation dominates
}

// Mirrors InvTwoSigmaSq in bilateral_filter3_d_tiling.cpp: 0.5/sigma^2 saturated at
// 1e30 so the value stays finite in fp32 while still underflowing exp() to zero.
constexpr float MAX_INV2_SIGMA = 1e30F;

float InvTwoSigmaSq(double sigma)
{
    const double inv2 = 0.5 / (sigma * sigma);
    if (inv2 > static_cast<double>(MAX_INV2_SIGMA)) {
        return MAX_INV2_SIGMA;
    }
    return static_cast<float>(inv2);
}

// Deterministic LCG, identical to the device E2E harness.
uint32_t LcgNext(uint32_t& state)
{
    state = state * 1664525U + 1013904223U;
    return state;
}

float LcgUniform(uint32_t& state) { return static_cast<float>(LcgNext(state) >> 8) * (1.0F / 16777216.0F); }

// Double-precision spec reference. inv2ss/inv2sc are the already-clamped fp32
// values the kernel actually receives, so the only intended divergence from the
// kernel is arithmetic precision.
void GoldenBilateral3D(const std::vector<float>& x, uint64_t nVolume, uint32_t d, uint32_t h, uint32_t w,
                       int32_t kernelSize, float inv2ss, float inv2sc, std::vector<double>& y)
{
    const int32_t r = kernelSize / 2;
    const int64_t dI = static_cast<int64_t>(d);
    const int64_t hI = static_cast<int64_t>(h);
    const int64_t wI = static_cast<int64_t>(w);
    const int64_t volSize = dI * hI * wI;
    y.assign(x.size(), 0.0);
    for (uint64_t v = 0; v < nVolume; ++v) {
        const int64_t base = static_cast<int64_t>(v) * volSize;
        for (int64_t z = 0; z < dI; ++z) {
            for (int64_t row = 0; row < hI; ++row) {
                for (int64_t col = 0; col < wI; ++col) {
                    const int64_t lin = base + z * hI * wI + row * wI + col;
                    const double c = x[static_cast<size_t>(lin)];
                    double num = 0.0;
                    double den = 0.0;
                    for (int32_t dz = -r; dz <= r; ++dz) {
                        const int64_t zz = z + dz;
                        if (zz < 0 || zz >= dI) {
                            continue;
                        }
                        for (int32_t dy = -r; dy <= r; ++dy) {
                            const int64_t yy = row + dy;
                            if (yy < 0 || yy >= hI) {
                                continue;
                            }
                            const int64_t rowBase = base + zz * hI * wI + yy * wI;
                            for (int32_t dx = -r; dx <= r; ++dx) {
                                const int64_t xx = col + dx;
                                if (xx < 0 || xx >= wI) {
                                    continue;
                                }
                                const double nb = x[static_cast<size_t>(rowBase + xx)];
                                const double dc = nb - c;
                                const double offSq = static_cast<double>(dz * dz + dy * dy + dx * dx);
                                const double wgt = std::exp(
                                    -(offSq * static_cast<double>(inv2ss) + dc * dc * static_cast<double>(inv2sc)));
                                num += wgt * nb;
                                den += wgt;
                            }
                        }
                    }
                    y[static_cast<size_t>(lin)] = num / den;
                }
            }
        }
    }
}

// Mirrors the host chunk-width derivation closely enough for the UT: the host also
// caps the chunk by the UB budget, which never binds at these tiny widths.
uint32_t DeriveChunkLen(uint32_t w)
{
    const uint32_t padW = (w + 7U) / 8U * 8U;
    return padW < optiling::BF3D_CHUNK_MAX ? padW : optiling::BF3D_CHUNK_MAX;
}

// chunkLen = 0 means "derive it like the host would"; a non-zero value forces a
// narrower chunk so that a tiny shape still exercises the multi-chunk control flow.
template <typename T>
void RunCase(uint64_t tilingKey, uint64_t nVolume, uint32_t d, uint32_t h, uint32_t w, uint32_t kernelSize,
             double sigmaSpace, double sigmaColor, uint32_t usedCoreNum, uint32_t chunkLen = 0U, uint32_t seed = 777U)
{
    const uint64_t totalRows = nVolume * static_cast<uint64_t>(d) * static_cast<uint64_t>(h);
    const uint64_t total = totalRows * static_cast<uint64_t>(w);
    ASSERT_GT(total, 0U);
    ASSERT_LE(usedCoreNum, totalRows) << "usedCoreNum must not exceed totalRows";
    const uint32_t effChunkLen = (chunkLen == 0U) ? DeriveChunkLen(w) : chunkLen;
    ASSERT_GT(effChunkLen, 0U);
    ASSERT_EQ(effChunkLen % 8U, 0U) << "chunkLen must be a multiple of 8 (32-byte alignment)";

    // Input spans [-1, 2) so that colour differences are O(1) relative to sigma.
    std::vector<T> input(total);
    std::vector<float> rounded(total);
    uint32_t state = seed;
    for (uint64_t i = 0; i < total; ++i) {
        input[i] = FromFloat<T>(-1.0F + 3.0F * LcgUniform(state));
        rounded[i] = ToFloat(input[i]);
    }

    const float inv2ss = InvTwoSigmaSq(sigmaSpace);
    const float inv2sc = InvTwoSigmaSq(sigmaColor);
    std::vector<double> expected;
    GoldenBilateral3D(rounded, nVolume, d, h, w, static_cast<int32_t>(kernelSize), inv2ss, inv2sc, expected);

    GmBuffer inputGm(total * sizeof(T));
    GmBuffer outputGm(total * sizeof(T));
    GmBuffer workspaceGm(1024);
    GmBuffer tilingGm(sizeof(optiling::BilateralFilter3DTilingData));
    ASSERT_NE(inputGm.Get(), nullptr);
    ASSERT_NE(outputGm.Get(), nullptr);
    ASSERT_NE(workspaceGm.Get(), nullptr);
    ASSERT_NE(tilingGm.Get(), nullptr);
    std::memcpy(inputGm.Get(), input.data(), total * sizeof(T));

    auto* tiling = reinterpret_cast<optiling::BilateralFilter3DTilingData*>(tilingGm.Get());
    std::memset(tiling, 0, sizeof(*tiling));
    tiling->totalRows = totalRows;
    tiling->d = d;
    tiling->h = h;
    tiling->w = w;
    tiling->kernelSize = kernelSize;
    tiling->usedCoreNum = usedCoreNum;
    tiling->chunkLen = effChunkLen;
    tiling->inv2ss = inv2ss;
    tiling->inv2sc = inv2sc;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(bilateral_filter3_d, usedCoreNum, inputGm.Get(), outputGm.Get(), workspaceGm.Get(), tilingGm.Get());

    const T* actual = reinterpret_cast<const T*>(outputGm.Get());
    const float tol = Tolerance<T>();
    for (uint64_t i = 0; i < total; ++i) {
        const float want = ToFloat(FromFloat<T>(static_cast<float>(expected[i])));
        const float got = ToFloat(actual[i]);
        ASSERT_FALSE(std::isnan(got)) << "NaN written at index " << i;
        // Mixed absolute/relative bound: the kernel accumulates in fp32 while the
        // reference accumulates in double, so error grows with the magnitude.
        EXPECT_NEAR(got, want, tol * (1.0F + std::fabs(want))) << "mismatch at index " << i;
    }
}
} // namespace

// ---------------------------------------------------------------------------
// Width / tiling-geometry coverage.
// ---------------------------------------------------------------------------

// W = 8 is exactly one aligned tile: no ragged tail, every dx stays in range for
// the interior columns.
TEST(BilateralFilter3DKernel, Fp32AlignedWidthK3)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 5U, 6U, 8U, 3U, 1.0, 1.0, 1U);
}

// W = 11 forces a second, partially valid tile (cnt = 3) and therefore exercises
// the prefix mask rows of the mask table.
TEST(BilateralFilter3DKernel, Fp32RaggedTailK5)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 4U, 5U, 11U, 5U, 1.0, 1.0, 1U);
}

TEST(BilateralFilter3DKernel, Fp16RaggedTailK5)
{
    RunCase<half>(optiling::BF3D_TILING_KEY_FP16, 1U, 4U, 5U, 11U, 5U, 0.8, 0.5, 1U);
}

// W = 5 < BF3D_TILE: headCnt clamps to W and the single tile is only partly valid.
TEST(BilateralFilter3DKernel, Fp32NarrowWidthK9)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 3U, 3U, 5U, 9U, 1.0, 1.0, 1U);
}

// radius 4 exceeds D and H, so zLow/zHigh and yLow/yHigh clamp on both sides for
// every row -- the renormalisation path.
TEST(BilateralFilter3DKernel, Fp32RadiusExceedsExtentK9)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 2U, 3U, 16U, 9U, 1.5, 2.0, 1U);
}

// W = 24 gives three tiles: a == 0 (negative-dx scalar fixup), a == 8 and a == 16
// (plain shifted GM loads).
TEST(BilateralFilter3DKernel, Fp32MultiTileK7)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 4U, 4U, 24U, 7U, 1.0, 1.0, 1U);
}

// ---------------------------------------------------------------------------
// Chunk decomposition. These force a chunk narrower than the row so that a tiny
// shape still produces a leading chunk, an interior chunk and a trailing chunk.
// The interior chunk is the one that takes the no-mask fast path, and getting its
// guard wrong (cb >= r && cb + cl + r <= W) fails silently, so it needs coverage
// at several radii.
// ---------------------------------------------------------------------------

// W=48, k=5 (r=2), chunk=16 -> cb=0 masked, cb=16 unmasked, cb=32 masked.
TEST(BilateralFilter3DKernel, Fp32MultiChunkUnmaskedInteriorK5)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 2U, 3U, 48U, 5U, 1.0, 1.0, 1U, 16U);
}

// W=20, k=3 (r=1), chunk=8 -> trailing chunk is ragged (cl=4 < chunkLen).
TEST(BilateralFilter3DKernel, Fp32MultiChunkRaggedLastChunkK3)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 3U, 3U, 20U, 3U, 1.0, 1.0, 1U, 8U);
}

// k=9 (r=4) with chunk=8: the radius is half the chunk, the tightest case for the
// no-mask guard (cb=8 and cb=16 unmasked, cb=24 masked at W=32).
TEST(BilateralFilter3DKernel, Fp32NarrowChunkK9)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 2U, 3U, 32U, 9U, 1.0, 1.0, 1U, 8U);
}

// fp16 path through the same control flow, plus multi-core and several volumes so
// the chunk loop and the row/volume decomposition are exercised together.
TEST(BilateralFilter3DKernel, Fp16MultiChunkMultiCoreMultiVolumeK7)
{
    RunCase<half>(optiling::BF3D_TILING_KEY_FP16, 2U, 2U, 3U, 40U, 7U, 1.0, 1.0, 3U, 16U);
}

// ---------------------------------------------------------------------------
// Core and volume decomposition.
// ---------------------------------------------------------------------------

// grid-stride over 24 rows on 3 cores: each core must touch a disjoint row set.
TEST(BilateralFilter3DKernel, Fp32MultiCoreGridStride)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 4U, 6U, 16U, 5U, 1.0, 1.0, 3U);
}

// Two leading volumes: row % (D*H) must keep dz from crossing the volume boundary.
TEST(BilateralFilter3DKernel, Fp32MultiVolumeBoundary)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 2U, 3U, 4U, 9U, 5U, 1.0, 1.0, 2U);
}

TEST(BilateralFilter3DKernel, Fp16MultiVolumeMultiCore)
{
    RunCase<half>(optiling::BF3D_TILING_KEY_FP16, 3U, 3U, 4U, 12U, 5U, 1.2, 2.0, 4U);
}

// ---------------------------------------------------------------------------
// Sigma extremes.
// ---------------------------------------------------------------------------

// inv2sc saturates at 1e30: every neighbour with a non-zero colour difference gets
// weight 0, so the output collapses onto the centre voxel.
TEST(BilateralFilter3DKernel, Fp32SigmaColorClampedToCentre)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 3U, 4U, 8U, 5U, 1.0, 1e-20, 1U);
}

// inv2ss saturates: only the zero offset survives, again leaving the centre voxel.
TEST(BilateralFilter3DKernel, Fp16SigmaSpaceClampedToCentre)
{
    RunCase<half>(optiling::BF3D_TILING_KEY_FP16, 1U, 3U, 4U, 9U, 5U, 1e-20, 0.5, 1U);
}

// Huge sigma_color makes the colour term vanish, reducing the filter to a
// spatial-only weighted box over the valid neighbourhood.
TEST(BilateralFilter3DKernel, Fp32SigmaColorHuge)
{
    RunCase<float>(optiling::BF3D_TILING_KEY_FP32, 1U, 4U, 4U, 11U, 5U, 1.0, 1e9, 1U);
}
