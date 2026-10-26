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
 * \file grid_sampler2_d_simt.h
 * \brief SIMT kernel implementation for grid_sampler2_d operator
 *
 * Performance optimizations applied (op_perf_skill Stage 1):
 *   R002: Constant divisions (WOut, HOut) replaced by Simt::UintDiv (magic + shift)
 *   R003: Index width templated (int32_t / int64_t) — 32-bit path when max address <= INT32_MAX
 *   R006: __launch_bounds__ templated on index width (1024 for int32, 512 for int64)
 *
 * TTK round 36 precision fix (bicubic + float32):
 *   Root cause: PyTorch's vectorized CPU kernel uses FMA (fused multiply-add)
 *   instructions for two critical computations:
 *   1. UnnormalizeNoClip: (coord+1)*scaling - 0.5 compiled as FMA(scaling, coord+1, -0.5)
 *      — single rounding vs our two separate roundings (mul then sub)
 *   2. BicubicWeightedSum: c0*v0 + c1*v1 + c2*v2 + c3*v3 compiled as FMA chain
 *      — each c*v+r step has single rounding vs our double rounding
 *   Fix: use fmaf() to match PyTorch's FMA behavior exactly.
 *   TTK round 37: attempted to also use fmaf() chains in CubicConvolution1/2
 *   to match PyTorch's x86 FMA disassembly, but NPU fmaf differs from x86
 *   FMA in rounding behavior, causing regression (95.3%->92.4%, +10 fails).
 *   Reverted CubicConvolution1/2 to round 36's volatile non-FMA impl.
 *   UnnormalizeNoClip and BicubicWeightedSum keep fmaf (validated in v36).
 *   ReflectCoordinates keeps non-FMA — FMA has zero effect there because
 *   doubleFlips*twiceSpan is typically exact (integer * power-of-2).
 */

#ifndef GRID_SAMPLER2_D_SIMT_H_
#define GRID_SAMPLER2_D_SIMT_H_

#include "grid_sampler2_d_tiling_data.h"
#include "grid_sampler2_d_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"

// Disable FMA (fused multiply-add) contraction for all float arithmetic.
//
// TTK round 38 precision fix (bicubic + float32):
//   Previous rounds used function-level `#pragma STDC FP_CONTRACT OFF` around
//   CubicConvolution1/2, but the NPU AI Core compiler (based on clang) may not
//   recognize the STDC variant — only `#pragma clang fp contract(off)` is
//   reliably honored (verified by image_projective_transform, which uses the
//   file-level clang pragma and achieves 100% bit-exact match with TF golden).
//   This means the FMA prohibition in CubicConvolution may have NEVER taken
//   effect in previous rounds, and the volatile float was the only real
//   safeguard.
//
//   This file-level pragma disables compiler-driven FMA contraction globally.
//   Strategy per computation:
//     - CubicConvolution1/2: volatile float (double insurance with pragma)
//       — separate mul/add with double rounding, matching x86 non-FMA path
//     - ReflectCoordinates: volatile float (same rationale)
//     - UnnormalizeNoClip: explicit fmaf() — library call, NOT affected by
//       this pragma; matches PyTorch's x86 FMA compilation
//     - BicubicWeightedSum: explicit fmaf() chain — same rationale
//
//   Explicit fmaf() calls are library function calls and are NOT affected by
//   `#pragma clang fp contract(off)` — they still produce correctly-rounded
//   FMA results. This allows us to selectively use FMA where PyTorch uses it
//   (UnnormalizeNoClip, BicubicWeightedSum) while prohibiting it elsewhere.
#pragma clang fp contract(off)

namespace NsGridSampler2D {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <typename INDEX_SIZE_T>
constexpr uint32_t THREAD_NUM_T = (sizeof(INDEX_SIZE_T) == 4) ? 1024 : 512;

template <typename T>
__simt_callee__ inline float ReadAsFloat(T val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
__simt_callee__ inline T CastToOutput(float val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else {
        return static_cast<T>(val);
    }
}

// ===== Coordinate transform helpers =====

__simt_callee__ inline float Unnormalize(float coord, int32_t size, int32_t alignCorners)
{
    if (alignCorners != 0) {
        return ((coord + 1.0f) / 2.0f) * static_cast<float>(size - 1);
    }
    float scalingFactor = static_cast<float>(size) / 2.0f;
    float result = (coord + 1.0f) * scalingFactor - 0.5f;
    if (isinf(result)) {
        return coord > 0.0f ? 3.4e38f : -3.4e38f;
    }
    return result;
}

// Unnormalize matching PyTorch's vectorized CPU kernel computation order:
//   (coord + 1) * (size / 2) - 0.5
// TTK round 36: use fmaf to match PyTorch's FMA compilation of this expression.
// PyTorch x86 compiler fuses (coord+1)*scaling - 0.5 into FMA(scaling, coord+1, -0.5)
// with single rounding. Our previous code did separate mul+sub (double rounding),
// causing ~1 ULP difference that amplifies through bicubic interpolation.
__simt_callee__ inline float UnnormalizeNoClip(float coord, int32_t size, int32_t alignCorners)
{
    if (alignCorners != 0) {
        return ((coord + 1.0f) / 2.0f) * static_cast<float>(size - 1);
    }
    float scalingFactor = static_cast<float>(size) / 2.0f;
    return fmaf(scalingFactor, coord + 1.0f, -0.5f);
}

// NaN → 0 matches aicpu grid_sampler_2d.cc and PyTorch clip_coordinates.
// v24 returned maxVal (size-1) which broke nearest+reflection for inf/NaN grid.
__simt_callee__ inline float ClipCoordinates(float coord, int32_t size)
{
    float maxVal = static_cast<float>(size - 1);
    if (isnan(coord)) {
        return 0.0f;
    }
    return fminf(fmaxf(coord, 0.0f), maxVal);
}

// PyTorch vectorized reflect_coordinates algorithm (from GridSamplerKernel.cpp).
//
// This replaces the previous ExactFmodWithParity + parity-based approach.
//
// Root cause of v30 bicubic 0% failures (1e10 grid + reflection):
//   The old ExactFmodWithParity computed the mathematically EXACT fmod
//   (e.g., fmod(3e10, 6) = 4.0), but PyTorch's CPU vectorized kernel uses
//   a DIFFERENT algorithm: naive fmod `a - truncf(a/b)*b` with twice_span
//   (2*size) as the divisor, plus `min(extra, twice_span - extra)` for flip
//   detection. For extreme values like 1e10, the naive fmod suffers
//   catastrophic cancellation (3e10 - 2.5e9*12 = 0 in float32), producing
//   extra=0 instead of the exact 4.0. PyTorch's min() then maps this to
//   index 0, while our exact fmod mapped to index 3 — causing 0% match.
//
// Fix: Replicate PyTorch's EXACT algorithm (twice_span + naive fmod + min flip)
//   so our float32 results match PyTorch CPU golden for ordinary normalized
//   coordinates. This is NOT a bit-identical guarantee for pathological
//   coordinates: for |grid| around 1e8 and beyond, UnnormalizeNoClip's
//   fmaf(scalingFactor, coord + 1, -0.5) and ReflectCoordinates' 2*size fmod
//   folding can diverge from torch's reflection semantics by an O(1) amount.
//
// Key differences from old code:
//   1. Uses twice_span = 2*size (NOT 2*size-1) as fmod divisor
//   2. Uses naive fmod: a - truncf(a/b)*b (NOT exact binary-GCD fmod)
//   3. Uses min(extra, twice_span-extra) for flip (NOT parity tracking)
//   4. No int32 overflow (stays in float, no static_cast<int>)
//
// volatile intermediates prevent FMA contraction, matching x86 float32
// two-rounding behavior (separate multiply then subtract).
__simt_callee__ inline float ReflectCoordinates(float coord, float low, float twiceSpan)
{
    if (twiceSpan == 0.0f) {
        return 0.0f;
    }
    float absIn = fabsf(coord - low);
    float fdoubleFlips = absIn / twiceSpan;
    float doubleFlips = truncf(fdoubleFlips);
    // Prevent FMA: compute product and subtraction as separate operations
    volatile float product = doubleFlips * twiceSpan;
    volatile float extra = absIn - product;
    volatile float complement = twiceSpan - extra;
    float result = fminf(extra, complement) + low;
    return result;
}

__simt_callee__ inline float ComputeCoordinates(float coord, int32_t size, int32_t paddingMode, int32_t alignCorners)
{
    if (paddingMode == 1) {
        coord = ClipCoordinates(coord, size);
    } else if (paddingMode == 2) {
        if (alignCorners != 0) {
            coord = ReflectCoordinates(coord, 0.0f, 2.0f * static_cast<float>(size - 1));
        } else {
            coord = ReflectCoordinates(coord, -0.5f, 2.0f * static_cast<float>(size));
        }
        coord = ClipCoordinates(coord, size);
    }
    return coord;
}

__simt_callee__ inline float ComputeSourceIndex(float coord, int32_t size, int32_t paddingMode, int32_t alignCorners)
{
    coord = UnnormalizeNoClip(coord, size, alignCorners);
    coord = ComputeCoordinates(coord, size, paddingMode, alignCorners);
    return coord;
}

__simt_callee__ inline float SafeDowngradeToIntRange(float x)
{
    if (x >= 2147483648.0f || x < -2147483648.0f || isnan(x) || isinf(x)) {
        return -100.0f;
    }
    return x;
}

// Non-finite grid coordinates (NaN/±Inf) follow the declared PyTorch reference
// semantics, verified against torch.nn.functional.grid_sample on CPU:
//   * bilinear: zeros keeps NaN, border/reflection clip to a finite index;
//   * bicubic : NaN for every padding mode;
//   * nearest : a finite (0) index under zeros, clipped under border/reflection.
// `IsNanOrInf` gates the bilinear/bicubic NaN branch; nearest relies on
// SafeDowngradeToIntRange mapping non-finite to an out-of-range index (-> 0).
__simt_callee__ inline bool IsNanOrInf(float x) { return isnan(x) || isinf(x); }

__simt_callee__ inline int32_t BankerRoundToInt(float x)
{
    float floorVal = floorf(x);
    float frac = x - floorVal;
    if (frac < 0.5f) {
        return static_cast<int32_t>(floorVal);
    }
    if (frac > 0.5f) {
        return static_cast<int32_t>(floorVal) + 1;
    }
    int32_t floorInt = static_cast<int32_t>(floorVal);
    if ((floorInt & 1) == 0) {
        return floorInt;
    }
    return floorInt + 1;
}

template <typename T, typename INDEX_SIZE_T>
__simt_callee__ inline void WriteNanToOutput(__gm__ T* yGm, int32_t n, int32_t C, int32_t HOut, int32_t WOut,
                                             int32_t hOut, int32_t wOut)
{
    INDEX_SIZE_T yBase = static_cast<INDEX_SIZE_T>(n) * C * HOut * WOut + static_cast<INDEX_SIZE_T>(hOut) * WOut + wOut;
    INDEX_SIZE_T yStep = static_cast<INDEX_SIZE_T>(HOut) * WOut;
    float nanVal = 0.0f / 0.0f;
    for (int32_t c = 0; c < C; c++) {
        yGm[yBase] = CastToOutput<T>(nanVal);
        if (c + 1 < C) {
            yBase += yStep;
        }
    }
}

__simt_callee__ inline bool WithinBounds2d(int32_t h, int32_t w, int32_t H, int32_t W)
{
    return h >= 0 && h < H && w >= 0 && w < W;
}

// ===== Bicubic helpers =====

// Bicubic coefficient helpers. PyTorch's get_cubic_upsample_coefficients
// evaluates coeffs[3] = cubic_convolution2(x2 + 1.0, A) with x2 = 1.0 - t.
// The cx3/cy3 arguments below use 2.0f - t (a single rounding of the argument)
// rather than the mathematically equivalent (1.0f - t) + 1.0f: round 34 tried
// PyTorch's two-rounding argument, but round 37/38 reverted it because the NPU
// fmaf rounding differs from x86 FMA and the two-rounding form regressed the
// differential comparison (95.3% -> 92.4%). The current form pairs 2.0f - t
// with the separate mul/add (volatile + file-level `#pragma clang fp
// contract(off)`) evaluation, which is the validated-on-NPU formulation.
// UnnormalizeNoClip and BicubicWeightedSum keep fmaf (validated in round 36).
constexpr float A = -0.75f;

__simt_callee__ inline float CubicConvolution1(float x, float a)
{
    volatile float t0 = (a + 2.0f) * x;
    volatile float t1 = t0 - (a + 3.0f);
    volatile float t2 = t1 * x;
    volatile float t3 = t2 * x;
    return t3 + 1.0f;
}

__simt_callee__ inline float CubicConvolution2(float x, float a)
{
    volatile float t0 = a * x;
    volatile float t1 = t0 - 5.0f * a;
    volatile float t2 = t1 * x;
    volatile float t3 = t2 + 8.0f * a;
    volatile float t4 = t3 * x;
    return t4 - 4.0f * a;
}

__simt_callee__ inline int32_t SafeFloatToInt32(float f)
{
    if (f >= 0.0f && f < 2147483648.0f) {
        return static_cast<int32_t>(f);
    }
    return -1;
}

// ===== Interpolation samplers =====

template <typename T, typename INDEX_SIZE_T>
__simt_callee__ inline void BilinearSample(__gm__ T* xGm, __gm__ T* yGm, int32_t n, int32_t C, int32_t HIn, int32_t WIn,
                                           int32_t HOut, int32_t WOut, int32_t hOut, int32_t wOut, float ix, float iy)
{
    float ixFloor = floorf(ix);
    float iyFloor = floorf(iy);
    int32_t ixNw = static_cast<int32_t>(ixFloor);
    int32_t iyNw = static_cast<int32_t>(iyFloor);
    int32_t ixNe = ixNw + 1;
    int32_t iySw = iyNw + 1;
    float nw = (ixFloor + 1.0f - ix) * (iyFloor + 1.0f - iy);
    float ne = (ix - ixFloor) * (iyFloor + 1.0f - iy);
    float sw = (ixFloor + 1.0f - ix) * (iy - iyFloor);
    float se = (ix - ixFloor) * (iy - iyFloor);
    INDEX_SIZE_T xBase = static_cast<INDEX_SIZE_T>(n) * C * HIn * WIn;
    INDEX_SIZE_T yBase = static_cast<INDEX_SIZE_T>(n) * C * HOut * WOut + static_cast<INDEX_SIZE_T>(hOut) * WOut + wOut;
    INDEX_SIZE_T xStep = static_cast<INDEX_SIZE_T>(HIn) * WIn;
    INDEX_SIZE_T yStep = static_cast<INDEX_SIZE_T>(HOut) * WOut;
    for (int32_t c = 0; c < C; c++) {
        float out = 0.0f;
        if (WithinBounds2d(iyNw, ixNw, HIn, WIn)) {
            out += ReadAsFloat<T>(xGm[xBase + static_cast<INDEX_SIZE_T>(iyNw) * WIn + ixNw]) * nw;
        }
        if (WithinBounds2d(iyNw, ixNe, HIn, WIn)) {
            out += ReadAsFloat<T>(xGm[xBase + static_cast<INDEX_SIZE_T>(iyNw) * WIn + ixNe]) * ne;
        }
        if (WithinBounds2d(iySw, ixNw, HIn, WIn)) {
            out += ReadAsFloat<T>(xGm[xBase + static_cast<INDEX_SIZE_T>(iySw) * WIn + ixNw]) * sw;
        }
        if (WithinBounds2d(iySw, ixNe, HIn, WIn)) {
            out += ReadAsFloat<T>(xGm[xBase + static_cast<INDEX_SIZE_T>(iySw) * WIn + ixNe]) * se;
        }
        yGm[yBase] = CastToOutput<T>(out);
        // Do not advance past the last element: with a bounded int32 index
        // width the final xBase/yBase increment could cross INT32_MAX and
        // overflow (signed overflow is undefined). The advanced values are not
        // read after the last iteration anyway.
        if (c + 1 < C) {
            xBase += xStep;
            yBase += yStep;
        }
    }
}

template <typename T, typename INDEX_SIZE_T>
__simt_callee__ inline void NearestSample(__gm__ T* xGm, __gm__ T* yGm, int32_t n, int32_t C, int32_t HIn, int32_t WIn,
                                          int32_t HOut, int32_t WOut, int32_t hOut, int32_t wOut, float ix, float iy)
{
    int32_t ixNear = BankerRoundToInt(ix);
    int32_t iyNear = BankerRoundToInt(iy);
    INDEX_SIZE_T xBase = static_cast<INDEX_SIZE_T>(n) * C * HIn * WIn;
    INDEX_SIZE_T yBase = static_cast<INDEX_SIZE_T>(n) * C * HOut * WOut + static_cast<INDEX_SIZE_T>(hOut) * WOut + wOut;
    INDEX_SIZE_T xStep = static_cast<INDEX_SIZE_T>(HIn) * WIn;
    INDEX_SIZE_T yStep = static_cast<INDEX_SIZE_T>(HOut) * WOut;
    for (int32_t c = 0; c < C; c++) {
        float out = 0.0f;
        if (WithinBounds2d(iyNear, ixNear, HIn, WIn)) {
            out = ReadAsFloat<T>(xGm[xBase + static_cast<INDEX_SIZE_T>(iyNear) * WIn + ixNear]);
        }
        yGm[yBase] = CastToOutput<T>(out);
        // Do not advance past the last element: with a bounded int32 index
        // width the final xBase/yBase increment could cross INT32_MAX and
        // overflow (signed overflow is undefined). The advanced values are not
        // read after the last iteration anyway.
        if (c + 1 < C) {
            xBase += xStep;
            yBase += yStep;
        }
    }
}

// TTK round 36: use fmaf chain to match PyTorch's FMA compilation of
// c0*v0 + c1*v1 + c2*v2 + c3*v3 (x86 vfmadd231ps chain).
// First step is a plain multiply (vmulps), steps 2-4 are FMA.
__simt_callee__ inline float BicubicWeightedSum(float c0, float c1, float c2, float c3, float v0, float v1, float v2,
                                                float v3)
{
    float r = c0 * v0;
    r = fmaf(c1, v1, r);
    r = fmaf(c2, v2, r);
    r = fmaf(c3, v3, r);
    return r;
}

template <typename T, typename INDEX_SIZE_T>
__simt_callee__ inline void BicubicSample(__gm__ T* xGm, __gm__ T* yGm, int32_t n, int32_t C, int32_t HIn, int32_t WIn,
                                          int32_t HOut, int32_t WOut, int32_t hOut, int32_t wOut, float ix, float iy,
                                          int32_t paddingMode, int32_t alignCorners)
{
    float ixFloor = floorf(ix);
    float iyFloor = floorf(iy);
    float tx = ix - ixFloor;
    float ty = iy - iyFloor;

    float fx0 = ComputeCoordinates(ixFloor - 1.0f, WIn, paddingMode, alignCorners);
    float fx1 = ComputeCoordinates(ixFloor, WIn, paddingMode, alignCorners);
    float fx2 = ComputeCoordinates(ixFloor + 1.0f, WIn, paddingMode, alignCorners);
    float fx3 = ComputeCoordinates(ixFloor + 2.0f, WIn, paddingMode, alignCorners);
    int32_t bx0 = SafeFloatToInt32(fx0);
    int32_t bx1 = SafeFloatToInt32(fx1);
    int32_t bx2 = SafeFloatToInt32(fx2);
    int32_t bx3 = SafeFloatToInt32(fx3);

    float fy0 = ComputeCoordinates(iyFloor - 1.0f, HIn, paddingMode, alignCorners);
    float fy1 = ComputeCoordinates(iyFloor, HIn, paddingMode, alignCorners);
    float fy2 = ComputeCoordinates(iyFloor + 1.0f, HIn, paddingMode, alignCorners);
    float fy3 = ComputeCoordinates(iyFloor + 2.0f, HIn, paddingMode, alignCorners);
    int32_t by0 = SafeFloatToInt32(fy0);
    int32_t by1 = SafeFloatToInt32(fy1);
    int32_t by2 = SafeFloatToInt32(fy2);
    int32_t by3 = SafeFloatToInt32(fy3);

    float cx0 = CubicConvolution2(tx + 1.0f, A);
    float cx1 = CubicConvolution1(tx, A);
    float cx2 = CubicConvolution1(1.0f - tx, A);
    float cx3 = CubicConvolution2(2.0f - tx, A);
    float cy0 = CubicConvolution2(ty + 1.0f, A);
    float cy1 = CubicConvolution1(ty, A);
    float cy2 = CubicConvolution1(1.0f - ty, A);
    float cy3 = CubicConvolution2(2.0f - ty, A);

    INDEX_SIZE_T xBase = static_cast<INDEX_SIZE_T>(n) * C * HIn * WIn;
    INDEX_SIZE_T yBase = static_cast<INDEX_SIZE_T>(n) * C * HOut * WOut + static_cast<INDEX_SIZE_T>(hOut) * WOut + wOut;
    INDEX_SIZE_T xStep = static_cast<INDEX_SIZE_T>(HIn) * WIn;
    INDEX_SIZE_T yStep = static_cast<INDEX_SIZE_T>(HOut) * WOut;
    for (int32_t c = 0; c < C; c++) {
        float r0 = 0.0f;
        if (by0 >= 0 && by0 < HIn) {
            INDEX_SIZE_T off = xBase + static_cast<INDEX_SIZE_T>(by0) * WIn;
            float v0 = (bx0 >= 0 && bx0 < WIn) ? ReadAsFloat<T>(xGm[off + bx0]) : 0.0f;
            float v1 = (bx1 >= 0 && bx1 < WIn) ? ReadAsFloat<T>(xGm[off + bx1]) : 0.0f;
            float v2 = (bx2 >= 0 && bx2 < WIn) ? ReadAsFloat<T>(xGm[off + bx2]) : 0.0f;
            float v3 = (bx3 >= 0 && bx3 < WIn) ? ReadAsFloat<T>(xGm[off + bx3]) : 0.0f;
            r0 = BicubicWeightedSum(cx0, cx1, cx2, cx3, v0, v1, v2, v3);
        }
        float r1 = 0.0f;
        if (by1 >= 0 && by1 < HIn) {
            INDEX_SIZE_T off = xBase + static_cast<INDEX_SIZE_T>(by1) * WIn;
            float v0 = (bx0 >= 0 && bx0 < WIn) ? ReadAsFloat<T>(xGm[off + bx0]) : 0.0f;
            float v1 = (bx1 >= 0 && bx1 < WIn) ? ReadAsFloat<T>(xGm[off + bx1]) : 0.0f;
            float v2 = (bx2 >= 0 && bx2 < WIn) ? ReadAsFloat<T>(xGm[off + bx2]) : 0.0f;
            float v3 = (bx3 >= 0 && bx3 < WIn) ? ReadAsFloat<T>(xGm[off + bx3]) : 0.0f;
            r1 = BicubicWeightedSum(cx0, cx1, cx2, cx3, v0, v1, v2, v3);
        }
        float r2 = 0.0f;
        if (by2 >= 0 && by2 < HIn) {
            INDEX_SIZE_T off = xBase + static_cast<INDEX_SIZE_T>(by2) * WIn;
            float v0 = (bx0 >= 0 && bx0 < WIn) ? ReadAsFloat<T>(xGm[off + bx0]) : 0.0f;
            float v1 = (bx1 >= 0 && bx1 < WIn) ? ReadAsFloat<T>(xGm[off + bx1]) : 0.0f;
            float v2 = (bx2 >= 0 && bx2 < WIn) ? ReadAsFloat<T>(xGm[off + bx2]) : 0.0f;
            float v3 = (bx3 >= 0 && bx3 < WIn) ? ReadAsFloat<T>(xGm[off + bx3]) : 0.0f;
            r2 = BicubicWeightedSum(cx0, cx1, cx2, cx3, v0, v1, v2, v3);
        }
        float r3 = 0.0f;
        if (by3 >= 0 && by3 < HIn) {
            INDEX_SIZE_T off = xBase + static_cast<INDEX_SIZE_T>(by3) * WIn;
            float v0 = (bx0 >= 0 && bx0 < WIn) ? ReadAsFloat<T>(xGm[off + bx0]) : 0.0f;
            float v1 = (bx1 >= 0 && bx1 < WIn) ? ReadAsFloat<T>(xGm[off + bx1]) : 0.0f;
            float v2 = (bx2 >= 0 && bx2 < WIn) ? ReadAsFloat<T>(xGm[off + bx2]) : 0.0f;
            float v3 = (bx3 >= 0 && bx3 < WIn) ? ReadAsFloat<T>(xGm[off + bx3]) : 0.0f;
            r3 = BicubicWeightedSum(cx0, cx1, cx2, cx3, v0, v1, v2, v3);
        }
        float out = BicubicWeightedSum(cy0, cy1, cy2, cy3, r0, r1, r2, r3);
        yGm[yBase] = CastToOutput<T>(out);
        // Do not advance past the last element: with a bounded int32 index
        // width the final xBase/yBase increment could cross INT32_MAX and
        // overflow (signed overflow is undefined). The advanced values are not
        // read after the last iteration anyway.
        if (c + 1 < C) {
            xBase += xStep;
            yBase += yStep;
        }
    }
}

// ===== Main SIMT VF kernel =====
template <typename T, uint32_t interpMode, typename INDEX_SIZE_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_T<INDEX_SIZE_T>) inline void OpGridSampler2dSimt(
    int32_t totalPixels, int32_t N, int32_t C, int32_t HIn, int32_t WIn, int32_t HOut, int32_t WOut,
    int32_t paddingMode, int32_t alignCorners, __gm__ T* xGm, __gm__ T* gridGm, __gm__ T* yGm, uint32_t magicW,
    uint32_t shiftW, uint32_t magicH, uint32_t shiftH)
{
    // 64-bit loop induction: totalPixels may equal INT32_MAX, and an int32_t
    // `idx += stride` would overflow (UB/wrap) on the final increment, so the
    // thread would not terminate. The 64-bit index also matches the widened
    // address arithmetic below.
    int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    int64_t baseIdx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                      static_cast<int64_t>(threadIdx.x);
    uint32_t uWOut = static_cast<uint32_t>(WOut);
    uint32_t uHOut = static_cast<uint32_t>(HOut);
    for (int64_t idx = baseIdx; idx < static_cast<int64_t>(totalPixels); idx += stride) {
        uint32_t uIdx = static_cast<uint32_t>(idx);
        uint32_t hw = Simt::UintDiv<uint32_t>(uIdx, magicW, shiftW);
        uint32_t wOut = uIdx - hw * uWOut;
        uint32_t n = Simt::UintDiv<uint32_t>(hw, magicH, shiftH);
        uint32_t hOut = hw - n * uHOut;
        INDEX_SIZE_T gridOff = (static_cast<INDEX_SIZE_T>(n) * HOut + hOut) * WOut + wOut;
        gridOff = gridOff * 2;
        float xCoord = ReadAsFloat<T>(gridGm[gridOff]);
        float yCoord = ReadAsFloat<T>(gridGm[gridOff + 1]);
        if constexpr (interpMode == GRID_SAMPLER_2D_BILINEAR) {
            float ix = ComputeSourceIndex(xCoord, WIn, paddingMode, alignCorners);
            float iy = ComputeSourceIndex(yCoord, HIn, paddingMode, alignCorners);
            if (IsNanOrInf(ix) || IsNanOrInf(iy)) {
                WriteNanToOutput<T, INDEX_SIZE_T>(yGm, static_cast<int32_t>(n), C, HOut, WOut,
                                                  static_cast<int32_t>(hOut), static_cast<int32_t>(wOut));
            } else {
                ix = SafeDowngradeToIntRange(ix);
                iy = SafeDowngradeToIntRange(iy);
                BilinearSample<T, INDEX_SIZE_T>(xGm, yGm, static_cast<int32_t>(n), C, HIn, WIn, HOut, WOut,
                                                static_cast<int32_t>(hOut), static_cast<int32_t>(wOut), ix, iy);
            }
        } else if constexpr (interpMode == GRID_SAMPLER_2D_NEAREST) {
            float ix = ComputeSourceIndex(xCoord, WIn, paddingMode, alignCorners);
            float iy = ComputeSourceIndex(yCoord, HIn, paddingMode, alignCorners);
            ix = SafeDowngradeToIntRange(ix);
            iy = SafeDowngradeToIntRange(iy);
            NearestSample<T, INDEX_SIZE_T>(xGm, yGm, static_cast<int32_t>(n), C, HIn, WIn, HOut, WOut,
                                           static_cast<int32_t>(hOut), static_cast<int32_t>(wOut), ix, iy);
        } else {
            float ix = UnnormalizeNoClip(xCoord, WIn, alignCorners);
            float iy = UnnormalizeNoClip(yCoord, HIn, alignCorners);
            if (IsNanOrInf(ix) || IsNanOrInf(iy)) {
                WriteNanToOutput<T, INDEX_SIZE_T>(yGm, static_cast<int32_t>(n), C, HOut, WOut,
                                                  static_cast<int32_t>(hOut), static_cast<int32_t>(wOut));
            } else {
                BicubicSample<T, INDEX_SIZE_T>(xGm, yGm, static_cast<int32_t>(n), C, HIn, WIn, HOut, WOut,
                                               static_cast<int32_t>(hOut), static_cast<int32_t>(wOut), ix, iy,
                                               paddingMode, alignCorners);
            }
        }
    }
}

template <typename T, uint32_t interpMode>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR grid, GM_ADDR y, const GridSampler2DTilingData* tilingData)
{
    int64_t totalPixels64 = static_cast<int64_t>(tilingData->N) * tilingData->H_out * tilingData->W_out;
    if (totalPixels64 <= 0 || totalPixels64 > 2147483647LL) {
        return;
    }
    int32_t totalPixels = static_cast<int32_t>(totalPixels64);
    __gm__ T* xGm = (__gm__ T*)x;
    __gm__ T* gridGm = (__gm__ T*)grid;
    __gm__ T* yGm = (__gm__ T*)y;

    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    uint32_t magicH = 0;
    uint32_t shiftH = 0;
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(tilingData->W_out));
    GetUintDivMagicAndShift<uint32_t>(magicH, shiftH, static_cast<uint32_t>(tilingData->H_out));

    int64_t maxGridOff = totalPixels64 * 2 - 1;
    int64_t maxXAddr = static_cast<int64_t>(tilingData->N) * tilingData->C * tilingData->H_in * tilingData->W_in - 1;
    int64_t maxYAddr = totalPixels64 * tilingData->C - 1;
    int64_t maxAddr = maxGridOff;
    if (maxXAddr > maxAddr) {
        maxAddr = maxXAddr;
    }
    if (maxYAddr > maxAddr) {
        maxAddr = maxYAddr;
    }

    if (maxAddr <= 2147483647LL) {
        asc_vf_call<OpGridSampler2dSimt<T, interpMode, int32_t>>(
            dim3(THREAD_NUM_T<int32_t>), totalPixels, tilingData->N, tilingData->C, tilingData->H_in, tilingData->W_in,
            tilingData->H_out, tilingData->W_out, tilingData->paddingMode, tilingData->alignCorners, xGm, gridGm, yGm,
            magicW, shiftW, magicH, shiftH);
    } else {
        asc_vf_call<OpGridSampler2dSimt<T, interpMode, int64_t>>(
            dim3(THREAD_NUM_T<int64_t>), totalPixels, tilingData->N, tilingData->C, tilingData->H_in, tilingData->W_in,
            tilingData->H_out, tilingData->W_out, tilingData->paddingMode, tilingData->alignCorners, xGm, gridGm, yGm,
            magicW, shiftW, magicH, shiftH);
    }
}

} // namespace NsGridSampler2D
#endif // GRID_SAMPLER2_D_SIMT_H_
