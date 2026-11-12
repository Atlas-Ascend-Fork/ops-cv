/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// rotated_box_decode_package/op_kernel/arch35/rotated_box_decode_simt.h
// =============================================================================
//
// ROLE: N==1 SIMT branch (UB_AXIS_SEL=I, tiling key 512) for RotatedBoxDecode.
//
// WHY: N==1 shapes give 1-box tiles on the VF path — 1/8 lane utilization,
//   ~30 fp16 unpack/cast/pack conversion instructions per repeat, and 3 tiny
//   MTE descriptors + 6 flag ops per single box (PRC-SIMT-001's "离散小包"
//   regime: 10B contiguous per box). The SIMT form replaces all of it with
//   one scalar pipeline per box: threads read GM directly (adjacent threads
//   coalesce into full lines), no MTE, no UB staging, no lane waste.
//
// ISOLATION: routed ONLY by host tiling when N==1 (both dtypes, any B>0).
//   The VF path (kernel.h) is untouched; every N>1 shape keeps byte-identical
//   scheduling. Multi-core = every AIV joins one global grid-stride pool:
//   host sets BlockDim(coreCount), each AIV calls asc_vf_call, and the
//   __simt_vf__ body strides by blockDim.x * gridDim.x (strided_slice_assign
//   house pattern).
//
// PRECISION — op-for-op scalar port of the VF chain (rotated_box_decode_kernel.h):
//   Every Reg API op maps to the SAME single IEEE operation, executed once
//   (no SIMD lane masking): Div→`/`, Muls/Mul→`*`, Add/Adds→`+`, Sub→`-`,
//   MulAddDst(d,a,b)→`d = fmaf(a, b, d)` (single rounding, == vmula),
//   Truncate<CAST_RINT>→rintf, Truncate<CAST_FLOOR>→floorf, Abs→fabsf,
//   Compares+Select→ternary, hw Exp tail→expf, b16 widen/narrow→
//   __half2float / __float2half_rn (RN == CAST_RINT).
//
//   NaN-semantics notes (mirror the VF behavior that the L1 suite blesses):
//   - Maxs/Mins clamps use comparison-ternary forms that PROPAGATE NaN
//     ((x < c) ? c : x — NaN fails the compare, keeps NaN), matching
//     torch.clamp/atan(NaN)=NaN. fminf/fmaxf would swallow NaN instead.
//   - The AtanImpl branch-selection Min() uses fminf: on NaN input all
//     branches are NaN (fminf(NaN,NaN)=NaN), and the +Inf overflow guard
//     needs min semantics, so fminf is correct there.
//   - RbdDivIeeeS keeps the Markstein correction + NaN fallback: with an
//     IEEE-correct scalar fdiv it degenerates to the correctly-rounded
//     quotient (r==exact residual, e≈0), i.e. bit-matches torch
//     true_divide either way; with a ≤1ULP fdiv it repairs exactly like
//     the VF path.
//
//   Known accepted divergences vs the VF path (documented risk, L1-gated):
//   - exp tail (|x|>87): expf (correctly-rounded-ish) vs hw vexp (≤2ULP) —
//     the band only matters for already-huge outputs; the VF path already
//     tolerates ULP noise there.
//   - Plain-Div sites (AtanImpl branch-2 reduction + asymptotic 1/|x|):
//     scalar fdiv is correctly rounded where vector vdiv is ≤1ULP — equal
//     or closer to the golden.
//
// =============================================================================

#ifndef ROTATED_BOX_DECODE_SIMT_H_
#define ROTATED_BOX_DECODE_SIMT_H_

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"

#include "rotated_box_decode_tiling_data.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace rbd_simt {
using namespace AscendC;

// __launch_bounds__ / dim3 want a uint32 constant expression (SIMT MDE rule:
// compile-time, never from tiling). RBD_SIMT_THREAD_NUM lives in
// rotated_box_decode_tiling_data.h so the host tiling can mirror it for
// core-count math.
constexpr uint32_t kSimtThreadNum = static_cast<uint32_t>(RBD_SIMT_THREAD_NUM);

// =============================================================================
// Scalar ports of the VF precision helpers — DO NOT reorder or reassociate
// any operation: each line is one IEEE op, mirroring the Reg API sequence in
// rotated_box_decode_kernel.h bit-for-bit (see the header comment).
// =============================================================================

// RbdDivIeeeS — correctly-rounded fp32 division (Markstein), scalar port of
// RbdDivIeee in kernel.h. NaN fallback keeps q0 for inf/NaN edges.
__simt_callee__ inline float RbdDivIeeeS(float d, float w)
{
    float q0 = d / w;          // Reg::Div
    float r = fmaf(q0, -w, d); // r = d − q0·w (FMA, exact residual)
    float e = r / w;
    float q = q0 + e;
    if (q != q) { // Compare NE + Select: NaN → q0
        q = q0;
    }
    return q;
}

// RbdExpDDS — SLEEF xexpf replica + tail, scalar port of RbdExpDD in kernel.h.
__simt_callee__ inline float RbdExpDDS(float x)
{
    constexpr float RBD_R_LN2 = 1.442695040888963407359924681001892137426645954152985934135449406931f;
    constexpr float RBD_L2U = 0.693145751953125f;
    constexpr float RBD_L2L = 1.428606765330187045e-06f;
    constexpr float RBD_PC0 = 0.000198527617612853646278381f;
    constexpr float RBD_PC1 = 0.00139304355252534151077271f;
    constexpr float RBD_PC2 = 0.00833336077630519866943359f;
    constexpr float RBD_PC3 = 0.0416664853692054748535156f;
    constexpr float RBD_PC4 = 0.166666671633720397949219f;
    constexpr float RBD_PC5 = 0.5f;
    constexpr float RBD_MAGIC = 12582912.0f;       // 1.5·2^23
    constexpr int32_t RBD_IADJ = 127 - 0x4B400000; // 127 − bits(1.5·2^23)
    constexpr float RBD_SLEEF_BOUND = 87.0f;

    float hw = expf(x); // hw result kept for the tail select
    // k = rint(x·log2e)
    float k = rintf(x * RBD_R_LN2);
    // r = x − k·L2U − k·L2L (Cody-Waite, 2 FMAs)
    float r = x;
    r = fmaf(k, -RBD_L2U, r);
    r = fmaf(k, -RBD_L2L, r);
    // u = deg-5 minimax poly on r (alternating Horner, same op order as VF)
    float u = RBD_PC0;
    float t = RBD_PC1;
    t = fmaf(u, r, t); // t = PC1 + PC0·r
    u = RBD_PC2;
    u = fmaf(t, r, u); // u = PC2 + t·r
    t = RBD_PC3;
    t = fmaf(u, r, t);
    u = RBD_PC4;
    u = fmaf(t, r, u);
    t = RBD_PC5;
    t = fmaf(u, r, t);
    // uu = 1 + fma(r·r, u, r)
    float ss = r * r;
    r = fmaf(ss, t, r);
    r = r + 1.0f;
    // 2^k via magic-add: bits(k + MAGIC) + IADJ, <<23, reinterpret as float
    union {
        float f;
        int32_t i;
    } magic;
    magic.f = k + RBD_MAGIC;
    magic.i = magic.i + RBD_IADJ;
    magic.i = magic.i << 23;
    float result = r * magic.f;
    // |x| ≤ 87 → SLEEF replica; else hw tail
    if (x > RBD_SLEEF_BOUND || x < -RBD_SLEEF_BOUND) {
        return hw;
    }
    return result;
}

// AtanImplS — atan(x), scalar port of AtanImpl in kernel.h (9-term Taylor
// multi-range + Min branch selection + asymptotic path + sign apply).
__simt_callee__ inline float AtanImplS(float x)
{
    constexpr float PI_BY_4 = 0.78539816339744830961566084581988f;
    constexpr float PI_BY_8 = 0.39269908169872415480783042290994f;
    constexpr float THREE_PI_BY_8 = 1.1780972450961724f; // fp32(3π/8), single rounding
    constexpr float TAN_PI_BY_8 = 0.4142135623730950f;
    constexpr float MAX_INPUT = 10000.0f;
    constexpr float MIN_INPUT = -10000.0f;
    constexpr float ASYM_THRESHOLD = 100.0f;
    constexpr float PI_2_VAL = 1.57079632679489661923132169164f;
    constexpr float SIGN_EPS = 2.168404344971009e-19f; // 2^-62
    constexpr float F0 = 1.0f;
    constexpr float F1 = -0.3333333333333333f;
    constexpr float F2 = 0.2f;
    constexpr float F3 = -0.14285714285714285f;
    constexpr float F4 = 0.1111111111111111f;
    constexpr float F5 = -0.09090909090909091f;
    constexpr float F6 = 0.07692307692307693f;
    constexpr float F7 = -0.06666666666666667f;
    constexpr float F8 = 0.058823529411764705f;

    // Clip to [-10000, 10000] — NaN-propagating ternaries (see header comment)
    float clip = (x > MAX_INPUT) ? MAX_INPUT : x;
    clip = (clip < MIN_INPUT) ? MIN_INPUT : clip;
    float absX = fabsf(clip);

    // x² (reused by all Taylor branches)
    float x2 = absX * absX;

    // --- Branch 1: x ∈ (0, tan(π/8)) — 9-term Taylor on clip-to-2 ---
    float absC = (absX > 2.0f) ? 2.0f : absX;
    float x2c = absC * absC;
    float t4 = F8;
    float tmp = t4 * x2c;
    t4 = tmp + F7;
    tmp = t4 * x2c;
    t4 = tmp + F6;
    tmp = t4 * x2c;
    t4 = tmp + F5;
    tmp = t4 * x2c;
    t4 = tmp + F4;
    tmp = t4 * x2c;
    t4 = tmp + F3;
    tmp = t4 * x2c;
    t4 = tmp + F2;
    tmp = t4 * x2c;
    t4 = tmp + F1;
    tmp = t4 * x2c;
    t4 = tmp + F0;
    t4 = t4 * absC;

    // --- Branch 2: x ∈ (tan(π/8), tan(π/4)) → π/8 + atan((x-c)/(1+xc)) ---
    float tmpB = absX * TAN_PI_BY_8;
    tmpB = tmpB + 1.0f;
    float r2 = absX - TAN_PI_BY_8;
    r2 = r2 / tmpB; // plain Div in the VF path (vdiv); scalar fdiv is correctly rounded
    r2 = fabsf(r2);
    float den2 = r2 * r2;
    float t6 = F8;
    tmp = t6 * den2;
    t6 = tmp + F7;
    tmp = t6 * den2;
    t6 = tmp + F6;
    tmp = t6 * den2;
    t6 = tmp + F5;
    tmp = t6 * den2;
    t6 = tmp + F4;
    tmp = t6 * den2;
    t6 = tmp + F3;
    tmp = t6 * den2;
    t6 = tmp + F2;
    tmp = t6 * den2;
    t6 = tmp + F1;
    tmp = t6 * den2;
    t6 = tmp + F0;
    t6 = t6 * r2;
    t6 = t6 + PI_BY_8;
    t4 = fminf(t4, t6);

    // --- Branch 3: x ∈ (tan(π/4), ∞) → π/4 + atan((|x|-1)/(|x|+1)) ---
    float tmp3 = absX + 1.0f;
    float r3 = absX - 1.0f;
    r3 = RbdDivIeeeS(r3, tmp3); // correctly-rounded r3 (arg-chain accuracy)
    r3 = fabsf(r3);
    float den3 = r3 * r3;
    t6 = F8;
    tmp = t6 * den3;
    t6 = tmp + F7;
    tmp = t6 * den3;
    t6 = tmp + F6;
    tmp = t6 * den3;
    t6 = tmp + F5;
    tmp = t6 * den3;
    t6 = tmp + F4;
    tmp = t6 * den3;
    t6 = tmp + F3;
    tmp = t6 * den3;
    t6 = tmp + F2;
    tmp = t6 * den3;
    t6 = tmp + F1;
    tmp = t6 * den3;
    t6 = tmp + F0;
    t6 = t6 * r3;
    t6 = t6 + PI_BY_4;
    t4 = fminf(t4, t6);

    // --- Branch 4: finer — 3π/8 + atan(transform) — FMA Horner (1 rounding/step) ---
    float tmp4 = r3 * TAN_PI_BY_8;
    tmp4 = tmp4 + 1.0f;
    float r4 = r3 - TAN_PI_BY_8;
    r4 = RbdDivIeeeS(r4, tmp4);
    r4 = fabsf(r4);
    float den4 = r4 * r4;
    t6 = F8;
    tmp = F7;
    tmp = fmaf(t6, den4, tmp); // tmp = F7 + F8·d (FMA, 1 rounding)
    t6 = F6;
    t6 = fmaf(tmp, den4, t6);
    tmp = F5;
    tmp = fmaf(t6, den4, tmp);
    t6 = F4;
    t6 = fmaf(tmp, den4, t6);
    tmp = F3;
    tmp = fmaf(t6, den4, tmp);
    t6 = F2;
    t6 = fmaf(tmp, den4, t6);
    tmp = F1;
    tmp = fmaf(t6, den4, tmp);
    t6 = F0;
    t6 = fmaf(tmp, den4, t6);
    t6 = t6 * r4;
    t6 = t6 + THREE_PI_BY_8; // fused constant: one rounding
    t4 = fminf(t4, t6);

    // --- Asymptotic path for |x| > 100: atan(x) = sign·(π/2 − atan(1/|x|)) ---
    float absSrc = fabsf(x);
    float inv = 1.0f / absSrc; // plain Div in the VF path; handles inf (1/inf=0)
    float x2a = inv * inv;
    float acc = F6;
    tmp = acc * x2a;
    acc = tmp + F5;
    tmp = acc * x2a;
    acc = tmp + F4;
    tmp = acc * x2a;
    acc = tmp + F3;
    tmp = acc * x2a;
    acc = tmp + F2;
    tmp = acc * x2a;
    acc = tmp + F1;
    tmp = acc * x2a;
    acc = tmp + F0;
    acc = acc * inv;
    float asym = PI_2_VAL - acc;
    if (absSrc > ASYM_THRESHOLD) { // NaN > 100 is false → NaN falls through
        t4 = asym;
    }

    // --- Apply sign: dst = sign(clip) · atan(|clip|) ---
    float sign = clip / (absX + SIGN_EPS);
    return t4 * sign;
}

// TanSleefFastS — bit-exact SLEEF xtanf_u1 fast-path replica, scalar port of
// TanSleefFast in kernel.h (double-single arithmetic preserved op-for-op).
__simt_callee__ inline float TanSleefFastS(float d)
{
    constexpr float RBD_2RPI = 0.63661975f;    // 0x3f22f983, fp32(2·M_1_PI)
    constexpr float RBD_PA2H = -1.5707397f;    // 0xbfc90e00, −PI_A2f·0.5
    constexpr float RBD_PB2H = -5.657971e-05f; // 0xb86d5000, −PI_B2f·0.5
    constexpr float RBD_PC2H = -9.920936e-10f; // 0xb0885a31, −PI_C2f·0.5
    constexpr float RBD_TP0 = 0.0044663646f;   // 0x3b925a95
    constexpr float RBD_TP1 = -8.392018e-05f;  // 0xb8affe4e
    constexpr float RBD_TP2 = 0.010963924f;    // 0x3c33a208
    constexpr float RBD_TP3 = 0.02123603f;     // 0x3cadf72f
    constexpr float RBD_TP4 = 0.054068714f;    // 0x3d5d7728
    constexpr float RBD_TA = 0.13332567f;      // 0x3e088686
    constexpr float RBD_TB = 0.3333336f;       // 0x3eaaaab4

    // q = rint(d·2/π)
    float q = rintf(d * RBD_2RPI);

    // v = fma(q, −π/2.A, d)
    float v = fmaf(q, RBD_PA2H, d);

    // s = dfadd2(v, fl(q·(−π/2.B))): s = v+y; t = s−v; lo = (v−(s−t)) + (y−t)
    float p = q * RBD_PB2H;
    float sh = v + p;
    float t1 = sh - v;
    float t2 = sh - t1;
    v = v - t2;
    p = p - t1;
    float sl = v + p;

    // s = dfadd(s, fl(q·(−π/2.C))): s2 = sh+y; lo = ((sh−s2)+y) + sl
    p = q * RBD_PC2H;
    t1 = sh + p;
    t2 = sh - t1;
    t2 = t2 + p;
    sl = t2 + sl;
    sh = t1;

    // odd = (floor(q/2) ≠ q/2) — exact for the integer-valued q
    t1 = q * 0.5f;
    t2 = floorf(t1);
    bool odd = (t1 != t2);

    // s₂ = dfnormalize(dfsqu(s)): sq = fl(sh²); lo = fma(2sh, sl, fl(sh²−sq))
    float s2h = sh * sh;
    t1 = -s2h;
    t1 = fmaf(sh, sh, t1); // sh·sh − sq
    t2 = sh + sh;
    t1 = fmaf(t2, sl, t1);
    float s2l = t1;
    // normalize: s = s2h+s2l; lo = (s2h−s)+s2l
    t1 = s2h + s2l;
    t2 = s2h - t1;
    s2l = t2 + s2l;
    s2h = t1;

    // u = 4-step fused Horner on s₂.hi
    float u = RBD_TP0;
    float c = RBD_TP1;
    c = fmaf(u, s2h, c);
    u = c;
    c = RBD_TP2;
    c = fmaf(u, s2h, c);
    u = c;
    c = RBD_TP3;
    c = fmaf(u, s2h, c);
    u = c;
    c = RBD_TP4;
    c = fmaf(u, s2h, c);
    u = c;

    // x = dfadd(A, fl(u·s₂.hi)): s = A+p; lo = (A−s)+p
    p = u * s2h;
    t1 = p + RBD_TA;
    t2 = RBD_TA - t1;
    float xl = t2 + p;
    float xh = t1;

    // i1 = dfmul(s₂, x): lo = fma(s2h, xl, fma(s2l, xh, fl(s2h·xh−s)))
    float ih = s2h * xh;
    t1 = -ih;
    t1 = fmaf(s2h, xh, t1);
    t1 = fmaf(s2l, xh, t1);
    t1 = fmaf(s2h, xl, t1);
    float il = t1;

    // i2 = dfadd(B, i1): s = B+ih; v = s−B; lo = ((B−(s−v)) + (ih−v)) + il
    t1 = ih + RBD_TB;
    t2 = t1 - RBD_TB; // v
    c = t1 - t2;      // s−v
    p = RBD_TB - c;   // B−(s−v)
    c = ih - t2;      // ih−v
    p = p + c;
    il = p + il;
    ih = t1;

    // i3 = dfmul(i2, s₂) (into xh/xl — i1's registers are dead)
    xh = ih * s2h;
    t1 = -xh;
    t1 = fmaf(ih, s2h, t1);
    t1 = fmaf(il, s2h, t1);
    t1 = fmaf(ih, s2l, t1);
    xl = t1;

    // x = dfadd(1, i3): s = 1+xh; lo = ((1−(s−v)) + (xh−v)) + xl
    t1 = xh + 1.0f;
    t2 = t1 - 1.0f; // v
    c = t1 - t2;    // s−v
    p = 1.0f - c;   // 1−(s−v)
    c = xh - t2;    // xh−v (i3.hi)
    p = p + c;
    xl = p + xl;
    xh = t1;

    // deferred sign flip: odd → (sh, sl) := −(sh, sl)
    if (odd) {
        sh = -sh;
        sl = -sl;
    }

    // x = dfmul(s, x): s = fl(sh·xh); lo = fma(sh, xl, fma(sl, xh, fl(sh·xh−s)))
    t1 = sh * xh;
    t2 = -t1;
    t2 = fmaf(sh, xh, t2);
    t2 = fmaf(sl, xh, t2);
    t2 = fmaf(sh, xl, t2);
    xh = t1;
    xl = t2;

    // odd → x = dfrec(x): s = 1/xh (IEEE); e1 = 1−xh·s; e2 = e1−xl·s; lo = s·e2
    if (odd) {
        t1 = RbdDivIeeeS(1.0f, xh);
        t2 = -xh;
        c = fmaf(t2, t1, 1.0f); // 1 − xh·s
        t2 = -xl;
        c = fmaf(t2, t1, c); // e2
        t2 = t1 * c;         // lo = s·e2
        xh = t1;
        xl = t2;
    }

    // out = xh + xl
    float out = xh + xl;

    // d == −0.0 → d (bit compare — matches SLEEF's visnegzero select)
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = d;
    if (bits.u == 0x80000000u) {
        return d;
    }
    return out;
}

// =============================================================================
// dtype helpers (house pattern: GridSampler2D _simt.h)
// =============================================================================
template <typename T>
__simt_callee__ inline float RbdLoadToFloat(T val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
__simt_callee__ inline T RbdFloatToStore(float val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __float2half_rn(val); // RN == VF store's CAST_RINT
    } else {
        return static_cast<T>(val);
    }
}

// =============================================================================
// RotatedBoxDecodeSimtCompute — per-box scalar decode, op-for-op mirror of
// RotatedBoxDecodeComputeVF VF0..VF7 with every lane active (one box per
// thread, no masking). GM addressing: (B,5,1) N==1 flat layout, box b at
// element offset b*5 + c — adjacent threads touch adjacent 10B/20B chunks,
// hardware-coalesced across the warp.
// =============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreadNum) inline void RotatedBoxDecodeSimtCompute(
    __gm__ const T* anchorGm, __gm__ const T* deltasGm, __gm__ T* yGm, uint64_t totalBoxes, float w0, float w1,
    float w2, float w3, float w4)
{
    // Use decimal literals so the compiler rounds directly to the correctly-
    // rounded fp32 constant — same literals as the VF chain (torch casts the
    // double-precision constants to fp32).
    constexpr float RBD_DEG2RAD = 0.017453292519943295f; // fp32(pi/180) = 0x3c8efa35
    constexpr float RBD_RAD2DEG = 57.29577951308232f;    // fp32(180/pi) = 0x42652ee1
    constexpr float EXTREME_ANGLE_DEG = 1.0e6f;
    const float NAN_VAL = __builtin_nanf(""); // fp32 NaN (matches torch.nan)

    for (uint64_t box = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x); box < totalBoxes;
         box += static_cast<uint64_t>(blockDim.x * gridDim.x)) {
        const uint64_t base = box * RBD_CHANNELS;

        // ===== VF0: widen b16→f32 (fp32: direct) =====
        const float aLx = RbdLoadToFloat<T>(anchorGm[base + 0]);
        const float aLy = RbdLoadToFloat<T>(anchorGm[base + 1]);
        const float aRx = RbdLoadToFloat<T>(anchorGm[base + 2]);
        const float aRy = RbdLoadToFloat<T>(anchorGm[base + 3]);
        const float aAng = RbdLoadToFloat<T>(anchorGm[base + 4]);
        const float dDx = RbdLoadToFloat<T>(deltasGm[base + 0]);
        const float dDy = RbdLoadToFloat<T>(deltasGm[base + 1]);
        const float dDw = RbdLoadToFloat<T>(deltasGm[base + 2]);
        const float dDh = RbdLoadToFloat<T>(deltasGm[base + 3]);
        const float dDt = RbdLoadToFloat<T>(deltasGm[base + 4]);

        // ===== VF1: corner -> center (max(·,1) clamp, NaN-propagating) =====
        float aW = aRx - aLx;
        aW = (aW < 1.0f) ? 1.0f : aW;
        float aH = aRy - aLy;
        aH = (aH < 1.0f) ? 1.0f : aH;
        const float halfW = aW * 0.5f;
        const float aCx = aLx + halfW;
        const float halfH = aH * 0.5f;
        const float aCy = aLy + halfH;

        // ===== VF2: delta normalization Δ' = deltas / weight (Markstein) =====
        const float dpX = RbdDivIeeeS(dDx, w0);
        const float dpY = RbdDivIeeeS(dDy, w1);
        const float dpW = RbdDivIeeeS(dDw, w2);
        const float dpH = RbdDivIeeeS(dDh, w3);
        const float dpT = RbdDivIeeeS(dDt, w4);

        // ===== VF3: decode center =====
        float tmp = dpX * aW;
        const float tCx = aCx + tmp;
        tmp = dpY * aH;
        const float tCy = aCy + tmp;

        // ===== VF4: decode w/h exp =====
        const float expW = RbdExpDDS(dpW);
        const float tW = expW * aW;
        const float expH = RbdExpDDS(dpH);
        const float tH = expH * aH;

        // ===== VF5: decode angle tan·atan (deg<->rad), extreme guard =====
        const bool extreme = (aAng > EXTREME_ANGLE_DEG) || (aAng < -EXTREME_ANGLE_DEG);
        const float thetaARad = aAng * RBD_DEG2RAD;
        const float tanA = TanSleefFastS(thetaARad);
        const float tanSum = tanA + dpT;
        const float thetaTRad = AtanImplS(tanSum);
        float thetaT = thetaTRad * RBD_RAD2DEG;
        if (extreme) {
            thetaT = NAN_VAL;
        }

        // ===== VF6: center -> corner =====
        const float hTW = tW * 0.5f;
        const float hTH = tH * 0.5f;
        const float outLx = tCx - hTW;
        const float outRx = tCx + hTW;
        const float outLy = tCy - hTH;
        const float outRy = tCy + hTH;

        // ===== VF7: narrow f32→b16 + store =====
        yGm[base + 0] = RbdFloatToStore<T>(outLx);
        yGm[base + 1] = RbdFloatToStore<T>(outLy);
        yGm[base + 2] = RbdFloatToStore<T>(outRx);
        yGm[base + 3] = RbdFloatToStore<T>(outRy);
        yGm[base + 4] = RbdFloatToStore<T>(thetaT);
    }
}

// =============================================================================
// RotatedBoxDecodeSimtProcess — host-side launcher (normal __aicore__ code).
//   Every AIV calls this; the asc_vf_call + grid-stride loop covers the full
//   box range collectively (blockIdx.x distinguishes the AIV blocks). No MTE,
//   no UB staging, no event flags — pure SIMT over GM.
// =============================================================================
template <typename T>
__aicore__ inline void RotatedBoxDecodeSimtProcess(GM_ADDR anchorBox, GM_ADDR deltas, GM_ADDR y,
                                                   const RotatedBoxDecodeTilingData* td)
{
    if (td->totalCount == 0 || td->B == 0) {
        return; // empty tensor: host routes empty shapes to the VF no-op path anyway
    }

    asc_vf_call<RotatedBoxDecodeSimtCompute<T>>(
        dim3(kSimtThreadNum), reinterpret_cast<__gm__ const T*>(anchorBox), reinterpret_cast<__gm__ const T*>(deltas),
        reinterpret_cast<__gm__ T*>(y), static_cast<uint64_t>(td->totalCount), td->weight[0], td->weight[1],
        td->weight[2], td->weight[3], td->weight[4]);
}

} // namespace rbd_simt

#endif // ROTATED_BOX_DECODE_SIMT_H_
