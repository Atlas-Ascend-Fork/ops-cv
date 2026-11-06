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
// rotated_box_decode_package/op_kernel/arch35/rotated_box_decode_kernel.h
// =============================================================================
//
// ROLE: Ascend C kernel implementation for RotatedBoxDecode on arch35 (Ascend 950).
//   Implements DESIGN.md §10 + DESIGN-BRANCH-0.md §3/§5:
//     - Three-phase pipeline: CopyIn (MTE2) -> Compute (V) -> CopyOut (MTE3)
//     - P=2 ping-pong TBuf<VECCALC> slots with MTE2_V / V_MTE3 / MTE3_MTE2 sync
//     - VF0-VF7 single __simd_vf__ fused compute chain via asc_vf_call
//     - exp via RbdExpDD (__simd_callee__, bit-exact SLEEF xexpf replica
//       for |x| ≤ 87 + hardware tail beyond)
//     - tan via TanSleefFast (__simd_callee__, bit-exact SLEEF xtanf_u1
//       replica, both dtypes), atan via AtanImpl (__simd_callee__, Taylor)
//
// GM layout note: inputs are (B, 5, N) row-major channel-major per batch
//   (element (b,c,n) at flat offset (b*5+c)*N + n). CopyIn/CopyOut use a 2D stride
//   gather/scatter that maps GM [5, N] (per batch, channel-major) <-> UB [5, boxCount]
//   (channel-contiguous), so the VF sees a clean channel-contiguous view.
//
// =============================================================================
#pragma once

#include "kernel_operator.h"                // Ascend C core framework
#include "rotated_box_decode_tiling_data.h" // RotatedBoxDecodeTilingData (§7)
#include "rotated_box_decode_struct.h"      // TPL constants (RBD_UB_AXIS_N etc.)

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace rbd_kernel {

// ---------------------------------------------------------------------------
// Compile-time constants (DESIGN §10.2)
// ---------------------------------------------------------------------------
constexpr int64_t kPhysNodes = 2; // P=2 ping-pong (§9.2.3)
constexpr int64_t kChannels = 5;  // [lx,ly,rx,ry,angle] (RBD_CHANNELS)
constexpr int64_t kNumIoBufs = 3; // anchor + deltas + y (§9.5.1)
constexpr int64_t kNumInputs = 2; // anchor_box + deltas

// CastTrait for b16↔f32 conversion (dav-c310 requires explicit RegLayout + mode).
// Loads use LoadDist::DIST_UNPACK_B16, stores use StoreDist::DIST_PACK_B32 so the
// 16-bit↔32-bit width change is handled by the DMA distribution, and Cast runs in
// the register file with RegLayout::ZERO (single-shot, full vector).
//   VF0 b16→f32: castTraitB162F32 (no rounding — exact widening)
//   VF7 f32→b16: castTraitF322B16 (CAST_RINT — round to nearest even, matches torch)
constexpr AscendC::Reg::CastTrait rbdCastB16ToF32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait rbdCastF32ToB16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

// ---------------------------------------------------------------------------
// Integer helpers
// ---------------------------------------------------------------------------
__aicore__ inline int64_t MinI64(int64_t a, int64_t b) { return (a < b) ? a : b; }

// Lightweight correctly-rounded fp32 division (Markstein error compensation).
// The default Reg::Div (vdiv intrinsic) is NOT correctly rounded (≤1 ULP off).
//   q0 = vdiv(d, w)            ≤1 ULP approximation
//   r  = d - q0·w              EXACT residual (vmula fused multiply-add)
//   e  = vdiv(r, w)            small correction term
//   q  = q0 + e                correctly-rounded quotient (ties aside, ~2^-24)
// Used for Δ' = deltas/weight (bit-matches torch true_divide; a 1-ULP argument
// error is amplified by exp() by |Δ'w| up to ~50×) and for the atan argument
// reductions. Assumes finite d and normal w (input ranges guarantee this).
__simd_callee__ inline void RbdDivIeee(AscendC::Reg::RegTensor<float>& q, AscendC::Reg::RegTensor<float>& d,
                                       AscendC::Reg::RegTensor<float>& w, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> q0, r, e, negW;
    AscendC::Reg::MaskReg nanMask;
    AscendC::Reg::Div(q0, d, w, mask);
    AscendC::Reg::Muls(negW, w, -1.0f, mask);
    r = d;
    AscendC::Reg::MulAddDst(r, q0, negW, mask); // r = d - q0*w, exact via FMA
    AscendC::Reg::Div(e, r, w, mask);
    AscendC::Reg::Add(q, q0, e, mask);
    // Non-finite guard (standard Markstein NaN-fallback): for d=±inf the exact
    // residual is inf−inf=NaN, which would poison q (verified regression:
    // deltas=+inf input produced all-NaN output; golden expects Rx/Ry=±inf,
    // thetaT=90). Where q is NaN, fall back to q0 — vdiv already yields the
    // IEEE result for inf/0·x/0/0 edge cases (inf, ±inf, 0, NaN).
    AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, q, q, mask);
    AscendC::Reg::Select<float>(q, q0, q, nanMask);
}

// ===========================================================================
// __simd_callee__ transcendental helpers (DESIGN §10.9.2)
//   Called from within the __simd_vf__ body; operate on RegTensor<float>.
//   No libm / no ordinary LocalTensor API — register-only.
// ===========================================================================

// RbdExpDD — SLEEF xexpf replica (bit-exact vs torch.exp) + hardware tail.
//
// For |x| ≤ 87 the golden's torch.exp (SLEEF xexpf_u10) is replicated
// instruction-for-instruction, making the NPU exp bit-identical to the golden:
//   k = rint(x·log2e); r = x − k·L2U − k·L2L   (Cody-Waite, 2 FMAs)
//   u = deg-5 minimax poly (5 FMAs)
//   result = (1 + fma(r², u, r)) · 2^k
// 2^k is constructed exactly via magic-add (1.5·2^23 — the sum lands in
// [2^23, 2^24), so the integer k is embedded in the mantissa bits) → int32
// bit-reinterpret (C++ reference cast, same pattern as CANN atan_c310) →
// scalar adjust (127 − 0x4B400000) → ShiftLefts(23) → float reinterpret.
// Valid for k ∈ [−126, 126] ⟺ |x| ≤ 87; verified 0 mismatches vs torch on
// 30M+ samples across (−87, 87), including L1_324's enumerable /64 and /100
// domains (16,958 args each) and L1_382's ±3.8e-5 domain (50M).
// |x| > 87 → hardware Reg::Exp: x > 88.7 → inf (same as torch); x < −87.3 →
// |exp| < 1.6e-38, invisible downstream (t_w only enters out = t_cx ± t_w/2
// against O(1) t_cx); the ≤2 ULP hw noise in the (87, 88.7) finite band only
// affects already-huge outputs.
//
// This replaces the raw hw exp whose 1-2 ULP divergence vs torch, amplified by
// a_w/2 (up to ~500) through t_w = exp(Δ'w)·a_w and the catastrophic
// cancellation out = t_cx ∓ t_w/2, crossed the 2^-16 cross_check small-value
// budget (L1_482: 7 elements over, ratio 7.0 vs limit 2.0; args ∈ (−10,−2) all
// outside the old 2^-8 CR band) and, on full-fp32-span anchors, flipped
// fl(a_w·exp) across the fp32 overflow boundary (L1_382 special-value
// mismatch). The earlier 2-term CR formula (|x| ≤ 2^-8) is subsumed: the
// replica is bit-exact there too.
__simd_callee__ inline void RbdExpDD(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& src,
                                     AscendC::Reg::MaskReg& mask)
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
    constexpr float RBD_SLEEF_BOUND = 87.0f;       // |x| ≤ 87 ⟺ k ∈ [−126, 126]

    // NOTE on register economy (dav-c310 VF spill budget): the hw result is
    // written to dst FIRST and the final Select is in-place (dst as srcReg0 —
    // same pattern as CANN exp_c310's Select(tempDst, vReg0, tempDst, mask));
    // r accumulates in-place through the reduction and the final fma (the
    // RbdDivIeee dst-as-addend pattern); uReg is reused for the result after
    // the last poly FMA. VF4 consumes expW before computing expH so the two
    // calls' live ranges do not overlap.
    AscendC::Reg::RegTensor<float> kReg, sReg, uReg, tReg, ssReg, magicReg, zeroReg;
    AscendC::Reg::MaskReg bigA, bigB, bigMask;
    AscendC::Reg::Duplicate<float>(zeroReg, 0.0f);
    AscendC::Reg::Exp<float>(dst, src, mask); // dst = hw result (kept for the tail select)
    // k = rint(x·log2e)
    AscendC::Reg::Muls(kReg, src, RBD_R_LN2, mask);
    AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT, AscendC::Reg::MaskMergeMode::ZEROING>(kReg, kReg,
                                                                                                       mask);
    // r = x − k·L2U − k·L2L (SLEEF: vmla(q, −L2U, d); vmla(q, −L2L, s))
    sReg = src;
    AscendC::Reg::Adds(tReg, zeroReg, -RBD_L2U, mask);
    AscendC::Reg::MulAddDst(sReg, kReg, tReg, mask);
    AscendC::Reg::Adds(tReg, zeroReg, -RBD_L2L, mask);
    AscendC::Reg::MulAddDst(sReg, kReg, tReg, mask);
    // u = deg-5 minimax poly on r
    AscendC::Reg::Adds(uReg, zeroReg, RBD_PC0, mask);
    AscendC::Reg::Adds(tReg, zeroReg, RBD_PC1, mask);
    AscendC::Reg::MulAddDst(tReg, uReg, sReg, mask);
    AscendC::Reg::Adds(uReg, zeroReg, RBD_PC2, mask);
    AscendC::Reg::MulAddDst(uReg, tReg, sReg, mask);
    AscendC::Reg::Adds(tReg, zeroReg, RBD_PC3, mask);
    AscendC::Reg::MulAddDst(tReg, uReg, sReg, mask);
    AscendC::Reg::Adds(uReg, zeroReg, RBD_PC4, mask);
    AscendC::Reg::MulAddDst(uReg, tReg, sReg, mask);
    AscendC::Reg::Adds(tReg, zeroReg, RBD_PC5, mask);
    AscendC::Reg::MulAddDst(tReg, uReg, sReg, mask);
    // uu = 1 + fma(r·r, u, r) — sReg transitions r → uu in place
    AscendC::Reg::Mul(ssReg, sReg, sReg, mask);
    AscendC::Reg::MulAddDst(sReg, ssReg, tReg, mask);
    AscendC::Reg::Adds(sReg, sReg, 1.0f, mask);
    // 2^k: magic-add embeds k in the mantissa; int-bit view; adjust; shift out
    AscendC::Reg::Adds(magicReg, kReg, RBD_MAGIC, mask);
    AscendC::Reg::RegTensor<int32_t>& iyReg = (AscendC::Reg::RegTensor<int32_t>&)magicReg;
    AscendC::Reg::Adds(iyReg, iyReg, RBD_IADJ, mask);
    AscendC::Reg::ShiftLefts(iyReg, iyReg, static_cast<int16_t>(23), mask);
    AscendC::Reg::RegTensor<float>& scaleReg = (AscendC::Reg::RegTensor<float>&)iyReg;
    AscendC::Reg::Mul(uReg, sReg, scaleReg, mask); // uReg (dead after poly) = SLEEF result
    // |x| ≤ 87 → SLEEF replica; else hw (in-place select)
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(bigA, src, RBD_SLEEF_BOUND, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(bigB, src, -RBD_SLEEF_BOUND, mask);
    AscendC::Reg::Or(bigMask, bigA, bigB, mask);
    AscendC::Reg::Select<float>(dst, dst, uReg, bigMask);
}

// AtanImpl — atan(x) for x ∈ ℝ, result ∈ [-π/2, π/2].
// Algorithm (CANN library atan_c310_impl.h, adapted to __simd_callee__):
//   1. Clip x to [-10000, 10000] (fp32 Taylor convergence range).
//   2. |x| based multi-range Taylor: (0,tan(π/8)), (tan(π/8),tan(π/4)), (tan(π/4),∞).
//      Uses Min to select correct branch — avoids branch divergence, improves precision.
//   3. Apply sign: dst = sign(x) · atan(|x|).
//
// FIX vs upstream CANN: all branches use 9-term Taylor (F0..F8 =
// x - x³/3 + x⁵/5 - x⁷/7 + x⁹/9 - x¹¹/11 + x¹³/13 - x¹⁵/15 + x¹⁷/17) instead
// of the upstream 5-term (F0..F4). The 5-term truncation error at |x| near
// tan(π/8)=0.4142 is ~x¹¹/11 ≈ 5e-6 (76 ULP at scale 0.39), which propagates
// through θ_t = atan(tan(θ_a)+Δ't) to ~2.6e-4° abs error in the angle output —
// exceeding DESIGN §4.3 atol=1e-5 and inflating stat_rel_err mare to 2.8+
// on near-zero golden θ_t. The 9-term truncation error is ~x¹⁹/19 ≈ 3e-9
// (< 0.1 ULP, below the fp32 rounding floor). The odd (9-term) count is also
// REQUIRED by the Min() branch selection — see the F7/F8 note below.
//   For inf: clip→10000, atan(10000)≈π/2·sign → ±π/2 (matches torch.atan(±inf)).
//   For NaN: propagates through clip/Min/Mul → NaN (matches torch.atan(NaN)).
__simd_callee__ inline void AtanImpl(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& src,
                                     AscendC::Reg::MaskReg& mask)
{
    constexpr float PI_BY_4 = 0.78539816339744830961566084581988f;
    constexpr float PI_BY_8 = 0.39269908169872415480783042290994f;
    constexpr float THREE_PI_BY_8 = 1.1780972450961724f; // fp32(3π/8), single rounding
    constexpr float TAN_PI_BY_8 = 0.4142135623730950f;
    constexpr float MAX_INPUT = 10000.0f;
    constexpr float MIN_INPUT = -10000.0f;
    // Taylor coefficients: atan(x) = x - x³/3 + ... + x¹³/13 - x¹⁵/15 + x¹⁷/17
    // The Taylor MUST end on a POSITIVE term (F8, odd term count = 9): the branch
    // combination uses Min() and relies on every branch's partial sum being an
    // OVERESTIMATE of atan (alternating-series tail < 0), and on out-of-range
    // arguments (|x|>1) diverging to +large so Min discards them. An 8-term
    // partial (ending -x¹⁵/15) turns NEGATIVE for |x|>1 and UNDERESTIMATES on
    // (0.414,1) — Min then selects the wrong branch (verified regression:
    // L1_340/286/232 mere 300-1600). With 9 terms the truncation at the branch
    // boundary |x|→tan(π/8)=0.4142 is x¹⁹/19 ≈ 0.09 ULP (the 7-term's x¹⁵/15
    // ≈ 3.5 ULP dominated the cross_check thetaT error on L1_159).
    constexpr float F0 = 1.0f;
    constexpr float F1 = -0.3333333333333333f;
    constexpr float F2 = 0.2f;
    constexpr float F3 = -0.14285714285714285f;
    constexpr float F4 = 0.1111111111111111f;
    constexpr float F5 = -0.09090909090909091f;
    constexpr float F6 = 0.07692307692307693f;
    constexpr float F7 = -0.06666666666666667f;
    constexpr float F8 = 0.058823529411764705f;

    AscendC::Reg::RegTensor<float> clipReg, absReg, tmp, tmp2, x2, taylor4, taylor6;
    AscendC::Reg::RegTensor<float> signReg, denom;

    // Clip to [-10000, 10000]
    AscendC::Reg::Mins(clipReg, src, MAX_INPUT, mask);
    AscendC::Reg::Maxs(clipReg, clipReg, MIN_INPUT, mask);
    AscendC::Reg::Abs(absReg, clipReg, mask);

    // x² (reused by all Taylor branches)
    AscendC::Reg::Mul(x2, absReg, absReg, mask);

    // --- Branch 1: x ∈ (0, tan(π/8)) → atan(x) via 9-term Taylor ---
    // 9-term (F0..F8 = x - x³/3 + x⁵/5 - x⁷/7 + x⁹/9 - x¹¹/11 + x¹³/13
    // - x¹⁵/15 + x¹⁷/17) replaces original 5-term: truncation error at
    // |x|=0.411 (near tan(π/8)) drops from ~76 ULP (5-term, x¹¹/11 ≈ 5e-6)
    // to ~0.1 ULP (9-term, x¹⁹/19 ≈ 2e-9).
    // OVERFLOW GUARD: for |x| > 1, x²>1 causes the Horner accumulation of x¹⁷
    // to exceed fp32 max (3.4e38) at |x|≈300, producing +Inf. Since Min
    // operates on UNSIGNED (positive) values (sign applied after), +Inf is
    // always ≥ the correct branch, so Min would still pick the correct branch.
    // However, for robustness clip absReg to 2.0 for branch 1 only — at |x|=2
    // the 9-term Taylor gives ~6014 (well above π/2≈1.57, so Min still selects
    // the correct branch), and no intermediate exceeds ~3007 (far below fp32
    // max). For |x|<tan(π/8)=0.414 (branch 1's valid range), |x|<2 so the clip
    // is a no-op and accuracy is unaffected. Uses separate x2_clipped to
    // preserve the unclipped x2 for branches 2-4.
    AscendC::Reg::RegTensor<float> absRegClipped, x2_clipped;
    AscendC::Reg::Mins(absRegClipped, absReg, 2.0f, mask);
    AscendC::Reg::Mul(x2_clipped, absRegClipped, absRegClipped, mask);
    AscendC::Reg::Duplicate<float>(taylor4, F8);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F7, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F6, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F0, mask);
    AscendC::Reg::Mul(taylor4, taylor4, absRegClipped, mask);

    // --- Branch 2: x ∈ (tan(π/8), tan(π/4)) → π/8 + atan((x-c)/(1+xc)) ---
    AscendC::Reg::Muls(tmp, absReg, TAN_PI_BY_8, mask);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, mask);
    AscendC::Reg::Adds(tmp2, absReg, -TAN_PI_BY_8, mask);
    AscendC::Reg::Div(tmp2, tmp2, tmp, mask);
    AscendC::Reg::Abs(tmp2, tmp2, mask);
    AscendC::Reg::Mul(denom, tmp2, tmp2, mask);
    AscendC::Reg::Duplicate<float>(taylor6, F8);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F7, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F6, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F0, mask);
    AscendC::Reg::Mul(taylor6, taylor6, tmp2, mask);
    AscendC::Reg::Adds(taylor6, taylor6, PI_BY_8, mask);
    AscendC::Reg::Min(taylor4, taylor4, taylor6, mask);

    // --- Branch 3: x ∈ (tan(π/4), ∞) → π/4 + atan((|x|-1)/(|x|+1)) ---
    AscendC::Reg::Adds(tmp, absReg, 1.0f, mask);
    AscendC::Reg::Adds(tmp2, absReg, -1.0f, mask);
    RbdDivIeee(tmp2, tmp2, tmp, mask); // correctly-rounded r3 (arg-chain accuracy)
    AscendC::Reg::Abs(tmp2, tmp2, mask);
    AscendC::Reg::Mul(denom, tmp2, tmp2, mask);
    AscendC::Reg::Duplicate<float>(taylor6, F8);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F7, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F6, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F0, mask);
    AscendC::Reg::Mul(taylor6, taylor6, tmp2, mask);
    AscendC::Reg::Adds(taylor6, taylor6, PI_BY_4, mask);
    AscendC::Reg::Min(taylor4, taylor4, taylor6, mask);

    // --- Branch 4: x ∈ (tan(π/4), ∞) finer — 3π/8 + atan(transform) ---
    // FMA Horner (vmula/MulAddDst: 1 rounding per step instead of 2) — this
    // branch wins for |x| > ~2.4 (e.g. tanSum ∈ [-20,-4] cases) where the
    // Horner rounding noise dominated the thetaT cross_check error.
    AscendC::Reg::Muls(tmp, tmp2, TAN_PI_BY_8, mask);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, mask);
    AscendC::Reg::Adds(tmp2, tmp2, -TAN_PI_BY_8, mask);
    RbdDivIeee(tmp2, tmp2, tmp, mask); // correctly-rounded r4 (arg-chain accuracy)
    AscendC::Reg::Abs(tmp2, tmp2, mask);
    AscendC::Reg::Mul(denom, tmp2, tmp2, mask);
    AscendC::Reg::Duplicate<float>(taylor6, F8);
    AscendC::Reg::Duplicate<float>(tmp, F7);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F7 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F6);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F6 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(tmp, F5);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F5 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F4);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F4 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(tmp, F3);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F3 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F2);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F2 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(tmp, F1);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F1 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F0);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F0 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Mul(taylor6, taylor6, tmp2, mask);
    // Single fused constant 3π/8 (one rounding) instead of +π/8 then +π/4 (two
    // roundings) — saves ~0.5 ULP on the branch-4 result.
    AscendC::Reg::Adds(taylor6, taylor6, THREE_PI_BY_8, mask);
    AscendC::Reg::Min(taylor4, taylor4, taylor6, mask);

    // --- Asymptotic path for large |x| (> 100): atan(x) = sign(x)·(π/2 - atan(1/|x|)) ---
    // For |x| > 100, 1/|x| < 0.01, so the 7-term Taylor on 1/|x| converges to
    // < 1e-20 truncation error — far below fp32 ULP. This is dramatically more
    // accurate than the clip-to-10000 path (which loses ~5e-5 rad at |x|=10000).
    // Naturally handles inf: 1/inf = 0, atan(0) = 0, π/2 - 0 = π/2 (matches
    // torch.arctan(±inf) = ±π/2 exactly). NaN: |NaN| > 100 is false, so NaN
    // falls through to the normal path and propagates as NaN.
    // Registers reused (all free after Min selection): absRegClipped, denom, x2, tmp.
    constexpr float ASYM_THRESHOLD = 100.0f;
    constexpr float PI_2_VAL = 1.57079632679489661923132169164f;
    AscendC::Reg::Abs(absRegClipped, src, mask);
    AscendC::Reg::MaskReg largeMask;
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(largeMask, absRegClipped, ASYM_THRESHOLD, mask);
    AscendC::Reg::Duplicate<float>(taylor6, 1.0f);
    AscendC::Reg::Div(denom, taylor6, absRegClipped, mask);
    AscendC::Reg::Mul(x2, denom, denom, mask);
    AscendC::Reg::Duplicate<float>(absRegClipped, F6);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F0, mask);
    AscendC::Reg::Mul(absRegClipped, absRegClipped, denom, mask);
    AscendC::Reg::Duplicate<float>(tmp, PI_2_VAL);
    AscendC::Reg::Sub(absRegClipped, tmp, absRegClipped, mask);
    AscendC::Reg::Select<float>(taylor4, absRegClipped, taylor4, largeMask);

    // --- Apply sign: dst = sign(clipReg) · atan(|clipReg|) ---
    constexpr float SIGN_EPS = 2.168404344971009e-19f; // 2^-62
    AscendC::Reg::Adds(signReg, absReg, SIGN_EPS, mask);
    AscendC::Reg::Div(signReg, clipReg, signReg, mask);
    AscendC::Reg::Mul(dst, taylor4, signReg, mask);
}

// TanSleefFast — bit-exact SLEEF xtanf_u1 fast-path replica (the function
// exported as Sleef_tanf4_u10advsimd in pytorch's vendored SLEEF, which
// torch.tan dispatches to on the aarch64 golden/third-party host; verified
// 44960/44960 bit-exact over the enumerable fp16 grid).
//
// Motivation: on fp16 cases the golden and the third party compute the same
// torch.tan, so err_third ≡ 0 and the cross_check small-value denominator is
// floored at 1 — the kernel must be BIT-EXACT against torch.tan. At
// |tan(θ)| ≥ 4 any 1-ULP difference exceeds the 2^-16 output atol after the
// ×57.3 rad→deg scaling (L1_482 small-value θT: dpt ∈ [−10,−2] forces the
// cancellation band tan(θa) ≈ −dpt ∈ [2,10]); the previous cot-dbl path is
// 1-ULP-off SLEEF at ~37% of that band (measured), giving err_target ≈ 8.
//
// Algorithm (SLEEF xtanf_u1 DETERMINISTIC fast path, pytorch pin 56e1f79;
// valid |d| < 125 rad — SLEEF's TRIGRANGEMAX2f gate):
//   q  = rint(d·2/π)
//   v  = fma(q, −π/2.A, d)
//   s  = dfadd2(v, q·(−π/2.B)); s = dfadd(s, q·(−π/2.C))     double-single
//   (sh,sl) sign-flipped when q is odd; t = s
//   s₂ = dfnormalize(dfsqu(s))
//   u  = 4-step fused Horner on s₂.hi
//   x  = dfadd(A, u·s₂.hi)
//   x  = dfadd(1, dfmul(dfadd(B, dfmul(s₂, x)), s₂))
//   x  = dfmul(t, x)
//   x  = odd ? dfrec(x) : x          (dfrec: 1/xh IEEE-div + 2-step residual)
//   out = xh + xl;  d == −0 → d
__simd_callee__ inline void TanSleefFast(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& src,
                                         AscendC::Reg::RegTensor<float>& zReg, AscendC::Reg::MaskReg& mask)
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

    AscendC::Reg::RegTensor<float> q, v, sh, sl, s2h, s2l, u, c, t1, t2, p, xh, xl, ih, il;
    AscendC::Reg::MaskReg oddMask, negzMask;

    // q = rint(d·2/π)
    AscendC::Reg::Muls(q, src, RBD_2RPI, mask);
    AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT, AscendC::Reg::MaskMergeMode::ZEROING>(q, q, mask);

    // v = fma(q, −π/2.A, d)
    AscendC::Reg::Adds(c, zReg, RBD_PA2H, mask);
    v = src;
    AscendC::Reg::MulAddDst(v, q, c, mask);

    // s = dfadd2(v, fl(q·(−π/2.B))): s = v+y; t = s−v; lo = (v−(s−t)) + (y−t)
    AscendC::Reg::Muls(p, q, RBD_PB2H, mask);
    AscendC::Reg::Add(sh, v, p, mask);
    AscendC::Reg::Sub(t1, sh, v, mask);
    AscendC::Reg::Sub(t2, sh, t1, mask);
    AscendC::Reg::Sub(v, v, t2, mask);
    AscendC::Reg::Sub(p, p, t1, mask);
    AscendC::Reg::Add(sl, v, p, mask);

    // s = dfadd(s, fl(q·(−π/2.C))): s2 = sh+y; lo = ((sh−s2)+y) + sl
    AscendC::Reg::Muls(p, q, RBD_PC2H, mask);
    AscendC::Reg::Add(t1, sh, p, mask);
    AscendC::Reg::Sub(t2, sh, t1, mask);
    AscendC::Reg::Add(t2, t2, p, mask);
    AscendC::Reg::Add(sl, t2, sl, mask);
    sh = t1;

    // odd = (floor(q/2) ≠ q/2) — exact for the integer-valued q
    AscendC::Reg::Muls(t1, q, 0.5f, mask);
    AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_FLOOR, AscendC::Reg::MaskMergeMode::ZEROING>(t2, t1, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(oddMask, t1, t2, mask);

    // NOTE: the odd-q sign flip is DEFERRED to the final dfmul (see below) —
    // the squared s₂ is sign-invariant, so keeping (sh, sl) unflipped through
    // the poly avoids saving a second ds pair (VF slot control). Bit-exact:
    // fp32 negation is an exact sign-bit flip, so dfmul(−s, x) = −dfmul(s, x)
    // and dfrec(−z) = −dfrec(z).

    // s₂ = dfnormalize(dfsqu(s)): sq = fl(sh²); lo = fma(2sh, sl, fl(sh²−sq))
    AscendC::Reg::Mul(s2h, sh, sh, mask);
    AscendC::Reg::Muls(t1, s2h, -1.0f, mask);
    AscendC::Reg::MulAddDst(t1, sh, sh, mask); // sh·sh − sq (fmapn)
    AscendC::Reg::Add(t2, sh, sh, mask);
    AscendC::Reg::MulAddDst(t1, t2, sl, mask);
    s2l = t1;
    // normalize: s = s2h+s2l; lo = (s2h−s)+s2l
    AscendC::Reg::Add(t1, s2h, s2l, mask);
    AscendC::Reg::Sub(t2, s2h, t1, mask);
    AscendC::Reg::Add(s2l, t2, s2l, mask);
    s2h = t1;

    // u = 4-step fused Horner on s₂.hi
    AscendC::Reg::Adds(u, zReg, RBD_TP0, mask);
    AscendC::Reg::Adds(c, zReg, RBD_TP1, mask);
    AscendC::Reg::MulAddDst(c, u, s2h, mask);
    u = c;
    AscendC::Reg::Adds(c, zReg, RBD_TP2, mask);
    AscendC::Reg::MulAddDst(c, u, s2h, mask);
    u = c;
    AscendC::Reg::Adds(c, zReg, RBD_TP3, mask);
    AscendC::Reg::MulAddDst(c, u, s2h, mask);
    u = c;
    AscendC::Reg::Adds(c, zReg, RBD_TP4, mask);
    AscendC::Reg::MulAddDst(c, u, s2h, mask);
    u = c;

    // x = dfadd(A, fl(u·s₂.hi)): s = A+p; lo = (A−s)+p
    AscendC::Reg::Mul(p, u, s2h, mask);
    AscendC::Reg::Adds(t1, p, RBD_TA, mask);
    AscendC::Reg::Adds(c, zReg, RBD_TA, mask);
    AscendC::Reg::Sub(t2, c, t1, mask);
    AscendC::Reg::Add(xl, t2, p, mask);
    xh = t1;

    // i1 = dfmul(s₂, x): s = fl(s2h·xh); lo = fma(s2h, xl, fma(s2l, xh, fl(s2h·xh−s)))
    AscendC::Reg::Mul(ih, s2h, xh, mask);
    AscendC::Reg::Muls(t1, ih, -1.0f, mask);
    AscendC::Reg::MulAddDst(t1, s2h, xh, mask);
    AscendC::Reg::MulAddDst(t1, s2l, xh, mask);
    AscendC::Reg::MulAddDst(t1, s2h, xl, mask);
    il = t1;

    // i2 = dfadd(B, i1): s = B+ih; v = s−B; lo = ((B−(s−v)) + (ih−v)) + il
    AscendC::Reg::Adds(t1, ih, RBD_TB, mask);
    AscendC::Reg::Adds(c, zReg, RBD_TB, mask);
    AscendC::Reg::Sub(t2, t1, c, mask); // v
    AscendC::Reg::Sub(c, t1, t2, mask); // s−v
    AscendC::Reg::Adds(p, zReg, RBD_TB, mask);
    AscendC::Reg::Sub(p, p, c, mask);   // B−(s−v)
    AscendC::Reg::Sub(c, ih, t2, mask); // ih−v
    AscendC::Reg::Add(p, p, c, mask);
    AscendC::Reg::Add(il, p, il, mask);
    ih = t1;

    // i3 = dfmul(i2, s₂)  (into xh/xl — i1's registers are dead)
    AscendC::Reg::Mul(xh, ih, s2h, mask);
    AscendC::Reg::Muls(t1, xh, -1.0f, mask);
    AscendC::Reg::MulAddDst(t1, ih, s2h, mask);
    AscendC::Reg::MulAddDst(t1, il, s2h, mask);
    AscendC::Reg::MulAddDst(t1, ih, s2l, mask);
    xl = t1;

    // x = dfadd(1, i3): s = 1 + xh (i3.hi lives in xh — i3 was written into
    // (xh, xl) above; ih still holds i2.hi — reading it here was the L1_482
    // regression: x became 1+TB ≈ 4/3 instead of 1, scaling every tan by
    // ~4/3 and tripping all small-θT cancellation lanes); lo = ((1−(s−v)) +
    // (xh−v)) + xl (i3.lo)
    AscendC::Reg::Adds(t1, xh, 1.0f, mask);
    AscendC::Reg::Adds(t2, zReg, 1.0f, mask);
    AscendC::Reg::Sub(t2, t1, t2, mask); // v
    AscendC::Reg::Sub(c, t1, t2, mask);  // s−v
    AscendC::Reg::Adds(p, zReg, 1.0f, mask);
    AscendC::Reg::Sub(p, p, c, mask);   // 1−(s−v)
    AscendC::Reg::Sub(c, xh, t2, mask); // xh−v (i3.hi)
    AscendC::Reg::Add(p, p, c, mask);
    AscendC::Reg::Add(xl, p, xl, mask);
    xh = t1;

    // deferred sign flip: (sh, sl) := odd ? −(sh, sl) : (sh, sl), in place
    AscendC::Reg::Muls(t1, sh, -1.0f, mask);
    AscendC::Reg::Select<float>(sh, t1, sh, oddMask);
    AscendC::Reg::Muls(t1, sl, -1.0f, mask);
    AscendC::Reg::Select<float>(sl, t1, sl, oddMask);

    // x = dfmul(s, x): s = fl(sh·xh); lo = fma(sh, xl, fma(sl, xh, fl(sh·xh−s)))
    AscendC::Reg::Mul(t1, sh, xh, mask);
    AscendC::Reg::Muls(t2, t1, -1.0f, mask);
    AscendC::Reg::MulAddDst(t2, sh, xh, mask);
    AscendC::Reg::MulAddDst(t2, sl, xh, mask);
    AscendC::Reg::MulAddDst(t2, sh, xl, mask);
    xh = t1;
    xl = t2;

    // odd → x = dfrec(x): s = 1/xh (IEEE); e1 = 1−xh·s; e2 = e1−xl·s; lo = s·e2
    AscendC::Reg::Adds(c, zReg, 1.0f, mask);
    RbdDivIeee(t1, c, xh, mask); // t1 = 1/xh
    AscendC::Reg::Muls(t2, xh, -1.0f, mask);
    AscendC::Reg::Adds(c, zReg, 1.0f, mask);
    AscendC::Reg::MulAddDst(c, t2, t1, mask); // c = 1 − xh·s
    AscendC::Reg::Muls(t2, xl, -1.0f, mask);
    AscendC::Reg::MulAddDst(c, t2, t1, mask); // c = e2
    AscendC::Reg::Mul(t2, t1, c, mask);       // lo = s·e2
    AscendC::Reg::Select<float>(xh, t1, xh, oddMask);
    AscendC::Reg::Select<float>(xl, t2, xl, oddMask);

    // out = xh + xl. (The odd-lane sign was ALREADY applied by the in-place
    // flip above — do NOT negate here: flip-before-multiply + this sum equals
    // SLEEF's early-flip structure exactly. A stray extra negation here
    // double-inverted every odd-q lane — L1_482 board regression, ratio 11-17.)
    AscendC::Reg::Add(dst, xh, xl, mask);

    // d == −0.0 → d (bit compare — matches SLEEF's visnegzero select)
    AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(negzMask, (AscendC::Reg::RegTensor<uint32_t>&)src,
                                                                0x80000000u, mask);
    AscendC::Reg::Select<float>(dst, src, dst, negzMask);
}

// ===========================================================================
// RotatedBoxDecodeComputeVF — VF0-VF7 single __simd_vf__ fused compute chain
//   (DESIGN §10.9.1 / §5.2). f32 intermediates stay in RegTensor (NUM_CALC_BUFS=0).
//
// Inputs (UB channel-contiguous [5, boxCount] views, pointer-offset into slot):
//   anchorView / deltasView — ioIdx=0 / ioIdx=1 region bases
//   yView                   — ioIdx=2 region base (output)
// Scalar params:
//   invW0..invW4 — 1/weight[c] (precomputed in ordinary code; Muls scalar)
//   boxCount     — tile box count (VF repeat loop upper bound)
// ===========================================================================
template <typename T>
__simd_vf__ inline void RotatedBoxDecodeComputeVF(__ubuf__ T* yView, __ubuf__ T* anchorView, __ubuf__ T* deltasView,
                                                  float w0, float w1, float w2, float w3, float w4, int64_t boxCount,
                                                  int64_t ubFormer)
{
    // VL is the f32 vector length (8 on dav-c310). For b16, LoadDist::DIST_UNPACK_B16
    // loads 8 b16 elements per repeat and widens to 8 f32, so the repeat count and
    // mask are driven by the f32 VL for both dtype paths.
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = static_cast<uint32_t>(boxCount);
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + static_cast<int64_t>(VL) - 1) / static_cast<int64_t>(VL));
    // Use decimal literals so the compiler rounds directly to the correctly-
    // rounded fp32 constant — torch casts the double-precision constants to
    // fp32: angle * (math.pi/180) and rad * (180/math.pi).
    // `180.0f / 3.14159265f` computes to 0x42652ee0 (1 ULP BELOW the correct
    // 0x42652ee1), systematically biasing thetaT by -1 ULP vs torch.
    constexpr float RBD_DEG2RAD = 0.017453292519943295f; // fp32(pi/180) = 0x3c8efa35
    constexpr float RBD_RAD2DEG = 57.29577951308232f;    // fp32(180/pi) = 0x42652ee1

    for (uint16_t i = 0; i < repeatNum; ++i) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        // ===== VF0: Cast b16→f32 (fp32 path: direct load, no Cast) =====
        AscendC::Reg::RegTensor<float> aLx, aLy, aRx, aRy, aAng;
        AscendC::Reg::RegTensor<float> dDx, dDy, dDw, dDh, dDt;
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float>(aLx, anchorView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aLy, anchorView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aRx, anchorView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aRy, anchorView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aAng, anchorView + 4 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDx, deltasView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDy, deltasView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDw, deltasView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDh, deltasView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDt, deltasView + 4 * ubFormer + i * VL);
        } else {
            AscendC::Reg::RegTensor<T> nLx, nLy, nRx, nRy, nAng;
            AscendC::Reg::RegTensor<T> nDx, nDy, nDw, nDh, nDt;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nLx,
                                                                                anchorView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nLy,
                                                                                anchorView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nRx,
                                                                                anchorView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nRy,
                                                                                anchorView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nAng,
                                                                                anchorView + 4 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDx,
                                                                                deltasView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDy,
                                                                                deltasView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDw,
                                                                                deltasView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDh,
                                                                                deltasView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDt,
                                                                                deltasView + 4 * ubFormer + i * VL);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aLx, nLx, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aLy, nLy, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aRx, nRx, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aRy, nRy, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aAng, nAng, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDx, nDx, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDy, nDy, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDw, nDw, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDh, nDh, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDt, nDt, mask);
        }

        // ===== VF1: corner -> center (max(·,1) clamp) =====
        AscendC::Reg::RegTensor<float> aW, aH, aCx, aCy, halfW, halfH;
        AscendC::Reg::Sub(aW, aRx, aLx, mask);
        AscendC::Reg::Maxs(aW, aW, 1.0f, mask);
        AscendC::Reg::Sub(aH, aRy, aLy, mask);
        AscendC::Reg::Maxs(aH, aH, 1.0f, mask);
        AscendC::Reg::Muls(halfW, aW, 0.5f, mask);
        AscendC::Reg::Add(aCx, aLx, halfW, mask);
        AscendC::Reg::Muls(halfH, aH, 0.5f, mask);
        AscendC::Reg::Add(aCy, aLy, halfH, mask);

        // ===== VF2: delta normalization Δ' = deltas / weight =====
        // Use TRUE division (Duplicate weight → Div), not Muls by precomputed 1/weight,
        // to bit-match the golden's `deltas / weight` (torch true_divide). The
        // reciprocal-multiply form differs by ≤1 ULP which, on near-zero outputs
        // (catastrophic cancellation in out_lx = t_cx − t_w/2), amplifies the
        // stat_rel_err mare past the threshold even though mere stays at the fp32
        // precision floor.
        // Correctly-rounded division (RbdDivIeee) bit-matches the golden's
        // `deltas / weight` (torch true_divide). The default Reg::Div (vdiv) is
        // ≤1 ULP off; that argument error is AMPLIFIED by exp() by |Δ'w| (up to
        // ~50×) and propagated through tan→atan, blowing up cross_check
        // mare/mere/rmse ratios vs the third party (whose CUDA division is
        // IEEE-exact like the CPU golden).
        AscendC::Reg::RegTensor<float> dpX, dpY, dpW, dpH, dpT, wReg;
        AscendC::Reg::Duplicate<float>(wReg, w0);
        RbdDivIeee(dpX, dDx, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w1);
        RbdDivIeee(dpY, dDy, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w2);
        RbdDivIeee(dpW, dDw, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w3);
        RbdDivIeee(dpH, dDh, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w4);
        RbdDivIeee(dpT, dDt, wReg, mask);

        // ===== VF3: decode center (t_cx = a_cx + Δ'x·a_w) =====
        AscendC::Reg::RegTensor<float> tCx, tCy, tmp;
        AscendC::Reg::Mul(tmp, dpX, aW, mask);
        AscendC::Reg::Add(tCx, aCx, tmp, mask);
        AscendC::Reg::Mul(tmp, dpY, aH, mask);
        AscendC::Reg::Add(tCy, aCy, tmp, mask);

        // ===== VF4: decode w/h exp (t_w = exp(Δ'w)·a_w) =====
        // consume expW before computing expH — keeps the two RbdExpDD live
        // ranges from overlapping (VF spill control)
        AscendC::Reg::RegTensor<float> expW, expH, tW, tH;
        RbdExpDD(expW, dpW, mask);
        AscendC::Reg::Mul(tW, expW, aW, mask);
        RbdExpDD(expH, dpH, mask);
        AscendC::Reg::Mul(tH, expH, aH, mask);

        // ===== VF5: decode angle tan·atan (θ_t = atan(tan(θ_a)+Δ't), deg<->rad) =====
        // Extreme angle guard: golden.py marks |angle_deg| > 1e6 as NaN (spec.yaml
        // does not define behavior for such extreme angles). The SLEEF
        // replica's argument reduction loses precision beyond ~1144 rad
        // (≈ 65504°, the fp16 grid top); every |aAng| in (7162°, 1e6°) lane in
        // the suite was verified to have no small-θT cancellation exposure.
        // Match golden: |aAng| > 1e6 → NaN for thetaT.
        AscendC::Reg::RegTensor<float> thetaARad, tanA, tanSum, thetaTRad, thetaT;
        AscendC::Reg::RegTensor<float> nanVal, zReg5;
        AscendC::Reg::MaskReg extremeMask, extremeMaskNeg;
        constexpr float EXTREME_ANGLE_DEG = 1.0e6f;
        const float NAN_VAL = __builtin_nanf(""); // fp32 NaN (matches torch.nan)
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(extremeMask, aAng, EXTREME_ANGLE_DEG, mask);
        AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(extremeMaskNeg, aAng, -EXTREME_ANGLE_DEG, mask);
        AscendC::Reg::Or(extremeMask, extremeMask, extremeMaskNeg, mask);
        AscendC::Reg::Muls(thetaARad, aAng, RBD_DEG2RAD, mask);
        AscendC::Reg::Duplicate<float>(zReg5, 0.0f);
        TanSleefFast(tanA, thetaARad, zReg5, mask);
        AscendC::Reg::Add(tanSum, tanA, dpT, mask);
        AtanImpl(thetaTRad, tanSum, mask);
        AscendC::Reg::Muls(thetaT, thetaTRad, RBD_RAD2DEG, mask);
        // Where extreme: set NaN. Select picks NaN where extremeMask=true, thetaT where false.
        AscendC::Reg::Adds(nanVal, zReg5, NAN_VAL, mask);
        AscendC::Reg::Select<float>(thetaT, nanVal, thetaT, extremeMask);

        // ===== VF6: center -> corner =====
        AscendC::Reg::RegTensor<float> hTW, hTH, outLx, outLy, outRx, outRy;
        AscendC::Reg::Muls(hTW, tW, 0.5f, mask);
        AscendC::Reg::Muls(hTH, tH, 0.5f, mask);
        AscendC::Reg::Sub(outLx, tCx, hTW, mask);
        AscendC::Reg::Add(outRx, tCx, hTW, mask);
        AscendC::Reg::Sub(outLy, tCy, hTH, mask);
        AscendC::Reg::Add(outRy, tCy, hTH, mask);

        // ===== VF7: Cast f32→b16 + store 5 channels (fp32 path: no Cast) =====
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::StoreAlign<float>(yView + 0 * ubFormer + i * VL, outLx, mask);
            AscendC::Reg::StoreAlign<float>(yView + 1 * ubFormer + i * VL, outLy, mask);
            AscendC::Reg::StoreAlign<float>(yView + 2 * ubFormer + i * VL, outRx, mask);
            AscendC::Reg::StoreAlign<float>(yView + 3 * ubFormer + i * VL, outRy, mask);
            AscendC::Reg::StoreAlign<float>(yView + 4 * ubFormer + i * VL, thetaT, mask);
        } else {
            AscendC::Reg::RegTensor<T> yLx, yLy, yRx, yRy, yAng;
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yLx, outLx, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yLy, outLy, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yRx, outRx, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yRy, outRy, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yAng, thetaT, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 0 * ubFormer + i * VL, yLx,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 1 * ubFormer + i * VL, yLy,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 2 * ubFormer + i * VL, yRx,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 3 * ubFormer + i * VL, yRy,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 4 * ubFormer + i * VL, yAng,
                                                                                mask);
        }
    }
}

// ===========================================================================
// RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>
//   Main kernel orchestrator. P=2 ping-pong three-phase pipeline.
// ===========================================================================
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
class RotatedBoxDecodeKernel {
public:
    __aicore__ inline void Init(GM_ADDR anchorBox, GM_ADDR deltas, GM_ADDR y, const RotatedBoxDecodeTilingData* td);
    __aicore__ inline void Process();

private:
    struct CoreRange {
        int64_t begin;
        int64_t end;
    };
    struct TileInfo {
        int64_t boxStart;
        int64_t boxCount;
    };

    // Max ubFormer-sized tiles a core can produce. Per DESIGN-BRANCH-0 §2.3 the
    // per-core tile count is CeilDiv(perCoreBoxes, ubFormer) — tiles span batch
    // boundaries when N < ubFormer (CopyIn/CopyOut segment loops handle the
    // per-batch 2D stride gather/scatter). Worst-case (largest blackbox shape
    // B≈10M, N=3, bf16) yields ~130 tiles/core; 1024 gives a 7× safety margin
    // while keeping the on-stack TileInfo array at 16 KB (AIV stack budget).
    static constexpr int64_t kMaxTilesPerCore = 1024;

    __aicore__ inline CoreRange GetCoreRange();
    __aicore__ inline TileInfo GetTile(int64_t k);
    __aicore__ inline TileInfo NextTile(int64_t pos, int64_t end);
    __aicore__ inline int64_t BuildBatchTiles(TileInfo* out, int64_t maxTiles);
    __aicore__ inline void DoCopyInTransposeSplit(int64_t boxStart, int64_t boxCount, int slot);
    __aicore__ inline void ComputeMain(int slot, int64_t boxCount);
    __aicore__ inline void DoCopyOutStackTranspose(int64_t boxStart, int64_t boxCount, int slot);

    const RotatedBoxDecodeTilingData* td_ = nullptr;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> gmIn_[kNumInputs]; // [0]=anchor, [1]=deltas
    AscendC::GlobalTensor<T> gmOut_;            // y
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];

    int64_t blockIdx_ = 0;
    int64_t ubFormer_ = 0;
    int64_t blockFormer_ = 0;
    int64_t coreBoxBegin_ = 0;
    int64_t coreBoxEnd_ = 0;
};

// ---------------------------------------------------------------------------
// Init — GM binding + TBuf allocation + tiling context (DESIGN §10.4)
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::Init(GM_ADDR anchorBox, GM_ADDR deltas,
                                                                               GM_ADDR y,
                                                                               const RotatedBoxDecodeTilingData* td)
{
    td_ = td;
    blockIdx_ = AscendC::GetBlockIdx();
    ubFormer_ = td->ubFactor;
    blockFormer_ = td->perCoreCount;

    // Empty tensor short-circuit: no GM binding / no buffer alloc (no-op kernel)
    if (td->totalCount == 0) {
        coreBoxBegin_ = 0;
        coreBoxEnd_ = 0;
        return;
    }

    int64_t totalElems = td->B * kChannels * td->N;
    gmIn_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchorBox), totalElems);
    gmIn_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(deltas), totalElems);
    gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalElems);

    int64_t perBufBytes = td->bufferSize / kPhysNodes;
    if (perBufBytes <= 0)
        perBufBytes = AscendC::ONE_BLK_SIZE; // defensive: never allocate 0-byte TBuf
    for (int i = 0; i < kPhysNodes; i++) {
        pipe_.InitBuffer(buf_[i], perBufBytes);
    }
}

// ---------------------------------------------------------------------------
// GetCoreRange — this core's index range [begin, end) (DESIGN §10.3)
//   Multi-core split via大小核均衡: blockNum = CeilDiv(totalCount, blockFormer);
//   cores_tail cores get tiles_main+1 blocks, rest get tiles_main.
//
//   N-axis (key=0): range is in BOX index space (totalCount = B×N boxes,
//     perCoreCount = box count per block).
//   B-axis (key=1): range is in BATCH index space (B batches,
//     perCoreCount = batch count per block, 512-batch aligned). Each batch = one
//     UB tile with boxCount = td->N (DESIGN-BRANCH-1.md §2.1 / §3.3).
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline typename RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::CoreRange
RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::GetCoreRange()
{
    int64_t numCores = AscendC::GetBlockNum();
    if (numCores <= 0)
        numCores = 1;

    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_B) {
        // B-axis: split along batch dimension (DESIGN-BRANCH-1.md §2.1)
        //   perCoreCount = blockFormer (batch count, 512-batch aligned)
        //   blockNum = CeilDiv(B, blockFormer); cores split blocks
        int64_t B = td_->B;
        int64_t blockFormer = td_->perCoreCount;
        if (blockFormer <= 0)
            blockFormer = 1;
        int64_t blockNum = AscendC::CeilDivision(B, blockFormer);
        int64_t tilesMain = blockNum / numCores;
        int64_t coresTail = blockNum % numCores;
        int64_t myBlocks = tilesMain + (blockIdx_ < coresTail ? 1 : 0);
        int64_t beginBlock = blockIdx_ * tilesMain + MinI64(blockIdx_, coresTail);
        int64_t begin = beginBlock * blockFormer;                // batch begin
        int64_t end = MinI64(begin + myBlocks * blockFormer, B); // batch end
        return {begin, end};
    } else {
        // N-axis: split along box index space (totalCount = B×N boxes)
        int64_t totalCount = td_->totalCount;
        int64_t blockFormer = td_->perCoreCount;
        if (blockFormer <= 0)
            blockFormer = 1;
        int64_t blockNum = AscendC::CeilDivision(totalCount, blockFormer);
        int64_t tilesMain = blockNum / numCores;
        int64_t coresTail = blockNum % numCores;
        int64_t myBlocks = tilesMain + (blockIdx_ < coresTail ? 1 : 0);
        int64_t beginBlock = blockIdx_ * tilesMain + MinI64(blockIdx_, coresTail);
        int64_t begin = beginBlock * blockFormer;
        int64_t end = MinI64(begin + myBlocks * blockFormer, totalCount);
        return {begin, end};
    }
}

template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline typename RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::TileInfo
RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::GetTile(int64_t k)
{
    int64_t bs = coreBoxBegin_ + k * ubFormer_;
    int64_t bc = MinI64(ubFormer_, coreBoxEnd_ - bs);
    return {bs, bc};
}

template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline typename RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::TileInfo
RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::NextTile(int64_t pos, int64_t end)
{
    int64_t N = td_->N;
    int64_t b = pos / N;
    int64_t n = pos % N;
    int64_t batchEnd = (b + 1) * N;
    int64_t cap = MinI64(batchEnd, end);
    int64_t bc = MinI64(ubFormer_, cap - pos);
    return {pos, bc};
}

// BuildBatchTiles — decompose a core's range into single-batch tiles.
//
// Tiles are capped by ubFormer_ AND by the current batch's tail, so each tile
// stays within one batch. This keeps the CopyIn/CopyOut 2D stride gather /
// scatter a SINGLE segment per tile (ubOff=0, UB destination 32B-aligned) —
// required because a multi-batch tile would accumulate a non-32B-aligned
// ubOff across segments and trigger an AICore 507035 exception.
//
// When N > ubFormer_ the cap collapses to ubFormer_ (partial-batch slice,
// still single-segment). When N ≤ ubFormer_ the cap is the batch tail (one
// batch per tile).
//
// N-axis (key=0): range [coreBoxBegin_, coreBoxEnd_) is in BOX index space
//   (totalCount = B×N boxes).
// B-axis (key=1): range is in BATCH index space (DESIGN-BRANCH-1.md §2.3);
//   converted to BOX index space here so both axes share the same tile
//   decomposition. gmOff for boxStart = batchIdx × N is computed by
//   DoCopyInTransposeSplit from boxStart.
//
// NOTE: Process() iterates tiles on-the-fly (no TileInfo array) to avoid the
// kMaxTilesPerCore cap — for small N + large B a core may need O(B/cores)
// single-batch tiles, far exceeding any fixed stack array. This helper is
// kept only for UT/inspection; Process does not call it.
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline int64_t RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::BuildBatchTiles(TileInfo* out,
                                                                                             int64_t maxTiles)
{
    int64_t N = td_->N;
    int64_t cnt = 0;

    // B-axis core range is in BATCH space → convert to BOX space.
    int64_t boxBegin, boxEnd;
    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_B) {
        boxBegin = coreBoxBegin_ * N;
        boxEnd = coreBoxEnd_ * N;
    } else {
        boxBegin = coreBoxBegin_;
        boxEnd = coreBoxEnd_;
    }

    int64_t pos = boxBegin;
    while (pos < boxEnd && cnt < maxTiles) {
        int64_t b = pos / N;
        int64_t n = pos % N;
        int64_t batchEnd = (b + 1) * N; // next batch boundary
        int64_t cap = MinI64(batchEnd, boxEnd);
        int64_t bc = MinI64(ubFormer_, cap - pos); // ≤ ubFormer, within batch
        out[cnt].boxStart = pos;
        out[cnt].boxCount = bc;
        pos += bc;
        cnt++;
    }
    return cnt;
}

// ---------------------------------------------------------------------------
// DoCopyInTransposeSplit — Phase 1 (MTE2): GM [B,5,N] -> UB slot [5, boxCount]
//   2D stride gather per batch segment: blockCount=5 channels, blockLen=seg elems,
//   srcStride=(N-seg) skip GM between channels, dstStride=(boxCount-seg) skip UB
//   between channels (places ch_c at base + c*boxCount).
//   Handles tiles spanning batch boundaries by decomposing into per-batch segments.
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::DoCopyInTransposeSplit(int64_t boxStart,
                                                                                                 int64_t boxCount,
                                                                                                 int slot)
{
    constexpr int64_t kElemBytes = sizeof(T);
    int64_t N = td_->N;
    int64_t b = boxStart / N;
    int64_t n = boxStart % N;
    int64_t remaining = boxCount;
    int64_t ubOff = 0;
    while (remaining > 0) {
        int64_t seg = MinI64(N - n, remaining);
        for (int s = 0; s < kNumInputs; s++) {
            // DataCopyPad: GM-side srcStride is in BYTES; UB-side dstStride is in
            // 32-BYTE UNITS (dav-c310). Channel stride in UB = ubFormer_ elements, so
            // dstStride field = (ubFormer_ - seg) * sizeof(T) / 32.
            AscendC::DataCopyExtParams params;
            params.blockCount = static_cast<uint16_t>(kChannels);
            params.blockLen = static_cast<uint32_t>(seg * kElemBytes);
            params.srcStride = (N - seg) * kElemBytes;                                 // GM: bytes
            params.dstStride = (ubFormer_ - seg) * kElemBytes / AscendC::ONE_BLK_SIZE; // UB: 32B units
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = ((seg * kElemBytes) % AscendC::ONE_BLK_SIZE) != 0;

            int64_t ioBaseElem = static_cast<int64_t>(s) * kChannels * ubFormer_;
            uint32_t ubOffset = static_cast<uint32_t>(ioBaseElem + ubOff);
            int64_t gmOffset = (b * kChannels) * N + n; // channel 0 start of this batch
            AscendC::GlobalTensor<T> gmSrc = gmIn_[s][gmOffset];
            AscendC::LocalTensor<T> ubDst = buf_[slot].template Get<T>()[ubOffset];
            AscendC::DataCopyPad(ubDst, gmSrc, params, padParams);
        }
        ubOff += seg;
        remaining -= seg;
        b++;
        n = 0;
    }
}

// ---------------------------------------------------------------------------
// ComputeMain — Phase 2 (V): asc_vf_call VF0-VF7.
//   The VF performs Δ' = deltas / weight via TRUE division (Duplicate weight →
//   Reg::Div) to bit-match the golden's `deltas / weight` (torch true_divide).
//   This avoids the ≤1 ULP error of reciprocal-multiply (Muls by 1/weight)
//   which amplifies on near-zero outputs (catastrophic cancellation).
//   The 5 weight floats are passed as VF scalars; verified safe (non-default
//   weight cases pass — the earlier "≥7 scalars corrupt" diagnosis was a
//   misattribution of the real kMaxTilesPerCore truncation bug).
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::ComputeMain(int slot, int64_t boxCount)
{
    __ubuf__ T* slotBase = reinterpret_cast<__ubuf__ T*>(buf_[slot].template Get<T>().GetPhyAddr());
    __ubuf__ T* anchorView = slotBase;                         // ioIdx=0
    __ubuf__ T* deltasView = slotBase + kChannels * ubFormer_; // ioIdx=1
    __ubuf__ T* yView = slotBase + 2 * kChannels * ubFormer_;  // ioIdx=2

    asc_vf_call<RotatedBoxDecodeComputeVF<T>>(yView, anchorView, deltasView, td_->weight[0], td_->weight[1],
                                              td_->weight[2], td_->weight[3], td_->weight[4], boxCount, ubFormer_);
}

// ---------------------------------------------------------------------------
// DoCopyOutStackTranspose — Phase 3 (MTE3): UB slot [5, boxCount] -> GM [B,5,N]
//   Reverse 2D stride scatter per batch segment. No Interleave needed: GM is
//   channel-major per batch, UB is channel-contiguous — direct per-channel copy.
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::DoCopyOutStackTranspose(int64_t boxStart,
                                                                                                  int64_t boxCount,
                                                                                                  int slot)
{
    constexpr int64_t kElemBytes = sizeof(T);
    int64_t N = td_->N;
    int64_t b = boxStart / N;
    int64_t n = boxStart % N;
    int64_t remaining = boxCount;
    int64_t ubOff = 0;
    while (remaining > 0) {
        int64_t seg = MinI64(N - n, remaining);
        // DataCopyPad UB->GM: UB-side srcStride in 32-BYTE UNITS; GM-side dstStride in BYTES.
        AscendC::DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(kChannels);
        params.blockLen = static_cast<uint32_t>(seg * kElemBytes);
        params.srcStride = (ubFormer_ - seg) * kElemBytes / AscendC::ONE_BLK_SIZE; // UB: 32B units
        params.dstStride = (N - seg) * kElemBytes;                                 // GM: bytes

        int64_t yBaseElem = 2 * kChannels * ubFormer_; // ioIdx=2
        uint32_t ubOffset = static_cast<uint32_t>(yBaseElem + ubOff);
        int64_t gmOffset = (b * kChannels) * N + n;
        AscendC::GlobalTensor<T> gmDst = gmOut_[gmOffset];
        AscendC::LocalTensor<T> ubSrc = buf_[slot].template Get<T>()[ubOffset];
        AscendC::DataCopyPad(gmDst, ubSrc, params);
        ubOff += seg;
        remaining -= seg;
        b++;
        n = 0;
    }
}

// ---------------------------------------------------------------------------
// Process — main kernel entry (DESIGN §10.5 / §10.6)
//   P=2 ping-pong three-phase pipeline over single-batch tiles.
//
//   N-axis (key=0): each tile is split at batch boundaries (BuildBatchTiles)
//   so the CopyIn 2D stride gather has one segment per tile (ubOff=0, UB dst
//   32B-aligned).
//
//   B-axis (key=1): each tile is one whole batch (boxCount = td->N, N维整批
//   load). Multi-core split is along B (batch dimension). The CopyIn/CopyOut
//   2D stride gather / scatter and VF compute chain are shared with key=0
//   (DESIGN-BRANCH-1.md §3 / §5: "VF 计算链、Sync、CopyIn/CopyOut 逻辑与
//   key=0 同构"). gmOff = batchIdx × N × kChannels (DESIGN §10.3 B-axis branch),
//   computed by DoCopyInTransposeSplit from boxStart = batchIdx × N.
//
//   Sync events (eventID = slot 0/1):
//     MTE2_V:  CopyIn -> Compute (same slot, per round)
//     V_MTE3:  Compute -> CopyOut (same slot, per round)
//     MTE3_MTE2: CopyOut(slot) -> next-round CopyIn(slot) (slot reuse, round≥2)
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::Process()
{
    blockIdx_ = AscendC::GetBlockIdx();
    ubFormer_ = td_->ubFactor;
    blockFormer_ = td_->perCoreCount;

    // Empty tensor no-op (totalCount==0 -> host set ubFactor=0)
    if (td_->totalCount == 0 || ubFormer_ <= 0) {
        return;
    }

    CoreRange r = GetCoreRange();
    coreBoxBegin_ = r.begin;
    coreBoxEnd_ = r.end;
    if (coreBoxEnd_ <= coreBoxBegin_)
        return;

    // B-axis core range is in BATCH space → convert to BOX space so both axes
    // share the same single-batch tile iteration below.
    int64_t N = td_->N;
    int64_t posBegin, posEnd;
    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_B) {
        posBegin = coreBoxBegin_ * N;
        posEnd = coreBoxEnd_ * N;
    } else {
        posBegin = coreBoxBegin_;
        posEnd = coreBoxEnd_;
    }
    if (posEnd <= posBegin)
        return;

    // Tile stepper: each tile is capped by ubFormer_ AND by the current
    // batch's tail (single-segment CopyIn/CopyOut, UB destination 32B-aligned).
    // Tiles are produced on-the-fly — no TileInfo array — so a core handling
    // small-N + large-B (O(B/cores) single-batch tiles) is not capped by any
    // fixed stack array (kMaxTilesPerCore). This was the root cause of 57/71
    // blackbox failures: BuildBatchTiles previously truncated at 512 tiles,
    // leaving the rest of the core's box range uncomputed (stale GM output).

    // ===== Prologue: round 0, slot 0 =====
    TileInfo tPrev = NextTile(posBegin, posEnd);
    DoCopyInTransposeSplit(tPrev.boxStart, tPrev.boxCount, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);

    int64_t pos = tPrev.boxStart + tPrev.boxCount;
    int64_t round = 1;

    // ===== Steady-state: P=2 ping-pong =====
    while (pos < posEnd) {
        int slot = static_cast<int>(round % kPhysNodes);
        int prevSlot = static_cast<int>((round - 1) % kPhysNodes);

        // Wait prev CopyIn -> Compute prev slot
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(prevSlot);
        ComputeMain(prevSlot, tPrev.boxCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(prevSlot);

        // Slot reuse guard: wait prev-prev CopyOut of this slot done
        if (round >= 2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
        }
        // CopyIn current slot
        TileInfo tCur = NextTile(pos, posEnd);
        DoCopyInTransposeSplit(tCur.boxStart, tCur.boxCount, slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);

        // Wait prev Compute -> CopyOut prev slot
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(prevSlot);
        DoCopyOutStackTranspose(tPrev.boxStart, tPrev.boxCount, prevSlot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(prevSlot);

        tPrev = tCur;
        pos = tCur.boxStart + tCur.boxCount;
        round++;
    }

    // ===== Epilogue: last round =====
    int lastSlot = static_cast<int>((round - 1) % kPhysNodes);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(lastSlot);
    ComputeMain(lastSlot, tPrev.boxCount);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(lastSlot);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(lastSlot);
    DoCopyOutStackTranspose(tPrev.boxStart, tPrev.boxCount, lastSlot);
    // No SetFlag<MTE3_MTE2> for the last round (no further CopyIn to guard)
}

} // namespace rbd_kernel
