/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_BOX_ENCODE_KERNEL_H
#define ROTATED_BOX_ENCODE_KERNEL_H

#include "kernel_operator.h"
#include "adv_api/math/tan.h"
#include "adv_api/math/sin.h"
#include "adv_api/math/cos.h"
#include "rotated_box_encode_tiling_data.h"

constexpr int32_t RBE_BOX_CHANNELS = 5; // spec.yaml shape[1]==5
constexpr float RBE_PI_OVER_180 = 3.14159265358979323846f / 180.0f;

// VF LoadAlign may read up to one vector (64 elements) past 5*ubStride on the
// last channel; every TBuf carries this many extra tail elements and the
// zero-init covers them so masked overread lanes always see zeros.
constexpr int32_t RBE_UB_GUARD_ELEMS = 64;

// Channel indices (x0, y0, x1, y1, θ_deg)
constexpr int32_t RBE_CH_X0 = 0;
constexpr int32_t RBE_CH_Y0 = 1;
constexpr int32_t RBE_CH_X1 = 2;
constexpr int32_t RBE_CH_Y1 = 3;
constexpr int32_t RBE_CH_ANG = 4;

// fp32-only __simd_vf__ compute functions shared by both dtype branches.
// Each operates on a [5, ubStride] channel-contiguous UB view; the per-channel
// stride equals ubStride. Callers pass anchor as `b1` and gt as `b2`.

// DxDwVF: dx (ch0) = (cx_g − cx_a) / w_a × wx, dw (ch2) = log1p((w_g − w_a) / w_a) × ww
__simd_vf__ inline void DxDwVF(__ubuf__ float* b3, // output [5, ubStride], writes ch0 & ch2
                               __ubuf__ float* b1, // anchor [5, ubStride], reads ch0 & ch2
                               __ubuf__ float* b2, // gt     [5, ubStride], reads ch0 & ch2
                               uint32_t boxCount,  // valid box count per channel (may be < ubStride)
                               uint32_t ubStride,  // channel stride (32B-aligned, >= boxCount)
                               float wx, float ww) // weight[0] (dx), weight[2] (dw)
{
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float); // 64 (fp32)
    uint32_t remaining = boxCount;
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + VL - 1) / VL);

    __ubuf__ float* ax0Ptr = b1 + 0 * ubStride;
    __ubuf__ float* ax2Ptr = b1 + 2 * ubStride;
    __ubuf__ float* gx0Ptr = b2 + 0 * ubStride;
    __ubuf__ float* gx2Ptr = b2 + 2 * ubStride;
    __ubuf__ float* dxPtr = b3 + 0 * ubStride;
    __ubuf__ float* dwPtr = b3 + 2 * ubStride;

    for (uint16_t i = 0; i < repeatNum; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::RegTensor<float> ax0, ax2, gx0, gx2;
        AscendC::Reg::LoadAlign(ax0, ax0Ptr + off);
        AscendC::Reg::LoadAlign(ax2, ax2Ptr + off);
        AscendC::Reg::LoadAlign(gx0, gx0Ptr + off);
        AscendC::Reg::LoadAlign(gx2, gx2Ptr + off);

        // w_a = Maxs(ax2 - ax0, 1.0)
        AscendC::Reg::RegTensor<float> wa;
        AscendC::Reg::Sub(wa, ax2, ax0, mask);
        AscendC::Reg::Maxs(wa, wa, 1.0f, mask);

        // w_g = Maxs(gx2 - gx0, 1.0)
        AscendC::Reg::RegTensor<float> wg;
        AscendC::Reg::Sub(wg, gx2, gx0, mask);
        AscendC::Reg::Maxs(wg, wg, 1.0f, mask);

        // cx_a = ax0 + w_a * 0.5
        AscendC::Reg::RegTensor<float> halfWa, cxa;
        AscendC::Reg::Muls(halfWa, wa, 0.5f, mask);
        AscendC::Reg::Add(cxa, ax0, halfWa, mask);

        // cx_g = gx0 + w_g * 0.5
        AscendC::Reg::RegTensor<float> halfWg, cxg;
        AscendC::Reg::Muls(halfWg, wg, 0.5f, mask);
        AscendC::Reg::Add(cxg, gx0, halfWg, mask);

        // dx = (cx_g - cx_a) / w_a * wx
        AscendC::Reg::RegTensor<float> dxdc, dxn, dx;
        AscendC::Reg::Sub(dxdc, cxg, cxa, mask);
        AscendC::Reg::Div(dxn, dxdc, wa, mask);
        AscendC::Reg::Muls(dx, dxn, wx, mask);
        AscendC::Reg::StoreAlign(dxPtr + off, dx, mask);

        // dw = Log1p((w_g - w_a) / w_a) * ww — log1p instead of ln(w_g)−ln(w_a)
        // avoids catastrophic cancellation when w_g ≈ w_a (near-zero dw is the
        // common case); w_g−w_a is exact by Sterbenz for w_g/w_a ∈ [0.5,2].
        // Taylor t−t²/2+t³/3 for |t| < 2^-6, hardware Ln(1+t) otherwise;
        // w_g,w_a ≥ 1.0 (Maxs clamp) ⇒ Ln input always positive.
        AscendC::Reg::RegTensor<float> t, tNeg, tAbs, poly, coef, onePlusT, lnVal, dwVal, dw;
        AscendC::Reg::Sub(t, wg, wa, mask);
        AscendC::Reg::Div(t, t, wa, mask);
        AscendC::Reg::Muls(tNeg, t, -1.0f, mask);
        AscendC::Reg::Max(tAbs, t, tNeg, mask); // |t|
        AscendC::Reg::MaskReg bigMask;
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(bigMask, tAbs, 0.015625f, mask);
        AscendC::Reg::Duplicate<float>(poly, 0.333333343267440796f); // 1/3
        AscendC::Reg::Duplicate<float>(coef, -0.5f);
        AscendC::Reg::FusedMulDstAdd<float>(poly, t, coef, mask);
        AscendC::Reg::Duplicate<float>(coef, 1.0f);
        AscendC::Reg::FusedMulDstAdd<float>(poly, t, coef, mask);
        AscendC::Reg::Mul(poly, poly, t, mask); // t − t²/2 + t³/3
        AscendC::Reg::Adds(onePlusT, t, 1.0f, mask);
        AscendC::Reg::Ln(lnVal, onePlusT, mask);
        AscendC::Reg::Select(dwVal, lnVal, poly, bigMask); // |t| ≥ 2^-6 ? Ln(1+t) : Taylor
        AscendC::Reg::Muls(dw, dwVal, ww, mask);
        AscendC::Reg::StoreAlign(dwPtr + off, dw, mask);
    }
}

// DyDhVF: dy (ch1) = (cy_g − cy_a) / h_a × wy, dh (ch3) = log1p((h_g − h_a) / h_a) × wh
__simd_vf__ inline void DyDhVF(__ubuf__ float* b3, __ubuf__ float* b1, __ubuf__ float* b2, uint32_t boxCount,
                               uint32_t ubStride, float wy, float wh)
{
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = boxCount;
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + VL - 1) / VL);

    __ubuf__ float* ay0Ptr = b1 + 1 * ubStride;
    __ubuf__ float* ay1Ptr = b1 + 3 * ubStride;
    __ubuf__ float* gy0Ptr = b2 + 1 * ubStride;
    __ubuf__ float* gy1Ptr = b2 + 3 * ubStride;
    __ubuf__ float* dyPtr = b3 + 1 * ubStride;
    __ubuf__ float* dhPtr = b3 + 3 * ubStride;

    for (uint16_t i = 0; i < repeatNum; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::RegTensor<float> ay0, ay1, gy0, gy1;
        AscendC::Reg::LoadAlign(ay0, ay0Ptr + off);
        AscendC::Reg::LoadAlign(ay1, ay1Ptr + off);
        AscendC::Reg::LoadAlign(gy0, gy0Ptr + off);
        AscendC::Reg::LoadAlign(gy1, gy1Ptr + off);

        // h_a = Maxs(ay1 - ay0, 1.0)
        AscendC::Reg::RegTensor<float> ha;
        AscendC::Reg::Sub(ha, ay1, ay0, mask);
        AscendC::Reg::Maxs(ha, ha, 1.0f, mask);

        // h_g = Maxs(gy1 - gy0, 1.0)
        AscendC::Reg::RegTensor<float> hg;
        AscendC::Reg::Sub(hg, gy1, gy0, mask);
        AscendC::Reg::Maxs(hg, hg, 1.0f, mask);

        // cy_a = ay0 + h_a * 0.5
        AscendC::Reg::RegTensor<float> halfHa, cya;
        AscendC::Reg::Muls(halfHa, ha, 0.5f, mask);
        AscendC::Reg::Add(cya, ay0, halfHa, mask);

        // cy_g = gy0 + h_g * 0.5
        AscendC::Reg::RegTensor<float> halfHg, cyg;
        AscendC::Reg::Muls(halfHg, hg, 0.5f, mask);
        AscendC::Reg::Add(cyg, gy0, halfHg, mask);

        // dy = (cy_g - cy_a) / h_a * wy
        AscendC::Reg::RegTensor<float> dydc, dyn, dy;
        AscendC::Reg::Sub(dydc, cyg, cya, mask);
        AscendC::Reg::Div(dyn, dydc, ha, mask);
        AscendC::Reg::Muls(dy, dyn, wy, mask);
        AscendC::Reg::StoreAlign(dyPtr + off, dy, mask);

        // dh = Log1p((h_g - h_a) / h_a) * wh — mirror of DxDwVF dw (log1p rationale)
        AscendC::Reg::RegTensor<float> t, tNeg, tAbs, poly, coef, onePlusT, lnVal, dhVal, dh;
        AscendC::Reg::Sub(t, hg, ha, mask);
        AscendC::Reg::Div(t, t, ha, mask);
        AscendC::Reg::Muls(tNeg, t, -1.0f, mask);
        AscendC::Reg::Max(tAbs, t, tNeg, mask); // |t|
        AscendC::Reg::MaskReg bigMask;
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(bigMask, tAbs, 0.015625f, mask);
        AscendC::Reg::Duplicate<float>(poly, 0.333333343267440796f); // 1/3
        AscendC::Reg::Duplicate<float>(coef, -0.5f);
        AscendC::Reg::FusedMulDstAdd<float>(poly, t, coef, mask);
        AscendC::Reg::Duplicate<float>(coef, 1.0f);
        AscendC::Reg::FusedMulDstAdd<float>(poly, t, coef, mask);
        AscendC::Reg::Mul(poly, poly, t, mask); // t − t²/2 + t³/3
        AscendC::Reg::Adds(onePlusT, t, 1.0f, mask);
        AscendC::Reg::Ln(lnVal, onePlusT, mask);
        AscendC::Reg::Select(dhVal, lnVal, poly, bigMask); // |t| ≥ 2^-6 ? Ln(1+t) : Taylor
        AscendC::Reg::Muls(dh, dhVal, wh, mask);
        AscendC::Reg::StoreAlign(dhPtr + off, dh, mask);
    }
}

// DThetaVF: dθ (ch4) = dθ_prewa × wa, in place. dθ_prewa is precomputed
// OUTSIDE the VF (S3/S5) as sin(θ_g−θ_a)/(cos(θ_g)×cos(θ_a)) — the tan
// difference identity, avoiding catastrophic cancellation near the 90° tan
// singularity. b1/b2 are unused, kept for call-site uniformity with
// DxDwVF/DyDhVF (which never read ch4, so the S3/S5 ch4 precompute is safe).
__simd_vf__ inline void DThetaVF(__ubuf__ float* b3, // output [5, ubStride], reads & writes ch4
                                 __ubuf__ float* b1, // anchor (UNUSED — signature compat with DxDwVF/DyDhVF)
                                 __ubuf__ float* b2, // gt     (UNUSED — signature compat)
                                 uint32_t boxCount,  // valid box count per channel (may be < ubStride)
                                 uint32_t ubStride,  // channel stride (32B-aligned, >= boxCount)
                                 float wa)           // weight[4] (dθ scalar multiplier)
{
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = boxCount;
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + VL - 1) / VL);

    // dθ_prewa lives at B_out.ch4 (offset 4*ubStride); multiply by wa in place.
    __ubuf__ float* dthetaPrewaPtr = b3 + 4 * ubStride;

    for (uint16_t i = 0; i < repeatNum; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::RegTensor<float> prewa;
        AscendC::Reg::LoadAlign(prewa, dthetaPrewaPtr + off);
        AscendC::Reg::RegTensor<float> dtheta;
        AscendC::Reg::Muls(dtheta, prewa, wa, mask);
        AscendC::Reg::StoreAlign(dthetaPrewaPtr + off, dtheta, mask);
    }
}

// KernelRotatedBoxEncode — arch35 / ascend950, both dtype branches.
// DTYPE selects the pipeline: FP16 → fp16-upcast with Cast staging (S1–S8),
// FP32 → fp32-direct without Cast (S1–S5). Both share the same Process()
// outer-loop skeleton and the same three __simd_vf__ VF bodies; selection is
// via `if constexpr` inside Init / ProcessTile.
template <int DTYPE>
class KernelRotatedBoxEncode {
public:
    __aicore__ inline KernelRotatedBoxEncode() {}

    __aicore__ inline void Init(GM_ADDR anchor_box, GM_ADDR gt_box, GM_ADDR y, const RotatedBoxEncodeTilingData& td)
    {
        // Defensive empty-tensor short-circuit: safe to call with an empty TilingData.
        if (td.dim0 == 0) {
            empty_ = true;
            return;
        }
        empty_ = false;
        td_ = &td;
        N_ = td.N;
        B_ = td.dim0 / N_; // host guarantees dim0 == B*N exactly

        // Aggregated tiles use the padded slot layout: channel stride per tile
        // is K*A elements (K = ubFormer/N batches, A = slotElems), which may
        // exceed ubFormer; buffers are sized to that stride. Non-aggregated
        // tiles keep the plain [5, ubFormer] layout.
        int64_t ubSlot = td.ubFormer;
        if (td.slotElems > 0 && td.N > 0) {
            ubSlot = (td.ubFormer / td.N) * td.slotElems;
        }

        // GM layout: [B, 5, N] row-major → element(b,c,n) at b*5N + c*N + n.
        // Every TBuf carries RBE_UB_GUARD_ELEMS of tail slack (see its
        // definition); the host tiling reserves the matching guard bytes when
        // computing maxBoxNum.
        if constexpr (DTYPE == ROTATED_BOX_ENCODE_DTYPE_FP16) {
            anchorGm_.SetGlobalBuffer((__gm__ half*)anchor_box);
            gtGm_.SetGlobalBuffer((__gm__ half*)gt_box);
            outGm_.SetGlobalBuffer((__gm__ half*)y);

            // All 4 TBuf sized to the full tile (ubSlot boxes × 5 channels).
            uint32_t tileBytesB0 = static_cast<uint32_t>((ubSlot * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS) *
                                                         sizeof(half));
            uint32_t tileBytesB1 = static_cast<uint32_t>((ubSlot * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS) *
                                                         sizeof(float));
            pipe_.InitBuffer(b0Buf_, tileBytesB0); // B0 fp16 IO staging
            pipe_.InitBuffer(b1Buf_, tileBytesB1); // B1 fp32 anchor (Cast target)
            pipe_.InitBuffer(b2Buf_, tileBytesB1); // B2 fp32 gt     (Cast target)
            pipe_.InitBuffer(b3Buf_, tileBytesB1); // B3 fp32 output (VF result)
        } else {
            // Branch-1 fp32-direct: 3 fp32 TBuf reusing the same handles
            // (TBuf is dtype-agnostic; dtype set by Get<float> at use site);
            // b3Buf_ is unused — no Cast staging buffer on this path.
            anchorGmF32_.SetGlobalBuffer((__gm__ float*)anchor_box);
            gtGmF32_.SetGlobalBuffer((__gm__ float*)gt_box);
            outGmF32_.SetGlobalBuffer((__gm__ float*)y);

            uint32_t tileBytesFp32 = static_cast<uint32_t>((ubSlot * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS) *
                                                           sizeof(float));
            pipe_.InitBuffer(b0Buf_, tileBytesFp32); // B0 fp32 anchor (CopyIn direct)
            pipe_.InitBuffer(b1Buf_, tileBytesFp32); // B1 fp32 gt     (CopyIn direct)
            pipe_.InitBuffer(b2Buf_, tileBytesFp32); // B2 fp32 output (VF result)
        }
    }

    __aicore__ inline void Process()
    {
        if (empty_) {
            return; // empty-tensor short-circuit
        }

        int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (blockIdx >= td_->blockNum) {
            return; // over-allocated cores: skip
        }

        bool isLastBlock = (blockIdx == td_->blockNum - 1);
        int64_t loopNum = isLastBlock ? td_->ubLoopOfTailBlock : td_->ubLoopOfFormerBlock;
        int64_t tailNum = isLastBlock ? td_->ubTailOfTailBlock : td_->ubTailOfFormerBlock;

        int64_t boxOffset = blockIdx * td_->blockFormer;

        // (loopNum - 1) full ubFormer tiles + 1 tail tile.
        for (int64_t i = 0; i < loopNum - 1; ++i) {
            ProcessTileRange(boxOffset, td_->ubFormer);
            boxOffset += td_->ubFormer;
        }
        ProcessTileRange(boxOffset, tailNum);
    }

    // Split a tile range at batch boundaries so every ProcessTile covers boxes
    // of ONE batch only — CopyIn/CopyOut then issue a single-segment 2D stride
    // DataCopyPad whose UB base stays 32B-aligned (a running non-32B-aligned
    // UB offset would trigger an AICore 507035 fault).
    // Fast path: with the aggregated slot layout (slotElems > 0, host tiling
    // aligns ubFormer/blockFormer to N), ProcessTile covers K = boxCount/N
    // batches at once and the per-channel 2D copies gather all batches in one
    // shot. This avoids one full pipeline pass (zero-init + barriers +
    // precompute + VF) per batch, which collapses throughput for
    // large-B/small-N shapes.
    __aicore__ inline void ProcessTileRange(int64_t boxOffset, int64_t boxCount)
    {
        if (td_->slotElems > 0 && boxCount > 0 && N_ > 0 && (boxOffset % N_) == 0 && (boxCount % N_) == 0) {
            ProcessTile(boxOffset, boxCount);
            return;
        }
        while (boxCount > 0) {
            int64_t nStart = boxOffset % N_;
            int64_t seg = N_ - nStart; // boxes left in this batch
            if (seg > boxCount) {
                seg = boxCount;
            }
            ProcessTile(boxOffset, seg);
            boxOffset += seg;
            boxCount -= seg;
        }
    }

private:
    // Dispatch to the per-branch tile pipeline:
    //   fp16: CopyIn → Cast fp16→fp32 → precompute → VF → Cast fp32→fp16 → CopyOut
    //   fp32: CopyIn → precompute → VF → CopyOut (no Cast)
    __aicore__ inline void ProcessTile(int64_t boxOffset, int64_t boxCount)
    {
        if constexpr (DTYPE == ROTATED_BOX_ENCODE_DTYPE_FP16) {
            ProcessTileFp16(boxOffset, boxCount);
        } else {
            ProcessTileFp32(boxOffset, boxCount);
        }
    }

    // Branch-0 fp16-upcast tile pipeline (S1–S8).
    __aicore__ inline void ProcessTileFp16(int64_t boxOffset, int64_t boxCount)
    {
        // Aggregated tile (slot layout): boxCount spans K whole batches; each
        // (batch, channel) segment occupies A = slotElems elements, so the
        // channel stride is K*A and the A-N pad lanes are zero (inert for the
        // encoding math: an all-zero box encodes to all-zero deltas). Non-
        // aggregated tiles keep the plain ceilAlign(boxCount, 16) stride.
        bool agg = (td_->slotElems > 0) && ((boxOffset % N_) == 0) && ((boxCount % N_) == 0);
        int64_t ubStride = agg ? (boxCount / N_) * td_->slotElems :
                                 (boxCount + 15) & ~static_cast<int64_t>(15); // ceilAlign(boxCount, 16)
        int64_t cnt64 = agg ? ubStride : boxCount; // vector op count (pads included when aggregated)

        // Zero-init B1/B2/B3 so Tan/VF never read uninitialized data beyond
        // boxCount (507035 faults from Div-by-zero/overflow); the valid region
        // is overwritten by Cast/VF. The length also covers the guard tail:
        // VF LoadAlign reads up to 5*ubStride+RBE_UB_GUARD_ELEMS, and masked
        // overread lanes must see zeros on tail tiles (previous-tile residue
        // would otherwise reach the Div/Sin/Cos pipeline). In the aggregated
        // layout B0 is zeroed too: the aggregated vector count spans the pad
        // lanes, so B0 pads must be zero before Cast even if the MTE
        // pad-fill behaviour ever changes; per-batch (non-agg) tiles leave B0
        // alone as before.
        {
            if (agg) {
                auto b0Local = b0Buf_.Get<half>();
                AscendC::Duplicate<half>(b0Local, static_cast<half>(0),
                                         static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
            }
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            auto b3Local = b3Buf_.Get<float>();
            AscendC::Duplicate<float>(b1Local, (float)0,
                                      static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
            AscendC::Duplicate<float>(b2Local, (float)0,
                                      static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
            AscendC::Duplicate<float>(b3Local, (float)0,
                                      static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S1 CopyIn anchor (MTE2) → B0 fp16 [5, ubStride] =====
        if (agg) {
            CopyInAgg<half>(anchorGm_, b0Buf_, boxOffset, boxCount, ubStride);
        } else {
            CopyInPerChannel(anchorGm_, boxOffset, boxCount, ubStride);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S2 Cast fp16→fp32 anchor (V, ordinary) → B1 [5, ubStride] =====
        // Cast ubStride*5 elements (includes padding zeros between channels).
        {
            auto b0Local = b0Buf_.Get<half>();
            auto b1Local = b1Buf_.Get<float>();
            AscendC::Cast<float, half>(b1Local, b0Local, AscendC::RoundMode::CAST_NONE,
                                       static_cast<uint32_t>(ubStride * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S3 CopyIn gt (MTE2) → B0 fp16 (reuse) =====
        if (agg) {
            CopyInAgg<half>(gtGm_, b0Buf_, boxOffset, boxCount, ubStride);
        } else {
            CopyInPerChannel(gtGm_, boxOffset, boxCount, ubStride);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S4 Cast fp16→fp32 gt (V, ordinary) → B2 [5, ubStride] =====
        {
            auto b0Local = b0Buf_.Get<half>();
            auto b2Local = b2Buf_.Get<float>();
            AscendC::Cast<float, half>(b2Local, b0Local, AscendC::RoundMode::CAST_NONE,
                                       static_cast<uint32_t>(ubStride * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S5 dθ_prewa precompute (V, ordinary, stable identity on ch4) =====
        // dθ_prewa = sin(θ_g_rad − θ_a_rad) / (cos(θ_g_rad) × cos(θ_a_rad)) —
        // the tan difference identity: equals tan(θ_g) − tan(θ_a) but avoids
        // catastrophic cancellation near the 90° singularity. Choreography:
        //   B1.ch4: θ_a → cos(θ_a);  B2.ch4: θ_g → cos(θ_g) → cos_g × cos_a;
        //   B3.ch4: Δθ → sin(Δθ) → dθ_prewa = sin(Δθ) / cos_prod.
        // DxDwVF/DyDhVF never read ch4 of B1/B2, so overwriting ch4 is safe.
        {
            auto b1Local = b1Buf_.Get<float>(); // anchor
            auto b2Local = b2Buf_.Get<float>(); // gt
            auto b3Local = b3Buf_.Get<float>(); // output
            uint32_t cnt = static_cast<uint32_t>(cnt64);
            uint32_t angOff = static_cast<uint32_t>(RBE_CH_ANG * ubStride);

            // Step 1-2: θ_a_rad = θ_a_deg × π/180, θ_g_rad = θ_g_deg × π/180
            AscendC::Muls<float, false>(b1Local[angOff], b1Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::Muls<float, false>(b2Local[angOff], b2Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 3: Δθ = θ_g_rad − θ_a_rad → B3.ch4
            AscendC::Sub<float>(b3Local[angOff], b2Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 4: sin(Δθ) → B3.ch4 (in-place)
            AscendC::Sin<float>(b3Local[angOff], b3Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 5-6: cos(θ_a_rad) → B1.ch4, cos(θ_g_rad) → B2.ch4 (in-place)
            AscendC::Cos<float>(b1Local[angOff], b1Local[angOff], cnt);
            AscendC::Cos<float>(b2Local[angOff], b2Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 7: cos_prod = cos(θ_g) × cos(θ_a) → B2.ch4 (in-place: B2 = B2 × B1)
            AscendC::Mul<float>(b2Local[angOff], b2Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 8: dθ_prewa = sin(Δθ) / cos_prod → B3.ch4 (in-place: B3 = B3 / B2)
            AscendC::Div<float>(b3Local[angOff], b3Local[angOff], b2Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 9: Inf→NaN guard for out-of-hardware-range angles. Hardware
            // Sin/Cos valid range is ±65504 rad; beyond it Cos→0 → dθ_prewa =
            // ±Inf. Convert ±Inf→NaN via the IEEE-754 identity inf*0=NaN
            // (finite values preserved); reuses B1.ch4 as scratch.
            AscendC::Muls<float, false>(b1Local[angOff], b3Local[angOff], (float)0, static_cast<int32_t>(cnt));
            AscendC::Add<float>(b3Local[angOff], b3Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S6 VF Compute (V, asc_vf_call) → B3 [5, ubStride] =====
        // VF uses ubStride as channel stride; the mask handles the boxCount tail.
        {
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            auto b3Local = b3Buf_.Get<float>();
            __ubuf__ float* b1Ptr = (__ubuf__ float*)b1Local.GetPhyAddr();
            __ubuf__ float* b2Ptr = (__ubuf__ float*)b2Local.GetPhyAddr();
            __ubuf__ float* b3Ptr = (__ubuf__ float*)b3Local.GetPhyAddr();
            uint32_t cnt = static_cast<uint32_t>(cnt64);
            uint32_t stride = static_cast<uint32_t>(ubStride);
            asc_vf_call<DxDwVF>(b3Ptr, b1Ptr, b2Ptr, cnt, stride, td_->weight[0], td_->weight[2]);
            asc_vf_call<DyDhVF>(b3Ptr, b1Ptr, b2Ptr, cnt, stride, td_->weight[1], td_->weight[3]);
            asc_vf_call<DThetaVF>(b3Ptr, b1Ptr, b2Ptr, cnt, stride, td_->weight[4]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S7 Cast fp32→fp16 output (V, ordinary) → B0 fp16 (reuse) =====
        {
            auto b0Local = b0Buf_.Get<half>();
            auto b3Local = b3Buf_.Get<float>();
            AscendC::Cast<half, float>(b0Local, b3Local, AscendC::RoundMode::CAST_NONE,
                                       static_cast<uint32_t>(ubStride * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S8 CopyOut (V→MTE3) → GM y =====
        if (agg) {
            CopyOutAgg<half>(outGm_, b0Buf_, boxOffset, boxCount, ubStride);
        } else {
            CopyOutPerChannel(boxOffset, boxCount, ubStride);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Branch-1 fp32-direct tile pipeline (S1–S5): B0 anchor / B1 gt / B2 output,
    // no Cast steps; VF bodies shared with the fp16 path.
    __aicore__ inline void ProcessTileFp32(int64_t boxOffset, int64_t boxCount)
    {
        // Aggregated tile (slot layout): see ProcessTileFp16.
        bool agg = (td_->slotElems > 0) && ((boxOffset % N_) == 0) && ((boxCount % N_) == 0);
        int64_t ubStride = agg ? (boxCount / N_) * td_->slotElems :
                                 (boxCount + 15) & ~static_cast<int64_t>(15); // ceilAlign(boxCount, 16)
        int64_t cnt64 = agg ? ubStride : boxCount; // vector op count (pads included when aggregated)

        // Zero-init B0/B1/B2 so Tan/VF never read uninitialized data beyond
        // boxCount (507035 faults); the valid region is overwritten by CopyIn/VF.
        // The length covers the guard tail for the same reason as the fp16 path.
        {
            auto b0Local = b0Buf_.Get<float>();
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            AscendC::Duplicate<float>(b0Local, (float)0,
                                      static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
            AscendC::Duplicate<float>(b1Local, (float)0,
                                      static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
            AscendC::Duplicate<float>(b2Local, (float)0,
                                      static_cast<int32_t>(ubStride * RBE_BOX_CHANNELS + RBE_UB_GUARD_ELEMS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S1 CopyIn anchor (MTE2) → B0 fp32 [5, ubStride] =====
        if (agg) {
            CopyInAgg<float>(anchorGmF32_, b0Buf_, boxOffset, boxCount, ubStride);
        } else {
            CopyInPerChannelFp32(anchorGmF32_, b0Buf_, boxOffset, boxCount, ubStride);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S2 CopyIn gt (MTE2) → B1 fp32 [5, ubStride] =====
        if (agg) {
            CopyInAgg<float>(gtGmF32_, b1Buf_, boxOffset, boxCount, ubStride);
        } else {
            CopyInPerChannelFp32(gtGmF32_, b1Buf_, boxOffset, boxCount, ubStride);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S3 dθ_prewa precompute (V, ordinary, stable identity on ch4) =====
        // Same algorithm as the fp16 path S5. Choreography (B0=anchor, B1=gt,
        // B2=output): B0.ch4 → cos(θ_a); B1.ch4 → cos(θ_g) → cos_g × cos_a;
        // B2.ch4 → Δθ → sin(Δθ) → dθ_prewa. DxDwVF/DyDhVF never read ch4 of
        // B0/B1, so overwriting ch4 is safe.
        {
            auto b0Local = b0Buf_.Get<float>(); // anchor
            auto b1Local = b1Buf_.Get<float>(); // gt
            auto b2Local = b2Buf_.Get<float>(); // output
            uint32_t cnt = static_cast<uint32_t>(cnt64);
            uint32_t angOff = static_cast<uint32_t>(RBE_CH_ANG * ubStride);

            // Step 1-2: θ_a_rad = θ_a_deg × π/180, θ_g_rad = θ_g_deg × π/180
            AscendC::Muls<float, false>(b0Local[angOff], b0Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::Muls<float, false>(b1Local[angOff], b1Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 3: Δθ = θ_g_rad − θ_a_rad → B2.ch4
            AscendC::Sub<float>(b2Local[angOff], b1Local[angOff], b0Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 4: sin(Δθ) → B2.ch4 (in-place)
            AscendC::Sin<float>(b2Local[angOff], b2Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 5-6: cos(θ_a_rad) → B0.ch4, cos(θ_g_rad) → B1.ch4 (in-place)
            AscendC::Cos<float>(b0Local[angOff], b0Local[angOff], cnt);
            AscendC::Cos<float>(b1Local[angOff], b1Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 7: cos_prod = cos(θ_g) × cos(θ_a) → B1.ch4 (in-place: B1 = B1 × B0)
            AscendC::Mul<float>(b1Local[angOff], b1Local[angOff], b0Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 8: dθ_prewa = sin(Δθ) / cos_prod → B2.ch4 (in-place: B2 = B2 / B1)
            AscendC::Div<float>(b2Local[angOff], b2Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 9: Inf→NaN guard for out-of-hardware-range angles (see fp16 path S5 step 9).
            AscendC::Muls<float, false>(b0Local[angOff], b2Local[angOff], (float)0, static_cast<int32_t>(cnt));
            AscendC::Add<float>(b2Local[angOff], b2Local[angOff], b0Local[angOff], static_cast<int32_t>(cnt));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S4 VF Compute (V, asc_vf_call) → B2 fp32 [5, ubStride] =====
        // VF signature (out, anchor, gt, ...): out=B2, anchor=B0, gt=B1.
        {
            auto b0Local = b0Buf_.Get<float>();
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            __ubuf__ float* b0Ptr = (__ubuf__ float*)b0Local.GetPhyAddr();
            __ubuf__ float* b1Ptr = (__ubuf__ float*)b1Local.GetPhyAddr();
            __ubuf__ float* b2Ptr = (__ubuf__ float*)b2Local.GetPhyAddr();
            uint32_t cnt = static_cast<uint32_t>(cnt64);
            uint32_t stride = static_cast<uint32_t>(ubStride);
            asc_vf_call<DxDwVF>(b2Ptr, b0Ptr, b1Ptr, cnt, stride, td_->weight[0], td_->weight[2]);
            asc_vf_call<DyDhVF>(b2Ptr, b0Ptr, b1Ptr, cnt, stride, td_->weight[1], td_->weight[3]);
            asc_vf_call<DThetaVF>(b2Ptr, b0Ptr, b1Ptr, cnt, stride, td_->weight[4]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S5 CopyOut (V→MTE3) → GM y =====
        if (agg) {
            CopyOutAgg<float>(outGmF32_, b2Buf_, boxOffset, boxCount, ubStride);
        } else {
            CopyOutPerChannelFp32(boxOffset, boxCount, ubStride);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Aggregated copies — batch-aligned tiles (slot layout).
    // The tile covers K = boxCount/N whole batches starting at b0. Channel c
    // keeps the [5, ubStride] view the VF bodies expect, with batch b's N
    // elements at c*ubStride + b*slotElems; the trailing pad lanes
    // [N, slotElems) per slot are zero (zero-inert for the encoding math).
    //
    // MTE 2D copy constraint: every UB block start must be 32B-aligned. With
    // blockLen = N*sizeof(T) and dstStride = 0, consecutive block starts
    // advance by ceilAlign(N*sizeof(T), 32B) = slotElems*sizeof(T) — exactly
    // the slot pitch — so all K blocks of one channel land correctly in a
    // single 2D DataCopyPad per channel.
    // Invariant: K <= MAX_COPY_BLOCK_COUNT (host tiling caps ubFormer/N).
    // Precondition: [boxOffset, boxOffset+boxCount) spans whole batches only.

    // CopyInAgg — MTE2 aggregated gather (GM → UB), slot layout.
    template <typename T>
    __aicore__ inline void CopyInAgg(const AscendC::GlobalTensor<T>& gm,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC>& dstBuf, int64_t boxOffset,
                                     int64_t boxCount, int64_t ubStride)
    {
        auto bLocal = dstBuf.Get<T>();
        int64_t b0 = boxOffset / N_;
        uint32_t batchCount = static_cast<uint32_t>(boxCount / N_);
        // Aggregated copies only run when the host tiling enabled the slot
        // layout (N <= ubFormer, a few thousand), so N * sizeof(T) fits
        // uint32_t blockLen; the GM stride is kept in int64_t end to end.
        uint32_t blockLen = static_cast<uint32_t>(N_ * sizeof(T));
        int64_t srcStride = (RBE_BOX_CHANNELS - 1) * N_ * sizeof(T); // GM: bytes

        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = (blockLen % AscendC::ONE_BLK_SIZE) != 0;

        for (int64_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = batchCount;
            copyParams.blockLen = blockLen;
            copyParams.srcStride = srcStride;
            copyParams.dstStride = 0; // UB: 32B units — slots are contiguous

            int64_t gmOff = b0 * RBE_BOX_CHANNELS * N_ + c * N_;
            AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(
                bLocal[static_cast<uint32_t>(c * ubStride)], gm[static_cast<uint64_t>(gmOff)], copyParams, padParams);
        }
    }

    // CopyOutAgg — MTE3 aggregated scatter (UB → GM), mirror of CopyInAgg:
    // blocks read contiguous slot lanes (srcStride = 0), GM blocks hop
    // 4N elements between batch segments (dstStride = 4N elements, bytes).
    template <typename T>
    __aicore__ inline void CopyOutAgg(AscendC::GlobalTensor<T>& outGm,
                                      AscendC::TBuf<AscendC::TPosition::VECCALC>& srcBuf, int64_t boxOffset,
                                      int64_t boxCount, int64_t ubStride)
    {
        auto bLocal = srcBuf.Get<T>();
        int64_t b0 = boxOffset / N_;
        uint32_t batchCount = static_cast<uint32_t>(boxCount / N_);
        // See CopyInAgg: the host slot-layout invariant bounds blockLen;
        // the GM stride stays int64_t end to end.
        uint32_t blockLen = static_cast<uint32_t>(N_ * sizeof(T));
        int64_t dstStride = (RBE_BOX_CHANNELS - 1) * N_ * sizeof(T); // GM: bytes

        for (int64_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = batchCount;
            copyParams.blockLen = blockLen;
            copyParams.srcStride = 0; // UB: 32B units — slots are contiguous
            copyParams.dstStride = dstStride;

            int64_t gmOff = b0 * RBE_BOX_CHANNELS * N_ + c * N_;
            AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(
                outGm[static_cast<uint64_t>(gmOff)], bLocal[static_cast<uint32_t>(c * ubStride)], copyParams);
        }
    }

    // CopyInPerChannel — MTE2 2D stride gather (fp16 path).
    // One DataCopyPad with blockCount=5 fetches all channels of the batch
    // segment: GM srcStride=(N-boxCount) hops between channel starts, UB
    // dstStride (32B units) hops between the [5, ubStride] channel slots —
    // it lands exactly on the channel boundary because ubStride*sizeof(T)
    // is 32B-aligned. GM side is byte-granular, so arbitrary N/nStart are fine.
    // Precondition: [boxOffset, boxOffset+boxCount) lies within ONE batch.
    __aicore__ inline void CopyInPerChannel(AscendC::GlobalTensor<half>& gm, int64_t boxOffset, int64_t boxCount,
                                            int64_t ubStride)
    {
        auto b0Local = b0Buf_.Get<half>();
        int64_t b = boxOffset / N_;
        int64_t nStart = boxOffset % N_;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = RBE_BOX_CHANNELS;
        copyParams.blockLen = static_cast<uint32_t>(boxCount * sizeof(half));
        copyParams.srcStride = (N_ - boxCount) * sizeof(half);                               // GM: bytes
        copyParams.dstStride = (ubStride - boxCount) * sizeof(half) / AscendC::ONE_BLK_SIZE; // UB: 32B units

        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = ((boxCount * sizeof(half)) % AscendC::ONE_BLK_SIZE) != 0;

        int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + nStart;
        AscendC::DataCopyPad<half, AscendC::PaddingMode::Normal>(b0Local[0], gm[static_cast<uint64_t>(gmOff)],
                                                                 copyParams, padParams);
    }

    // CopyOutPerChannel — MTE3 2D stride scatter (fp16 path), mirror of
    // CopyInPerChannel. GM writes are byte-granular, so no read-modify-write
    // overlap can occur between concurrent blocks.
    // Precondition: [boxOffset, boxOffset+boxCount) lies within ONE batch.
    __aicore__ inline void CopyOutPerChannel(int64_t boxOffset, int64_t boxCount, int64_t ubStride)
    {
        auto b0Local = b0Buf_.Get<half>();
        int64_t b = boxOffset / N_;
        int64_t nStart = boxOffset % N_;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = RBE_BOX_CHANNELS;
        copyParams.blockLen = static_cast<uint32_t>(boxCount * sizeof(half));
        copyParams.srcStride = (ubStride - boxCount) * sizeof(half) / AscendC::ONE_BLK_SIZE; // UB: 32B units
        copyParams.dstStride = (N_ - boxCount) * sizeof(half);                               // GM: bytes

        int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + nStart;
        AscendC::DataCopyPad<half, AscendC::PaddingMode::Normal>(outGm_[static_cast<uint64_t>(gmOff)], b0Local[0],
                                                                 copyParams);
    }

    // CopyInPerChannelFp32 — MTE2 2D stride gather (fp32 path), fp32 mirror of
    // CopyInPerChannel. The destination buffer is passed in (B0 for anchor,
    // B1 for gt — distinct, no reuse unlike fp16's B0 double-use).
    // Precondition: [boxOffset, boxOffset+boxCount) lies within ONE batch.
    __aicore__ inline void CopyInPerChannelFp32(AscendC::GlobalTensor<float>& gm,
                                                AscendC::TBuf<AscendC::TPosition::VECCALC>& dstBuf, int64_t boxOffset,
                                                int64_t boxCount, int64_t ubStride)
    {
        auto bLocal = dstBuf.Get<float>();
        int64_t b = boxOffset / N_;
        int64_t nStart = boxOffset % N_;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = RBE_BOX_CHANNELS;
        copyParams.blockLen = static_cast<uint32_t>(boxCount * sizeof(float));
        copyParams.srcStride = (N_ - boxCount) * sizeof(float);                               // GM: bytes
        copyParams.dstStride = (ubStride - boxCount) * sizeof(float) / AscendC::ONE_BLK_SIZE; // UB: 32B units

        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = ((boxCount * sizeof(float)) % AscendC::ONE_BLK_SIZE) != 0;

        int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + nStart;
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(bLocal[0], gm[static_cast<uint64_t>(gmOff)],
                                                                  copyParams, padParams);
    }

    // CopyOutPerChannelFp32 — MTE3 2D stride scatter (fp32 path), mirror of
    // CopyInPerChannelFp32; reads VF output from B2 and scatters to GM y.
    // Precondition: [boxOffset, boxOffset+boxCount) lies within ONE batch.
    __aicore__ inline void CopyOutPerChannelFp32(int64_t boxOffset, int64_t boxCount, int64_t ubStride)
    {
        auto b2Local = b2Buf_.Get<float>();
        int64_t b = boxOffset / N_;
        int64_t nStart = boxOffset % N_;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = RBE_BOX_CHANNELS;
        copyParams.blockLen = static_cast<uint32_t>(boxCount * sizeof(float));
        copyParams.srcStride = (ubStride - boxCount) * sizeof(float) / AscendC::ONE_BLK_SIZE; // UB: 32B units
        copyParams.dstStride = (N_ - boxCount) * sizeof(float);                               // GM: bytes

        int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + nStart;
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(outGmF32_[static_cast<uint64_t>(gmOff)], b2Local[0],
                                                                  copyParams);
    }

    // --- state ---
    AscendC::TPipe pipe_;
    // fp16 path: b0=fp16 IO / b1=fp32 anchor / b2=fp32 gt / b3=fp32 output.
    // fp32 path: b0=fp32 anchor / b1=fp32 gt / b2=fp32 output (b3 unused).
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b3Buf_; // fp16 path only

    AscendC::GlobalTensor<half> anchorGm_; // fp16 path (branch-0)
    AscendC::GlobalTensor<half> gtGm_;
    AscendC::GlobalTensor<half> outGm_;
    AscendC::GlobalTensor<float> anchorGmF32_; // fp32 path (branch-1)
    AscendC::GlobalTensor<float> gtGmF32_;
    AscendC::GlobalTensor<float> outGmF32_;

    const RotatedBoxEncodeTilingData* td_ = nullptr;
    int64_t N_ = 0;
    int64_t B_ = 0;
    bool empty_ = true;
};

#endif // ROTATED_BOX_ENCODE_KERNEL_H
