/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BOUNDING_BOX_DECODE_KERNEL_H
#define BOUNDING_BOX_DECODE_KERNEL_H

#include "kernel_operator.h"
#include "bounding_box_decode_tiling_data.h"

// Cast traits for fp16<->fp32 conversion inside VF functions
constexpr AscendC::MicroAPI::CastTrait kCastB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr AscendC::MicroAPI::CastTrait kCastB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr int64_t kPhysNodes = 5;
constexpr int64_t kMaxInputSlots = 2;
constexpr int64_t kMaxOutputSlots = 1;
static constexpr uint32_t VL_F32 = 256U / sizeof(float); // 64

// High-precision exp constants (Cody-Waite range reduction for fp32 path).
// exp(dw) = 2^n * exp(r), where dw = n*ln2 + r, |r| <= ln2/2 ≈ 0.347.
// The 2^n scaling is exact (power-of-2 multiply = IEEE exponent shift, zero
// mantissa rounding). NPU Reg::Exp on the reduced range [-0.347, 0.347] is
// more accurate than on the full input range, matching libm expf more closely.
constexpr float kInvLn2 = 1.44269502162933349609375f; // 0x3fb8aa3b, 1/ln(2)
constexpr float kLn2Hi = 0.693145751953125f; // 0x3f317000, ln(2) high (12 sig bits → n*ln2Hi exact for |n|<2048)
constexpr float kLn2Lo = 1.428606765330187045e-06f; // ln(2) - kLn2Hi, low residual
constexpr int32_t kExpBias = 127;
constexpr int16_t kExpShift = 23;
constexpr uint32_t kF32PInf = 0x7F800000u;
constexpr uint32_t kF32NInf = 0xFF800000u;

constexpr AscendC::MicroAPI::CastTrait kCastF2IRound = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
constexpr AscendC::MicroAPI::CastTrait kCastI2FRound = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

// =========================================================================
// VF 1: AnchorPreVF — pw=aHi-aLo+1, pcx=(aLo+aHi)*0.5
//   Loads T from channel-contiguous aChLo/aChHi (fp16 upcast to fp32),
//   stores fp32 pw/pcx to calc slots. DESIGN §5.2 S3a/S3f.
// =========================================================================
template <typename T>
__simd_vf__ inline void AnchorPreVF(__ubuf__ float* pwOut, __ubuf__ float* pcxOut, __ubuf__ T* aChLo, __ubuf__ T* aChHi,
                                    uint32_t count, uint16_t repeatTime)
{
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> aLo, aHi, pw, pcx;
        if constexpr (std::is_same_v<T, half>) {
            AscendC::Reg::RegTensor<half> hLo, hHi;
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hLo, aChLo + off);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hHi, aChHi + off);
            AscendC::Reg::Cast<float, half, kCastB162B32>(aLo, hLo, mask);
            AscendC::Reg::Cast<float, half, kCastB162B32>(aHi, hHi, mask);
        } else {
            AscendC::Reg::LoadAlign(aLo, aChLo + off);
            AscendC::Reg::LoadAlign(aHi, aChHi + off);
        }
        AscendC::Reg::Sub<float>(pw, aHi, aLo, mask);
        AscendC::Reg::Adds<float>(pw, pw, 1.0f, mask);
        AscendC::Reg::Add<float>(pcx, aLo, aHi, mask);
        AscendC::Reg::Muls<float>(pcx, pcx, 0.5f, mask);
        AscendC::Reg::StoreAlign(pwOut + off, pw, mask);
        AscendC::Reg::StoreAlign(pcxOut + off, pcx, mask);
    }
}

// =========================================================================
// VF 2: DeltaDeStdVF — dx=dLo*stdsLo+meansLo, dw=clamp(dHi*stdsHi+meansHi, ±whMaxRatio)
//   Loads T from channel-contiguous dChLo/dChHi (fp16 upcast), stores fp32
//   dx/dw. DESIGN §5.2 S3b/S3g.
//   dw/dh 裁剪对齐 A2 基线 canndev tbe clamp_denorm_detal（主链无条件调用）：
//   whMaxRatio = |ln(wh_ratio_clip)| 由 Host 预计算经 TilingData 下发。
// =========================================================================
template <typename T>
__simd_vf__ inline void DeltaDeStdVF(__ubuf__ float* dxOut, __ubuf__ float* dwOut, __ubuf__ T* dChLo, __ubuf__ T* dChHi,
                                     float stdsLo, float stdsHi, float meansLo, float meansHi, float whMaxRatio,
                                     uint32_t count, uint16_t repeatTime)
{
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> dLo, dHi, dx, dw;
        if constexpr (std::is_same_v<T, half>) {
            AscendC::Reg::RegTensor<half> hLo, hHi;
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hLo, dChLo + off);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hHi, dChHi + off);
            AscendC::Reg::Cast<float, half, kCastB162B32>(dLo, hLo, mask);
            AscendC::Reg::Cast<float, half, kCastB162B32>(dHi, hHi, mask);
        } else {
            AscendC::Reg::LoadAlign(dLo, dChLo + off);
            AscendC::Reg::LoadAlign(dHi, dChHi + off);
        }
        AscendC::Reg::Muls<float>(dx, dLo, stdsLo, mask);
        AscendC::Reg::Adds<float>(dx, dx, meansLo, mask);
        AscendC::Reg::Muls<float>(dw, dHi, stdsHi, mask);
        AscendC::Reg::Adds<float>(dw, dw, meansHi, mask);
        // A2 基线 clamp_denorm_detal：dw = min(dw, +whMaxRatio) 后 max(dw, -whMaxRatio)。
        // 采用张量 Compare+Select（与 BoxClipVF 同款原语）；NaN 比较恒假 → NaN 原样透传，
        // 与既有 NaN 处理路径（BoxClipVF 收敛）保持一致。
        AscendC::Reg::MaskReg gtMask, ltMask;
        AscendC::Reg::RegTensor<float> hiReg, loReg;
        AscendC::Reg::Duplicate<float>(hiReg, whMaxRatio);
        AscendC::Reg::Duplicate<float>(loReg, -whMaxRatio);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(gtMask, dw, hiReg, mask);
        AscendC::Reg::Select(dw, hiReg, dw, gtMask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(ltMask, dw, loReg, mask);
        AscendC::Reg::Select(dw, loReg, dw, ltMask);
        AscendC::Reg::StoreAlign(dxOut + off, dx, mask);
        AscendC::Reg::StoreAlign(dwOut + off, dw, mask);
    }
}

// =========================================================================
// VF 3: DecodeGwGxVF<T> — gw=pw*exp(dw), gx=pcx+pw*dx  (pure fp32)
//   Reads pw/pcx/dx/dw from calc slots, writes gw/gx. Slot reuse safe:
//   all inputs loaded to registers before any store. DESIGN §5.2 S3c/S3h.
//
//   Template parameter T selects the exp implementation:
//   - T=float (fp32 path): custom Cody-Waite+Taylor FMA exp (~0.5 ULP/step,
//     matching libm expf to meet fp32 rtol=atol=1e-5, DESIGN §4.3/§7.2).
//   - T=half (fp16 path): NPU Reg::Exp hardware instruction. The fp16
//     tolerance (rtol=atol=1e-3) is 100× more lenient than fp32, so the
//     hardware exp's ~1 ULP error is sufficient. Using Reg::Exp for fp16
//     also aligns the kernel's exp with np.exp (which uses the same vexp
//     instruction family on SIMD platforms), reducing catastrophic-
//     cancellation-induced mare spikes at near-zero outputs.
// =========================================================================
template <typename T>
__simd_vf__ inline void DecodeGwGxVF(__ubuf__ float* gwOut, __ubuf__ float* gxOut, __ubuf__ float* pwIn,
                                     __ubuf__ float* pcxIn, __ubuf__ float* dxIn, __ubuf__ float* dwIn, uint32_t count,
                                     uint16_t repeatTime)
{
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> pw, pcx, dx, dw, gw, gx, expDw, tmp;
        AscendC::Reg::LoadAlign(pw, pwIn + off);
        AscendC::Reg::LoadAlign(pcx, pcxIn + off);
        AscendC::Reg::LoadAlign(dx, dxIn + off);
        AscendC::Reg::LoadAlign(dw, dwIn + off);
        // Common ±Inf/overflow correction masks and constants (shared by
        // both exp paths below).
        AscendC::Reg::MaskReg infMask, ninfMask, ovfMask, udfMask;
        AscendC::Reg::RegTensor<float> zeroReg, infReg;
        AscendC::Reg::Duplicate<float>(zeroReg, 0.0f);
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<uint32_t>&)infReg, kF32PInf);
        constexpr float kExpOvfThreshold = 89.0f;
        constexpr float kExpUdfThreshold = -88.0f;
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(ovfMask, dw, kExpOvfThreshold, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(udfMask, dw, kExpUdfThreshold, mask);
        AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(infMask, (AscendC::Reg::RegTensor<uint32_t>&)dw,
                                                                    kF32PInf, mask);
        AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(ninfMask, (AscendC::Reg::RegTensor<uint32_t>&)dw,
                                                                    kF32NInf, mask);

        if constexpr (std::is_same_v<T, float>) {
            // ---- fp32 path: high-precision Cody-Waite+Taylor FMA exp ----
            // dw = n*ln2 + r (|r| <= ln2/2),  exp(dw) = 2^n * exp(r)
            // 2^n is exact (IEEE exponent shift). FMA Horner gives 0.5 ULP
            // per step, matching libm expf for fp32 rtol=atol=1e-5.
            // Range guards fix 2^n bit-pattern wrap for |dw| > ~88.7.
            AscendC::Reg::RegTensor<float> t, nF, r, poly, coef;
            AscendC::Reg::RegTensor<int32_t> nI, biased;
            AscendC::Reg::Muls<float>(t, dw, kInvLn2, mask);
            AscendC::Reg::Cast<int32_t, float, kCastF2IRound>(nI, t, mask);
            AscendC::Reg::Cast<float, int32_t, kCastI2FRound>(nF, nI, mask);
            AscendC::Reg::Muls<float>(r, nF, kLn2Hi, mask);
            AscendC::Reg::Sub<float>(r, dw, r, mask);
            AscendC::Reg::Muls<float>(t, nF, kLn2Lo, mask);
            AscendC::Reg::Sub<float>(r, r, t, mask);
            // exp(r) via degree-7 Taylor, FMA Horner
            AscendC::Reg::Duplicate<float>(poly, 1.9841269841269841e-04f);
            AscendC::Reg::Duplicate<float>(coef, 1.3888888888888889e-03f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Duplicate<float>(coef, 8.3333333333333333e-03f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Duplicate<float>(coef, 4.1666666666666667e-02f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Duplicate<float>(coef, 1.6666666666666667e-01f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Duplicate<float>(coef, 5.0000000000000000e-01f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Duplicate<float>(coef, 1.0000000000000000e+00f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Duplicate<float>(coef, 1.0000000000000000e+00f);
            AscendC::Reg::FusedMulDstAdd<float>(poly, r, coef, mask);
            AscendC::Reg::Adds<int32_t>(biased, nI, kExpBias, mask);
            AscendC::Reg::ShiftLefts<int32_t, int16_t>(biased, biased, kExpShift, mask);
            AscendC::Reg::Mul<float>(expDw, poly, (AscendC::Reg::RegTensor<float>&)biased, mask);
        } else {
            // ---- fp16 path: NPU hardware Reg::Exp ----
            // fp16 tolerance (1e-3) is 100× more lenient than fp32 (1e-5),
            // so the hardware exp's ~1 ULP error is well within tolerance.
            // Using the same vexp instruction family as np.exp reduces
            // ULP divergence at near-zero outputs (catastrophic cancellation).
            AscendC::Reg::Exp<float>(expDw, dw, mask);
        }
        // Override for ±Inf and |dw| beyond valid exp range (both paths).
        AscendC::Reg::Select(expDw, infReg, expDw, ovfMask);   // dw >= 89 → +Inf
        AscendC::Reg::Select(expDw, zeroReg, expDw, udfMask);  // dw <= -88 → 0
        AscendC::Reg::Select(expDw, infReg, expDw, infMask);   // dw == +Inf → +Inf
        AscendC::Reg::Select(expDw, zeroReg, expDw, ninfMask); // dw == -Inf → 0
        AscendC::Reg::Mul<float>(gw, pw, expDw, mask);
        AscendC::Reg::Mul<float>(tmp, pw, dx, mask);
        AscendC::Reg::Add<float>(gx, pcx, tmp, mask);
        AscendC::Reg::StoreAlign(gwOut + off, gw, mask);
        AscendC::Reg::StoreAlign(gxOut + off, gx, mask);
    }
}

// =========================================================================
// VF 4: BoxClipVF<T> — clip(gx+halfSign*gw+offset, clipLo, clipHi) → outCh
//   Loads fp32 gw/gx from calc, clips, casts to T on store (fp16 downcast).
//   DESIGN §5.2 S3d/e/S3i/j. clipLo=0.0f (clipped_non_negative invariant).
// =========================================================================
template <typename T>
__simd_vf__ inline void BoxClipVF(__ubuf__ T* outCh, __ubuf__ float* gwIn, __ubuf__ float* gxIn, float halfSign,
                                  float offset, float clipLo, float clipHi, uint32_t count, uint16_t repeatTime)
{
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> gw, gx, tmp, loReg, hiReg, out;
        AscendC::Reg::MaskReg nanMask;
        AscendC::Reg::LoadAlign(gw, gwIn + off);
        AscendC::Reg::LoadAlign(gx, gxIn + off);
        AscendC::Reg::Duplicate(loReg, clipLo);
        AscendC::Reg::Duplicate(hiReg, clipHi);
        AscendC::Reg::Muls<float>(tmp, gw, halfSign, mask);
        AscendC::Reg::Add<float>(tmp, gx, tmp, mask);
        AscendC::Reg::Adds<float>(tmp, tmp, offset, mask);
        // min-then-max 与 CANN 内置 tbe clamp_result 的 vmin→vmax 顺序一致：
        // 正常输入等价 clamp(x, clipLo, clipHi)；退化 clipHi<clipLo 时收敛到 clipLo，
        // 保持 clipped_non_negative 不变量（max-then-min 会输出 clipLo 以下的负值）。
        AscendC::Reg::Min<float>(tmp, tmp, hiReg, mask);
        AscendC::Reg::Max<float>(out, tmp, loReg, mask);
        // NaN→0: extreme anchor_box (±3.4e38) causes Inf-Inf=NaN in coordinate
        // computation. Max/Min propagate NaN on NPU. Replace NaN with clipLo (0)
        // to satisfy clipped_non_negative invariant (DESIGN §4.3).
        // IEEE 754: NaN==NaN is false → EQ mask is false for NaN → Select picks loReg.
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(nanMask, out, out, mask);
        AscendC::Reg::Select(out, out, loReg, nanMask);
        if constexpr (std::is_same_v<T, half>) {
            AscendC::Reg::RegTensor<half> hOut;
            AscendC::Reg::Cast<half, float, kCastB322B16>(hOut, out, mask);
            AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(outCh + off, hOut, mask);
        } else {
            AscendC::Reg::StoreAlign(outCh + off, out, mask);
        }
    }
}

// =========================================================================
// Kernel Class
//   buf_[0]: B_anchor  [4, ubFormer] channel-contiguous (T)
//   buf_[1]: B_deltas  [4, ubFormer] channel-contiguous (T)
//   buf_[2]: B_calc0   [4, ubFormer] fp32 (fp16 only; slots pw|pcx|gw|gx)
//   buf_[3]: B_calc1   [4, ubFormer] fp32 (fp16 only; slots dx|dw|..)
//   buf_[4]: B_boxes   [4, ubFormer] channel-contiguous (T)
// fp32 path: buf_[2]/buf_[3] not allocated; calc aliases IO views (K=0).
// =========================================================================
template <typename T>
class BoundingBoxDecodeKernel {
public:
    __aicore__ inline void Init(GM_ADDR anchor_box, GM_ADDR deltas, GM_ADDR boxes,
                                const BoundingBoxDecodeTilingData* td)
    {
        td_ = td;
        // Empty-tensor short-circuit (DESIGN-BRANCH-1 §3/§5, DESIGN §10.4):
        // dim0==0 (N==0) returns early, skipping all buffer allocation and
        // GM setup — no CopyIn/Compute/CopyOut, no UB occupancy.
        // (was compile-time kIsEmpty; changed to runtime check to fix aclnn
        //  e2e tiling buffer allocation failure — see struct.h comment.)
        if (td_->dim0 == 0) {
            return;
        }
        gmIn_[0].SetGlobalBuffer((__gm__ T*)anchor_box);
        gmIn_[1].SetGlobalBuffer((__gm__ T*)deltas);
        gmOut_[0].SetGlobalBuffer((__gm__ T*)boxes);

        const int64_t cap = td_->ubFormer; // full tile capacity (box count)
        const int64_t ioBytes = cap * kElemsPerBox * sizeof(T);
        const int64_t calcBytes = cap * kElemsPerBox * sizeof(float);
        pipe_.InitBuffer(buf_[0], static_cast<uint32_t>(ioBytes));
        pipe_.InitBuffer(buf_[1], static_cast<uint32_t>(ioBytes));
        if constexpr (std::is_same_v<T, half>) {
            pipe_.InitBuffer(buf_[2], static_cast<uint32_t>(calcBytes));
            pipe_.InitBuffer(buf_[3], static_cast<uint32_t>(calcBytes));
        }
        pipe_.InitBuffer(buf_[4], static_cast<uint32_t>(ioBytes));
    }

    __aicore__ inline void Process()
    {
        // Empty-tensor short-circuit (DESIGN-BRANCH-1 §3/§5, DESIGN §10.3):
        // dim0==0 (N==0) returns early — no CopyIn/Compute/CopyOut pipeline.
        if (td_->dim0 == 0) {
            return;
        }
        const int64_t blockIdx = AscendC::GetBlockIdx();
        // 512-box alignment in ComputeMultiCoreSplit can make blockNum < coreNum;
        // idle cores (blockIdx >= blockNum) have no valid data — return early to
        // avoid GM OOB read/write beyond tensor bounds.
        if (blockIdx >= td_->blockNum) {
            return;
        }
        const int64_t isLastBlock = (blockIdx == td_->blockNum - 1);
        const int64_t boxBase = blockIdx * td_->blockFormer;
        const int64_t loopNum = isLastBlock ? td_->ubLoopOfTailBlock : td_->ubLoopOfFormerBlock;
        const int64_t tailNum = isLastBlock ? td_->ubTailOfTailBlock : td_->ubTailOfFormerBlock;

        int64_t offset = 0;
        for (int64_t i = 0; i < loopNum - 1; ++i) {
            ProcessTile(boxBase + offset, td_->ubFormer);
            offset += td_->ubFormer;
        }
        if (tailNum > 0) {
            ProcessTile(boxBase + offset, tailNum);
        }
    }

private:
    __aicore__ inline void ProcessTile(int64_t boxOffset, int64_t boxCount)
    {
        // ===== CopyIn: NDDMA 2D stride gather, 4 channels × 2 inputs =====
        // DESIGN §3.2 / §10.7. GM [N,4] interleaved → UB [4,N] ch-contiguous.
        // blockCount=boxCount, blockLen=sizeof(T), srcStride=(4-1)*sizeof(T) bytes,
        // dstStride=0 (UB contiguous packing). Channel view stride = ubFormer.
        CopyInChannels(buf_[0], gmIn_[0], boxOffset, boxCount);
        CopyInChannels(buf_[1], gmIn_[1], boxOffset, boxCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        // ===== Compute: x-chain (ch0,ch2) then y-chain (ch1,ch3) =====
        // clip 上界为 max_shape-1（最后有效像素），与 CANN 内置 BoundingBoxDecode
        // (tbe clamp_result: 0<=x<=max_shape[1]-1) 及 CheckValid(x<=W*r-1) 语义闭合。
        // int64 域先减 1 再转 fp32（与 A2 基线 Python 整数运算一致）：若先转 float 再减
        // 1.0f，奇数 W>2^24（如 16777217）会因舍入使上界再小 1。
        ProcessAxis(boxCount, 0, 2, td_->stds[0], td_->stds[2], td_->means[0], td_->means[2],
                    static_cast<float>(td_->maxShapeW - 1));
        ProcessAxis(boxCount, 1, 3, td_->stds[1], td_->stds[3], td_->means[1], td_->means[3],
                    static_cast<float>(td_->maxShapeH - 1));

        // ===== CopyOut: 2D stride scatter, 4 channels → GM [N,4] =====
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        CopyOutChannels(buf_[4], gmOut_[0], boxOffset, boxCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

    // One axis (x or y): chLo/chHi are the two channels for this axis.
    __aicore__ inline void ProcessAxis(int64_t boxCount, int32_t chLo, int32_t chHi, float stdsLo, float stdsHi,
                                       float meansLo, float meansHi, float clipHi)
    {
        const uint32_t cap = static_cast<uint32_t>(td_->ubFormer);
        const uint32_t count = static_cast<uint32_t>(boxCount);
        const uint16_t rep = static_cast<uint16_t>((count + VL_F32 - 1) / VL_F32);

        __ubuf__ T* aBase = (__ubuf__ T*)buf_[0].Get<T>().GetPhyAddr();
        __ubuf__ T* dBase = (__ubuf__ T*)buf_[1].Get<T>().GetPhyAddr();
        __ubuf__ T* oBase = (__ubuf__ T*)buf_[4].Get<T>().GetPhyAddr();

        // Channel views (stride = ubFormer for 32B alignment).
        __ubuf__ T* aChLo = aBase + static_cast<uint64_t>(chLo) * cap;
        __ubuf__ T* aChHi = aBase + static_cast<uint64_t>(chHi) * cap;
        __ubuf__ T* dChLo = dBase + static_cast<uint64_t>(chLo) * cap;
        __ubuf__ T* dChHi = dBase + static_cast<uint64_t>(chHi) * cap;

        // Calc slot pointers.
        __ubuf__ float* pwOut;
        __ubuf__ float* pcxOut;
        __ubuf__ float* dxOut;
        __ubuf__ float* dwOut;
        __ubuf__ float* gwIn;
        __ubuf__ float* gxIn;
        if constexpr (std::is_same_v<T, half>) {
            // fp16: separate fp32 calc buffers. 4 slots in c0 (pw|pcx|gw|gx), 2 in c1 (dx|dw).
            __ubuf__ float* c0 = (__ubuf__ float*)buf_[2].Get<float>().GetPhyAddr();
            __ubuf__ float* c1 = (__ubuf__ float*)buf_[3].Get<float>().GetPhyAddr();
            pwOut = c0 + 0 * cap;
            pcxOut = c0 + 1 * cap;
            dxOut = c1 + 0 * cap;
            dwOut = c1 + 1 * cap;
            gwIn = c0 + 2 * cap;
            gxIn = c0 + 3 * cap;
        } else {
            // fp32: calc aliases IO buffer views (K=0, no separate calc TBuf).
            // pw→a_chLo, pcx→a_chHi, dx→d_chLo, dw→d_chHi; gw reuses pw slot, gx reuses pcx slot.
            pwOut = (__ubuf__ float*)aChLo;
            pcxOut = (__ubuf__ float*)aChHi;
            dxOut = (__ubuf__ float*)dChLo;
            dwOut = (__ubuf__ float*)dChHi;
            gwIn = (__ubuf__ float*)aChLo;
            gxIn = (__ubuf__ float*)aChHi;
        }

        // S3a/S3f: anchor preprocess → pw, pcx
        asc_vf_call<AnchorPreVF<T>>(pwOut, pcxOut, aChLo, aChHi, count, rep);
        // S3b/S3g: deltas de-standardize + A2 基线 wh_ratio 裁剪（dw/dh → ±whMaxRatio）→ dx, dw
        asc_vf_call<DeltaDeStdVF<T>>(dxOut, dwOut, dChLo, dChHi, stdsLo, stdsHi, meansLo, meansHi, td_->whMaxRatio,
                                     count, rep);
        // S3c/S3h: decode gw, gx (reads pw,pcx,dx,dw; writes gw,gx)
        asc_vf_call<DecodeGwGxVF<T>>(gwIn, gxIn, pwOut, pcxOut, dxOut, dwOut, count, rep);
        // S3d: ox1/oy1 = clip(gx - gw*0.5 + 0.5, 0, clipHi) → B_boxes chLo
        asc_vf_call<BoxClipVF<T>>(oBase + static_cast<uint64_t>(chLo) * cap, gwIn, gxIn, -0.5f, 0.5f, 0.0f, clipHi,
                                  count, rep);
        // S3e: ox2/oy2 = clip(gx + gw*0.5 - 0.5, 0, clipHi) → B_boxes chHi
        asc_vf_call<BoxClipVF<T>>(oBase + static_cast<uint64_t>(chHi) * cap, gwIn, gxIn, 0.5f, -0.5f, 0.0f, clipHi,
                                  count, rep);
    }

    // NDDMA 2D stride gather: GM [N,4] interleaved → UB [4,N] ch-contiguous.
    // 4 DataCopyPad calls, one per channel. dst channel view stride = ubFormer.
    // PaddingMode::Compact packs sub-32B blocks contiguously in UB (Normal mode
    // would 32B-align each block, breaking channel-contiguous layout).
    __aicore__ inline void CopyInChannels(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf, AscendC::GlobalTensor<T>& gm,
                                          int64_t boxOffset, int64_t boxCount)
    {
        AscendC::DataCopyExtParams params(static_cast<uint16_t>(boxCount), static_cast<uint32_t>(sizeof(T)),
                                          static_cast<int64_t>((kElemsPerBox - 1) * sizeof(T)), // srcStride bytes (GM)
                                          0, // dstStride 0 (UB contiguous)
                                          0);
        AscendC::DataCopyPadExtParams<T> pad(false, 0, 0, 0);
        const uint32_t cap = static_cast<uint32_t>(td_->ubFormer);
        for (int32_t c = 0; c < kElemsPerBox; ++c) {
            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                buf.Get<T>()[static_cast<uint32_t>(c) * cap], gm[static_cast<uint64_t>(boxOffset * kElemsPerBox + c)],
                params, pad);
        }
    }

    // MTE3 2D stride scatter: UB [4,N] ch-contiguous → GM [N,4] interleaved.
    // 4 DataCopyPad calls, one per channel. src channel view stride = ubFormer.
    // PaddingMode::Compact reads sub-32B blocks contiguously from UB.
    __aicore__ inline void CopyOutChannels(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
                                           AscendC::GlobalTensor<T>& gm, int64_t boxOffset, int64_t boxCount)
    {
        AscendC::DataCopyExtParams params(static_cast<uint16_t>(boxCount), static_cast<uint32_t>(sizeof(T)),
                                          0, // srcStride 0 (UB contiguous)
                                          static_cast<int64_t>((kElemsPerBox - 1) * sizeof(T)), // dstStride bytes (GM)
                                          0);
        const uint32_t cap = static_cast<uint32_t>(td_->ubFormer);
        for (int32_t c = 0; c < kElemsPerBox; ++c) {
            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                gm[static_cast<uint64_t>(boxOffset * kElemsPerBox + c)], buf.Get<T>()[static_cast<uint32_t>(c) * cap],
                params);
        }
    }

    AscendC::TPipe pipe_;
    const BoundingBoxDecodeTilingData* td_;
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
};

#endif // BOUNDING_BOX_DECODE_KERNEL_H
