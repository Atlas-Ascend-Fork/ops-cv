/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAMMA_CORRECTION_IMPL_HPP_
#define GAMMA_CORRECTION_IMPL_HPP_

#include "gamma_correction_tiling_def.h"
#include "kernel_operator.h"

namespace NsGammaCorrection {
using namespace AscendC;

// sRGB IEC 61966-2-1 constants (linear -> sRGB encoding curve). Plain constexpr:
// __aicore__ is a function attribute and warns when applied to variables; these
// compile-time constants are inlined into the __aicore__ functions that use them.
constexpr float SRGB_THRESHOLD = 0.0031308F; // x <= thr -> linear branch
constexpr float SRGB_LINEAR_SLOPE = 12.92F;
constexpr float SRGB_POW_COEFF = 1.055F;
constexpr float SRGB_POW_OFFSET = 0.055F;
constexpr float SRGB_INV_GAMMA = 1.0F / 2.4F;

// AscendC vector Compare requires a 256-byte aligned element count.
constexpr uint32_t COMPARE_ALIGN_BYTES = 256U;

__aicore__ inline uint32_t AlignUp32(uint64_t bytes) { return static_cast<uint32_t>((bytes + 31U) / 32U * 32U); }

// Round a real element count up to the 256-byte boundary required by Compare.
__aicore__ inline uint32_t AlignCompareCount(uint32_t count, uint32_t typeBytes)
{
    const uint32_t align = COMPARE_ALIGN_BYTES / typeBytes;
    return (count + align - 1U) / align * align;
}

template <typename T>
struct Convert;

template <>
struct Convert<half> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<half>& src, uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }

    __aicore__ static inline void FromFloat(const LocalTensor<half>& dst, const LocalTensor<float>& src, uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
};

template <>
struct Convert<bfloat16_t> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<bfloat16_t>& src,
                                          uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }

    __aicore__ static inline void FromFloat(const LocalTensor<bfloat16_t>& dst, const LocalTensor<float>& src,
                                            uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
};

template <>
struct Convert<float> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
    {
        // fp32->fp32 Cast does not support CAST_NONE on dav_c220 (asserts); use a
        // scalar add of 0 as the element-wise copy instead.
        Adds(dst, src, 0.0F, count);
    }

    __aicore__ static inline void FromFloat(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                            uint32_t count)
    {
        Adds(dst, src, 0.0F, count);
    }
};

// Integer dtype conversions. The dav_c220 vector Cast unit only implements a
// subset of fp<->int transitions directly:
//   int8_t  <-> half, uint8_t <-> half  (no direct fp32 path)
//   int16_t <-> float                    (direct)
//   uint16_t has NO cast path at all     (int16 reinterpretation + FP32 wrap fix-up)
// int8/uint8 bridge through half; the half scratch must be a separate buffer of
// at least count*sizeof(half) bytes because the typed input/output buffers are
// only count*sizeof(T) bytes (too small for the half intermediate). The caller
// supplies a large enough scratch (an FP32 UB buffer, free during Convert).
// The caller clamps to the dtype range first (see WriteBack/IntRange).
template <>
struct Convert<int8_t> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<int8_t>& src, uint32_t count,
                                          const LocalTensor<half>& scratch)
    {
        Cast(scratch, src, RoundMode::CAST_NONE, count); // int8 -> half
        PipeBarrier<PIPE_V>();
        Cast(dst, scratch, RoundMode::CAST_NONE, count); // half -> float
    }

    __aicore__ static inline void FromFloat(const LocalTensor<int8_t>& dst, const LocalTensor<float>& src,
                                            uint32_t count, const LocalTensor<half>& scratch)
    {
        Cast(scratch, src, RoundMode::CAST_RINT, count); // float -> half
        PipeBarrier<PIPE_V>();
        Cast(dst, scratch, RoundMode::CAST_NONE, count); // half -> int8
    }
};

template <>
struct Convert<uint8_t> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<uint8_t>& src,
                                          uint32_t count, const LocalTensor<half>& scratch)
    {
        Cast(scratch, src, RoundMode::CAST_NONE, count); // uint8 -> half
        PipeBarrier<PIPE_V>();
        Cast(dst, scratch, RoundMode::CAST_NONE, count); // half -> float
    }

    __aicore__ static inline void FromFloat(const LocalTensor<uint8_t>& dst, const LocalTensor<float>& src,
                                            uint32_t count, const LocalTensor<half>& scratch)
    {
        Cast(scratch, src, RoundMode::CAST_RINT, count); // float -> half
        PipeBarrier<PIPE_V>();
        Cast(dst, scratch, RoundMode::CAST_NONE, count); // half -> uint8
    }
};

template <>
struct Convert<int16_t> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<int16_t>& src,
                                          uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_NONE, count); // int16 -> float (direct)
    }

    __aicore__ static inline void FromFloat(const LocalTensor<int16_t>& dst, const LocalTensor<float>& src,
                                            uint32_t count)
    {
        Cast(dst, src, RoundMode::CAST_RINT, count); // float -> int16 (direct)
    }
};

// uint16 has no cast path, so it goes through the int16 bit pattern: values in
// [32768, 65535] share their bits with int16 values v - 65536. The +/-65536 fix-up
// is done with FP32 min/max arithmetic (exact for integers below 2^24), so no
// Compare/Select and no 256-byte count alignment is needed. scratch is an FP32
// buffer of at least count elements.
constexpr float UINT16_WRAP = 65536.0F;
constexpr float UINT16_RINT_SPLIT = 32767.5F; // RINT(v) >= 32768  <=>  v >= 32767.5
constexpr float UINT16_SPLIT_SCALE = 512.0F;  // 1 / fp32 spacing (2^-9) just below 32767.5

template <>
struct Convert<uint16_t> {
    __aicore__ static inline void ToFloat(const LocalTensor<float>& dst, const LocalTensor<uint16_t>& src,
                                          uint32_t count, const LocalTensor<float>& scratch)
    {
        // Bits >= 0x8000 read back as v - 65536 in [-32768, -1].
        LocalTensor<int16_t> tmp = src.ReinterpretCast<int16_t>();
        Cast(dst, tmp, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        // wrapped = min(-min(v, 0), 1): 1 for the wrapped half, 0 otherwise.
        Mins(scratch, dst, 0.0F, count);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, -1.0F, count);
        PipeBarrier<PIPE_V>();
        Mins(scratch, scratch, 1.0F, count);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, UINT16_WRAP, count);
        PipeBarrier<PIPE_V>();
        Add(dst, dst, scratch, count);
    }

    // src must already be clamped to [0, 65535] (see WriteBack).
    __aicore__ static inline void FromFloat(const LocalTensor<uint16_t>& dst, const LocalTensor<float>& src,
                                            uint32_t count, const LocalTensor<float>& scratch)
    {
        // high = clamp((v - 32767.5) * 512 + 1, 0, 1): 1 exactly when v >= 32767.5. The
        // fp32 neighbour just below 32767.5 is 2^-9 away, so it maps to 0.
        Adds(scratch, src, -UINT16_RINT_SPLIT, count);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, UINT16_SPLIT_SCALE, count);
        PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, 1.0F, count);
        PipeBarrier<PIPE_V>();
        Maxs(scratch, scratch, 0.0F, count);
        PipeBarrier<PIPE_V>();
        Mins(scratch, scratch, 1.0F, count);
        PipeBarrier<PIPE_V>();
        // Shift the high half into int16 range; its int16 bits then equal the uint16 value.
        Muls(scratch, scratch, UINT16_WRAP, count);
        PipeBarrier<PIPE_V>();
        Sub(scratch, src, scratch, count);
        PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> tmp = dst.ReinterpretCast<int16_t>();
        Cast(tmp, scratch, RoundMode::CAST_RINT, count);
    }
};

// Compile-time clamping range for the integer dtype T. Image modes already clamp
// the FP32 result to [0, maxValue]; Mode 0 relies on this clamp alone. It narrows
// the result to the dtype range so the narrowing Cast does not rely on implicit
// saturation.
template <typename T>
struct IntRange {
    static constexpr float LO = 0.0F;
    static constexpr float HI = 0.0F;
};
template <>
struct IntRange<int8_t> {
    static constexpr float LO = -128.0F;
    static constexpr float HI = 127.0F;
};
template <>
struct IntRange<uint8_t> {
    static constexpr float LO = 0.0F;
    static constexpr float HI = 255.0F;
};
template <>
struct IntRange<int16_t> {
    static constexpr float LO = -32768.0F;
    static constexpr float HI = 32767.0F;
};
template <>
struct IntRange<uint16_t> {
    static constexpr float LO = 0.0F;
    static constexpr float HI = 65535.0F;
};

// True when T is an integer image dtype (needs the dtype-range clamp before cast).
template <typename T>
struct IsInteger {
    static constexpr bool VALUE = false;
};
template <>
struct IsInteger<int8_t> {
    static constexpr bool VALUE = true;
};
template <>
struct IsInteger<uint8_t> {
    static constexpr bool VALUE = true;
};
template <>
struct IsInteger<int16_t> {
    static constexpr bool VALUE = true;
};
template <>
struct IsInteger<uint16_t> {
    static constexpr bool VALUE = true;
};

// True when T bridges through half (int8/uint8) and so needs a half scratch tensor.
template <typename T>
struct NeedsHalfScratch {
    static constexpr bool VALUE = false;
};
template <>
struct NeedsHalfScratch<int8_t> {
    static constexpr bool VALUE = true;
};
template <>
struct NeedsHalfScratch<uint8_t> {
    static constexpr bool VALUE = true;
};

// True when T needs an FP32 scratch tensor for its wrap fix-up (uint16).
template <typename T>
struct NeedsFloatScratch {
    static constexpr bool VALUE = false;
};
template <>
struct NeedsFloatScratch<uint16_t> {
    static constexpr bool VALUE = true;
};

// Dispatch helpers so call sites can stay dtype-agnostic. scratch is an FP32 UB
// buffer that is free during the cast: int8/uint8 use it as a half bridge, uint16
// as an FP32 fix-up buffer; other dtypes ignore it.
template <typename T>
__aicore__ inline void ConvertToFloat(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t count,
                                      const LocalTensor<half>& scratch)
{
    if constexpr (NeedsHalfScratch<T>::VALUE) {
        Convert<T>::ToFloat(dst, src, count, scratch);
    } else if constexpr (NeedsFloatScratch<T>::VALUE) {
        Convert<T>::ToFloat(dst, src, count, scratch.ReinterpretCast<float>());
    } else {
        Convert<T>::ToFloat(dst, src, count);
    }
}

template <typename T>
__aicore__ inline void ConvertFromFloat(const LocalTensor<T>& dst, const LocalTensor<float>& src, uint32_t count,
                                        const LocalTensor<half>& scratch)
{
    if constexpr (NeedsHalfScratch<T>::VALUE) {
        Convert<T>::FromFloat(dst, src, count, scratch);
    } else if constexpr (NeedsFloatScratch<T>::VALUE) {
        Convert<T>::FromFloat(dst, src, count, scratch.ReinterpretCast<float>());
    } else {
        Convert<T>::FromFloat(dst, src, count);
    }
}

template <typename T>
class GammaCorrectionKernel {
public:
    __aicore__ inline GammaCorrectionKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const optiling::GammaCorrectionTilingData* tiling)
    {
        totalNum_ = tiling->totalNum;
        blockFactor_ = tiling->blockFactor;
        ubFactor_ = tiling->ubFactor;
        usedCoreNum_ = tiling->usedCoreNum;
        gammaMode_ = tiling->gammaMode;
        integerExponent_ = tiling->integerExponent;
        correctionMode_ = tiling->correctionMode;
        gamma_ = tiling->gamma;
        maxValue_ = tiling->maxValue;
        invMax_ = tiling->invMax;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum_);

        const uint32_t typedBytes = AlignUp32(ubFactor_ * sizeof(T));
        const uint32_t floatBytes = AlignUp32(ubFactor_ * sizeof(float));
        // Compare operates on the 256-byte-aligned tile, which is <= ubFactor_ for
        // every dtype (ubFactor is itself 256/typeBytes aligned), so floatBytes
        // covers the predicate scratch too.
        const uint32_t maskBytes = AlignUp32(ubFactor_ * sizeof(uint8_t));
        pipe_.InitBuffer(inQueue_, 2, typedBytes);
        pipe_.InitBuffer(outQueue_, 1, typedBytes);
        pipe_.InitBuffer(xFloatBuf_, floatBytes); // input -> fp32
        pipe_.InitBuffer(resultBuf_, floatBytes); // primary compute accumulator
        pipe_.InitBuffer(baseBuf_, floatBytes);   // integer-exponent scratch
        pipe_.InitBuffer(normBuf_, floatBytes);   // normalized [0,1] copy
        pipe_.InitBuffer(linBuf_, floatBytes);    // sRGB linear branch 12.92*x
        pipe_.InitBuffer(powBuf_, floatBytes);    // sRGB pow branch / generic scratch
        pipe_.InitBuffer(maskBuf_, maskBytes);    // uint8 predicate
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        if (core >= usedCoreNum_) {
            return;
        }
        const uint64_t coreOffset = core * blockFactor_;
        if (coreOffset >= totalNum_) {
            return;
        }
        const uint64_t remaining = totalNum_ - coreOffset;
        const uint64_t coreLength = remaining < blockFactor_ ? remaining : blockFactor_;
        for (uint64_t localOffset = 0; localOffset < coreLength; localOffset += ubFactor_) {
            const uint64_t tileRemaining = coreLength - localOffset;
            const uint32_t count = static_cast<uint32_t>(tileRemaining < ubFactor_ ? tileRemaining : ubFactor_);
            ProcessTile(coreOffset + localOffset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(input, xGm_[offset], copy, pad);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[offset], output, copy);
        outQueue_.FreeTensor(output);
    }

    // xFloat becomes scratch after the original input is copied to base.
    __aicore__ inline void ComputeIntegerPow(const LocalTensor<float>& xFloat, const LocalTensor<float>& result,
                                             const LocalTensor<float>& base, uint32_t count)
    {
        Adds(base, xFloat, 0.0F, count);
        Duplicate(result, 1.0F, count);
        uint32_t exponent = integerExponent_;
        while (exponent > 0U) {
            if ((exponent & 1U) != 0U) {
                Mul(xFloat, result, base, count);
                Adds(result, xFloat, 0.0F, count);
            }
            exponent >>= 1U;
            if (exponent > 0U) {
                Mul(xFloat, base, base, count);
                Adds(base, xFloat, 0.0F, count);
            }
        }
    }

    // y = pow(x, gamma) written into result. x is clamped to >= 0 by the caller in
    // image modes, so Log is in domain. NaN/Inf propagate by IEEE rules here; the
    // image modes clean them up afterwards.
    __aicore__ inline void ComputePow(const LocalTensor<float>& x, const LocalTensor<float>& result,
                                      const LocalTensor<float>& scratch, uint32_t count)
    {
        if (correctionMode_ == optiling::GC_CORRECTION_PURE) {
            if (gammaMode_ == optiling::GC_MODE_IDENTITY) {
                Adds(result, x, 0.0F, count);
            } else if (gammaMode_ == optiling::GC_MODE_SQUARE) {
                Mul(result, x, x, count);
            } else if (gammaMode_ == optiling::GC_MODE_SQRT) {
                Sqrt(result, x, count);
            } else if (gammaMode_ == optiling::GC_MODE_INTEGER) {
                ComputeIntegerPow(x, result, scratch, count);
            } else {
                Log(result, x, count);
                Muls(scratch, result, gamma_, count);
                Exp(result, scratch, count);
            }
        } else {
            // Image modes always use the exp(g*ln) generic path so the result lands
            // in [0, 1] after normalization regardless of gamma's exact value.
            Log(result, x, count);
            Muls(scratch, result, gamma_, count);
            Exp(result, scratch, count);
        }
    }

    // Clamp an FP32 tensor in-place to [lo, hi] using the binary-scalar APIs.
    __aicore__ inline void ClampRange(const LocalTensor<float>& t, float lo, float hi, uint32_t count)
    {
        Mins(t, t, hi, count);
        PipeBarrier<PIPE_V>();
        Maxs(t, t, lo, count);
        PipeBarrier<PIPE_V>();
    }

    // Replace any NaN element with 0 on an FP32 tensor. NaN is detected via the
    // IEEE rule (x != x => Compare(EQ) is false). Inf is NOT separately handled:
    // both image modes call ClampRange(x, 0, M) before reaching here, which maps
    // any +/-Inf to the finite bound M (Mins/Maxs saturate Inf), so no Inf can
    // survive into this routine. Mode 0 with a float dtype propagates NaN/Inf by
    // IEEE semantics and skips this; Mode 0 with an integer dtype calls it and lets
    // WriteBack saturate any Inf to the dtype bound. count must be 256-byte aligned.
    __aicore__ inline void ZeroNonFinite(const LocalTensor<float>& t, const LocalTensor<float>& zero,
                                         const LocalTensor<uint8_t>& mask, uint32_t alignedCount)
    {
        // NaN: x != x => Compare(EQ) returns false there. mask==true => finite.
        Compare(mask, t, t, CMPMODE::EQ, alignedCount);
        PipeBarrier<PIPE_V>();
        // dst = mask ? t : zero  (mask false at NaN -> zero).
        Select(t, mask, t, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCount);
        PipeBarrier<PIPE_V>();
    }

    // sRGB encoding curve on an already-clamped [0,1] tensor t, result in `out`:
    //   t <= 0.0031308 -> 12.92 * t
    //   else           -> 1.055 * t^(1/2.4) - 0.055
    __aicore__ inline void ComputeSRgb(const LocalTensor<float>& t, const LocalTensor<float>& lin,
                                       const LocalTensor<float>& powTmp, const LocalTensor<float>& out,
                                       const LocalTensor<uint8_t>& mask, uint32_t alignedCount)
    {
        // Linear branch: 12.92 * t.
        Muls(lin, t, SRGB_LINEAR_SLOPE, alignedCount);
        PipeBarrier<PIPE_V>();
        // Pow branch: t^(1/2.4) = exp((1/2.4) * ln(t)). t is clamped >= 0 above; the
        // subsequent denormalize+clamp maps any -inf from ln(0) back to 0.
        Log(powTmp, t, alignedCount);
        PipeBarrier<PIPE_V>();
        Muls(powTmp, powTmp, SRGB_INV_GAMMA, alignedCount);
        PipeBarrier<PIPE_V>();
        Exp(out, powTmp, alignedCount);
        PipeBarrier<PIPE_V>();
        Muls(out, out, SRGB_POW_COEFF, alignedCount);
        PipeBarrier<PIPE_V>();
        Adds(out, out, -SRGB_POW_OFFSET, alignedCount);
        PipeBarrier<PIPE_V>();
        // Predicate: mask = (t <= threshold). dst = mask ? lin : out.
        CompareScalar(mask, t, SRGB_THRESHOLD, CMPMODE::LE, alignedCount);
        PipeBarrier<PIPE_V>();
        Select(out, mask, lin, out, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        LocalTensor<float> xFloat = xFloatBuf_.Get<float>();
        LocalTensor<float> result = resultBuf_.Get<float>();
        LocalTensor<float> base = baseBuf_.Get<float>();
        LocalTensor<float> norm = normBuf_.Get<float>();
        LocalTensor<float> lin = linBuf_.Get<float>();
        LocalTensor<float> powTmp = powBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        // Conversion scratch (int8/uint8 half bridge, uint16 FP32 fix-up); lives in
        // the FP32 powBuf region, which is free during both conversions.
        LocalTensor<half> halfScratch = powBuf_.Get<half>();
        ConvertToFloat<T>(xFloat, input, count, halfScratch);
        PipeBarrier<PIPE_V>();

        // Compare requires a 256-byte-aligned count; round up only on paths that use
        // it (image modes, and the Mode 0 NaN cleanup for integer dtypes).
        const bool needsAlignedCount = correctionMode_ != optiling::GC_CORRECTION_PURE || IsInteger<T>::VALUE;
        const uint32_t alignCount = needsAlignedCount ? AlignCompareCount(count, sizeof(float)) : count;

        if (correctionMode_ == optiling::GC_CORRECTION_PURE) {
            // Mode 0: pure pow(x, gamma). Float dtypes keep IEEE NaN/Inf propagation.
            // Integer dtypes map NaN (negative base, non-integer gamma) to 0, then
            // WriteBack saturates to the dtype range before the narrowing cast.
            ComputePow(xFloat, result, base, count);
            PipeBarrier<PIPE_V>();
            if constexpr (IsInteger<T>::VALUE) {
                Duplicate(lin, 0.0F, alignCount);
                PipeBarrier<PIPE_V>();
                ZeroNonFinite(result, lin, mask, alignCount);
            }
            WriteBack(output, result, count, halfScratch);
        } else if (correctionMode_ == optiling::GC_CORRECTION_CLAMP) {
            // Mode 1: t = clamp(x, 0, M); y = clamp(pow(t/M, gamma) * M, 0, M).
            ClampRange(xFloat, 0.0F, maxValue_, count);
            Muls(norm, xFloat, invMax_, count); // normalize to [0,1]
            PipeBarrier<PIPE_V>();
            ComputePow(norm, result, powTmp, count);
            PipeBarrier<PIPE_V>();
            // Build a zero tensor for the NaN/Inf cleanup (Select tensor-tensor form).
            Duplicate(lin, 0.0F, alignCount);
            PipeBarrier<PIPE_V>();
            ZeroNonFinite(result, lin, mask, alignCount); // NaN/Inf from -0/ln -> 0
            Muls(result, result, maxValue_, count);       // denormalize
            PipeBarrier<PIPE_V>();
            ClampRange(result, 0.0F, maxValue_, count);
            WriteBack(output, result, count, halfScratch);
        } else {
            // Mode 2: sRGB encoding. t = clamp(x,0,M)/M; sRGB(t); *M; clamp.
            ClampRange(xFloat, 0.0F, maxValue_, count);
            Muls(norm, xFloat, invMax_, count);
            PipeBarrier<PIPE_V>();
            ComputeSRgb(norm, lin, powTmp, result, mask, alignCount);
            // Clean NaN/Inf produced by NaN inputs or ln(0) before denormalizing,
            // matching the host golden (isfinite(u) ? u : 0).
            Duplicate(lin, 0.0F, alignCount);
            PipeBarrier<PIPE_V>();
            ZeroNonFinite(result, lin, mask, alignCount);
            Muls(result, result, maxValue_, count);
            PipeBarrier<PIPE_V>();
            ClampRange(result, 0.0F, maxValue_, count);
            WriteBack(output, result, count, halfScratch);
        }

        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    // Narrow an FP32 result back to the typed output, applying the dtype-range
    // clamp required for integer casts (every mode goes through here).
    // halfScratch is the conversion scratch (free at this point).
    __aicore__ inline void WriteBack(const LocalTensor<T>& output, const LocalTensor<float>& result, uint32_t count,
                                     const LocalTensor<half>& halfScratch)
    {
        if constexpr (IsInteger<T>::VALUE) {
            ClampRange(result, IntRange<T>::LO, IntRange<T>::HI, count);
        }
        ConvertFromFloat<T>(output, result, count, halfScratch);
    }

    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count)
    {
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> xFloatBuf_;
    TBuf<TPosition::VECCALC> resultBuf_;
    TBuf<TPosition::VECCALC> baseBuf_;
    TBuf<TPosition::VECCALC> normBuf_;
    TBuf<TPosition::VECCALC> linBuf_;
    TBuf<TPosition::VECCALC> powBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint64_t totalNum_ = 0;
    uint64_t blockFactor_ = 0;
    uint64_t ubFactor_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t gammaMode_ = optiling::GC_MODE_GENERIC;
    uint32_t integerExponent_ = 0;
    uint32_t correctionMode_ = optiling::GC_CORRECTION_PURE;
    float gamma_ = 0.0F;
    float maxValue_ = 1.0F;
    float invMax_ = 1.0F;
};
} // namespace NsGammaCorrection

#endif // GAMMA_CORRECTION_IMPL_HPP_
