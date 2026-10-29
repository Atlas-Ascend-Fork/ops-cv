/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "gamma_correction_tiling_def.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void gamma_correction(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

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
    uint8_t* Get() const { return ptr_; }

private:
    uint8_t* ptr_ = nullptr;
};

template <typename T>
constexpr bool IsIntegerType()
{
    return std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
           std::is_same_v<T, uint16_t>;
}

template <typename T>
float ToFloat(T value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToFloat(value);
    }
    return static_cast<float>(value);
}

// Saturating cast from float into an integer image dtype, matching the kernel's
// clamp-before-cast (CAST_RINT) semantics: NaN -> 0, +/-Inf saturate to the bounds.
template <typename T>
T FromFloat(float value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToBfloat16(value);
    } else if constexpr (IsIntegerType<T>()) {
        if (std::isnan(value)) {
            value = 0.0F;
        }
        constexpr float lo = std::numeric_limits<T>::lowest();
        constexpr float hi = static_cast<float>(std::numeric_limits<T>::max());
        if (value < lo) {
            value = lo;
        } else if (value > hi) {
            value = hi;
        }
        return static_cast<T>(std::nearbyint(value));
    }
    return static_cast<T>(value);
}

template <typename T>
float Tolerance()
{
    if constexpr (IsIntegerType<T>()) {
        // Integer pixel output: allow 1 ULP for the final rounding difference.
        return 1.0F;
    } else if constexpr (std::is_same_v<T, float>) {
        return 5.0e-5F;
    } else if constexpr (std::is_same_v<T, half>) {
        return 2.0e-3F;
    }
    return 8.0e-3F; // BF16
}

// Mode 0 input pattern for integer dtypes: spans the whole dtype range (incl. the
// uint16 values above 32767).
template <typename T>
float IntegerRangeSpan(uint64_t i, uint64_t totalNum)
{
    if constexpr (IsIntegerType<T>()) {
        const float lo = static_cast<float>(std::numeric_limits<T>::lowest());
        const float hi = static_cast<float>(std::numeric_limits<T>::max());
        return lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(totalNum > 1U ? totalNum - 1U : 1U);
    } else {
        return 0.0F;
    }
}

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1U) / alignment * alignment; }

// Default value-range upper bound when maxValue is omitted, matching the tiling rule.
template <typename T>
float DefaultMaxValue()
{
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
        return 255.0F;
    } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) {
        return 65535.0F;
    } else {
        return 1.0F;
    }
}

// Host reference mirroring golden.py: pow with NaN/Inf/over/under-flow propagation.
float PurePowGolden(float x, float gamma) { return std::pow(x, gamma); }

// Mode 1: t = clamp(x,0,M); y = clamp(pow(t/M, gamma) * M, 0, M) with NaN/Inf -> 0.
float ClampModeGolden(float x, float gamma, float maxValue)
{
    float t = x < 0.0F ? 0.0F : (x > maxValue ? maxValue : x);
    float p = std::pow(t / maxValue, gamma);
    if (std::isnan(p) || std::isinf(p)) {
        p = 0.0F;
    }
    float y = p * maxValue;
    return y < 0.0F ? 0.0F : (y > maxValue ? maxValue : y);
}

// Mode 2: sRGB encoding curve.
float SrgbModeGolden(float x, float maxValue)
{
    float t = x < 0.0F ? 0.0F : (x > maxValue ? maxValue : x);
    t /= maxValue;
    float u = (t <= 0.0031308F) ? 12.92F * t : 1.055F * std::pow(t, 1.0F / 2.4F) - 0.055F;
    if (std::isnan(u) || std::isinf(u)) {
        u = 0.0F;
    }
    float y = u * maxValue;
    return y < 0.0F ? 0.0F : (y > maxValue ? maxValue : y);
}

template <typename T>
void RunCase(uint64_t tilingKey, uint32_t gammaMode, float gamma, uint32_t integerExponent, uint64_t totalNum,
             uint32_t usedCoreNum, uint32_t correctionMode = optiling::GC_CORRECTION_PURE, float maxValue = 0.0F,
             bool includeSpecialValues = false)
{
    const float effectiveMax = (maxValue > 0.0F) ? maxValue : DefaultMaxValue<T>();

    std::vector<T> input(totalNum);
    for (uint64_t i = 0; i < totalNum; ++i) {
        // For image modes keep inputs within a sane pixel range; for the pure-pow
        // mode use the original signed test pattern.
        float value;
        if (correctionMode == optiling::GC_CORRECTION_PURE && IsIntegerType<T>()) {
            value = IntegerRangeSpan<T>(i, totalNum);
        } else if (correctionMode == optiling::GC_CORRECTION_PURE) {
            value = gammaMode == optiling::GC_MODE_SQRT ? static_cast<float>(i % 17U) / 16.0F :
                                                          static_cast<float>(static_cast<int32_t>(i % 17U) - 8) / 8.0F;
        } else {
            // Span [0, maxValue] across the tile.
            value = static_cast<float>(i) / static_cast<float>(totalNum > 1U ? totalNum - 1U : 1U) * effectiveMax;
        }
        input[i] = FromFloat<T>(value);
    }
    if (includeSpecialValues && totalNum >= 5) {
        if constexpr (IsIntegerType<T>()) {
            // Integers cannot represent NaN/Inf; use the dtype-specific edge values.
            input[0] = FromFloat<T>(0.0F);
            input[1] = FromFloat<T>(-1.0F); // below range -> clamps to 0
            input[2] = FromFloat<T>(effectiveMax);
            input[3] = FromFloat<T>(effectiveMax + 10.0F); // above range -> clamps
            input[4] = FromFloat<T>(1.0F);
        } else {
            input[0] = FromFloat<T>(0.0F);
            input[1] = FromFloat<T>(-0.25F);
            input[2] = FromFloat<T>(std::numeric_limits<float>::infinity());
            input[3] = FromFloat<T>(std::numeric_limits<float>::quiet_NaN());
            input[4] = FromFloat<T>(1.0F);
        }
    }

    std::vector<float> expected(totalNum);
    for (uint64_t i = 0; i < totalNum; ++i) {
        const float x = ToFloat(input[i]);
        if (correctionMode == optiling::GC_CORRECTION_PURE) {
            expected[i] = PurePowGolden(x, gamma);
        } else if (correctionMode == optiling::GC_CORRECTION_CLAMP) {
            expected[i] = ClampModeGolden(x, gamma, effectiveMax);
        } else {
            expected[i] = SrgbModeGolden(x, effectiveMax);
        }
    }

    GmBuffer inputGm(totalNum * sizeof(T));
    GmBuffer outputGm(totalNum * sizeof(T));
    GmBuffer workspaceGm(32);
    GmBuffer tilingGm(sizeof(optiling::GammaCorrectionTilingData));
    ASSERT_NE(inputGm.Get(), nullptr);
    ASSERT_NE(outputGm.Get(), nullptr);
    ASSERT_NE(tilingGm.Get(), nullptr);
    std::memcpy(inputGm.Get(), input.data(), totalNum * sizeof(T));

    // ubFactor must be a multiple of 256/typeBytes so the vector Compare predicate
    // stays on its required 256-byte boundary for every dtype.
    constexpr uint32_t typeBytes = sizeof(T);
    const uint32_t ubAlignElements = static_cast<uint32_t>(256U / typeBytes);
    const uint32_t ubFactor = std::max<uint32_t>(ubAlignElements, 128U);
    ASSERT_EQ(ubFactor % ubAlignElements, 0U);

    auto* tiling = reinterpret_cast<optiling::GammaCorrectionTilingData*>(tilingGm.Get());
    std::memset(tiling, 0, sizeof(*tiling));
    tiling->totalNum = totalNum;
    tiling->blockFactor = AlignUp((totalNum + usedCoreNum - 1U) / usedCoreNum, 512);
    tiling->ubFactor = ubFactor;
    tiling->usedCoreNum = usedCoreNum;
    tiling->gammaMode = gammaMode;
    tiling->integerExponent = integerExponent;
    tiling->correctionMode = correctionMode;
    tiling->gamma = gamma;
    tiling->maxValue = effectiveMax;
    tiling->invMax = 1.0F / effectiveMax;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(gamma_correction, usedCoreNum, inputGm.Get(), outputGm.Get(), workspaceGm.Get(), tilingGm.Get());

    const T* actual = reinterpret_cast<const T*>(outputGm.Get());
    for (uint64_t i = 0; i < totalNum; ++i) {
        const float roundedExpected = ToFloat(FromFloat<T>(expected[i]));
        const float actualValue = ToFloat(actual[i]);
        if (std::isnan(roundedExpected)) {
            EXPECT_TRUE(std::isnan(actualValue)) << "expected NaN at index " << i;
        } else if (std::isinf(roundedExpected)) {
            EXPECT_EQ(actualValue, roundedExpected) << "infinity mismatch at index " << i;
        } else {
            EXPECT_NEAR(actualValue, roundedExpected, Tolerance<T>()) << "mismatch at index " << i;
        }
    }
}
} // namespace

// ---- Mode 0 (pure pow) regression: original behaviour preserved. ----

TEST(GammaCorrectionKernel, Fp32GenericTailAndMulticore)
{
    RunCase<float>(optiling::GC_TILING_KEY_FP32, optiling::GC_MODE_GENERIC, 2.2F, 0, 1031, 2,
                   optiling::GC_CORRECTION_PURE, 0.0F, true);
}

TEST(GammaCorrectionKernel, Fp16SqrtTail)
{
    RunCase<half>(optiling::GC_TILING_KEY_FP16, optiling::GC_MODE_SQRT, 0.5F, 0, 37, 1);
}

TEST(GammaCorrectionKernel, Bf16SquareNegativeInputs)
{
    RunCase<bfloat16_t>(optiling::GC_TILING_KEY_BF16, optiling::GC_MODE_SQUARE, 2.0F, 0, 65, 1);
}

TEST(GammaCorrectionKernel, Fp32IntegerCubeNegativeInputs)
{
    RunCase<float>(optiling::GC_TILING_KEY_FP32, optiling::GC_MODE_INTEGER, 3.0F, 3, 257, 1);
}

TEST(GammaCorrectionKernel, Fp16Identity)
{
    RunCase<half>(optiling::GC_TILING_KEY_FP16, optiling::GC_MODE_IDENTITY, 1.0F, 0, 33, 1);
}

// ---- Mode 1 (normalize + pow + clamp): image gamma correction on [0, M]. ----

TEST(GammaCorrectionKernel, Fp32ClampModeGamma22)
{
    RunCase<float>(optiling::GC_TILING_KEY_FP32, optiling::GC_MODE_GENERIC, 2.2F, 0, 257, 1,
                   optiling::GC_CORRECTION_CLAMP, 1.0F, true);
}

TEST(GammaCorrectionKernel, Fp16ClampModeGamma05)
{
    RunCase<half>(optiling::GC_TILING_KEY_FP16, optiling::GC_MODE_GENERIC, 0.5F, 0, 193, 1,
                  optiling::GC_CORRECTION_CLAMP, 1.0F, true);
}

TEST(GammaCorrectionKernel, Uint8ClampModePixelImage)
{
    // 8-bit image: maxValue derived from dtype (255). Multi-core + tail.
    RunCase<uint8_t>(optiling::GC_TILING_KEY_UINT8, optiling::GC_MODE_GENERIC, 2.2F, 0, 513, 2,
                     optiling::GC_CORRECTION_CLAMP, 0.0F, true);
}

// ---- Mode 2 (sRGB encoding curve). ----

TEST(GammaCorrectionKernel, Fp32SrgbEncoding)
{
    RunCase<float>(optiling::GC_TILING_KEY_FP32, optiling::GC_MODE_GENERIC, 2.2F, 0, 257, 1,
                   optiling::GC_CORRECTION_SRGB, 1.0F, true);
}

TEST(GammaCorrectionKernel, Uint8SrgbEncoding)
{
    RunCase<uint8_t>(optiling::GC_TILING_KEY_UINT8, optiling::GC_MODE_GENERIC, 2.2F, 0, 257, 1,
                     optiling::GC_CORRECTION_SRGB, 0.0F, true);
}

// ---- Integer dtypes: Mode 0 clamp-before-cast and full-range uint16. ----

TEST(GammaCorrectionKernel, Uint8PurePowSaturates)
{
    // 255^2.2 ~ 2.0e5 would overflow the half bridge; the dtype clamp keeps it at 255.
    RunCase<uint8_t>(optiling::GC_TILING_KEY_UINT8, optiling::GC_MODE_GENERIC, 2.2F, 0, 257, 1);
}

TEST(GammaCorrectionKernel, Int8PureSquareSaturates)
{
    RunCase<int8_t>(optiling::GC_TILING_KEY_INT8, optiling::GC_MODE_SQUARE, 2.0F, 0, 256, 1);
}

TEST(GammaCorrectionKernel, Int16PureGenericNegativeToZero)
{
    // Negative base with non-integer gamma is NaN in FP32 and must come out as 0.
    RunCase<int16_t>(optiling::GC_TILING_KEY_INT16, optiling::GC_MODE_GENERIC, 2.2F, 0, 257, 1);
}

TEST(GammaCorrectionKernel, Uint16PureIdentityFullRange)
{
    RunCase<uint16_t>(optiling::GC_TILING_KEY_UINT16, optiling::GC_MODE_IDENTITY, 1.0F, 0, 513, 2);
}

TEST(GammaCorrectionKernel, Uint16PureSqrtFullRange)
{
    RunCase<uint16_t>(optiling::GC_TILING_KEY_UINT16, optiling::GC_MODE_SQRT, 0.5F, 0, 257, 1);
}

TEST(GammaCorrectionKernel, Uint16ClampModeFullRange)
{
    // maxValue derived from dtype (65535); outputs cover both halves of the range.
    RunCase<uint16_t>(optiling::GC_TILING_KEY_UINT16, optiling::GC_MODE_GENERIC, 2.2F, 0, 513, 2,
                      optiling::GC_CORRECTION_CLAMP, 0.0F, true);
}

TEST(GammaCorrectionKernel, Uint16SrgbEncodingFullRange)
{
    RunCase<uint16_t>(optiling::GC_TILING_KEY_UINT16, optiling::GC_MODE_GENERIC, 2.2F, 0, 257, 1,
                      optiling::GC_CORRECTION_SRGB, 0.0F, true);
}
