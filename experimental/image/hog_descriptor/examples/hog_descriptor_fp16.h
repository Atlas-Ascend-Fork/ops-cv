/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOG_DESCRIPTOR_EXAMPLE_FP16_H_
#define HOG_DESCRIPTOR_EXAMPLE_FP16_H_

#include <cmath>
#include <cstdint>
#include <cstring>

namespace HogExample {
inline uint16_t Fp32ToFp16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 0x1U;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = x & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign << 15);
        }
        mant |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t sub = mant >> shift;
        const uint32_t rem = mant & ((1U << shift) - 1U);
        const uint32_t half = 1U << (shift - 1U);
        uint32_t rounded = sub + ((rem > half || (rem == half && (sub & 1U))) ? 1U : 0U);
        return static_cast<uint16_t>((sign << 15) + rounded);
    }
    if (exp >= 31) {
        const uint32_t nanPayload = (x & 0x7fffffffU) > 0x7f800000U ? 0x200U : 0U;
        return static_cast<uint16_t>((sign << 15) | (0x1fU << 10) | nanPayload);
    }
    uint32_t half = mant & 0x1fffU;
    mant >>= 13;
    if (half > 0x1000U || (half == 0x1000U && (mant & 1U) != 0U)) {
        mant += 1U;
        if (mant == 0x400U) {
            mant = 0U;
            return static_cast<uint16_t>((sign << 15) | (((exp + 1) & 0x1fU) << 10) | mant);
        }
    }
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
}

inline float Fp16ToFp32(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 0x1U;
    const uint32_t exp = (h >> 10) & 0x1fU;
    const uint32_t mant = h & 0x3ffU;
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        const float value = std::ldexp(static_cast<float>(mant), -24);
        return sign != 0U ? -value : value;
    }
    if (exp == 31) {
        const uint32_t x = (sign << 31) | (0xffU << 23) | (mant ? (1U << 22) : 0U);
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }
    const uint32_t x = (sign << 31) | ((exp - 15U + 127U) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

} // namespace HogExample

#endif // HOG_DESCRIPTOR_EXAMPLE_FP16_H_
