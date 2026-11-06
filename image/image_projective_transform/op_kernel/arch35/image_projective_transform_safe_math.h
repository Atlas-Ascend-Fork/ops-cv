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
 * \file image_projective_transform_safe_math.h
 * \brief Safe arithmetic helpers for image_projective_transform SIMT kernel
 */

#ifndef IMAGE_PROJECTIVE_TRANSFORM_SAFE_MATH_H_
#define IMAGE_PROJECTIVE_TRANSFORM_SAFE_MATH_H_

#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"

namespace NsImageProjectiveTransform {

using namespace AscendC;

// R006: __launch_bounds__ templated on index width.
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? 1024 : 512;

static constexpr int64_t INT32_MAX_VAL = 0x7FFFFFFFLL;
static constexpr float FLT_MAX_VAL = 3.4028234663852886e38f;

// ===== IEEE 754 special-value construction (union bit-cast, no traps) =====
__simt_callee__ inline float MakeInf(bool neg)
{
    union {
        uint32_t u;
        float f;
    } c;
    c.u = neg ? 0xFF800000U : 0x7F800000U;
    return c.f;
}

__simt_callee__ inline float MakeZero(bool neg)
{
    union {
        uint32_t u;
        float f;
    } c;
    c.u = neg ? 0x80000000U : 0x00000000U;
    return c.f;
}

__simt_callee__ inline float MakeNan()
{
    union {
        uint32_t u;
        float f;
    } c;
    c.u = 0x7FC00000U;
    return c.f;
}

// ===== Safe multiply: a*b following IEEE 754, never trapping the NPU =====
__simt_callee__ inline float SafeMul(float a, float b)
{
    if (isnan(a) || isnan(b)) {
        return MakeNan();
    }
    bool neg = (signbit(a) ^ signbit(b)) != 0;
    if (isinf(a)) {
        if (b == 0.0f) {
            return MakeNan();
        }
        return MakeInf(neg);
    }
    if (isinf(b)) {
        if (a == 0.0f) {
            return MakeNan();
        }
        return MakeInf(neg);
    }
    float aa = fabsf(a);
    float bb = fabsf(b);
    if (bb == 0.0f) {
        return 0.0f;
    }
    if (aa > (FLT_MAX_VAL / bb)) {
        return MakeInf(neg);
    }
    return a * b;
}

// ===== Safe add: a+b following IEEE 754, never trapping the NPU =====
__simt_callee__ inline float SafeAdd(float a, float b)
{
    if (isnan(a) || isnan(b)) {
        return MakeNan();
    }
    if (isinf(a)) {
        if (isinf(b) && (signbit(a) != signbit(b))) {
            return MakeNan();
        }
        return a;
    }
    if (isinf(b)) {
        return b;
    }
    if (signbit(a) == signbit(b)) {
        float ra = fabsf(a) / FLT_MAX_VAL;
        float rb = fabsf(b) / FLT_MAX_VAL;
        if (ra + rb > 1.0f) {
            return MakeInf(signbit(a));
        }
    }
    return a + b;
}

__simt_callee__ inline float FlushSubnormal(float x)
{
    constexpr float FLT_MIN_VAL = 1.1754942e-38f;
    if (x != 0.0f && fabsf(x) < FLT_MIN_VAL) {
        return MakeZero(signbit(x));
    }
    return x;
}

__simt_callee__ inline float SafeSum3(float x, float y, float z) { return SafeAdd(SafeAdd(x, y), z); }

// ===== Safe & accurate division (IEEE 754 special cases + corrective step) =====
// For ordinary finite operands it equals a/b plus the same corrective step the
// fast path used; non-finite divisors/quotients follow IEEE 754 (finite/inf == 0)
// so the caller's non-finite interception sees the same declared semantics on
// every path.
__simt_callee__ inline float SafeDiv(float a, float b)
{
    bool sa = signbit(a);
    bool sb = signbit(b);
    bool neg = (sa ^ sb) != 0;
    if (isnan(a) || isnan(b)) {
        return MakeNan();
    }
    if (isinf(b)) {
        if (isinf(a)) {
            return MakeNan();
        }
        return MakeZero(neg);
    }
    if (isinf(a)) {
        return MakeInf(neg);
    }
    if (b == 0.0f) {
        if (a == 0.0f) {
            return MakeNan();
        }
        return MakeInf(neg);
    }
    float q = a / b;
    // A finite/finite division may overflow to +/-Inf. The Newton correction
    // would then evaluate Inf - Inf and turn the correct IEEE result into NaN.
    if (isinf(q) || isnan(q)) {
        return q;
    }
    float residual = fmaf(q, b, -a);
    return q - residual / b;
}

// ===== Safe float -> int32 casts =====
__simt_callee__ inline int32_t SafeFloorToInt32(float f)
{
    if (f > 2.1474835e9f) {
        // INT32_MAX is never a valid index (valid range is [0, len-1] with
        // len <= INT32_MAX), so it stays OOB even when len == INT32_MAX.
        return 2147483647;
    }
    if (f < -2.1474835e9f) {
        return -2147483647;
    }
    return static_cast<int32_t>(floorf(f));
}

__simt_callee__ inline int32_t SafeRoundToInt32(float f)
{
    if (f > 2.1474835e9f) {
        // INT32_MAX is never a valid index (valid range is [0, len-1] with
        // len <= INT32_MAX), so it stays OOB even when len == INT32_MAX.
        return 2147483647;
    }
    if (f < -2.1474835e9f) {
        return -2147483647;
    }
    return static_cast<int32_t>(roundf(f));
}

// ===== Safe float -> int32 truncation (matches x86 cvttss2si overflow behavior) =====
// NaN/Inf/overflow (|val| >= 2^31) return INT_MIN, matching x86 cvttss2si.
// Uses a single integer comparison on the IEEE 754 magnitude (absU):
//   2^31.0f encodes as 0x4F000000; any |val| >= 2^31, Inf, or NaN has
//   absU >= 0x4F000000, so one comparison covers all three cases.
__simt_callee__ inline int32_t SafeTruncToInt32(float val)
{
    uint32_t absU = __float_as_uint(val) & 0x7FFFFFFFu;
    if (absU >= 0x4F000000U) {
        return INT32_MIN;
    }
    return static_cast<int32_t>(val);
}

// TF 2.21 实测：INT64 输出按 float32 插值后向零截断；NaN/Inf 或 |值| >= 2^63
// （float32 无法表示的 int64 范围外值）输出 INT64_MIN。
__simt_callee__ inline int64_t SafeTruncToInt64(float val)
{
    uint32_t absU = __float_as_uint(val) & 0x7FFFFFFFu;
    if (absU >= 0x5F000000U) { // |val| >= 2^63（含 Inf/NaN 的指数位）
        return INT64_MIN;
    }
    return static_cast<int64_t>(val);
}

// ===== Fast-path selector =====
__aicore__ inline bool IsFiniteAndBounded(float f, uint32_t threshAbsU)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    uint32_t absU = c.u & 0x7FFFFFFFu;
    uint32_t exp = (absU >> 23) & 0xFFu;
    return exp != 0xFFu && absU <= threshAbsU;
}

__aicore__ inline bool IsFinite(float f)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    uint32_t absU = c.u & 0x7FFFFFFFu;
    uint32_t exp = (absU >> 23) & 0xFFu;
    return exp != 0xFFu;
}

__aicore__ inline bool IsFastPath(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7)
{
    union {
        float f;
        uint32_t u;
    } th;
    th.f = 1e30f;
    uint32_t threshAbs = th.u & 0x7FFFFFFFu;
    return IsFiniteAndBounded(a0, threshAbs) && IsFiniteAndBounded(a1, threshAbs) && IsFinite(a2) &&
           IsFiniteAndBounded(a3, threshAbs) && IsFiniteAndBounded(a4, threshAbs) && IsFinite(a5) &&
           IsFiniteAndBounded(a6, threshAbs) && IsFiniteAndBounded(a7, threshAbs);
}

} // namespace NsImageProjectiveTransform

#endif // IMAGE_PROJECTIVE_TRANSFORM_SAFE_MATH_H_
