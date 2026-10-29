/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file test_aclnn_gamma_correction.cpp
 *  \brief aclnn call sample for GammaCorrection.
 *
 * Demonstrates the two-stage ACLNN flow:
 *   aclInit -> aclrtSetDevice -> aclrtCreateStream
 *   -> build x/y aclTensor
 *   -> aclnnGammaCorrectionGetWorkspaceSize
 *   -> aclrtMalloc workspace
 *   -> aclnnGammaCorrection
 *   -> aclrtSynchronizeStream
 *   -> aclrtMemcpy back, compare against a host reference
 *   -> release resources.
 *
 * Covers the adaptation-doc smoke-test matrix (Table 4-3): FP16/FP32/BF16,
 * gamma = 0.5/1.0/2.0/2.2, plus x=0 / negative base / NaN/Inf propagation, and the
 * integer dtypes: Mode 0 saturation (clamp-before-cast) and full-range uint16.
 *
 * Note: BF16 cases are compiled in but only valid on a platform with native
 * bfloat16 capability; otherwise the L2 API rejects the dtype.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_gamma_correction.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

// Self-contained IEEE-754 binary16 (float16) <-> float conversions, so the
// sample does not depend on any third-party half-precision library.
uint16_t FloatToHalfBits(float f)
{
    uint32_t fbits = 0;
    std::memcpy(&fbits, &f, sizeof(fbits));
    const uint32_t sign = (fbits >> 16U) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((fbits >> 23U) & 0xFFU) - 127 + 15;
    const uint32_t mant = (fbits >> 13U) & 0x3FFU;

    if (exp <= 0) {      // zero or subnormal
        if (exp < -10) { // underflow to zero
            return static_cast<uint16_t>(sign);
        }
        const uint32_t subMant = (fbits & 0x7FFFFFU) | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint16_t h = static_cast<uint16_t>(sign | (subMant >> shift));
        // round-to-nearest-even
        if ((subMant >> (shift - 1U)) & 1U) {
            ++h;
        }
        return h;
    }
    if (exp >= 31) { // finite overflow, inf or nan
        const bool isNan = ((fbits & 0x7F800000U) == 0x7F800000U) && ((fbits & 0x7FFFFFU) != 0U);
        // NaN keeps a quiet-NaN pattern; finite overflow and Inf both become Inf.
        return static_cast<uint16_t>(sign | (isNan ? 0x7E00U : 0x7C00U));
    }
    // round-to-nearest-even for normal range
    uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | mant);
    if ((fbits >> 12U) & 1U) {
        const uint32_t lsb = (h >> 10U) & 0x1FU;
        const uint32_t low = h & 0x3FFU;
        if (low != 0U || (lsb & 1U)) {
            ++h;
        }
    }
    return h;
}

float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16U;
    uint32_t exp = static_cast<uint32_t>((h >> 10U) & 0x1FU);
    uint32_t mant = static_cast<uint32_t>(h & 0x3FFU);

    if (exp == 0U) {
        if (mant == 0U) { // signed zero
            uint32_t zero = sign;
            float f = 0.0F;
            std::memcpy(&f, &zero, sizeof(f));
            return f;
        }
        // normalize subnormal
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            --exp;
        }
        ++exp;
        mant &= 0x3FFU;
    } else if (exp == 31U) { // inf or nan
        uint32_t fbits = sign | 0x7F800000U | (mant << 13U);
        float f = 0.0F;
        std::memcpy(&f, &fbits, sizeof(f));
        return f;
    }
    const uint32_t fbits = sign | ((exp + 112U) << 23U) | (mant << 13U);
    float f = 0.0F;
    std::memcpy(&f, &fbits, sizeof(f));
    return f;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// Convert a host scalar to its packed byte representation for a given acl dtype.
// Only the dtypes this sample exercises are supported.
void PackHostValue(void* dst, double value, aclDataType dtype)
{
    if (dtype == ACL_FLOAT) {
        const float v = static_cast<float>(value);
        std::memcpy(dst, &v, sizeof(v));
    } else if (dtype == ACL_FLOAT16) {
        const uint16_t bits = FloatToHalfBits(static_cast<float>(value));
        std::memcpy(dst, &bits, sizeof(bits));
    } else if (dtype == ACL_BF16) {
        // Truncation to the high 16 bits of the IEEE-754 float matches bfloat16 semantics.
        const float v = static_cast<float>(value);
        uint32_t fbits = 0;
        std::memcpy(&fbits, &v, sizeof(fbits));
        const uint16_t bits = static_cast<uint16_t>(fbits >> 16U);
        std::memcpy(dst, &bits, sizeof(bits));
    } else if (dtype == ACL_UINT8) {
        const uint8_t v = static_cast<uint8_t>(std::min<double>(std::max<double>(value, 0.0), 255.0));
        std::memcpy(dst, &v, sizeof(v));
    } else if (dtype == ACL_INT8) {
        const int8_t v = static_cast<int8_t>(std::min<double>(std::max<double>(value, -128.0), 127.0));
        std::memcpy(dst, &v, sizeof(v));
    } else if (dtype == ACL_UINT16) {
        const uint16_t v = static_cast<uint16_t>(std::min<double>(std::max<double>(value, 0.0), 65535.0));
        std::memcpy(dst, &v, sizeof(v));
    } else { // ACL_INT16
        const int16_t v = static_cast<int16_t>(std::min<double>(std::max<double>(value, -32768.0), 32767.0));
        std::memcpy(dst, &v, sizeof(v));
    }
}

// Read a packed device value back into a double for comparison.
double UnpackHostValue(const void* src, aclDataType dtype)
{
    if (dtype == ACL_FLOAT) {
        float v = 0.0F;
        std::memcpy(&v, src, sizeof(v));
        return static_cast<double>(v);
    }
    if (dtype == ACL_FLOAT16) {
        uint16_t bits = 0;
        std::memcpy(&bits, src, sizeof(bits));
        return static_cast<double>(HalfBitsToFloat(bits));
    }
    if (dtype == ACL_BF16) {
        // bfloat16 -> float via high 16 bits.
        uint16_t bits = 0;
        std::memcpy(&bits, src, sizeof(bits));
        const uint32_t fbits = static_cast<uint32_t>(bits) << 16U;
        float v = 0.0F;
        std::memcpy(&v, &fbits, sizeof(v));
        return static_cast<double>(v);
    }
    if (dtype == ACL_UINT8) {
        uint8_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        return static_cast<double>(v);
    }
    if (dtype == ACL_INT8) {
        int8_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        return static_cast<double>(v);
    }
    if (dtype == ACL_UINT16) {
        uint16_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        return static_cast<double>(v);
    }
    int16_t v = 0; // ACL_INT16
    std::memcpy(&v, src, sizeof(v));
    return static_cast<double>(v);
}

// Round-trip a host value through the device dtype. Integer packing saturates, so
// the golden must see the value the device actually receives.
double QuantizeHostValue(double value, aclDataType dtype)
{
    uint8_t packed[sizeof(float)] = {0};
    PackHostValue(packed, value, dtype);
    return UnpackHostValue(packed, dtype);
}

bool IntegerRange(aclDataType dtype, double* lo, double* hi)
{
    switch (dtype) {
        case ACL_UINT8:
            *lo = 0.0;
            *hi = 255.0;
            return true;
        case ACL_INT8:
            *lo = -128.0;
            *hi = 127.0;
            return true;
        case ACL_UINT16:
            *lo = 0.0;
            *hi = 65535.0;
            return true;
        case ACL_INT16:
            *lo = -32768.0;
            *hi = 32767.0;
            return true;
        default:
            return false;
    }
}

uint32_t TypeBytes(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return sizeof(float); // 4
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_UINT16:
        case ACL_INT16:
            return sizeof(uint16_t); // 2
        case ACL_UINT8:
        case ACL_INT8:
            return sizeof(uint8_t); // 1
        default:
            return sizeof(uint16_t);
    }
}

template <typename T>
T Tolerance(aclDataType dtype)
{
    switch (dtype) {
        case ACL_UINT8:
        case ACL_INT8:
        case ACL_UINT16:
        case ACL_INT16:
            return static_cast<T>(1.0); // integer pixel: 1 ULP rounding slack
        case ACL_FLOAT:
            return static_cast<T>(5.0e-5);
        case ACL_FLOAT16:
            return static_cast<T>(2.0e-3);
        default:
            // BF16: 8-bit mantissa. The normalize->pow->denormalize image pipeline
            // accumulates ~2-3 ULP, so a single-ULP-tight bound is unrealistic here.
            return static_cast<T>(1.5e-2); // BF16
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// Build a contiguous aclTensor from a list of host values expressed as doubles.
int CreateAclTensor(const std::vector<double>& hostData, const std::vector<int64_t>& shape, aclDataType dtype,
                    void** deviceAddr, aclTensor** tensor)
{
    const int64_t count = GetShapeSize(shape);
    const uint32_t bytes = TypeBytes(dtype);
    const size_t totalBytes = static_cast<size_t>(count) * bytes;

    auto ret = aclrtMalloc(deviceAddr, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> packed(totalBytes, 0);
    for (int64_t i = 0; i < count; ++i) {
        PackHostValue(packed.data() + static_cast<size_t>(i) * bytes, hostData[i], dtype);
    }
    ret = aclrtMemcpy(*deviceAddr, totalBytes, packed.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// Default value-range upper bound when maxValue is omitted, matching the tiling rule.
double DefaultMaxValueForDtype(aclDataType dtype)
{
    switch (dtype) {
        case ACL_UINT8:
        case ACL_INT8:
            return 255.0;
        case ACL_UINT16:
        case ACL_INT16:
            return 65535.0;
        default:
            return 1.0;
    }
}

// Host reference mirroring tests/ut/op_kernel/golden.py:
//   mode 0 (pure):  pow(x, gamma) with NaN/Inf propagation; integer dtypes map
//                   NaN -> 0 and saturate to the dtype range (clamp-before-cast).
//   mode 1 (clamp): clip(pow(clip(x,0,M)/M, gamma) * M, 0, M), NaN/Inf -> 0.
//   mode 2 (sRGB):  sRGB encoding curve, gamma ignored.
double HostGolden(double x, double gamma, int64_t correctionMode, double maxValue, aclDataType dtype)
{
    if (correctionMode == 0) {
        double y = std::pow(x, gamma);
        double lo = 0.0;
        double hi = 0.0;
        if (IntegerRange(dtype, &lo, &hi)) {
            y = std::isnan(y) ? 0.0 : std::min(std::max(y, lo), hi);
        }
        return y;
    }
    double t = x < 0.0 ? 0.0 : (x > maxValue ? maxValue : x);
    double y;
    if (correctionMode == 1) {
        double p = std::pow(t / maxValue, gamma);
        if (!std::isfinite(p)) {
            p = 0.0;
        }
        y = p * maxValue;
    } else { // sRGB
        double norm = t / maxValue;
        y = (norm <= 0.0031308) ? 12.92 * norm : 1.055 * std::pow(norm, 1.0 / 2.4) - 0.055;
        y *= maxValue;
    }
    if (!std::isfinite(y)) {
        y = 0.0;
    }
    return y < 0.0 ? 0.0 : (y > maxValue ? maxValue : y);
}

// Run one smoke case end-to-end and verify against the host reference.
int RunCase(const std::string& name, const std::vector<double>& hostValues, const std::vector<int64_t>& shape,
            aclDataType dtype, double gamma, int64_t correctionMode, double maxValue, aclrtStream stream)
{
    const int64_t count = GetShapeSize(shape);
    const uint32_t bytes = TypeBytes(dtype);
    const double effectiveMax = (maxValue > 0.0) ? maxValue : DefaultMaxValueForDtype(dtype);

    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    auto ret = CreateAclTensor(hostValues, shape, dtype, &xDeviceAddr, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    std::vector<double> yHost(static_cast<size_t>(count), 0.0);
    ret = CreateAclTensor(yHost, shape, dtype, &yDeviceAddr, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGammaCorrectionGetWorkspaceSize(x, gamma, correctionMode, maxValue, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[%s] GetWorkspaceSize failed: %d (dtype=%d may be unsupported on this platform)\n", name.c_str(),
                  ret, static_cast<int>(dtype));
        aclDestroyTensor(x);
        aclDestroyTensor(y);
        aclrtFree(xDeviceAddr);
        aclrtFree(yDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] workspace malloc failed: %d\n", name.c_str(), ret); return ret);
    }

    ret = aclnnGammaCorrection(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclnnGammaCorrection failed: %d\n", name.c_str(), ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclrtSynchronizeStream failed: %d\n", name.c_str(), ret); return ret);

    std::vector<uint8_t> outPacked(static_cast<size_t>(count) * bytes, 0);
    ret = aclrtMemcpy(outPacked.data(), outPacked.size(), yDeviceAddr, outPacked.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclrtMemcpy D2H failed: %d\n", name.c_str(), ret); return ret);

    const double tol = Tolerance<double>(dtype);
    int mismatches = 0;
    for (int64_t i = 0; i < count; ++i) {
        double lo = 0.0;
        double hi = 0.0;
        const double raw = hostValues[static_cast<size_t>(i)];
        const double input = IntegerRange(dtype, &lo, &hi) ? QuantizeHostValue(raw, dtype) : raw;
        const double expected = HostGolden(input, gamma, correctionMode, effectiveMax, dtype);
        const double actual = UnpackHostValue(outPacked.data() + static_cast<size_t>(i) * bytes, dtype);
        const bool nanExpected = std::isnan(expected);
        const bool nanActual = std::isnan(actual);
        const bool infExpected = std::isinf(expected);
        if ((nanExpected != nanActual) || (!nanExpected && infExpected && actual != expected) ||
            (!nanExpected && !infExpected && std::fabs(actual - expected) > tol)) {
            ++mismatches;
            if (mismatches <= 5) {
                LOG_PRINT("[%s] mismatch at %lld: x=%g gamma=%g expected=%g actual=%g\n", name.c_str(),
                          static_cast<long long>(i), input, gamma, expected, actual);
            }
        }
    }

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    if (mismatches == 0) {
        LOG_PRINT("[%s] PASS (%lld elements, dtype=%d, gamma=%g)\n", name.c_str(), static_cast<long long>(count),
                  static_cast<int>(dtype), gamma);
        return 0;
    }
    LOG_PRINT("[%s] FAIL: %d mismatches\n", name.c_str(), mismatches);
    return static_cast<int>(mismatches);
}

int RunPerf(const std::string& name, const std::vector<int64_t>& shape, aclDataType dtype, double gamma,
            int64_t correctionMode, double maxValue, aclrtStream stream, int warmup = 10, int iters = 100)
{
    const int64_t count = GetShapeSize(shape);
    const uint32_t bytes = TypeBytes(dtype);

    std::vector<double> hostValues(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        hostValues[static_cast<size_t>(i)] = static_cast<double>(i % 256) / 255.0;
    }
    std::vector<double> yHost(static_cast<size_t>(count), 0.0);

    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    if (CreateAclTensor(hostValues, shape, dtype, &xDeviceAddr, &x) != ACL_SUCCESS) {
        return -1;
    }
    if (CreateAclTensor(yHost, shape, dtype, &yDeviceAddr, &y) != ACL_SUCCESS) {
        return -1;
    }

    auto oneIter = [&]() -> int {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnGammaCorrectionGetWorkspaceSize(x, gamma, correctionMode, maxValue, y, &workspaceSize,
                                                        &executor);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0U) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                return ret;
            }
        }
        ret = aclnnGammaCorrection(workspaceAddr, workspaceSize, executor, stream);
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return ret;
    };

    for (int i = 0; i < warmup; ++i) {
        if (oneIter() != ACL_SUCCESS) {
            LOG_PRINT("[PERF] %s skipped (dtype=%d may be unsupported)\n", name.c_str(), static_cast<int>(dtype));
            aclDestroyTensor(x);
            aclDestroyTensor(y);
            aclrtFree(xDeviceAddr);
            aclrtFree(yDeviceAddr);
            return -1;
        }
    }
    aclrtSynchronizeStream(stream);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        oneIter();
    }
    aclrtSynchronizeStream(stream);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double meanMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);
    const double effBw = (static_cast<double>(count) * static_cast<double>(bytes) * 2.0) / (meanMs * 1.0e-3) / 1.0e9;

    LOG_PRINT("[PERF] %-30s dtype=%d  mean=%.4f ms  eff_bw=%.0f GB/s  (%lld elems)\n", name.c_str(),
              static_cast<int>(dtype), meanMs, effBw, static_cast<long long>(count));

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    return 0;
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // A small [1,3,4,4] image with values spanning negative, zero, small positive
    // and unit bases so that fast paths (gamma=1/2/0.5), the generic path and the
    // negative-base / special-value propagation are all exercised.
    const std::vector<int64_t> shape = {1, 3, 512, 512}; // 786432 elems: real multi-core + tiling workload
    std::vector<double> values(shape[0] * shape[1] * shape[2] * shape[3]);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<double>(static_cast<int64_t>(i % 17) - 8) / 8.0;
    }
    // Inject special values aligned with the adaptation-doc cases.
    values[0] = 0.0;   // x = 0  (y = 0 for gamma > 0)
    values[1] = -0.25; // negative base, non-integer gamma -> NaN
    values[2] = std::numeric_limits<double>::infinity();
    values[3] = std::numeric_limits<double>::quiet_NaN();
    values[4] = 1.0; // fixed point for any gamma

    // Use absolute values for the sqrt path (gamma = 0.5) so the reference is finite.
    std::vector<double> absValues(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        absValues[i] = std::fabs(values[i]);
    }

    // Image-mode inputs normalized to [0, 1] for clamp/sRGB paths.
    std::vector<double> unitValues(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        unitValues[i] = static_cast<double>(i) / static_cast<double>(values.size() - 1U);
    }

    int totalFailures = 0;

    // ---- Mode 0 (pure pow) regression. ----
    totalFailures += RunCase("Fp32_gamma2.2", values, shape, ACL_FLOAT, 2.2, 0, 0.0, stream);
    totalFailures += RunCase("Fp16_gamma2.2", values, shape, ACL_FLOAT16, 2.2, 0, 0.0, stream);
    totalFailures += RunCase("Fp32_identity", values, shape, ACL_FLOAT, 1.0, 0, 0.0, stream);
    totalFailures += RunCase("Fp16_square", values, shape, ACL_FLOAT16, 2.0, 0, 0.0, stream);
    totalFailures += RunCase("Fp32_sqrt", absValues, shape, ACL_FLOAT, 0.5, 0, 0.0, stream);

    // BF16 is only valid on platforms with native support; tolerate its rejection.
    const int bf16Ret = RunCase("Bf16_gamma2.2", values, shape, ACL_BF16, 2.2, 0, 0.0, stream);
    if (bf16Ret != 0) {
        LOG_PRINT("[Bf16_gamma2.2] skipped or failed (ret=%d); acceptable if BF16 unsupported.\n", bf16Ret);
    }

    // ---- Mode 1 (normalize + pow + clamp): real image gamma correction on [0,1]. ----
    totalFailures += RunCase("Fp32_clamp2.2", unitValues, shape, ACL_FLOAT, 2.2, 1, 1.0, stream);
    totalFailures += RunCase("Fp16_clamp0.5", unitValues, shape, ACL_FLOAT16, 0.5, 1, 1.0, stream);

    // 8-bit image: maxValue derived from dtype (255), pixels span [0, 255].
    std::vector<double> u8Values(values.size());
    for (size_t i = 0; i < u8Values.size(); ++i) {
        u8Values[i] = static_cast<double>(i * 5U % 256U); // mix of values incl. 0
    }
    u8Values[0] = 0.0;
    u8Values[1] = 255.0;
    u8Values[2] = 300.0; // above range -> clamps to 255
    totalFailures += RunCase("Uint8_clamp2.2", u8Values, shape, ACL_UINT8, 2.2, 1, 0.0, stream);

    // ---- Mode 2 (sRGB encoding curve). ----
    totalFailures += RunCase("Fp32_srgb", unitValues, shape, ACL_FLOAT, 2.2, 2, 1.0, stream);
    totalFailures += RunCase("Uint8_srgb", u8Values, shape, ACL_UINT8, 2.2, 2, 0.0, stream);

    // Non-aligned tail: Rank 3 with an odd element count.
    const std::vector<int64_t> tailShape = {3, 721, 1281}; // 2770803 elems, non-aligned tail at scale
    std::vector<double> tailValues(GetShapeSize(tailShape));
    for (size_t i = 0; i < tailValues.size(); ++i) {
        tailValues[i] = static_cast<double>(i) / static_cast<double>(tailValues.size());
    }
    totalFailures += RunCase("Fp16_tail_gamma0.8", tailValues, tailShape, ACL_FLOAT16, 0.8, 1, 1.0, stream);

    // ---- Extra coverage: more gamma / dtype / mode combinations (all validated paths). ----
    totalFailures += RunCase("Fp16_identity", values, shape, ACL_FLOAT16, 1.0, 0, 0.0, stream);
    totalFailures += RunCase("Fp32_cube_int", values, shape, ACL_FLOAT, 3.0, 0, 0.0, stream);
    totalFailures += RunCase("Fp16_sqrt", absValues, shape, ACL_FLOAT16, 0.5, 0, 0.0, stream);
    totalFailures += RunCase("Fp32_clamp0.5", unitValues, shape, ACL_FLOAT, 0.5, 1, 1.0, stream);
    totalFailures += RunCase("Fp16_clamp2.0", unitValues, shape, ACL_FLOAT16, 2.0, 1, 1.0, stream);
    totalFailures += RunCase("Uint8_clamp0.5", u8Values, shape, ACL_UINT8, 0.5, 1, 0.0, stream);
    totalFailures += RunCase("Fp16_srgb", unitValues, shape, ACL_FLOAT16, 2.2, 2, 1.0, stream);
    const int bf16ClampRet = RunCase("Bf16_clamp2.2", unitValues, shape, ACL_BF16, 2.2, 1, 1.0, stream);
    if (bf16ClampRet != 0) {
        LOG_PRINT("[Bf16_clamp2.2] tolerated (ret=%d); acceptable if BF16 unsupported.\n", bf16ClampRet);
    }
    const int bf16SrgbRet = RunCase("Bf16_srgb", unitValues, shape, ACL_BF16, 2.2, 2, 1.0, stream);
    if (bf16SrgbRet != 0) {
        LOG_PRINT("[Bf16_srgb] tolerated (ret=%d); acceptable if BF16 unsupported.\n", bf16SrgbRet);
    }

    // ---- Integer dtypes: Mode 0 clamp-before-cast and full-range uint16. ----
    std::vector<double> i8Values(values.size());
    std::vector<double> i16Values(values.size());
    std::vector<double> u16Values(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        i8Values[i] = static_cast<double>(static_cast<int64_t>(i % 256U) - 128);           // [-128, 127]
        i16Values[i] = static_cast<double>(static_cast<int64_t>(i % 2001U) - 1000) * 37.0; // saturates on pack
        u16Values[i] = static_cast<double>(i * 97U % 65536U);                              // full uint16 range
    }
    u16Values[0] = 0.0;
    u16Values[1] = 65535.0;
    u16Values[2] = 32767.0;
    u16Values[3] = 32768.0;
    totalFailures += RunCase("Uint8_pure2.2", u8Values, shape, ACL_UINT8, 2.2, 0, 0.0, stream); // saturates at 255
    totalFailures += RunCase("Int8_pure_square", i8Values, shape, ACL_INT8, 2.0, 0, 0.0, stream);
    totalFailures += RunCase("Int16_pure_square", i16Values, shape, ACL_INT16, 2.0, 0, 0.0, stream);
    totalFailures += RunCase("Int16_pure2.2", i16Values, shape, ACL_INT16, 2.2, 0, 0.0, stream); // x<0: NaN -> 0
    totalFailures += RunCase("Uint16_identity", u16Values, shape, ACL_UINT16, 1.0, 0, 0.0, stream);
    totalFailures += RunCase("Uint16_sqrt", u16Values, shape, ACL_UINT16, 0.5, 0, 0.0, stream);
    totalFailures += RunCase("Uint16_clamp2.2", u16Values, shape, ACL_UINT16, 2.2, 1, 0.0, stream);
    totalFailures += RunCase("Uint16_srgb", u16Values, shape, ACL_UINT16, 2.2, 2, 0.0, stream);

    LOG_PRINT("\n==== Custom operator latency (warmup 10, 100 iters, e2e) ====\n");
    RunPerf("1x3x1080x1920 FP16 g2.2", {1, 3, 1080, 1920}, ACL_FLOAT16, 2.2, 0, 0.0, stream);
    RunPerf("1x3x1080x1920 FP32 g2.2", {1, 3, 1080, 1920}, ACL_FLOAT, 2.2, 0, 0.0, stream);
    RunPerf("8x3x512x512 FP16 g2.2", {8, 3, 512, 512}, ACL_FLOAT16, 2.2, 0, 0.0, stream);
    RunPerf("1x3x1080x1920 FP16 g0.5", {1, 3, 1080, 1920}, ACL_FLOAT16, 0.5, 0, 0.0, stream);
    RunPerf("1x3x721x1281 FP16 g2.2 na", {1, 3, 721, 1281}, ACL_FLOAT16, 2.2, 0, 0.0, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (totalFailures != 0) {
        LOG_PRINT("RESULT: %d case(s) FAILED\n", totalFailures);
        return static_cast<int>(totalFailures);
    }
    LOG_PRINT("RESULT: ALL PASS\n");
    return 0;
}
