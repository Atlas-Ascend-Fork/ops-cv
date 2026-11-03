/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

// GaussianFilter3D NPU单算子测试：
//   test_aclnn_gaussian_filter_3_d <dtype:float32|float16|bfloat16> <D> <H> <W> <sigma> <MODE> [batch] [seed]
// 与内置CPU fp64可分离高斯参考（语义对齐scipy.ndimage.gaussian_filter, truncate=4.0）比对。

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include <cstring>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_gaussian_filter_3_d.h"

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

constexpr uint32_t MODE_REFLECT = 0;
constexpr uint32_t MODE_CONSTANT = 1;
constexpr uint32_t MODE_NEAREST = 2;
constexpr uint32_t MODE_MIRROR = 3;
constexpr uint32_t MODE_WRAP = 4;

uint32_t ParseMode(const std::string& s)
{
    std::string up;
    for (auto c : s) {
        up += static_cast<char>(toupper(static_cast<unsigned char>(c)));
    }
    if (up == "REFLECT")
        return MODE_REFLECT;
    if (up == "CONSTANT")
        return MODE_CONSTANT;
    if (up == "NEAREST")
        return MODE_NEAREST;
    if (up == "MIRROR")
        return MODE_MIRROR;
    if (up == "WRAP")
        return MODE_WRAP;
    return 99;
}

int64_t MapIdx(int64_t i, int64_t n, uint32_t mode)
{
    if (i >= 0 && i < n)
        return i;
    if (mode == MODE_CONSTANT)
        return -1;
    if (n == 1)
        return 0;
    if (mode == MODE_NEAREST)
        return i < 0 ? 0 : n - 1;
    if (mode == MODE_REFLECT) {
        int64_t p = 2 * n, j = i % p;
        if (j < 0)
            j += p;
        return j >= n ? p - 1 - j : j;
    }
    if (mode == MODE_MIRROR) {
        int64_t p = 2 * n - 2, j = i % p;
        if (j < 0)
            j += p;
        return j >= n ? p - j : j;
    }
    int64_t j = i % n;
    return j < 0 ? j + n : j;
}

void RefGaussian3D(const float* in, float* out, int64_t d, int64_t h, int64_t w, const std::vector<double>& wts,
                   uint32_t r, uint32_t mode)
{
    int64_t elem = d * h * w;
    std::vector<double> a(in, in + elem);
    std::vector<double> b(elem, 0.0);
    double* src = a.data();
    double* dst = b.data();
    const int64_t axisLen[3] = {w, h, d};
    const int64_t axisStride[3] = {1, w, h * w};
    for (int32_t pass = 0; pass < 3; pass++) {
        int64_t n = axisLen[pass], st = axisStride[pass];
        int64_t outer = elem / n;
        for (int64_t oi = 0; oi < outer; oi++) {
            int64_t base = (oi / st) * (n * st) + (oi % st);
            for (int64_t i = 0; i < n; i++) {
                double acc = 0.0;
                for (int64_t k = 0; k < static_cast<int64_t>(2 * r + 1); k++) {
                    int64_t j = MapIdx(i + k - static_cast<int64_t>(r), n, mode);
                    if (j < 0)
                        continue;
                    acc += wts[k] * src[base + j * st];
                }
                dst[base + i * st] = acc;
            }
        }
        std::swap(src, dst);
    }
    for (int64_t i = 0; i < elem; i++)
        out[i] = static_cast<float>(src[i]);
}

float Bits16ToFloat(uint16_t h, bool isBf16)
{
    uint32_t bits = 0;
    if (isBf16) {
        bits = static_cast<uint32_t>(h) << 16;
    } else {
        uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
        uint32_t exp = (h & 0x7C00) >> 10;
        uint32_t man = h & 0x03FF;
        if (exp == 0) {
            if (man == 0) {
                bits = sign;
            } else {
                uint32_t e = 127 - 15 + 1;
                while ((man & 0x0400) == 0) {
                    man <<= 1;
                    e--;
                }
                man &= 0x03FF;
                bits = sign | (e << 23) | (man << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000 | (man << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
        }
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

uint16_t FloatToBits16(float f, bool isBf16)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    if (isBf16)
        return static_cast<uint16_t>((bits >> 16) & 0xFFFF);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = bits & 0x007FFFFF;
    if (((bits >> 23) & 0xFF) == 0xFF)
        return static_cast<uint16_t>(sign | 0x7C00);
    if (exp >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10)
            return static_cast<uint16_t>(sign);
        man = (man | 0x00800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((man + 0x1000) >> 13));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | ((man + 0x1000) >> 13));
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto i : shape)
        s *= i;
    return s;
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

} // namespace

int main(int argc, char** argv)
{
    // usage: <dtype> <D> <H> <W> <sigma> <mode> [batch=1] [seed=12345] [pattern=rand|zero|neg|extreme]
    std::string dtype = (argc > 1) ? argv[1] : "float32";
    int64_t D = (argc > 2) ? std::stoll(argv[2]) : 4;
    int64_t H = (argc > 3) ? std::stoll(argv[3]) : 16;
    int64_t W = (argc > 4) ? std::stoll(argv[4]) : 16;
    double sigma = (argc > 5) ? std::stod(argv[5]) : 1.0;
    std::string modeStr = (argc > 6) ? argv[6] : "REFLECT";
    int64_t batch = (argc > 7) ? std::stoll(argv[7]) : 1;
    int64_t seed = (argc > 8) ? std::stoll(argv[8]) : 12345;
    std::string pattern = (argc > 9) ? argv[9] : "rand";

    uint32_t mode = ParseMode(modeStr);
    if (mode == 99) {
        LOG_PRINT("unsupported mode %s\n", modeStr.c_str());
        return -1;
    }
    bool isFp32 = dtype == "float32";
    bool isBf16 = dtype == "bfloat16";
    if (!isFp32 && !isBf16 && dtype != "float16") {
        LOG_PRINT("unsupported dtype %s\n", dtype.c_str());
        return -1;
    }

    const uint32_t r = static_cast<uint32_t>(4.0 * sigma + 0.5);
    std::vector<int64_t> shape = {batch, D, H, W};
    int64_t numel = GetShapeSize(shape);

    // CPU参考
    std::vector<double> wts(2 * r + 1, 0.0);
    double sum = 0.0;
    for (int64_t k = 0; k < static_cast<int64_t>(2 * r + 1); k++) {
        int64_t i = k - static_cast<int64_t>(r);
        wts[k] = (sigma == 0.0) ? (i == 0 ? 1.0 : 0.0) : exp(-0.5 * static_cast<double>(i) * i / (sigma * sigma));
        sum += wts[k];
    }
    for (auto& v : wts)
        v /= sum;

    std::vector<float> xVal(numel);
    uint64_t state = static_cast<uint64_t>(seed) * 2654435761ULL + 1;
    for (int64_t i = 0; i < numel; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float rnd = static_cast<float>((state >> 40) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
        if (pattern == "zero") {
            xVal[i] = 0.0f;
        } else if (pattern == "neg") {
            xVal[i] = -fabs(rnd) - 0.5f;
        } else if (pattern == "extreme") {
            float maxv = isBf16 ? 3.0e38f : 6.0e4f;
            xVal[i] = (i % 2 == 0 ? 1.0f : -1.0f) * (maxv * (0.9f + 0.1f * fabs(rnd)));
        } else {
            xVal[i] = rnd;
        }
    }
    if (!isFp32) {
        for (auto& v : xVal)
            v = Bits16ToFloat(FloatToBits16(v, isBf16), isBf16);
    }
    std::vector<float> golden(numel, 0.0f);
    {
        // batch=N*C 合并维：逐体积做参考滤波
        int64_t vol = D * H * W;
        for (int64_t b = 0; b < batch; b++) {
            RefGaussian3D(xVal.data() + b * vol, golden.data() + b * vol, D, H, W, wts, r, mode);
        }
    }
    if (!isFp32) {
        // kernel输出做饱和转换，参考侧同样clamp到dtype范围
        float clampMax = isBf16 ? 3.3895314e38f : 65504.0f;
        for (auto& v : golden)
            v = std::max(-clampMax, std::min(clampMax, v));
    }

    // NPU执行
    std::vector<uint16_t> xHost16;
    std::vector<float> xHost32;
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    aclDataType aclDtype = isFp32 ? ACL_FLOAT : (isBf16 ? ACL_BF16 : ACL_FLOAT16);
    size_t elemSize = isFp32 ? 4 : 2;
    if (isFp32) {
        xHost32 = xVal;
        ret = aclrtMalloc(&xDeviceAddr, numel * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = aclrtMemcpy(xDeviceAddr, numel * 4, xHost32.data(), numel * 4, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
        xHost16.resize(numel);
        for (int64_t i = 0; i < numel; i++)
            xHost16[i] = FloatToBits16(xVal[i], isBf16);
        ret = aclrtMalloc(&xDeviceAddr, numel * 2, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = aclrtMemcpy(xDeviceAddr, numel * 2, xHost16.data(), numel * 2, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclrtMalloc(&yDeviceAddr, numel * elemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
        strides[i] = shape[i + 1] * strides[i + 1];
    x = aclCreateTensor(shape.data(), shape.size(), aclDtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                        shape.size(), xDeviceAddr);
    y = aclCreateTensor(shape.data(), shape.size(), aclDtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                        shape.size(), yDeviceAddr);
    CHECK_RET(x != nullptr && y != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGaussianFilter3DGetWorkspaceSize(x, sigma, modeStr.c_str(), y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize=%llu r=%u\n", (unsigned long long)workspaceSize, r);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGaussianFilter3D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGaussianFilter3D failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("sync failed. ERROR: %d\n", ret); return ret);

    std::vector<uint16_t> yHost16(numel);
    std::vector<float> yHost32(numel);
    if (isFp32) {
        ret = aclrtMemcpy(yHost32.data(), numel * 4, yDeviceAddr, numel * 4, ACL_MEMCPY_DEVICE_TO_HOST);
    } else {
        ret = aclrtMemcpy(yHost16.data(), numel * 2, yDeviceAddr, numel * 2, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy result failed. ERROR: %d\n", ret); return ret);

    double atol = isFp32 ? 1e-4 : 1e-2;
    double rtol = isFp32 ? 1e-3 : (isBf16 ? 5e-2 : 1e-2);
    bool pass = true;
    double worst = 0.0;
    for (int64_t i = 0; i < numel; i++) {
        float out = isFp32 ? yHost32[i] : Bits16ToFloat(yHost16[i], isBf16);
        double g = golden[i];
        double diff = fabs(static_cast<double>(out) - g);
        double lim = atol + rtol * fabs(g);
        if (diff > lim) {
            pass = false;
            if (diff > worst)
                worst = diff;
        }
    }
    LOG_PRINT("shape=[%lld,%lld,%lld,%lld] dtype=%s sigma=%f mode=%s pattern=%s worstDiff=%.3e\n", (long long)batch,
              (long long)D, (long long)H, (long long)W, dtype.c_str(), sigma, modeStr.c_str(), pattern.c_str(), worst);
    if (pass) {
        LOG_PRINT("PASS\n");
    } else {
        LOG_PRINT("FAIL (worstDiff=%f)\n", worst);
    }

    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}
