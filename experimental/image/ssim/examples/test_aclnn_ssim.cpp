/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_ssim.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char** argv)
{
    // usage: test_aclnn_ssim <dtype:uint8|float32> <rank2|rank3|rank4> H W [C] [N] [seed]
    std::string dtype = (argc > 1) ? argv[1] : "float32";
    std::string mode = (argc > 2) ? argv[2] : "rank2";
    int64_t H = (argc > 3) ? std::stoll(argv[3]) : 64;
    int64_t W = (argc > 4) ? std::stoll(argv[4]) : 64;
    int64_t C = (argc > 5) ? std::stoll(argv[5]) : 1;
    int64_t N = (argc > 6) ? std::stoll(argv[6]) : 1;
    int64_t seed = (argc > 7) ? std::stoll(argv[7]) : 12345;
    if (dtype != "uint8" && dtype != "float32") {
        LOG_PRINT("unsupported dtype %s\n", dtype.c_str());
        return -1;
    }
    if (H < 11 || W < 11) {
        LOG_PRINT("H/W must be >= 11\n");
        return -1;
    }
    if (C < 1 || N < 1) {
        LOG_PRINT("C/N must be >= 1\n");
        return -1;
    }

    std::vector<int64_t> shape;
    if (mode == "rank2") {
        shape = {H, W};
        N = 1;
        C = 1;
    } else if (mode == "rank3") {
        shape = {H, W, C};
        N = 1;
    } else if (mode == "rank4") {
        shape = {N, H, W, C};
    } else {
        LOG_PRINT("unsupported mode %s\n", mode.c_str());
        return -1;
    }

    // Reject empty tensors before any device-side resource is created.
    const int64_t numel = GetShapeSize(shape);
    if (numel <= 0) {
        LOG_PRINT("empty input is not supported\n");
        return -1;
    }

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* yOut = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> yOutShape = {1};
    std::vector<float> yOutHost(1, 0.0f);
    aclDataType aclDtype = (dtype == "uint8") ? aclDataType::ACL_UINT8 : aclDataType::ACL_FLOAT;

    int ret = Init(deviceId, &stream);
    if (ret != 0) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    if (dtype == "uint8") {
        std::vector<uint8_t> x1Host(numel);
        std::vector<uint8_t> x2Host(numel);
        uint64_t state = static_cast<uint64_t>(seed) * 2654435761ULL + 1;
        for (int64_t i = 0; i < numel; i++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            uint8_t v = static_cast<uint8_t>((state >> 33) & 0xFF);
            x1Host[i] = v;
            x2Host[i] = static_cast<uint8_t>((static_cast<uint32_t>(v) * 3 + static_cast<uint32_t>(i % 7)) & 0xFF);
        }
        ret = CreateAclTensor(x1Host, shape, &x1DeviceAddr, aclDtype, &x1);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create x1 tensor failed. ERROR: %d\n", ret);
            goto cleanup;
        }
        ret = CreateAclTensor(x2Host, shape, &x2DeviceAddr, aclDtype, &x2);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create x2 tensor failed. ERROR: %d\n", ret);
            goto cleanup;
        }
    } else {
        std::vector<float> x1Host(numel);
        std::vector<float> x2Host(numel);
        uint64_t state = static_cast<uint64_t>(seed) * 2654435761ULL + 1;
        for (int64_t i = 0; i < numel; i++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            float v = static_cast<float>((state >> 40) & 0xFFFF) / 65535.0f;
            x1Host[i] = v;
            x2Host[i] = v + 0.03f * static_cast<float>((state >> 24) & 0xFF) / 255.0f;
        }
        LOG_PRINT("host x1[0..13]=");
        for (int64_t i = 0; i < 14 && i < numel; i++) {
            LOG_PRINT("%.6f ", x1Host[i]);
        }
        LOG_PRINT("\nhost x2[0..13]=");
        for (int64_t i = 0; i < 14 && i < numel; i++) {
            LOG_PRINT("%.6f ", x2Host[i]);
        }
        LOG_PRINT("\n");
        ret = CreateAclTensor(x1Host, shape, &x1DeviceAddr, aclDtype, &x1);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create x1 tensor failed. ERROR: %d\n", ret);
            goto cleanup;
        }
        ret = CreateAclTensor(x2Host, shape, &x2DeviceAddr, aclDtype, &x2);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create x2 tensor failed. ERROR: %d\n", ret);
            goto cleanup;
        }
    }

    ret = CreateAclTensor(yOutHost, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT, &yOut);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("create yOut tensor failed. ERROR: %d\n", ret);
        goto cleanup;
    }

    ret = aclnnSsimGetWorkspaceSize(x1, x2, yOut, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnSsimGetWorkspaceSize failed. ERROR: %d\n", ret);
        goto cleanup;
    }
    LOG_PRINT("workspaceSize=%llu\n", (unsigned long long)workspaceSize);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret);
            goto cleanup;
        }
    }

    ret = aclnnSsim(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnSsim failed. ERROR: %d\n", ret);
        goto cleanup;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        goto cleanup;
    }

    ret = aclrtMemcpy(yOutHost.data(), sizeof(float), yOutDeviceAddr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", ret);
        goto cleanup;
    }

    LOG_PRINT("ssim_result=%.9f\n", yOutHost[0]);
    LOG_PRINT("shape=[%lld,%lld,%lld,%lld] dtype=%s seed=%lld\n", (long long)shape[0],
              (long long)(shape.size() > 1 ? shape[1] : 0), (long long)(shape.size() > 2 ? shape[2] : 0),
              (long long)(shape.size() > 3 ? shape[3] : 0), dtype.c_str(), (long long)seed);

cleanup:
    // Single exit path: every resource created above is released even when an ACLNN call fails.
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (yOut != nullptr) {
        aclDestroyTensor(yOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (yOutDeviceAddr != nullptr) {
        aclrtFree(yOutDeviceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
