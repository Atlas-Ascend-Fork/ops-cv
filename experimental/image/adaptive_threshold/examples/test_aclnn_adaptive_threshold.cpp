/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_adaptive_threshold.h"

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
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int main()
{
    // 1. 初始化 device 和 stream，可按实际环境修改 deviceId。
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 2. 构造 UINT8 单通道图像及同形状输出，格式为 ND。
    const std::vector<int64_t> shape = {4, 4};
    const std::vector<uint8_t> xHostData = {10, 10, 10, 10, 10, 200, 200, 10, 10, 200, 200, 10, 10, 10, 10, 10};
    std::vector<uint8_t> yHostData(static_cast<size_t>(GetShapeSize(shape)), 0);
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    // 正常结束及出错时均释放已申请的资源。
    auto cleanup = [&]() {
        aclrtSynchronizeStream(stream);
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (y != nullptr) {
            aclDestroyTensor(y);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (yDeviceAddr != nullptr) {
            aclrtFree(yDeviceAddr);
        }
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    ret = CreateAclTensor(xHostData, shape, &xDeviceAddr, ACL_UINT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(yHostData, shape, &yDeviceAddr, ACL_UINT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);

    // 3. 调用第一段接口获取 workspace 大小，随后申请 workspace。
    const double maxValue = 255.0;
    char adaptiveMethod[] = "MEAN";
    char thresholdType[] = "THRESH_BINARY";
    const int64_t blockSize = 3;
    const double c = 2.0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAdaptiveThresholdGetWorkspaceSize(x, maxValue, adaptiveMethod, thresholdType, blockSize, c, y,
                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveThresholdGetWorkspaceSize failed. ERROR: %d\n", ret);
              cleanup(); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); cleanup(); return ret);
    }

    // 4. 调用第二段接口执行算子，并同步等待执行结束。
    ret = aclnnAdaptiveThreshold(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveThreshold failed. ERROR: %d\n", ret); cleanup(); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); cleanup(); return ret);

    // 5. 将结果拷回 host 并打印。此输入的中央 2×2 区域为 200，其余位置为 10。
    const size_t outputBytes = yHostData.size() * sizeof(uint8_t);
    ret = aclrtMemcpy(yHostData.data(), outputBytes, yDeviceAddr, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ERROR: %d\n", ret); cleanup(); return ret);
    for (size_t i = 0; i < yHostData.size(); ++i) {
        LOG_PRINT("result[%zu] is: %u\n", i, static_cast<unsigned int>(yHostData[i]));
    }

    // 6. 释放 Tensor、device 内存、stream 和运行时资源。
    cleanup();
    return ACL_SUCCESS;
}
