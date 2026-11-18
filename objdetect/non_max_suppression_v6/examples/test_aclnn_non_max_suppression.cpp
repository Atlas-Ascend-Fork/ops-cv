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
#include "acl/acl.h"
#include "aclnnop/aclnn_non_max_suppression.h"

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

template <typename T>
int64_t GetShapeSize(const std::vector<T>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray)
{
    auto size = GetShapeSize(hostData) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 调用aclCreateIntArray接口创建aclIntArray
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> boxesShape = {1, 7, 4};
    std::vector<int64_t> scoresShape = {1, 1, 7};
    std::vector<int64_t> maxSizePerClassShape = {3};
    std::vector<int64_t> selectedIndicesShape = {3, 3};

    void* boxesDeviceAddr = nullptr;
    void* scoresDeviceAddr = nullptr;
    void* maxSizePerClassDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* boxes = nullptr;
    aclTensor* scores = nullptr;
    aclIntArray* maxOutputBoxesPerClass = nullptr;
    aclFloatArray* iouThd = nullptr;
    aclFloatArray* scoresThd = nullptr;
    aclTensor* selectedIndices = nullptr;

    std::vector<float> boxesHostData = {
        49.1, 32.4, 51.0, 35.9, 49.3, 32.9, 51.0, 35.3, 49.2, 31.8, 51.0, 35.4, 35.1, 11.5,
        39.1, 15.7, 35.6, 11.8, 39.3, 14.2, 35.3, 11.5, 39.9, 14.5, 35.2, 11.7, 39.7, 15.7,
    };
    std::vector<float> scoresHostData = {0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3};
    std::vector<int64_t> maxOutputBoxesPerClassHostData = {3};
    std::vector<float> iouThresholdHostData = {0.6};
    std::vector<float> scoreThresholdHostData = {0};
    std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 创建aclTensor: boxes
    ret = CreateAclTensor(boxesHostData, boxesShape, &boxesDeviceAddr, aclDataType::ACL_FLOAT, &boxes);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建aclTensor: scores
    ret = CreateAclTensor(scoresHostData, scoresShape, &scoresDeviceAddr, aclDataType::ACL_FLOAT, &scores);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建AclIntArray: maxOutputBoxesPerClass
    ret = CreateAclIntArray(maxOutputBoxesPerClassHostData, &maxSizePerClassDeviceAddr, &maxOutputBoxesPerClass);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建AclFloatArray: iouThreshold
    iouThd = aclCreateFloatArray(iouThresholdHostData.data(), iouThresholdHostData.size());
    CHECK_RET(iouThd != nullptr, return 0);

    // 创建AclFloatArray: scoresThreshold
    scoresThd = aclCreateFloatArray(scoreThresholdHostData.data(), scoreThresholdHostData.size());
    CHECK_RET(scoresThd != nullptr, return 0);

    // 创建aclTensor: selectedIndices
    ret = CreateAclTensor(outHostData, selectedIndicesShape, &outDeviceAddr, aclDataType::ACL_INT32, &selectedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建attr int: centerPointBox
    int32_t centerPointBox = 0;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnNonMaxSuppression第一段接口
    ret = aclnnNonMaxSuppressionGetWorkspaceSize(boxes, scores, maxOutputBoxesPerClass, iouThd, scoresThd,
                                                 centerPointBox, selectedIndices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppressionGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnNonMaxSuppression第二段接口
    ret = aclnnNonMaxSuppression(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNonMaxSuppression failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(selectedIndicesShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(boxes);
    aclDestroyTensor(scores);
    aclDestroyIntArray(maxOutputBoxesPerClass);
    aclDestroyFloatArray(iouThd);
    aclDestroyFloatArray(scoresThd);
    aclDestroyTensor(selectedIndices);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(boxesDeviceAddr);
    aclrtFree(scoresDeviceAddr);
    aclrtFree(maxSizePerClassDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
