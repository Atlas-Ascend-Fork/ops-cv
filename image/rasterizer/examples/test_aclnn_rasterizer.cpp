/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_rasterizer.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> vShape = {3, 4};
    std::vector<int64_t> fShape = {1, 3};
    std::vector<int64_t> dShape = {10, 10};
    std::vector<int64_t> findicesShape = {10, 10};
    std::vector<int64_t> baryShape = {10, 10, 3};
    int64_t height = 10;
    int64_t width = 10;
    double occlusionTruncation = 0.0;
    int64_t useDepthPrior = 0;
    std::vector<float> vData = {6.0f,       4.0f,       1.0f,      6.9f,       7.0928106f, 0.3491799f,
                                3.0046327f, 6.6574745f, 7.308903f, 7.6934705f, 0.1315008f, 3.9899914f};
    std::vector<int32_t> fData = {2, 1, 0};
    std::vector<float> dData(100, 0.0f);
    std::vector<int32_t> findicesData(100, 0);
    std::vector<float> baryData(10 * 10 * 3, 0.0f);

    void* vDeviceAddr = nullptr;
    void* fDeviceAddr = nullptr;
    void* dDeviceAddr = nullptr;
    void* findicesDeviceAddr = nullptr;
    void* baryDeviceAddr = nullptr;

    aclTensor* v = nullptr;
    aclTensor* f = nullptr;
    aclTensor* d = nullptr;
    aclTensor* findices = nullptr;
    aclTensor* barycentric = nullptr;

    ret = CreateAclTensor(vData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(fData, fShape, &fDeviceAddr, aclDataType::ACL_INT32, &f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dData, dShape, &dDeviceAddr, aclDataType::ACL_FLOAT, &d);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(findicesData, findicesShape, &findicesDeviceAddr, aclDataType::ACL_INT32, &findices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(baryData, baryShape, &baryDeviceAddr, aclDataType::ACL_FLOAT, &barycentric);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRasterizer第一段接口
    ret = aclnnRasterizerGetWorkspaceSize(v, f, d, width, height, occlusionTruncation, useDepthPrior, findices,
                                          barycentric, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRasterizerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnRasterizer第二段接口
    ret = aclnnRasterizer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRasterizer failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto findicesSize = GetShapeSize(findicesShape);
    std::vector<int32_t> findicesOutData(findicesSize, 0);
    ret = aclrtMemcpy(findicesOutData.data(), findicesSize * sizeof(findicesOutData[0]), findicesDeviceAddr,
                      findicesSize * sizeof(findicesOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    auto barycentricSize = GetShapeSize(baryShape);
    std::vector<float> baryOutData(barycentricSize, 0);
    ret = aclrtMemcpy(baryOutData.data(), barycentricSize * sizeof(baryOutData[0]), baryDeviceAddr,
                      barycentricSize * sizeof(baryOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < findicesSize; i++) {
        LOG_PRINT("findices[%ld] is: %d\n", i, findicesOutData[i]);
    }
    for (int64_t i = 0; i < barycentricSize; i++) {
        LOG_PRINT("barycentric[%ld] is: %f\n", i, baryOutData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(v);
    aclDestroyTensor(f);
    aclDestroyTensor(d);
    aclDestroyTensor(findices);
    aclDestroyTensor(barycentric);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
