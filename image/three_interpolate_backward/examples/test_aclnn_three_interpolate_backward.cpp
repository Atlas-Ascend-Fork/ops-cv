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
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnThreeInterpolateBackward的第二段接口，用于执行计算。
 */
aclnnStatus aclnnThreeInterpolateBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

/**
 * @brief aclnnThreeInterpolateBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor* grad_x, const aclTensor* idx,
                                                          const aclTensor* weight, int m, aclTensor* grad_y,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor);
#ifdef __cplusplus
}
#endif

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    uint32_t bs = 1;
    uint32_t cs = 2;
    uint32_t ms = 6;
    uint32_t ns = 5;
    uint32_t c0 = 16;

    auto c1 = (cs + c0 - 1) / c0;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // Ascend 950PR/Ascend 950DT：grad_x、grad_y以ND格式、3D shape传入，分别为（b, c, n）、（b, c, m）
    std::vector<int64_t> gradXShape = {bs, cs, ns}; // nd
    std::vector<int64_t> idxShape = {bs, ns, 3};    // nd
    std::vector<int64_t> weightShape = {bs, ns, 3}; // nd
    std::vector<int64_t> gradYShape = {bs, cs, ms}; // nd

    void* gradXDeviceAddr = nullptr;
    void* idxDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* gradYDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    aclTensor* gradX = nullptr;
    aclTensor* idx = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* gradY = nullptr;

    std::vector<float> gradXHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int32_t> idxHostData = {0, 3, 2, 3, 5, 1, 0, 3, 0, 0, 3, 5, 0, 4, 1};
    std::vector<float> weightHostData = {3.2081969,  -0.86573875, -1.0929844, 5.6045847, 5.328887,
                                         -2.0821328, 0.98572457,  9.612394,   2.216394,  1.8722068,
                                         4.2976365,  -7.019285,   -8.4070425, 1.0159919, -0.78814566};
    std::vector<float> gradYHostData(bs * cs * ms, 0.0f);

    aclOpExecutor* executor = nullptr;

    // 统一资源回收：失败路径与正常路径共用，避免中途返回泄漏已申请的
    // aclTensor、device 内存、workspace、aclOpExecutor、stream 与 aclInit 上下文
    auto cleanup = [&]() {
        if (gradX != nullptr) {
            aclDestroyTensor(gradX);
        }
        if (idx != nullptr) {
            aclDestroyTensor(idx);
        }
        if (weight != nullptr) {
            aclDestroyTensor(weight);
        }
        if (gradY != nullptr) {
            aclDestroyTensor(gradY);
        }
        if (gradXDeviceAddr != nullptr) {
            aclrtFree(gradXDeviceAddr);
        }
        if (idxDeviceAddr != nullptr) {
            aclrtFree(idxDeviceAddr);
        }
        if (weightDeviceAddr != nullptr) {
            aclrtFree(weightDeviceAddr);
        }
        if (gradYDeviceAddr != nullptr) {
            aclrtFree(gradYDeviceAddr);
        }
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(idxHostData, idxShape, &idxDeviceAddr, aclDataType::ACL_INT32, &idx,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;

    // 调用aclnnThreeInterpolateBackward第一段接口（executor 已在上方声明并纳入 cleanup）
    ret = aclnnThreeInterpolateBackwardGetWorkspaceSize(gradX, idx, weight, ms, gradY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              cleanup(); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); cleanup(); return ret);
    }

    // 调用aclnnThreeInterpolateBackward第二段接口
    ret = aclnnThreeInterpolateBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThreeInterpolateBackward failed. ERROR: %d\n", ret); cleanup();
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); cleanup(); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(gradYShape, &gradYDeviceAddr);

    // 6/7. 释放aclTensor与device资源
    cleanup();
    return 0;
}
