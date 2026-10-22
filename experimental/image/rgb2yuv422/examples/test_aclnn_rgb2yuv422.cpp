/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_rgb2yuv422.h"

namespace {
constexpr int32_t DEVICE_ID = 0;
constexpr int64_t HEIGHT = 2;
constexpr int64_t WIDTH = 4;
constexpr size_t PIXELS = HEIGHT * WIDTH;
constexpr float TOLERANCE = 1.0e-5F;

struct DeviceContext {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;

    ~DeviceContext()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(DEVICE_ID);
        }
        if (initialized) {
            aclFinalize();
        }
    }

    bool Init()
    {
        if (aclInit(nullptr) != ACL_SUCCESS) {
            return false;
        }
        initialized = true;
        if (aclrtSetDevice(DEVICE_ID) != ACL_SUCCESS) {
            return false;
        }
        deviceSet = true;
        return aclrtCreateStream(&stream) == ACL_SUCCESS;
    }
};

struct Tensor {
    void* address = nullptr;
    aclTensor* value = nullptr;

    ~Tensor()
    {
        if (value != nullptr) {
            aclDestroyTensor(value);
        }
        if (address != nullptr) {
            aclrtFree(address);
        }
    }

    bool Init(const std::vector<float>& host, const std::vector<int64_t>& shape)
    {
        const size_t bytes = host.size() * sizeof(float);
        if (aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMemcpy(address, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return false;
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = shape[i] * strides[i];
        }
        // Tensor storage stays ND; the separate attribute selects image layout.
        value = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                shape.size(), address);
        return value != nullptr;
    }
};

struct Workspace {
    void* address = nullptr;
    ~Workspace()
    {
        if (address != nullptr) {
            aclrtFree(address);
        }
    }
};

bool CheckOutput(const std::vector<float>& output, bool channelsFirst)
{
    for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
        const size_t yIndex = channelsFirst ? pixel : pixel * 2;
        const size_t uvIndex = channelsFirst ? PIXELS + pixel : pixel * 2 + 1;
        const float expectedUv = pixel % 2 == 0 ? -0.16874F : 0.5F;
        if (!std::isfinite(output[yIndex]) || !std::isfinite(output[uvIndex]) ||
            std::abs(output[yIndex] - 0.299F) > TOLERANCE || std::abs(output[uvIndex] - expectedUv) > TOLERANCE) {
            std::cerr << "Output differs from the red-pixel YUV422 reference at pixel " << pixel << '\n';
            return false;
        }
    }
    return true;
}

bool Invoke(Tensor& input, Tensor& output, char* dataFormat, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (aclnnRgb2yuv422GetWorkspaceSize(input.value, dataFormat, output.value, &workspaceSize, &executor) !=
        ACL_SUCCESS) {
        std::cerr << "aclnnRgb2yuv422GetWorkspaceSize failed\n";
        return false;
    }
    Workspace workspace;
    if (workspaceSize > 0 && aclrtMalloc(&workspace.address, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    const auto status = aclnnRgb2yuv422(workspace.address, workspaceSize, executor, stream);
    const auto syncStatus = aclrtSynchronizeStream(stream);
    return status == ACL_SUCCESS && syncStatus == ACL_SUCCESS;
}

bool RunCase(bool channelsFirst, aclrtStream stream)
{
    const std::vector<int64_t> inputShape = channelsFirst ? std::vector<int64_t>{1, 3, HEIGHT, WIDTH} :
                                                            std::vector<int64_t>{1, HEIGHT, WIDTH, 3};
    const std::vector<int64_t> outputShape = channelsFirst ? std::vector<int64_t>{1, 2, HEIGHT, WIDTH} :
                                                             std::vector<int64_t>{1, HEIGHT, WIDTH, 2};
    std::vector<float> inputData(PIXELS * 3, 0.0F);
    std::vector<float> outputData(PIXELS * 2, 0.0F);
    for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
        inputData[channelsFirst ? pixel : pixel * 3] = 1.0F;
    }
    Tensor input;
    Tensor output;
    if (!input.Init(inputData, inputShape) || !output.Init(outputData, outputShape)) {
        return false;
    }
    char nhwc[] = "NHWC";
    char nchw[] = "NCHW";
    if (!Invoke(input, output, channelsFirst ? nchw : nhwc, stream)) {
        return false;
    }
    const size_t bytes = outputData.size() * sizeof(float);
    if (aclrtMemcpy(outputData.data(), bytes, output.address, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    const bool passed = CheckOutput(outputData, channelsFirst);
    std::cout << (channelsFirst ? "NCHW" : "NHWC") << ": " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}
} // namespace

int main()
{
    DeviceContext context;
    if (!context.Init()) {
        std::cerr
            << "ACL initialization failed; this example requires a supported Ascend 950 device and operator package\n";
        return 1;
    }
    const bool nhwcPassed = RunCase(false, context.stream);
    const bool nchwPassed = RunCase(true, context.stream);
    return nhwcPassed && nchwPassed ? 0 : 1;
}
