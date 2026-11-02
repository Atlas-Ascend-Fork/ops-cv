/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_distance_transform.h"

namespace {
constexpr float TOLERANCE = 1e-5F;
constexpr int FAILED_CODE = -1;

int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t result = 1;
    for (const int64_t dim : shape) {
        result *= dim;
    }
    return result;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
int CreateTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                 void** deviceAddress, aclTensor** tensor)
{
    const size_t bytes = hostData.size() * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMalloc failed: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddress, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMemcpy host-to-device failed: %d\n", ret);
        aclrtFree(*deviceAddress);
        *deviceAddress = nullptr;
        return ret;
    }
    const std::vector<int64_t> strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddress);
    if (*tensor == nullptr) {
        std::printf("aclCreateTensor failed\n");
        aclrtFree(*deviceAddress);
        *deviceAddress = nullptr;
        return FAILED_CODE;
    }
    return ACL_SUCCESS;
}

std::vector<float> Reference(const std::vector<uint8_t>& x, int64_t height, int64_t width, const std::string& mode)
{
    std::vector<std::pair<int64_t, int64_t>> background;
    for (int64_t row = 0; row < height; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            if (x[row * width + col] == 0U) {
                background.emplace_back(row, col);
            }
        }
    }
    const float sentinel = mode == "L1"         ? static_cast<float>(height + width) :
                           mode == "CHESSBOARD" ? static_cast<float>(std::max(height, width)) :
                                                  std::hypot(static_cast<float>(height), static_cast<float>(width));
    std::vector<float> result(x.size(), sentinel);
    for (int64_t row = 0; row < height; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            for (const auto& point : background) {
                const int64_t dr = std::abs(row - point.first);
                const int64_t dc = std::abs(col - point.second);
                const float distance = mode == "L1" ? static_cast<float>(dr + dc) :
                                       mode == "CHESSBOARD" ?
                                                      static_cast<float>(std::max(dr, dc)) :
                                                      std::hypot(static_cast<float>(dr), static_cast<float>(dc));
                result[row * width + col] = std::min(result[row * width + col], distance);
            }
        }
    }
    return result;
}

int RunMode(aclrtStream stream, const aclTensor* x, const aclTensor* y, void* yDevice,
            const std::vector<uint8_t>& input, int64_t height, int64_t width, const std::string& mode)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::vector<char> distanceType(mode.begin(), mode.end());
    distanceType.push_back('\0');
    aclnnStatus ret = aclnnDistanceTransformGetWorkspaceSize(x, distanceType.data(), 0, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* recentError = aclGetRecentErrMsg();
        std::printf("[%s] aclnnDistanceTransformGetWorkspaceSize failed: %d, x=%p, y=%p, mode=%p\n"
                    "  recent error: %s\n",
                    mode.c_str(), ret, static_cast<const void*>(x), static_cast<const void*>(y),
                    static_cast<void*>(distanceType.data()), recentError == nullptr ? "<none>" : recentError);
        return ret;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("[%s] workspace allocation failed: %d\n", mode.c_str(), ret);
            return ret;
        }
    }
    ret = aclnnDistanceTransform(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }

    std::vector<float> actual(input.size(), 0.0F);
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), yDevice, actual.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("[%s] execution failed: %d\n", mode.c_str(), ret);
        return ret;
    }

    const std::vector<float> expected = Reference(input, height, width, mode);
    float maxError = 0.0F;
    bool printedFirstFailure = false;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        maxError = std::max(maxError, error);
        if (!printedFirstFailure && error > TOLERANCE) {
            std::printf("  first mismatch: row=%zu col=%zu actual=%.6f expected=%.6f diff=%.6f\n",
                        i / static_cast<size_t>(width), i % static_cast<size_t>(width), actual[i], expected[i], error);
            printedFirstFailure = true;
        }
    }
    const bool passed = maxError <= TOLERANCE;
    std::printf("[%s] mode=%-10s shape=(%ld,%ld) max_abs_diff=%.6g\n", passed ? "PASS" : "FAIL", mode.c_str(), height,
                width, maxError);
    return passed ? ACL_SUCCESS : FAILED_CODE;
}
int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

void FinalizeAcl(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclError ret = InitAcl(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> shape = {3, 5};
    const std::vector<uint8_t> input = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    };
    std::vector<float> output(Numel(shape), 0.0F);
    void* xDevice = nullptr;
    void* yDevice = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    ret = CreateTensor(input, shape, ACL_UINT8, &xDevice, &x);
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(output, shape, ACL_FLOAT, &yDevice, &y);
    }
    int failures = 0;
    if (ret == ACL_SUCCESS) {
        for (const std::string mode : {"L1", "L2", "CHESSBOARD"}) {
            failures += RunMode(stream, x, y, yDevice, input, shape[0], shape[1], mode) == ACL_SUCCESS ? 0 : 1;
        }
    } else {
        failures = 1;
    }

    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (xDevice != nullptr) {
        aclrtFree(xDevice);
    }
    if (yDevice != nullptr) {
        aclrtFree(yDevice);
    }
    FinalizeAcl(deviceId, stream);
    std::printf("DistanceTransform ACLNN example: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
