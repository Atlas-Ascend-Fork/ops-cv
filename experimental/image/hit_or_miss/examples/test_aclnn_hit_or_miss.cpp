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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnn_hit_or_miss.h"

namespace {
constexpr int SUCCESS = 0;
constexpr int FAILED = 1;

#define CHECK_ACL(expr)                                                                                            \
    do {                                                                                                           \
        const auto _ret = (expr);                                                                                  \
        if (_ret != ACL_SUCCESS) {                                                                                 \
            std::fprintf(stderr, "[ERROR] %s failed, ret=%d, line=%d\n", #expr, static_cast<int>(_ret), __LINE__); \
            return FAILED;                                                                                         \
        }                                                                                                          \
    } while (0)

struct TestCase {
    std::string name;
    std::vector<int64_t> inputShape;
    std::vector<uint8_t> inputData;
    aclDataType inputType = ACL_UINT8;
    std::vector<int64_t> kernelShape;
    std::vector<int8_t> kernelData;
    bool useAxes = false;
    std::vector<int64_t> axes;
    bool useAnchor = false;
    std::vector<int64_t> anchor;
    bool borderValue = true;
};

int64_t NumElements(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

int CreateTensor(const void* hostData, size_t bytes, const std::vector<int64_t>& shape, aclDataType dataType,
                 void** deviceAddr, aclTensor** tensor)
{
    CHECK_ACL(aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(*deviceAddr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    const auto strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        std::fprintf(stderr, "[ERROR] aclCreateTensor returned nullptr\n");
        return FAILED;
    }
    return SUCCESS;
}

void DestroyTensor(aclTensor*& tensor, void*& deviceAddr)
{
    if (tensor != nullptr) {
        (void)aclDestroyTensor(tensor);
        tensor = nullptr;
    }
    if (deviceAddr != nullptr) {
        (void)aclrtFree(deviceAddr);
        deviceAddr = nullptr;
    }
}

std::vector<uint8_t> ReferenceHitOrMiss(const TestCase& tc)
{
    const size_t rank = tc.inputShape.size();
    std::vector<int64_t> axes = tc.useAxes ? tc.axes : std::vector<int64_t>{-2, -1};
    std::vector<int64_t> anchor = tc.useAnchor ? tc.anchor : std::vector<int64_t>{-1, -1};

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
    }

    const int64_t kernelH = tc.kernelShape[0];
    const int64_t kernelW = tc.kernelShape[1];
    if (anchor[0] == -1) {
        anchor[0] = kernelH / 2;
    }
    if (anchor[1] == -1) {
        anchor[1] = kernelW / 2;
    }

    const bool hasActiveConstraint = std::any_of(tc.kernelData.begin(), tc.kernelData.end(),
                                                 [](int8_t value) { return value == 1 || value == -1; });

    const auto strides = MakeStrides(tc.inputShape);
    const int64_t total = NumElements(tc.inputShape);
    std::vector<uint8_t> expected(static_cast<size_t>(total), 0);
    std::vector<int64_t> coordinates(rank, 0);

    for (int64_t flat = 0; flat < total; ++flat) {
        if (!hasActiveConstraint) {
            expected[static_cast<size_t>(flat)] = tc.inputData[static_cast<size_t>(flat)] != 0;
            continue;
        }

        int64_t remainder = flat;
        for (size_t dim = 0; dim < rank; ++dim) {
            coordinates[dim] = remainder / strides[dim];
            remainder %= strides[dim];
        }

        bool matched = true;
        for (int64_t kh = 0; kh < kernelH && matched; ++kh) {
            for (int64_t kw = 0; kw < kernelW; ++kw) {
                const int8_t kernelValue = tc.kernelData[static_cast<size_t>(kh * kernelW + kw)];
                if (kernelValue != 1 && kernelValue != -1) {
                    continue;
                }

                const int64_t coord0 = coordinates[static_cast<size_t>(axes[0])] + kh - anchor[0];
                const int64_t coord1 = coordinates[static_cast<size_t>(axes[1])] + kw - anchor[1];
                bool currentMatch = tc.borderValue;

                if (coord0 >= 0 && coord0 < tc.inputShape[static_cast<size_t>(axes[0])] && coord1 >= 0 &&
                    coord1 < tc.inputShape[static_cast<size_t>(axes[1])]) {
                    int64_t inputFlat = flat;
                    inputFlat += (coord0 - coordinates[static_cast<size_t>(axes[0])]) *
                                 strides[static_cast<size_t>(axes[0])];
                    inputFlat += (coord1 - coordinates[static_cast<size_t>(axes[1])]) *
                                 strides[static_cast<size_t>(axes[1])];
                    const bool pixel = tc.inputData[static_cast<size_t>(inputFlat)] != 0;
                    currentMatch = kernelValue == 1 ? pixel : !pixel;
                }

                if (!currentMatch) {
                    matched = false;
                    break;
                }
            }
        }
        expected[static_cast<size_t>(flat)] = matched ? 1 : 0;
    }
    return expected;
}

void PrintVector(const char* label, const std::vector<uint8_t>& data)
{
    std::printf("%s:", label);
    for (const auto value : data) {
        std::printf(" %u", static_cast<unsigned>(value));
    }
    std::printf("\n");
}

int RunCase(const TestCase& tc, aclrtStream stream)
{
    void* inputDevice = nullptr;
    void* kernelDevice = nullptr;
    void* outputDevice = nullptr;
    void* workspace = nullptr;
    aclTensor* inputTensor = nullptr;
    aclTensor* kernelTensor = nullptr;
    aclTensor* outputTensor = nullptr;
    aclIntArray* axesArray = nullptr;
    aclIntArray* anchorArray = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int result = FAILED;

    const int64_t inputElements = NumElements(tc.inputShape);
    const int64_t kernelElements = NumElements(tc.kernelShape);
    std::vector<uint8_t> outputHost(static_cast<size_t>(inputElements), 0);

    if (inputElements <= 0 || kernelElements <= 0 || static_cast<int64_t>(tc.inputData.size()) != inputElements ||
        static_cast<int64_t>(tc.kernelData.size()) != kernelElements) {
        std::fprintf(stderr, "[ERROR] %s has invalid test data sizes\n", tc.name.c_str());
        goto cleanup;
    }

    if (CreateTensor(tc.inputData.data(), tc.inputData.size() * sizeof(uint8_t), tc.inputShape, tc.inputType,
                     &inputDevice, &inputTensor) != SUCCESS) {
        goto cleanup;
    }
    if (CreateTensor(tc.kernelData.data(), tc.kernelData.size() * sizeof(int8_t), tc.kernelShape, ACL_INT8,
                     &kernelDevice, &kernelTensor) != SUCCESS) {
        goto cleanup;
    }
    if (CreateTensor(outputHost.data(), outputHost.size() * sizeof(uint8_t), tc.inputShape, ACL_BOOL, &outputDevice,
                     &outputTensor) != SUCCESS) {
        goto cleanup;
    }

    if (tc.useAxes) {
        axesArray = aclCreateIntArray(tc.axes.data(), tc.axes.size());
        if (axesArray == nullptr) {
            std::fprintf(stderr, "[ERROR] aclCreateIntArray(axes) returned nullptr\n");
            goto cleanup;
        }
    }
    if (tc.useAnchor) {
        anchorArray = aclCreateIntArray(tc.anchor.data(), tc.anchor.size());
        if (anchorArray == nullptr) {
            std::fprintf(stderr, "[ERROR] aclCreateIntArray(anchor) returned nullptr\n");
            goto cleanup;
        }
    }

    {
        const auto ret = aclnnHitOrMissGetWorkspaceSize(inputTensor, kernelTensor, axesArray, anchorArray,
                                                        tc.borderValue, outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s GetWorkspaceSize failed, ret=%d\n", tc.name.c_str(),
                         static_cast<int>(ret));
            goto cleanup;
        }
    }

    if (workspaceSize > 0) {
        const auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s workspace allocation failed, ret=%d\n", tc.name.c_str(),
                         static_cast<int>(ret));
            goto cleanup;
        }
    }

    {
        const auto ret = aclnnHitOrMiss(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s execute failed, ret=%d\n", tc.name.c_str(), static_cast<int>(ret));
            goto cleanup;
        }
    }
    {
        const auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s stream sync failed, ret=%d\n", tc.name.c_str(), static_cast<int>(ret));
            goto cleanup;
        }
    }
    {
        const auto ret = aclrtMemcpy(outputHost.data(), outputHost.size() * sizeof(uint8_t), outputDevice,
                                     outputHost.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s output copy failed, ret=%d\n", tc.name.c_str(), static_cast<int>(ret));
            goto cleanup;
        }
    }

    {
        const auto expected = ReferenceHitOrMiss(tc);
        if (outputHost != expected) {
            std::fprintf(stderr, "[FAIL] %s output mismatch\n", tc.name.c_str());
            PrintVector("actual  ", outputHost);
            PrintVector("expected", expected);
            goto cleanup;
        }
    }

    std::printf("[PASS] %-32s workspace=%llu bytes\n", tc.name.c_str(), static_cast<unsigned long long>(workspaceSize));
    result = SUCCESS;

cleanup:
    if (axesArray != nullptr) {
        (void)aclDestroyIntArray(axesArray);
    }
    if (anchorArray != nullptr) {
        (void)aclDestroyIntArray(anchorArray);
    }
    if (workspace != nullptr) {
        (void)aclrtFree(workspace);
    }
    DestroyTensor(outputTensor, outputDevice);
    DestroyTensor(kernelTensor, kernelDevice);
    DestroyTensor(inputTensor, inputDevice);
    return result;
}

std::vector<TestCase> BuildCases()
{
    std::vector<TestCase> cases;

    cases.push_back({
        "default_mixed_uint8",
        {5, 5},
        {
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        },
        ACL_UINT8,
        {3, 3},
        {
            0,
            1,
            0,
            -1,
            1,
            -1,
            0,
            1,
            0,
        },
        false,
        {},
        false,
        {},
        true,
    });

    cases.push_back({
        "all_zero_kernel_bool",
        {3, 4},
        {
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
        },
        ACL_BOOL,
        {3, 3},
        {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        },
        false,
        {},
        false,
        {},
        false,
    });

    cases.push_back({
        "all_minus_one_background",
        {7, 7},
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        ACL_UINT8,
        {3, 3},
        {
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        },
        true,
        {-2, -1},
        true,
        {-1, -1},
        false,
    });

    cases.push_back({
        "arbitrary_axes_rank3",
        {2, 3, 4},
        {
            0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1,
        },
        ACL_UINT8,
        {2, 3},
        {
            1,
            0,
            -1,
            0,
            1,
            0,
        },
        true,
        {0, 2},
        true,
        {0, 1},
        true,
    });

    cases.push_back({
        "even_kernel_default_anchor",
        {4, 5},
        {
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        ACL_UINT8,
        {2, 2},
        {
            1,
            0,
            0,
            -1,
        },
        false,
        {},
        false,
        {},
        true,
    });

    return cases;
}
} // namespace

int main(int argc, char** argv)
{
    const int32_t deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
    aclrtStream stream = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    int result = SUCCESS;
    for (const auto& tc : BuildCases()) {
        if (RunCase(tc, stream) != SUCCESS) {
            result = FAILED;
            break;
        }
    }

    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();

    if (result == SUCCESS) {
        std::printf("ALL HIT_OR_MISS ACLNN SMOKE TESTS PASSED\n");
    }
    return result;
}
