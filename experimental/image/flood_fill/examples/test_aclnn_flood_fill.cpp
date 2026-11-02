/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "acl/acl.h"
#include "aclnn_flood_fill.h"

namespace {
bool Check(int status, const char* operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %d\n", operation, status);
    return false;
}

// Each invocation owns separate input/output buffers and its workspace.
struct Resources {
    std::array<void*, 4> buffers{};
    std::array<aclTensor*, 4> tensors{};
    aclIntArray* seed = nullptr;
    aclFloatArray* value = nullptr;
    aclFloatArray* tolerance = nullptr;
    void* workspace = nullptr;

    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources()
    {
        for (auto tensor : tensors) {
            if (tensor != nullptr) {
                aclDestroyTensor(tensor);
            }
        }
        for (auto buffer : buffers) {
            if (buffer != nullptr) {
                aclrtFree(buffer);
            }
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (seed != nullptr) {
            aclDestroyIntArray(seed);
        }
        if (value != nullptr) {
            aclDestroyFloatArray(value);
        }
        if (tolerance != nullptr) {
            aclDestroyFloatArray(tolerance);
        }
    }

    bool Create(size_t index, int64_t height, int64_t width, aclDataType type, size_t bytes, const void* input)
    {
        if (!Check(aclrtMalloc(&buffers[index], bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
            return false;
        }
        const int status = input != nullptr ?
                               aclrtMemcpy(buffers[index], bytes, input, bytes, ACL_MEMCPY_HOST_TO_DEVICE) :
                               aclrtMemset(buffers[index], bytes, 0xaa, bytes);
        if (!Check(status, "initialize tensor")) {
            return false;
        }
        const int64_t shape[] = {height, width};
        const int64_t strides[] = {width, 1};
        tensors[index] = aclCreateTensor(shape, 2, type, strides, 0, ACL_FORMAT_ND, shape, 2, buffers[index]);
        if (tensors[index] == nullptr) {
            std::fprintf(stderr, "aclCreateTensor failed\n");
            return false;
        }
        return true;
    }
};

template <typename T>
bool RunCase(aclrtStream stream, aclDataType type, int64_t connectivity, bool fixedRange, int maskKind, bool bindMask)
{
    // Seed [1,3] is 12. The diagonal endpoints distinguish 4/8 connectivity;
    // the 12 -> 11 -> 10 chain distinguishes FIXED_RANGE from NEIGHBOR.
    const std::array<T, 15> input = {10, 0, 0, 0, 0, 0, 10, 11, 12, 0, 0, 0, 0, 0, 12};
    std::array<uint8_t, 35> inputMask{};
    if (maskKind == 1) {
        inputMask[2 * 7 + 3] = 7; // Barrier at image [1,2].
    } else if (maskKind == 2) {
        inputMask[2 * 7 + 4] = 9; // Block the seed itself.
    }

    // Explicit expected region for this small fixture, independent of the kernel algorithm.
    std::array<bool, 15> filled{};
    if (maskKind != 2) {
        filled[8] = true;
        filled[14] = connectivity == 8;
        if (maskKind == 0) {
            filled[7] = true;
            filled[6] = !fixedRange;
            filled[0] = !fixedRange && connectivity == 8;
        }
    }
    auto expected = input;
    auto expectedMask = inputMask;
    for (size_t p = 0; p < input.size(); ++p) {
        if (filled[p]) {
            expected[p] = static_cast<T>(200);
            expectedMask[(p / 5 + 1) * 7 + p % 5 + 1] = 1;
        }
    }
    for (size_t p = 0; p < expectedMask.size(); ++p) {
        if (p < 7 || p >= 28 || p % 7 == 0 || p % 7 == 6) {
            expectedMask[p] = 1;
        }
    }

    Resources data;
    if (!data.Create(0, 3, 5, type, sizeof(input), input.data()) ||
        !data.Create(1, 3, 5, type, sizeof(input), nullptr) ||
        (maskKind != 0 && !data.Create(2, 5, 7, ACL_UINT8, inputMask.size(), inputMask.data())) ||
        (bindMask && !data.Create(3, 5, 7, ACL_UINT8, expectedMask.size(), nullptr))) {
        return false;
    }
    const int64_t seedValues[] = {1, 3}; // row, col
    const float fillValues[] = {200.0F};
    const float toleranceValues[] = {1.0F};
    data.seed = aclCreateIntArray(seedValues, 2);
    data.value = aclCreateFloatArray(fillValues, 1);
    data.tolerance = aclCreateFloatArray(toleranceValues, 1);
    if (data.seed == nullptr || data.value == nullptr || data.tolerance == nullptr) {
        std::fprintf(stderr, "attribute creation failed\n");
        return false;
    }
    char neighbor[] = "NEIGHBOR";
    char fixed[] = "FIXED_RANGE";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (!Check(aclnnFloodFillGetWorkspaceSize(data.tensors[0], data.tensors[2], data.seed, data.value, data.tolerance,
                                              data.tolerance, connectivity, fixedRange ? fixed : neighbor,
                                              data.tensors[1], data.tensors[3], &workspaceSize, &executor),
               "aclnnFloodFillGetWorkspaceSize")) {
        return false;
    }
    if (workspaceSize > 0 &&
        !Check(aclrtMalloc(&data.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation")) {
        return false;
    }
    const bool launched = Check(aclnnFloodFill(data.workspace, workspaceSize, executor, stream), "aclnnFloodFill");
    const bool synchronized = Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    if (!launched || !synchronized) {
        return false;
    }
    std::array<T, 15> actual{}, unchanged{};
    std::array<uint8_t, 35> actualMask{}, unchangedMask{};
    if (!Check(aclrtMemcpy(actual.data(), sizeof(actual), data.buffers[1], sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST),
               "copy output") ||
        !Check(aclrtMemcpy(unchanged.data(), sizeof(unchanged), data.buffers[0], sizeof(unchanged),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "copy input") ||
        (bindMask && !Check(aclrtMemcpy(actualMask.data(), actualMask.size(), data.buffers[3], actualMask.size(),
                                        ACL_MEMCPY_DEVICE_TO_HOST),
                            "copy output mask")) ||
        (maskKind != 0 && !Check(aclrtMemcpy(unchangedMask.data(), unchangedMask.size(), data.buffers[2],
                                             unchangedMask.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                                 "copy input mask"))) {
        return false;
    }
    const bool pass = actual == expected && unchanged == input && (!bindMask || actualMask == expectedMask) &&
                      (maskKind == 0 || unchangedMask == inputMask);
    std::printf("[%s] dtype=%d connectivity=%lld mode=%s mask_kind=%d mask_out=%d\n", pass ? "PASS" : "FAIL",
                static_cast<int>(type), static_cast<long long>(connectivity), fixedRange ? fixed : neighbor, maskKind,
                bindMask);
    return pass;
}
} // namespace

int main()
{
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    if (!Check(aclrtSetDevice(0), "aclrtSetDevice")) {
        aclFinalize();
        return 1;
    }
    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
    unsigned passed = 0;
    for (int64_t connectivity : {4, 8}) {
        for (bool fixedRange : {false, true}) {
            for (int maskKind : {0, 1, 2}) {
                for (bool bindMask : {false, true}) {
                    passed += RunCase<uint8_t>(stream, ACL_UINT8, connectivity, fixedRange, maskKind, bindMask);
                    passed += RunCase<int32_t>(stream, ACL_INT32, connectivity, fixedRange, maskKind, bindMask);
                    passed += RunCase<float>(stream, ACL_FLOAT, connectivity, fixedRange, maskKind, bindMask);
                }
            }
        }
    }
    std::printf("SUMMARY: %u/72 PASS\n", passed);
    bool success = passed == 72;
    success = Check(aclrtDestroyStream(stream), "aclrtDestroyStream") && success;
    success = Check(aclrtResetDevice(0), "aclrtResetDevice") && success;
    success = Check(aclFinalize(), "aclFinalize") && success;
    return success ? 0 : 1;
}
