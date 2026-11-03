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
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_retinex_enhance.h"

namespace {
bool IsSupportedChannels(int64_t channels) { return channels == 1 || channels == 3 || channels == 4; }

bool CheckAcl(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(status));
    return false;
}

bool ParsePositiveDimension(const char* text, int64_t& value)
{
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0) {
        return false;
    }
    value = static_cast<int64_t>(parsed);
    return true;
}

bool ParseIterationCount(const char* text, bool allowZero, uint32_t& value)
{
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max() ||
        (!allowZero && parsed == 0)) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

float Percentile(std::vector<float> samples, double quantile)
{
    std::sort(samples.begin(), samples.end());
    const double position = quantile * static_cast<double>(samples.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return static_cast<float>(samples[lower] * (1.0 - fraction) + samples[upper] * fraction);
}

bool ReadInput(const std::string& path, size_t elementCount, std::vector<float>& data)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open input: %s\n", path.c_str());
        return false;
    }
    const std::streamsize expectedBytes = static_cast<std::streamsize>(elementCount * sizeof(float));
    if (file.tellg() != expectedBytes) {
        std::fprintf(stderr, "input size mismatch: expected %lld bytes, got %lld\n",
                     static_cast<long long>(expectedBytes), static_cast<long long>(file.tellg()));
        return false;
    }
    data.resize(elementCount);
    file.seekg(0);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(data.data()), expectedBytes));
}

bool WriteOutput(const std::string& path, const std::vector<float>& data)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::fprintf(stderr, "failed to open output: %s\n", path.c_str());
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(file);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 6 && argc != 8) {
        std::fprintf(stderr, "usage: %s H W C input_fp32.bin output_fp32.bin [warmup_iterations measured_iterations]\n",
                     argv[0]);
        return 2;
    }

    int64_t height = 0;
    int64_t width = 0;
    int64_t channels = 0;
    if (!ParsePositiveDimension(argv[1], height) || !ParsePositiveDimension(argv[2], width) ||
        !ParsePositiveDimension(argv[3], channels) || !IsSupportedChannels(channels)) {
        std::fprintf(stderr, "H/W must be positive and C must be one of 1, 3, or 4\n");
        return 2;
    }
    uint32_t warmupIterations = 0;
    uint32_t measuredIterations = 1;
    if (argc == 8 && (!ParseIterationCount(argv[6], true, warmupIterations) ||
                      !ParseIterationCount(argv[7], false, measuredIterations))) {
        std::fprintf(stderr, "warmup must be non-negative and measured iterations must be positive\n");
        return 2;
    }
    if (warmupIterations > std::numeric_limits<uint32_t>::max() - measuredIterations) {
        std::fprintf(stderr, "total iteration count is too large\n");
        return 2;
    }
    const uint64_t pixelCount = static_cast<uint64_t>(height) * static_cast<uint64_t>(width);
    if (pixelCount > std::numeric_limits<size_t>::max() / static_cast<uint64_t>(channels) ||
        pixelCount * static_cast<uint64_t>(channels) > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "input shape is too large\n");
        return 2;
    }
    const size_t elementCount = static_cast<size_t>(pixelCount * static_cast<uint64_t>(channels));
    const size_t dataBytes = elementCount * sizeof(float);

    std::vector<float> input;
    if (!ReadInput(argv[4], elementCount, input)) {
        return 2;
    }
    std::vector<float> output(elementCount, 0.0f);

    aclrtStream stream = nullptr;
    void* inputDevice = nullptr;
    void* outputDevice = nullptr;
    void* workspaceDevice = nullptr;
    aclTensor* inputTensor = nullptr;
    aclTensor* outputTensor = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool deviceSet = false;
    int result = 1;

    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) {
        goto cleanup;
    }
    deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMalloc(&inputDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)") ||
        !CheckAcl(aclrtMalloc(&outputDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(output)")) {
        goto cleanup;
    }

    {
        std::chrono::steady_clock::time_point transferStart = std::chrono::steady_clock::now();
        if (!CheckAcl(aclrtMemcpy(inputDevice, dataBytes, input.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                      "aclrtMemcpy(H2D)")) {
            goto cleanup;
        }
        const double h2dMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                                 transferStart)
                                           .count();

        {
            const int64_t shape[] = {height, width, channels};
            const int64_t strides[] = {width * channels, channels, 1};
            inputTensor = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 3, inputDevice);
            outputTensor = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 3, outputDevice);
        }
        if (inputTensor == nullptr || outputTensor == nullptr) {
            std::fprintf(stderr, "aclCreateTensor failed\n");
            goto cleanup;
        }

        if (!CheckAcl(aclnnRetinexEnhanceGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor),
                      "aclnnRetinexEnhanceGetWorkspaceSize")) {
            goto cleanup;
        }
        if (workspaceSize > 0 && !CheckAcl(aclrtMalloc(&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
                                           "aclrtMalloc(workspace)")) {
            goto cleanup;
        }

        std::vector<float> kernelSamples;
        kernelSamples.reserve(measuredIterations);
        const uint64_t allocatedWorkspaceSize = workspaceSize;
        const uint32_t totalIterations = warmupIterations + measuredIterations;
        for (uint32_t iteration = 0; iteration < totalIterations; ++iteration) {
            if (iteration != 0) {
                workspaceSize = 0;
                executor = nullptr;
                if (!CheckAcl(aclnnRetinexEnhanceGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor),
                              "aclnnRetinexEnhanceGetWorkspaceSize") ||
                    workspaceSize > allocatedWorkspaceSize) {
                    std::fprintf(stderr, "workspace size changed across iterations\n");
                    goto cleanup;
                }
            }

            if (iteration < warmupIterations) {
                if (!CheckAcl(aclnnRetinexEnhance(workspaceDevice, workspaceSize, executor, stream),
                              "aclnnRetinexEnhance(warmup)") ||
                    !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(warmup)")) {
                    goto cleanup;
                }
                continue;
            }

            aclrtEvent startEvent = nullptr;
            aclrtEvent endEvent = nullptr;
            if (!CheckAcl(aclrtCreateEvent(&startEvent), "aclrtCreateEvent(start)") ||
                !CheckAcl(aclrtCreateEvent(&endEvent), "aclrtCreateEvent(end)")) {
                if (startEvent != nullptr) {
                    aclrtDestroyEvent(startEvent);
                }
                if (endEvent != nullptr) {
                    aclrtDestroyEvent(endEvent);
                }
                goto cleanup;
            }
            float elapsedMilliseconds = 0.0f;
            const bool measured = CheckAcl(aclrtRecordEvent(startEvent, stream), "aclrtRecordEvent(start)") &&
                                  CheckAcl(aclnnRetinexEnhance(workspaceDevice, workspaceSize, executor, stream),
                                           "aclnnRetinexEnhance") &&
                                  CheckAcl(aclrtRecordEvent(endEvent, stream), "aclrtRecordEvent(end)") &&
                                  CheckAcl(aclrtSynchronizeEvent(endEvent), "aclrtSynchronizeEvent(end)") &&
                                  CheckAcl(aclrtEventElapsedTime(&elapsedMilliseconds, startEvent, endEvent),
                                           "aclrtEventElapsedTime");
            aclrtDestroyEvent(endEvent);
            aclrtDestroyEvent(startEvent);
            if (!measured) {
                goto cleanup;
            }
            kernelSamples.push_back(elapsedMilliseconds);
        }

        transferStart = std::chrono::steady_clock::now();
        if (!CheckAcl(aclrtMemcpy(output.data(), dataBytes, outputDevice, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                      "aclrtMemcpy(D2H)")) {
            goto cleanup;
        }
        const double d2hMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                                 transferStart)
                                           .count();

        std::printf("workspace_bytes=%llu\n", static_cast<unsigned long long>(allocatedWorkspaceSize));
        std::printf("h2d_ms=%.6f\n", h2dMilliseconds);
        std::printf("d2h_ms=%.6f\n", d2hMilliseconds);
        std::printf("kernel_iterations=%u\n", measuredIterations);
        std::printf("kernel_ms_median=%.6f\n", Percentile(kernelSamples, 0.5));
        std::printf("kernel_ms_p95=%.6f\n", Percentile(kernelSamples, 0.95));
        std::printf("kernel_ms_min=%.6f\n", *std::min_element(kernelSamples.begin(), kernelSamples.end()));
        std::printf("kernel_ms_max=%.6f\n", *std::max_element(kernelSamples.begin(), kernelSamples.end()));

        if (!WriteOutput(argv[5], output)) {
            goto cleanup;
        }
        result = 0;
    }

cleanup:
    if (inputTensor != nullptr) {
        aclDestroyTensor(inputTensor);
    }
    if (outputTensor != nullptr) {
        aclDestroyTensor(outputTensor);
    }
    if (workspaceDevice != nullptr) {
        aclrtFree(workspaceDevice);
    }
    if (outputDevice != nullptr) {
        aclrtFree(outputDevice);
    }
    if (inputDevice != nullptr) {
        aclrtFree(inputDevice);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(0);
    }
    aclFinalize();
    return result;
}
