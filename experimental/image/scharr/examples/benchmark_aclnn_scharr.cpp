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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_scharr.h"

namespace {

struct CaseConfig {
    const char* name;
    aclDataType dtype;
    int64_t height;
    int64_t width;
    int64_t batch;
    const char* direction;
    const char* border;
    int warmup;
    int repeats;
};

std::vector<aclFloat16> ToFloat16(const std::vector<float>& values)
{
    std::vector<aclFloat16> encoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        encoded[i] = aclFloatToFloat16(values[i]);
    }
    return encoded;
}

std::vector<uint8_t> ToUint8(const std::vector<float>& values)
{
    std::vector<uint8_t> encoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        encoded[i] = static_cast<uint8_t>((i * 17 + 3) % 256);
    }
    return encoded;
}

struct TensorResource {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
};

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

template <typename T>
int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype,
                 TensorResource& resource)
{
    const size_t bytes = host.size() * sizeof(T);
    int ret = aclrtMalloc(&resource.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(resource.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), resource.device);
    return resource.tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

void Destroy(TensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
    }
    if (resource.device != nullptr) {
        aclrtFree(resource.device);
    }
}

double Percentile(const std::vector<double>& sorted, double fraction)
{
    const size_t index = static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size()))) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
}

int RunCase(aclrtStream stream, const CaseConfig& config)
{
    const bool correctnessOnly = std::getenv("SCHARR_CORRECTNESS_ONLY") != nullptr;
    const int warmup = correctnessOnly ? 0 : config.warmup;
    const int repeats = correctnessOnly ? 1 : config.repeats;

    std::vector<int64_t> shape = {config.height, config.width};
    if (config.batch > 1) {
        shape = {config.batch, config.height, config.width};
    }
    const int64_t elementCount = ShapeSize(shape);

    std::vector<float> source(static_cast<size_t>(elementCount));
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>((i * 17 + 3) % 29) / 7.0f;
    }

    TensorResource inputResource;
    TensorResource outputResource;
    int ret = ACL_SUCCESS;
    if (config.dtype == ACL_FLOAT16) {
        ret = CreateTensor(ToFloat16(source), shape, ACL_FLOAT16, inputResource);
    } else if (config.dtype == ACL_UINT8) {
        ret = CreateTensor(ToUint8(source), shape, ACL_UINT8, inputResource);
    } else {
        ret = CreateTensor(source, shape, ACL_FLOAT, inputResource);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(std::vector<float>(static_cast<size_t>(elementCount), 0.0f), shape, ACL_FLOAT,
                           outputResource);
    }
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "tensor allocation failed for %s: %d\n", config.name, ret);
        Destroy(inputResource);
        Destroy(outputResource);
        return ret;
    }

    char* directionArg = const_cast<char*>(config.direction);
    char* borderArg = const_cast<char*>(config.border);

    uint64_t allocatedWorkspaceSize = 0;
    void* workspace = nullptr;
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    if (ret == ACL_SUCCESS) {
        ret = aclrtCreateEvent(&startEvent);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtCreateEvent(&endEvent);
    }
    std::vector<double> hostSamples;
    std::vector<double> deviceSamples;
    hostSamples.reserve(repeats);
    deviceSamples.reserve(repeats);
    const int totalIterations = warmup + repeats;
    for (int iteration = 0; ret == ACL_SUCCESS && iteration < totalIterations; ++iteration) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnScharrGetWorkspaceSize(inputResource.tensor, borderArg, directionArg, outputResource.tensor,
                                          &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            break;
        }
        if (workspaceSize > allocatedWorkspaceSize) {
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            workspace = nullptr;
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            allocatedWorkspaceSize = workspaceSize;
            if (ret != ACL_SUCCESS) {
                break;
            }
        }

        ret = aclrtRecordEvent(startEvent, stream);
        const auto hostStart = std::chrono::steady_clock::now();
        if (ret == ACL_SUCCESS) {
            ret = aclnnScharr(workspace, workspaceSize, executor, stream);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclrtRecordEvent(endEvent, stream);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
        }
        const auto hostStop = std::chrono::steady_clock::now();
        float deviceMilliseconds = 0.0f;
        if (ret == ACL_SUCCESS) {
            ret = aclrtEventElapsedTime(&deviceMilliseconds, startEvent, endEvent);
        }
        if (ret == ACL_SUCCESS && iteration >= warmup) {
            hostSamples.push_back(std::chrono::duration<double, std::micro>(hostStop - hostStart).count());
            deviceSamples.push_back(static_cast<double>(deviceMilliseconds) * 1000.0);
        }
    }

    if (ret == ACL_SUCCESS) {
        std::sort(hostSamples.begin(), hostSamples.end());
        std::sort(deviceSamples.begin(), deviceSamples.end());
        const double hostMean = std::accumulate(hostSamples.begin(), hostSamples.end(), 0.0) / hostSamples.size();
        const double deviceMean = std::accumulate(deviceSamples.begin(), deviceSamples.end(), 0.0) /
                                  deviceSamples.size();
        std::printf("%s,%s,%ldx%ld,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%lu\n", config.name,
                    config.dtype == ACL_FLOAT16 ? "float16" : (config.dtype == ACL_UINT8 ? "uint8" : "float32"),
                    config.height, config.width, config.direction, config.border, repeats, hostSamples.front(),
                    Percentile(hostSamples, 0.5), Percentile(hostSamples, 0.95), hostMean, deviceSamples.front(),
                    Percentile(deviceSamples, 0.5), Percentile(deviceSamples, 0.95), deviceMean,
                    allocatedWorkspaceSize);

        const char* dumpDir = std::getenv("SCHARR_DUMP_DIR");
        if (dumpDir != nullptr && dumpDir[0] != '\0') {
            const std::string prefix = std::string(dumpDir) + "/" + config.name;
            std::vector<float> output(static_cast<size_t>(elementCount));
            ret = aclrtMemcpy(output.data(), output.size() * sizeof(float), outputResource.device,
                              output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == ACL_SUCCESS) {
                std::ofstream(prefix + ".out.f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
                std::printf("dumped=%s\n", prefix.c_str());
            }
        }
    } else {
        std::fprintf(stderr, "benchmark failed for %s: %d\n", config.name, ret);
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (startEvent != nullptr) {
        aclrtDestroyEvent(startEvent);
    }
    if (endEvent != nullptr) {
        aclrtDestroyEvent(endEvent);
    }
    Destroy(inputResource);
    Destroy(outputResource);
    return ret;
}

} // namespace

int main(int argc, char** argv)
{
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    constexpr int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    if (ret == ACL_SUCCESS) {
        ret = aclrtCreateStream(&stream);
    }

    const std::vector<CaseConfig> cases = {
        // uint8
        {"uint8_64x64_X_REFLECT", ACL_UINT8, 64, 64, 1, "X", "REFLECT_101", 10, 100},
        {"uint8_128x128_X_REFLECT", ACL_UINT8, 128, 128, 1, "X", "REFLECT_101", 10, 100},
        {"uint8_256x256_X_REFLECT", ACL_UINT8, 256, 256, 1, "X", "REFLECT_101", 10, 100},
        {"uint8_256x256_Y_CONSTANT", ACL_UINT8, 256, 256, 1, "Y", "CONSTANT", 10, 100},
        {"uint8_512x512_X_REFLECT", ACL_UINT8, 512, 512, 1, "X", "REFLECT_101", 5, 20},
        {"uint8_512x512_Y_REPLICATE", ACL_UINT8, 512, 512, 1, "Y", "REPLICATE", 5, 20},
        {"uint8_1024x1024_X_REFLECT", ACL_UINT8, 1024, 1024, 1, "X", "REFLECT_101", 3, 10},
        {"uint8_720x1280_X_REFLECT", ACL_UINT8, 720, 1280, 1, "X", "REFLECT_101", 2, 10},
        {"uint8_1080x1920_X_REFLECT", ACL_UINT8, 1080, 1920, 1, "X", "REFLECT_101", 2, 10},
        {"uint8_2048x2048_X_REFLECT", ACL_UINT8, 2048, 2048, 1, "X", "REFLECT_101", 2, 10},
        {"uint8_2048x2048_Y_REPLICATE", ACL_UINT8, 2048, 2048, 1, "Y", "REPLICATE", 2, 10},
        {"uint8_257x4096_X_REFLECT", ACL_UINT8, 257, 4096, 1, "X", "REFLECT_101", 2, 10},
        {"uint8_721x1283_Y_CONSTANT", ACL_UINT8, 721, 1283, 1, "Y", "CONSTANT", 2, 10},
        {"uint8_batch4_512x512_X_REFLECT", ACL_UINT8, 512, 512, 4, "X", "REFLECT_101", 2, 10},
        // float16
        {"fp16_64x64_X_REFLECT", ACL_FLOAT16, 64, 64, 1, "X", "REFLECT_101", 10, 100},
        {"fp16_128x128_X_REFLECT", ACL_FLOAT16, 128, 128, 1, "X", "REFLECT_101", 10, 100},
        {"fp16_256x256_X_REFLECT", ACL_FLOAT16, 256, 256, 1, "X", "REFLECT_101", 10, 100},
        {"fp16_256x256_Y_CONSTANT", ACL_FLOAT16, 256, 256, 1, "Y", "CONSTANT", 10, 100},
        {"fp16_512x512_X_REFLECT", ACL_FLOAT16, 512, 512, 1, "X", "REFLECT_101", 5, 20},
        {"fp16_512x512_Y_REPLICATE", ACL_FLOAT16, 512, 512, 1, "Y", "REPLICATE", 5, 20},
        {"fp16_1024x1024_X_REFLECT", ACL_FLOAT16, 1024, 1024, 1, "X", "REFLECT_101", 3, 10},
        {"fp16_720x1280_X_REFLECT", ACL_FLOAT16, 720, 1280, 1, "X", "REFLECT_101", 2, 10},
        {"fp16_1080x1920_X_REFLECT", ACL_FLOAT16, 1080, 1920, 1, "X", "REFLECT_101", 2, 10},
        {"fp16_2048x2048_X_REFLECT", ACL_FLOAT16, 2048, 2048, 1, "X", "REFLECT_101", 2, 10},
        {"fp16_2048x2048_Y_REPLICATE", ACL_FLOAT16, 2048, 2048, 1, "Y", "REPLICATE", 2, 10},
        {"fp16_257x4096_X_REFLECT", ACL_FLOAT16, 257, 4096, 1, "X", "REFLECT_101", 2, 10},
        {"fp16_721x1283_Y_CONSTANT", ACL_FLOAT16, 721, 1283, 1, "Y", "CONSTANT", 2, 10},
        {"fp16_batch4_512x512_X_REFLECT", ACL_FLOAT16, 512, 512, 4, "X", "REFLECT_101", 2, 10},
        // float32
        {"f32_64x64_X_REFLECT", ACL_FLOAT, 64, 64, 1, "X", "REFLECT_101", 10, 100},
        {"f32_128x128_X_REFLECT", ACL_FLOAT, 128, 128, 1, "X", "REFLECT_101", 10, 100},
        {"f32_256x256_X_REFLECT", ACL_FLOAT, 256, 256, 1, "X", "REFLECT_101", 10, 100},
        {"f32_256x256_Y_CONSTANT", ACL_FLOAT, 256, 256, 1, "Y", "CONSTANT", 10, 100},
        {"f32_512x512_X_REFLECT", ACL_FLOAT, 512, 512, 1, "X", "REFLECT_101", 5, 20},
        {"f32_512x512_Y_REPLICATE", ACL_FLOAT, 512, 512, 1, "Y", "REPLICATE", 5, 20},
        {"f32_1024x1024_X_REFLECT", ACL_FLOAT, 1024, 1024, 1, "X", "REFLECT_101", 3, 10},
        {"f32_720x1280_X_REFLECT", ACL_FLOAT, 720, 1280, 1, "X", "REFLECT_101", 2, 10},
        {"f32_1080x1920_X_REFLECT", ACL_FLOAT, 1080, 1920, 1, "X", "REFLECT_101", 2, 10},
        {"f32_2048x2048_X_REFLECT", ACL_FLOAT, 2048, 2048, 1, "X", "REFLECT_101", 2, 10},
        {"f32_2048x2048_Y_REPLICATE", ACL_FLOAT, 2048, 2048, 1, "Y", "REPLICATE", 2, 10},
        {"f32_257x4096_X_REFLECT", ACL_FLOAT, 257, 4096, 1, "X", "REFLECT_101", 2, 10},
        {"f32_721x1283_Y_CONSTANT", ACL_FLOAT, 721, 1283, 1, "Y", "CONSTANT", 2, 10},
        {"f32_batch4_512x512_X_REFLECT", ACL_FLOAT, 512, 512, 4, "X", "REFLECT_101", 2, 10},
    };

    std::puts("case,dtype,shape,direction,border,repeats,host_min_us,host_p50_us,host_p95_us,host_mean_us,"
              "device_min_us,device_p50_us,device_p95_us,device_mean_us,workspace_bytes");
    const std::string requestedCase = argc > 1 ? argv[1] : "";
    bool foundRequestedCase = requestedCase.empty();
    for (const CaseConfig& config : cases) {
        if (!requestedCase.empty() && requestedCase != config.name) {
            continue;
        }
        foundRequestedCase = true;
        if (ret == ACL_SUCCESS) {
            ret = RunCase(stream, config);
        }
    }
    if (!foundRequestedCase) {
        std::fprintf(stderr, "unknown benchmark case: %s\n", requestedCase.c_str());
        ret = ACL_ERROR_INVALID_PARAM;
    }

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret == ACL_SUCCESS ? 0 : 1;
}
