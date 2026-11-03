/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file benchmark_aclnn_lbp_feature.cpp
 * \brief LbpFeature ACLNN 性能基准：dtype × 形状 × (radius, neighbors) 矩阵，
 *        逐用例输出 host/device min/p50/p95/mean 与 workspace 字节数（CSV → stdout）。
 *        用法: ./benchmark_aclnn_lbp_feature [case_name]
 *        环境变量: LBP_CORRECTNESS_ONLY=1 时 warmup=0、repeats=1；
 *                  LBP_DUMP_DIR=<dir> 时导出输出 bin（int32）。
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
#include "aclnn_lbp_feature.h"

namespace {

struct CaseConfig {
    const char* name;
    aclDataType dtype;
    int64_t height;
    int64_t width;
    int64_t batch;
    int64_t radius;
    int64_t neighbors;
    int warmup;
    int repeats;
};

std::vector<uint8_t> ToUint8(size_t count)
{
    std::vector<uint8_t> encoded(count);
    for (size_t i = 0; i < count; ++i) {
        encoded[i] = static_cast<uint8_t>((i * 17 + 3) % 256);
    }
    return encoded;
}

std::vector<float> ToFloat32(size_t count)
{
    std::vector<float> encoded(count);
    for (size_t i = 0; i < count; ++i) {
        encoded[i] = static_cast<float>((i * 17 + 3) % 29) / 7.0f;
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

std::vector<int64_t> BuildShape(int64_t height, int64_t width, int64_t batch)
{
    return batch > 1 ? std::vector<int64_t>{batch, height, width} : std::vector<int64_t>{height, width};
}

int RunCase(aclrtStream stream, const CaseConfig& config)
{
    const bool correctnessOnly = std::getenv("LBP_CORRECTNESS_ONLY") != nullptr;
    const int warmup = correctnessOnly ? 0 : config.warmup;
    const int repeats = correctnessOnly ? 1 : config.repeats;

    const std::vector<int64_t> inputShape = BuildShape(config.height, config.width, config.batch);
    const std::vector<int64_t> outputShape = BuildShape(config.height - 2 * config.radius,
                                                        config.width - 2 * config.radius, config.batch);
    const int64_t inputCount = ShapeSize(inputShape);
    const int64_t outputCount = ShapeSize(outputShape);

    TensorResource inputResource;
    TensorResource outputResource;
    int ret = ACL_SUCCESS;
    if (config.dtype == ACL_UINT8) {
        ret = CreateTensor(ToUint8(static_cast<size_t>(inputCount)), inputShape, ACL_UINT8, inputResource);
    } else {
        ret = CreateTensor(ToFloat32(static_cast<size_t>(inputCount)), inputShape, ACL_FLOAT, inputResource);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(std::vector<int32_t>(static_cast<size_t>(outputCount), 0), outputShape, ACL_INT32,
                           outputResource);
    }
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "tensor allocation failed for %s: %d\n", config.name, ret);
        Destroy(inputResource);
        Destroy(outputResource);
        return ret;
    }

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
        ret = aclnnLbpFeatureGetWorkspaceSize(inputResource.tensor, config.radius, config.neighbors,
                                              outputResource.tensor, &workspaceSize, &executor);
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
            ret = aclnnLbpFeature(workspace, workspaceSize, executor, stream);
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
        std::printf("%s,%s,%ldx%ld,%ld,%ld,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%lu\n", config.name,
                    config.dtype == ACL_UINT8 ? "uint8" : "float32", config.height, config.width, config.radius,
                    config.neighbors, repeats, hostSamples.front(), Percentile(hostSamples, 0.5),
                    Percentile(hostSamples, 0.95), hostMean, deviceSamples.front(), Percentile(deviceSamples, 0.5),
                    Percentile(deviceSamples, 0.95), deviceMean, allocatedWorkspaceSize);

        const char* dumpDir = std::getenv("LBP_DUMP_DIR");
        if (dumpDir != nullptr && dumpDir[0] != '\0') {
            const std::string prefix = std::string(dumpDir) + "/" + config.name;
            std::vector<int32_t> output(static_cast<size_t>(outputCount));
            ret = aclrtMemcpy(output.data(), output.size() * sizeof(int32_t), outputResource.device,
                              output.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == ACL_SUCCESS) {
                std::ofstream(prefix + ".out.i32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(output.data()),
                           static_cast<std::streamsize>(output.size() * sizeof(int32_t)));
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
        // uint8 × (radius, neighbors) × 分辨率
        {"u8_64x64_r1n8", ACL_UINT8, 64, 64, 1, 1, 8, 10, 100},
        {"u8_64x64_r1n4", ACL_UINT8, 64, 64, 1, 1, 4, 10, 100},
        {"u8_64x64_r2n12", ACL_UINT8, 64, 64, 1, 2, 12, 10, 100},
        {"u8_64x64_r3n16", ACL_UINT8, 64, 64, 1, 3, 16, 10, 100},
        {"u8_128x128_r1n8", ACL_UINT8, 128, 128, 1, 1, 8, 10, 100},
        {"u8_128x128_r2n16", ACL_UINT8, 128, 128, 1, 2, 16, 10, 100},
        {"u8_256x256_r1n8", ACL_UINT8, 256, 256, 1, 1, 8, 10, 100},
        {"u8_256x256_r3n24", ACL_UINT8, 256, 256, 1, 3, 24, 10, 100},
        {"u8_512x512_r1n8", ACL_UINT8, 512, 512, 1, 1, 8, 5, 20},
        {"u8_512x512_r2n12", ACL_UINT8, 512, 512, 1, 2, 12, 5, 20},
        {"u8_720x1280_r1n8", ACL_UINT8, 720, 1280, 1, 1, 8, 2, 10},
        {"u8_1080x1920_r1n8", ACL_UINT8, 1080, 1920, 1, 1, 8, 2, 10},
        {"u8_2048x2048_r1n8", ACL_UINT8, 2048, 2048, 1, 1, 8, 2, 10},
        {"u8_2048x2048_r2n16", ACL_UINT8, 2048, 2048, 1, 2, 16, 2, 10},
        {"u8_257x4096_r1n8", ACL_UINT8, 257, 4096, 1, 1, 8, 2, 10},
        {"u8_721x1283_r1n8", ACL_UINT8, 721, 1283, 1, 1, 8, 2, 10},
        {"u8_batch4_512x512_r1n8", ACL_UINT8, 512, 512, 4, 1, 8, 2, 10},
        {"u8_33x45_r1n8", ACL_UINT8, 33, 45, 1, 1, 8, 10, 100},
        {"u8_63x65_r4n16", ACL_UINT8, 63, 65, 1, 4, 16, 10, 100},
        {"u8_300x100_r2n12", ACL_UINT8, 300, 100, 1, 2, 12, 10, 100},
        // float32 × (radius, neighbors) × 分辨率
        {"f32_64x64_r1n8", ACL_FLOAT, 64, 64, 1, 1, 8, 10, 100},
        {"f32_64x64_r1n4", ACL_FLOAT, 64, 64, 1, 1, 4, 10, 100},
        {"f32_64x64_r2n12", ACL_FLOAT, 64, 64, 1, 2, 12, 10, 100},
        {"f32_64x64_r3n16", ACL_FLOAT, 64, 64, 1, 3, 16, 10, 100},
        {"f32_128x128_r1n8", ACL_FLOAT, 128, 128, 1, 1, 8, 10, 100},
        {"f32_128x128_r2n16", ACL_FLOAT, 128, 128, 1, 2, 16, 10, 100},
        {"f32_256x256_r1n8", ACL_FLOAT, 256, 256, 1, 1, 8, 10, 100},
        {"f32_256x256_r3n24", ACL_FLOAT, 256, 256, 1, 3, 24, 10, 100},
        {"f32_512x512_r1n8", ACL_FLOAT, 512, 512, 1, 1, 8, 5, 20},
        {"f32_512x512_r2n12", ACL_FLOAT, 512, 512, 1, 2, 12, 5, 20},
        {"f32_720x1280_r1n8", ACL_FLOAT, 720, 1280, 1, 1, 8, 2, 10},
        {"f32_1080x1920_r1n8", ACL_FLOAT, 1080, 1920, 1, 1, 8, 2, 10},
        {"f32_2048x2048_r1n8", ACL_FLOAT, 2048, 2048, 1, 1, 8, 2, 10},
        {"f32_2048x2048_r2n16", ACL_FLOAT, 2048, 2048, 1, 2, 16, 2, 10},
        {"f32_257x4096_r1n8", ACL_FLOAT, 257, 4096, 1, 1, 8, 2, 10},
        {"f32_721x1283_r1n8", ACL_FLOAT, 721, 1283, 1, 1, 8, 2, 10},
        {"f32_batch4_512x512_r1n8", ACL_FLOAT, 512, 512, 4, 1, 8, 2, 10},
        {"f32_33x45_r1n8", ACL_FLOAT, 33, 45, 1, 1, 8, 10, 100},
        {"f32_63x65_r4n16", ACL_FLOAT, 63, 65, 1, 4, 16, 10, 100},
        {"f32_300x100_r2n12", ACL_FLOAT, 300, 100, 1, 2, 12, 10, 100},
    };

    std::puts("case,dtype,shape,radius,neighbors,repeats,host_min_us,host_p50_us,host_p95_us,host_mean_us,"
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
