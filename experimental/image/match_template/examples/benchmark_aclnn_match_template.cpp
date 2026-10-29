/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <cstring>
#include <functional>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_match_template.h"

namespace {

struct CaseConfig {
    const char* name;
    aclDataType dtype;
    int64_t imageH;
    int64_t imageW;
    int64_t channels;
    int64_t templateH;
    int64_t templateW;
    int64_t method;
    int warmup;
    int repeats;
    int64_t batch = 1;
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
        encoded[i] = static_cast<uint8_t>(std::lround(values[i] * 32.0f));
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
    const bool correctnessOnly = std::getenv("MATCH_TEMPLATE_CORRECTNESS_ONLY") != nullptr;
    const int warmup = correctnessOnly ? 0 : config.warmup;
    const int repeats = correctnessOnly ? 1 : config.repeats;
    const int64_t outputH = config.imageH - config.templateH + 1;
    const int64_t outputW = config.imageW - config.templateW + 1;
    const std::vector<int64_t> imageShape = {config.batch, config.imageH, config.imageW, config.channels};
    const std::vector<int64_t> templateShape = {1, config.templateH, config.templateW, config.channels};
    const std::vector<int64_t> scoreShape = {config.batch, outputH, outputW};
    const std::vector<int64_t> bestScoreShape = {config.batch};
    const std::vector<int64_t> bestLocationShape = {config.batch, 2};

    std::vector<float> image(ShapeSize(imageShape));
    std::vector<float> templ(ShapeSize(templateShape));
    const char* inputMode = std::getenv("MATCH_TEMPLATE_INPUT_MODE");
    const bool signedInput = inputMode != nullptr && std::strcmp(inputMode, "signed") == 0;
    const bool zeroInput = inputMode != nullptr && std::strcmp(inputMode, "zero") == 0;
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = zeroInput ?
                       0.0f :
                       static_cast<float>(static_cast<int64_t>((i * 17 + 3) % 29) - (signedInput ? 14 : 0)) / 7.0f;
    }
    for (size_t i = 0; i < templ.size(); ++i) {
        templ[i] = zeroInput ?
                       0.0f :
                       static_cast<float>(static_cast<int64_t>((i * 11 + 5) % 23) - (signedInput ? 11 : 0)) / 5.0f;
    }

    TensorResource imageResource;
    TensorResource templateResource;
    TensorResource scoreResource;
    TensorResource bestScoreResource;
    TensorResource bestLocationResource;
    int ret = ACL_SUCCESS;
    if (config.dtype == ACL_FLOAT16) {
        const std::vector<aclFloat16> imageF16 = ToFloat16(image);
        const std::vector<aclFloat16> templateF16 = ToFloat16(templ);
        ret = CreateTensor(imageF16, imageShape, ACL_FLOAT16, imageResource);
        if (ret == ACL_SUCCESS) {
            ret = CreateTensor(templateF16, templateShape, ACL_FLOAT16, templateResource);
        }
    } else if (config.dtype == ACL_UINT8) {
        const std::vector<uint8_t> imageU8 = ToUint8(image);
        const std::vector<uint8_t> templateU8 = ToUint8(templ);
        ret = CreateTensor(imageU8, imageShape, ACL_UINT8, imageResource);
        if (ret == ACL_SUCCESS) {
            ret = CreateTensor(templateU8, templateShape, ACL_UINT8, templateResource);
        }
    } else {
        ret = CreateTensor(image, imageShape, ACL_FLOAT, imageResource);
        if (ret == ACL_SUCCESS) {
            ret = CreateTensor(templ, templateShape, ACL_FLOAT, templateResource);
        }
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(std::vector<float>(ShapeSize(scoreShape), 0.0f), scoreShape, ACL_FLOAT, scoreResource);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(std::vector<float>(ShapeSize(bestScoreShape), 0.0f), bestScoreShape, ACL_FLOAT,
                           bestScoreResource);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(std::vector<int64_t>(ShapeSize(bestLocationShape), 0), bestLocationShape, ACL_INT64,
                           bestLocationResource);
    }
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "tensor allocation failed for %s: %d\n", config.name, ret);
        Destroy(imageResource);
        Destroy(templateResource);
        Destroy(scoreResource);
        Destroy(bestScoreResource);
        Destroy(bestLocationResource);
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
        ret = aclnnMatchTemplateGetWorkspaceSize(imageResource.tensor, templateResource.tensor, config.method,
                                                 scoreResource.tensor, bestScoreResource.tensor,
                                                 bestLocationResource.tensor, &workspaceSize, &executor);
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
            ret = aclnnMatchTemplate(workspace, workspaceSize, executor, stream);
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
        std::printf("%s,%s,%ldx%ldx%ldx%ld,%ldx%ld,%ld,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%lu\n", config.name,
                    config.dtype == ACL_FLOAT16 ? "float16" : (config.dtype == ACL_UINT8 ? "uint8" : "float32"),
                    config.batch, config.imageH, config.imageW, config.channels, config.templateH, config.templateW,
                    config.method, repeats, hostSamples.front(), Percentile(hostSamples, 0.5),
                    Percentile(hostSamples, 0.95), hostMean, deviceSamples.front(), Percentile(deviceSamples, 0.5),
                    Percentile(deviceSamples, 0.95), deviceMean, allocatedWorkspaceSize);

        const char* dumpDir = std::getenv("MATCH_TEMPLATE_DUMP_DIR");
        if (dumpDir != nullptr && dumpDir[0] != '\0') {
            const std::string prefix = std::string(dumpDir) + "/" + config.name;
            const int64_t scoreElements = ShapeSize(scoreShape);
            std::vector<float> scoreHost(static_cast<size_t>(scoreElements));
            std::vector<float> bestScoreHost(static_cast<size_t>(config.batch));
            std::vector<int64_t> bestLocationHost(static_cast<size_t>(config.batch * 2));
            ret = aclrtMemcpy(scoreHost.data(), scoreHost.size() * sizeof(float), scoreResource.device,
                              scoreHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == ACL_SUCCESS) {
                ret = aclrtMemcpy(bestScoreHost.data(), bestScoreHost.size() * sizeof(float), bestScoreResource.device,
                                  bestScoreHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            }
            if (ret == ACL_SUCCESS) {
                ret = aclrtMemcpy(bestLocationHost.data(), bestLocationHost.size() * sizeof(int64_t),
                                  bestLocationResource.device, bestLocationHost.size() * sizeof(int64_t),
                                  ACL_MEMCPY_DEVICE_TO_HOST);
            }
            if (ret == ACL_SUCCESS) {
                std::ofstream(prefix + ".score.f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(scoreHost.data()), scoreHost.size() * sizeof(float));
                std::ofstream(prefix + ".best_score.f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(bestScoreHost.data()), bestScoreHost.size() * sizeof(float));
                std::ofstream(prefix + ".best_location.i64", std::ios::binary)
                    .write(reinterpret_cast<const char*>(bestLocationHost.data()),
                           bestLocationHost.size() * sizeof(int64_t));
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
    Destroy(imageResource);
    Destroy(templateResource);
    Destroy(scoreResource);
    Destroy(bestScoreResource);
    Destroy(bestLocationResource);
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
        {"f32_mode_sqdiff", ACL_FLOAT, 128, 128, 1, 16, 16, 0, 10, 100},
        {"f32_mode_sqdiff_normed", ACL_FLOAT, 128, 128, 1, 16, 16, 1, 10, 100},
        {"f32_mode_ccorr", ACL_FLOAT, 128, 128, 1, 16, 16, 2, 10, 100},
        {"f32_mode_ccorr_normed", ACL_FLOAT, 128, 128, 1, 16, 16, 3, 10, 100},
        {"f32_mode_ccoeff", ACL_FLOAT, 128, 128, 1, 16, 16, 4, 10, 100},
        {"f32_mode_ccoeff_normed", ACL_FLOAT, 128, 128, 1, 16, 16, 5, 10, 100},
        {"f32_small_sqdiff", ACL_FLOAT, 64, 64, 1, 8, 8, 0, 10, 100},
        {"f32_color_ccoeff_normed", ACL_FLOAT, 96, 128, 3, 8, 8, 5, 10, 100},
        {"f32_large_sqdiff", ACL_FLOAT, 256, 256, 1, 16, 16, 0, 10, 100},
        {"f32_large_template_ccoeff_normed", ACL_FLOAT, 96, 96, 1, 65, 65, 5, 10, 100},
        {"fp16_mode_sqdiff", ACL_FLOAT16, 128, 128, 1, 16, 16, 0, 10, 100},
        {"fp16_mode_sqdiff_normed", ACL_FLOAT16, 128, 128, 1, 16, 16, 1, 10, 100},
        {"fp16_mode_ccorr", ACL_FLOAT16, 128, 128, 1, 16, 16, 2, 10, 100},
        {"fp16_mode_ccorr_normed", ACL_FLOAT16, 128, 128, 1, 16, 16, 3, 10, 100},
        {"fp16_mode_ccoeff", ACL_FLOAT16, 128, 128, 1, 16, 16, 4, 10, 100},
        {"fp16_mode_ccoeff_normed", ACL_FLOAT16, 128, 128, 1, 16, 16, 5, 10, 100},
        {"fp16_small_sqdiff", ACL_FLOAT16, 64, 64, 1, 8, 8, 0, 10, 100},
        {"fp16_color_ccoeff_normed", ACL_FLOAT16, 96, 128, 3, 8, 8, 5, 10, 100},
        {"fp16_large_sqdiff", ACL_FLOAT16, 256, 256, 1, 16, 16, 0, 10, 100},
        {"fp16_large_template_ccoeff_normed", ACL_FLOAT16, 96, 96, 1, 65, 65, 5, 10, 100},
        {"uint8_mode_sqdiff", ACL_UINT8, 128, 128, 1, 16, 16, 0, 10, 100},
        {"uint8_mode_sqdiff_normed", ACL_UINT8, 128, 128, 1, 16, 16, 1, 10, 100},
        {"uint8_mode_ccorr", ACL_UINT8, 128, 128, 1, 16, 16, 2, 10, 100},
        {"uint8_mode_ccorr_normed", ACL_UINT8, 128, 128, 1, 16, 16, 3, 10, 100},
        {"uint8_mode_ccoeff", ACL_UINT8, 128, 128, 1, 16, 16, 4, 10, 100},
        {"uint8_mode_ccoeff_normed", ACL_UINT8, 128, 128, 1, 16, 16, 5, 10, 100},
        {"uint8_small_sqdiff", ACL_UINT8, 64, 64, 1, 8, 8, 0, 10, 100},
        {"uint8_color_ccoeff_normed", ACL_UINT8, 96, 128, 3, 8, 8, 5, 10, 100},
        {"uint8_large_sqdiff", ACL_UINT8, 256, 256, 1, 16, 16, 0, 10, 100},
        {"uint8_large_template_ccoeff_normed", ACL_UINT8, 96, 96, 1, 65, 65, 5, 10, 100},
        {"uint8_tiny_sqdiff", ACL_UINT8, 32, 32, 1, 3, 3, 0, 10, 100},
        {"uint8_medium_512_sqdiff", ACL_UINT8, 512, 512, 1, 16, 16, 0, 5, 20},
        {"uint8_cube_512_ccorr", ACL_UINT8, 512, 512, 3, 16, 16, 2, 3, 10},
        {"uint8_color_512_ccoeff_normed", ACL_UINT8, 512, 512, 3, 16, 16, 5, 3, 10},
        {"uint8_large_template_512_ccoeff_normed", ACL_UINT8, 512, 512, 1, 65, 65, 5, 3, 10},
        {"uint8_720p_sqdiff", ACL_UINT8, 720, 1280, 1, 32, 32, 0, 2, 10},
        {"uint8_1080p_sqdiff", ACL_UINT8, 1080, 1920, 1, 16, 16, 0, 2, 10},
        {"uint8_template_129_ccoeff_normed", ACL_UINT8, 160, 160, 1, 129, 129, 5, 2, 10},
        {"f32_720p_c1_t16_sqdiff", ACL_FLOAT, 720, 1280, 1, 16, 16, 0, 2, 10},
        {"f32_1080p_c1_t16_sqdiff", ACL_FLOAT, 1080, 1920, 1, 16, 16, 0, 2, 10},
        {"f32_4k_c1_t16_sqdiff", ACL_FLOAT, 2160, 3840, 1, 16, 16, 0, 2, 10},
        {"f32_1080p_c3_t16_ccoeff_normed", ACL_FLOAT, 1080, 1920, 3, 16, 16, 5, 2, 10},
        {"f32_4k_c3_t16_ccoeff_normed", ACL_FLOAT, 2160, 3840, 3, 16, 16, 5, 1, 7},
        {"f32_1080p_c1_t65_ccoeff_normed", ACL_FLOAT, 1080, 1920, 1, 65, 65, 5, 1, 7},
        {"fp16_720p_c1_t16_sqdiff", ACL_FLOAT16, 720, 1280, 1, 16, 16, 0, 2, 10},
        {"fp16_1080p_c1_t16_sqdiff", ACL_FLOAT16, 1080, 1920, 1, 16, 16, 0, 2, 10},
        {"fp16_4k_c1_t16_sqdiff", ACL_FLOAT16, 2160, 3840, 1, 16, 16, 0, 2, 10},
        {"fp16_1080p_c3_t16_ccoeff_normed", ACL_FLOAT16, 1080, 1920, 3, 16, 16, 5, 2, 10},
        {"fp16_4k_c3_t16_ccoeff_normed", ACL_FLOAT16, 2160, 3840, 3, 16, 16, 5, 1, 7},
        {"fp16_1080p_c1_t65_ccoeff_normed", ACL_FLOAT16, 1080, 1920, 1, 65, 65, 5, 1, 7},
        {"uint8_720p_c1_t16_sqdiff", ACL_UINT8, 720, 1280, 1, 16, 16, 0, 2, 10},
        {"uint8_1080p_c1_t16_sqdiff", ACL_UINT8, 1080, 1920, 1, 16, 16, 0, 2, 10},
        {"uint8_4k_c1_t16_sqdiff", ACL_UINT8, 2160, 3840, 1, 16, 16, 0, 2, 10},
        {"uint8_1080p_c3_t16_ccoeff_normed", ACL_UINT8, 1080, 1920, 3, 16, 16, 5, 2, 10},
        {"uint8_4k_c3_t16_ccoeff_normed", ACL_UINT8, 2160, 3840, 3, 16, 16, 5, 1, 7},
        {"uint8_1080p_c1_t65_ccoeff_normed", ACL_UINT8, 1080, 1920, 1, 65, 65, 5, 1, 7},
        {"f32_non_aligned_c1_t15x17_sqdiff", ACL_FLOAT, 721, 1283, 1, 15, 17, 0, 2, 10},
        {"fp16_non_aligned_c1_t31x33_ccoeff_normed", ACL_FLOAT16, 1081, 1919, 1, 31, 33, 5, 1, 7},
        {"uint8_non_aligned_c3_t31x33_ccoeff_normed", ACL_UINT8, 1081, 1919, 3, 31, 33, 5, 1, 7},
        {"uint8_non_aligned_c1_t15x17_sqdiff", ACL_UINT8, 721, 1283, 1, 15, 17, 0, 2, 10},
        {"fp16_probe_n1_512_c3_t16_ccorr", ACL_FLOAT16, 512, 512, 3, 16, 16, 2, 2, 10, 1},
        {"fp16_probe_n1_512_c3_t16_ccorr_normed", ACL_FLOAT16, 512, 512, 3, 16, 16, 3, 2, 10, 1},
        {"fp16_probe_n1_512_c3_t16_ccoeff", ACL_FLOAT16, 512, 512, 3, 16, 16, 4, 2, 10, 1},
        {"fp16_probe_n1_512_c3_t16_ccoeff_normed", ACL_FLOAT16, 512, 512, 3, 16, 16, 5, 2, 10, 1},
        {"f32_batch2_512_c1_t16_sqdiff", ACL_FLOAT, 512, 512, 1, 16, 16, 0, 2, 10, 2},
        {"fp16_batch2_512_c3_t16_ccoeff_normed", ACL_FLOAT16, 512, 512, 3, 16, 16, 5, 2, 10, 2},
        {"uint8_batch4_1080p_c1_t16_sqdiff", ACL_UINT8, 1080, 1920, 1, 16, 16, 0, 1, 5, 4},
    };

    std::puts("case,dtype,image,template,method,repeats,host_min_us,host_p50_us,host_p95_us,host_mean_us,"
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
