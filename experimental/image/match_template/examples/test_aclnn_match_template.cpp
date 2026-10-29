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
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_match_template.h"

namespace {

constexpr float EPSILON = 1e-12f;

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype, void** device,
                 aclTensor** tensor, aclFormat format = ACL_FORMAT_ND)
{
    const size_t bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(T);
    int ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, format, shape.data(), shape.size(),
                              *device);
    return *tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

struct ReferenceResult {
    std::vector<float> score;
    std::vector<float> bestScore;
    std::vector<int64_t> bestLocation;
};

float Normalize(float numerator, float product, float lower, float upper, float zeroValue)
{
    if (product <= 0.0f) {
        return zeroValue;
    }
    const float denominator = std::sqrt(product);
    if (denominator <= EPSILON) {
        return zeroValue;
    }
    return std::max(lower, std::min(upper, numerator / denominator));
}

ReferenceResult Reference(const std::vector<float>& image, const std::vector<float>& templ,
                          const std::vector<int64_t>& imageShape, const std::vector<int64_t>& templateShape,
                          int64_t method)
{
    const int64_t nCount = imageShape[0];
    const int64_t imageH = imageShape[1];
    const int64_t imageW = imageShape[2];
    const int64_t channels = imageShape[3];
    const int64_t templateH = templateShape[1];
    const int64_t templateW = templateShape[2];
    const int64_t outputH = imageH - templateH + 1;
    const int64_t outputW = imageW - templateW + 1;
    const int64_t spatialSize = templateH * templateW;
    const int64_t outputCount = outputH * outputW;

    ReferenceResult result;
    result.score.resize(nCount * outputCount);
    result.bestScore.resize(nCount);
    result.bestLocation.resize(nCount * 2);
    for (int64_t n = 0; n < nCount; ++n) {
        bool first = true;
        float best = 0.0f;
        int64_t bestIndex = 0;
        for (int64_t outputIndex = 0; outputIndex < outputCount; ++outputIndex) {
            const int64_t outY = outputIndex / outputW;
            const int64_t outX = outputIndex % outputW;
            float cross = 0.0f;
            float imageSquare = 0.0f;
            float templateSquare = 0.0f;
            float squareDifference = 0.0f;
            float coefficient = 0.0f;
            float imageVariance = 0.0f;
            float templateVariance = 0.0f;
            for (int64_t c = 0; c < channels; ++c) {
                float channelCross = 0.0f;
                float channelImage = 0.0f;
                float channelImageSquare = 0.0f;
                float channelTemplate = 0.0f;
                float channelTemplateSquare = 0.0f;
                for (int64_t ty = 0; ty < templateH; ++ty) {
                    for (int64_t tx = 0; tx < templateW; ++tx) {
                        const int64_t imageIndex = ((n * imageH + outY + ty) * imageW + outX + tx) * channels + c;
                        const int64_t templateIndex = (ty * templateW + tx) * channels + c;
                        const float imageValue = image[imageIndex];
                        const float templateValue = templ[templateIndex];
                        const float difference = imageValue - templateValue;
                        channelCross += imageValue * templateValue;
                        channelImage += imageValue;
                        channelImageSquare += imageValue * imageValue;
                        channelTemplate += templateValue;
                        channelTemplateSquare += templateValue * templateValue;
                        squareDifference += difference * difference;
                    }
                }
                cross += channelCross;
                imageSquare += channelImageSquare;
                templateSquare += channelTemplateSquare;
                coefficient += channelCross - channelImage * channelTemplate / static_cast<float>(spatialSize);
                imageVariance += channelImageSquare - channelImage * channelImage / static_cast<float>(spatialSize);
                templateVariance += channelTemplateSquare -
                                    channelTemplate * channelTemplate / static_cast<float>(spatialSize);
            }

            float value = 0.0f;
            if (method == 0) {
                value = squareDifference;
            } else if (method == 1) {
                const float zeroValue = squareDifference <= EPSILON ? 0.0f : 1.0f;
                value = Normalize(squareDifference, imageSquare * templateSquare, 0.0f, 1.0f, zeroValue);
            } else if (method == 2) {
                value = cross;
            } else if (method == 3) {
                value = Normalize(cross, imageSquare * templateSquare, -1.0f, 1.0f, 0.0f);
            } else if (method == 4) {
                value = coefficient;
            } else {
                imageVariance = std::max(imageVariance, 0.0f);
                templateVariance = std::max(templateVariance, 0.0f);
                value = Normalize(coefficient, imageVariance * templateVariance, -1.0f, 1.0f, 0.0f);
            }
            result.score[n * outputCount + outputIndex] = value;
            const bool minimize = method <= 1;
            if (first || (minimize ? value < best : value > best)) {
                first = false;
                best = value;
                bestIndex = outputIndex;
            }
        }
        result.bestScore[n] = best;
        result.bestLocation[n * 2] = bestIndex % outputW;
        result.bestLocation[n * 2 + 1] = bestIndex / outputW;
    }
    return result;
}

struct DeviceTensors {
    void* imageDevice = nullptr;
    void* templateDevice = nullptr;
    void* scoreDevice = nullptr;
    void* bestScoreDevice = nullptr;
    void* bestLocationDevice = nullptr;
    aclTensor* imageTensor = nullptr;
    aclTensor* templateTensor = nullptr;
    aclTensor* scoreTensor = nullptr;
    aclTensor* bestScoreTensor = nullptr;
    aclTensor* bestLocationTensor = nullptr;
};

void Destroy(DeviceTensors& data)
{
    aclDestroyTensor(data.imageTensor);
    aclDestroyTensor(data.templateTensor);
    aclDestroyTensor(data.scoreTensor);
    aclDestroyTensor(data.bestScoreTensor);
    aclDestroyTensor(data.bestLocationTensor);
    aclrtFree(data.imageDevice);
    aclrtFree(data.templateDevice);
    aclrtFree(data.scoreDevice);
    aclrtFree(data.bestScoreDevice);
    aclrtFree(data.bestLocationDevice);
}

template <typename T>
int RunCase(aclrtStream stream, const char* dtypeName, aclDataType dtype, const std::vector<T>& encodedImage,
            const std::vector<T>& encodedTemplate, const std::vector<float>& image, const std::vector<float>& templ,
            const std::vector<int64_t>& imageShape, const std::vector<int64_t>& templateShape, int64_t method)
{
    const int64_t outputH = imageShape[1] - templateShape[1] + 1;
    const int64_t outputW = imageShape[2] - templateShape[2] + 1;
    const std::vector<int64_t> scoreShape = {imageShape[0], outputH, outputW};
    const std::vector<int64_t> bestScoreShape = {imageShape[0]};
    const std::vector<int64_t> bestLocationShape = {imageShape[0], 2};
    std::vector<float> zeroScore(ShapeSize(scoreShape), 0.0f);
    std::vector<float> zeroBestScore(ShapeSize(bestScoreShape), 0.0f);
    std::vector<int64_t> zeroBestLocation(ShapeSize(bestLocationShape), 0);

    DeviceTensors data;
    int ret = CreateTensor(encodedImage, imageShape, dtype, &data.imageDevice, &data.imageTensor);
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(encodedTemplate, templateShape, dtype, &data.templateDevice, &data.templateTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(zeroScore, scoreShape, ACL_FLOAT, &data.scoreDevice, &data.scoreTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(zeroBestScore, bestScoreShape, ACL_FLOAT, &data.bestScoreDevice, &data.bestScoreTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(zeroBestLocation, bestLocationShape, ACL_INT64, &data.bestLocationDevice,
                           &data.bestLocationTensor);
    }
    if (ret != ACL_SUCCESS) {
        Destroy(data);
        return ret;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatchTemplateGetWorkspaceSize(data.imageTensor, data.templateTensor, method, data.scoreTensor,
                                             data.bestScoreTensor, data.bestLocationTensor, &workspaceSize, &executor);
    if (std::getenv("MATCH_TEMPLATE_PRINT_WORKSPACE") != nullptr) {
        std::printf("[WORKSPACE] dtype=%s method=%ld bytes=%lu\n", dtypeName, method, workspaceSize);
    }
    void* workspace = nullptr;
    if (ret == ACL_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclnnMatchTemplate(workspace, workspaceSize, executor, stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }

    std::vector<float> actualScore(zeroScore.size());
    std::vector<float> actualBestScore(zeroBestScore.size());
    std::vector<int64_t> actualBestLocation(zeroBestLocation.size());
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(actualScore.data(), actualScore.size() * sizeof(float), data.scoreDevice,
                          actualScore.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(actualBestScore.data(), actualBestScore.size() * sizeof(float), data.bestScoreDevice,
                          actualBestScore.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(actualBestLocation.data(), actualBestLocation.size() * sizeof(int64_t),
                          data.bestLocationDevice, actualBestLocation.size() * sizeof(int64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    }

    const char* dumpPrefix = std::getenv("MATCH_TEMPLATE_DUMP_PREFIX");
    if (ret == ACL_SUCCESS && dumpPrefix != nullptr) {
        const std::string prefix(dumpPrefix);
        const bool dumpWorkspace = std::getenv("MATCH_TEMPLATE_DUMP_WORKSPACE") != nullptr;
        std::vector<uint8_t> hostWorkspace(dumpWorkspace ? workspaceSize : 0);
        bool workspaceCopied = true;
        if (dumpWorkspace && workspaceSize > 0) {
            workspaceCopied = aclrtMemcpy(hostWorkspace.data(), hostWorkspace.size(), workspace, workspaceSize,
                                          ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        }
        FILE* scoreFile = std::fopen((prefix + ".score.f32").c_str(), "wb");
        FILE* bestScoreFile = std::fopen((prefix + ".best_score.f32").c_str(), "wb");
        FILE* bestLocationFile = std::fopen((prefix + ".best_location.i64").c_str(), "wb");
        FILE* workspaceFile = dumpWorkspace ? std::fopen((prefix + ".workspace.bin").c_str(), "wb") : nullptr;
        const bool opened = scoreFile != nullptr && bestScoreFile != nullptr && bestLocationFile != nullptr &&
                            (!dumpWorkspace || workspaceFile != nullptr);
        const bool written = opened &&
                             std::fwrite(actualScore.data(), sizeof(float), actualScore.size(), scoreFile) ==
                                 actualScore.size() &&
                             std::fwrite(actualBestScore.data(), sizeof(float), actualBestScore.size(),
                                         bestScoreFile) == actualBestScore.size() &&
                             std::fwrite(actualBestLocation.data(), sizeof(int64_t), actualBestLocation.size(),
                                         bestLocationFile) == actualBestLocation.size() &&
                             workspaceCopied &&
                             (!dumpWorkspace || std::fwrite(hostWorkspace.data(), 1, hostWorkspace.size(),
                                                            workspaceFile) == hostWorkspace.size());
        if (scoreFile != nullptr) {
            std::fclose(scoreFile);
        }
        if (bestScoreFile != nullptr) {
            std::fclose(bestScoreFile);
        }
        if (bestLocationFile != nullptr) {
            std::fclose(bestLocationFile);
        }
        if (workspaceFile != nullptr) {
            std::fclose(workspaceFile);
        }
        std::printf("[DUMP] prefix=%s score_elements=%zu status=%s\n", dumpPrefix, actualScore.size(),
                    written ? "OK" : "FAILED");
        std::fflush(stdout);
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        Destroy(data);
        return written ? ACL_SUCCESS : 1;
    }

    const ReferenceResult reference = Reference(image, templ, imageShape, templateShape, method);
    float maxError = 0.0f;
    for (size_t i = 0; i < actualScore.size(); ++i) {
        maxError = std::max(maxError, std::fabs(actualScore[i] - reference.score[i]));
    }
    for (size_t i = 0; i < actualBestScore.size(); ++i) {
        maxError = std::max(maxError, std::fabs(actualBestScore[i] - reference.bestScore[i]));
    }
    const bool locationMatches = actualBestLocation == reference.bestLocation;
    const float tolerance = method == 0 || method == 2 || method == 4 ? 1e-3f : 2e-5f;
    const bool passed = ret == ACL_SUCCESS && maxError <= tolerance && locationMatches;
    std::printf("[%s] dtype=%-7s method=%ld max_error=%.3e location=%s ret=%d\n", passed ? "PASS" : "FAIL", dtypeName,
                method, maxError, locationMatches ? "OK" : "BAD", ret);
    if (!passed && dtype == ACL_FLOAT && method == 0 && actualScore.size() <= 32) {
        for (size_t i = 0; i < actualScore.size(); ++i) {
            std::printf("  score[%zu]: actual=%g reference=%g\n", i, actualScore[i], reference.score[i]);
        }
        for (size_t i = 0; i < actualBestScore.size(); ++i) {
            std::printf("  best[%zu]: actual=%g reference=%g, location=(%ld,%ld) reference=(%ld,%ld)\n", i,
                        actualBestScore[i], reference.bestScore[i], actualBestLocation[i * 2],
                        actualBestLocation[i * 2 + 1], reference.bestLocation[i * 2],
                        reference.bestLocation[i * 2 + 1]);
        }
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    Destroy(data);
    return passed ? ACL_SUCCESS : 1;
}

int RunInvalidMethodCase(int64_t method)
{
    const std::vector<int64_t> imageShape = {1, 2, 2, 1};
    const std::vector<int64_t> templateShape = {1, 1, 1, 1};
    const std::vector<int64_t> scoreShape = {1, 2, 2};
    const std::vector<int64_t> bestScoreShape = {1};
    const std::vector<int64_t> bestLocationShape = {1, 2};
    const std::vector<float> image = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> templ = {1.0f};
    const std::vector<float> score(4, 0.0f);
    const std::vector<float> bestScore(1, 0.0f);
    const std::vector<int64_t> bestLocation(2, 0);

    DeviceTensors data;
    int ret = CreateTensor(image, imageShape, ACL_FLOAT, &data.imageDevice, &data.imageTensor);
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(templ, templateShape, ACL_FLOAT, &data.templateDevice, &data.templateTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(score, scoreShape, ACL_FLOAT, &data.scoreDevice, &data.scoreTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(bestScore, bestScoreShape, ACL_FLOAT, &data.bestScoreDevice, &data.bestScoreTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(bestLocation, bestLocationShape, ACL_INT64, &data.bestLocationDevice,
                           &data.bestLocationTensor);
    }
    if (ret != ACL_SUCCESS) {
        Destroy(data);
        return ret;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatchTemplateGetWorkspaceSize(data.imageTensor, data.templateTensor, method, data.scoreTensor,
                                             data.bestScoreTensor, data.bestLocationTensor, &workspaceSize, &executor);
    const bool passed = ret != ACL_SUCCESS;
    std::printf("[%s] invalid method=%ld ret=%d\n", passed ? "PASS" : "FAIL", method, ret);
    Destroy(data);
    return passed ? ACL_SUCCESS : 1;
}

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
        encoded[i] = static_cast<uint8_t>(values[i]);
    }
    return encoded;
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
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::string requestedCase = argc > 1 ? argv[1] : "";
    const bool cubeOnly = requestedCase == "cube_64_ccorr" || requestedCase == "cube_512_ccorr";
    if (cubeOnly) {
        const int64_t imageSize = requestedCase == "cube_64_ccorr" ? 64 : 512;
        const std::vector<int64_t> cubeImageShape = {1, imageSize, imageSize, 3};
        const std::vector<int64_t> cubeTemplateShape = {1, 16, 16, 3};
        std::vector<uint8_t> cubeImage(ShapeSize(cubeImageShape));
        std::vector<uint8_t> cubeTemplate(ShapeSize(cubeTemplateShape));
        std::vector<float> cubeImageReference(cubeImage.size());
        std::vector<float> cubeTemplateReference(cubeTemplate.size());
        for (size_t i = 0; i < cubeImage.size(); ++i) {
            cubeImage[i] = static_cast<uint8_t>(std::lround((static_cast<float>((i * 17 + 3) % 29) / 7.0f) * 32.0f));
            cubeImageReference[i] = static_cast<float>(cubeImage[i]);
        }
        for (size_t i = 0; i < cubeTemplate.size(); ++i) {
            cubeTemplate[i] = static_cast<uint8_t>(std::lround((static_cast<float>((i * 11 + 5) % 23) / 5.0f) * 32.0f));
            cubeTemplateReference[i] = static_cast<float>(cubeTemplate[i]);
        }
        const int cubeRet = RunCase(stream, "cube-u8", ACL_UINT8, cubeImage, cubeTemplate, cubeImageReference,
                                    cubeTemplateReference, cubeImageShape, cubeTemplateShape, 2);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        std::printf("SUMMARY: %d/1 passed\n", cubeRet == ACL_SUCCESS ? 1 : 0);
        return cubeRet == ACL_SUCCESS ? 0 : 1;
    }

    const std::vector<int64_t> imageShape = {2, 3, 3, 1};
    const std::vector<int64_t> templateShape = {1, 2, 2, 1};
    const std::vector<float> image = {1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    const std::vector<float> templ = {5, 6, 8, 9};
    const std::vector<uint8_t> imageU8 = ToUint8(image);
    const std::vector<uint8_t> templateU8 = ToUint8(templ);
    const std::vector<aclFloat16> imageF16 = ToFloat16(image);
    const std::vector<aclFloat16> templateF16 = ToFloat16(templ);

    int failures = 0;
    int totalCases = 0;
    auto Record = [&](int result) {
        ++totalCases;
        failures += result != ACL_SUCCESS;
    };
    for (int64_t method = 0; method < 6; ++method) {
        Record(RunCase(stream, "float32", ACL_FLOAT, image, templ, image, templ, imageShape, templateShape, method));
        Record(RunCase(stream, "float16", ACL_FLOAT16, imageF16, templateF16, image, templ, imageShape, templateShape,
                       method));
        Record(
            RunCase(stream, "uint8", ACL_UINT8, imageU8, templateU8, image, templ, imageShape, templateShape, method));
    }

    const std::vector<int64_t> zeroImageShape = {1, 3, 3, 1};
    const std::vector<float> zeros(9, 0.0f);
    const std::vector<float> zeroTemplate(4, 0.0f);
    for (int64_t method : {1, 3, 5}) {
        Record(RunCase(stream, "zero", ACL_FLOAT, zeros, zeroTemplate, zeros, zeroTemplate, zeroImageShape,
                       templateShape, method));
    }

    // 7x7 output exercises adaptive tiling, non-divisible right/bottom edges, C=3, and cross-batch scheduling.
    const std::vector<int64_t> colorImageShape = {2, 9, 10, 3};
    const std::vector<int64_t> colorTemplateShape = {1, 3, 4, 3};
    std::vector<float> colorImage(ShapeSize(colorImageShape));
    std::vector<float> colorTemplate(ShapeSize(colorTemplateShape));
    for (size_t i = 0; i < colorImage.size(); ++i) {
        colorImage[i] = static_cast<float>((i * 17 + 3) % 23);
    }
    for (size_t i = 0; i < colorTemplate.size(); ++i) {
        colorTemplate[i] = static_cast<float>((i * 7 + 5) % 19);
    }
    const std::vector<aclFloat16> colorImageF16 = ToFloat16(colorImage);
    const std::vector<aclFloat16> colorTemplateF16 = ToFloat16(colorTemplate);
    const std::vector<uint8_t> colorImageU8 = ToUint8(colorImage);
    const std::vector<uint8_t> colorTemplateU8 = ToUint8(colorTemplate);
    for (int64_t method = 0; method < 6; ++method) {
        Record(RunCase(stream, "color-f32", ACL_FLOAT, colorImage, colorTemplate, colorImage, colorTemplate,
                       colorImageShape, colorTemplateShape, method));
        Record(RunCase(stream, "color-f16", ACL_FLOAT16, colorImageF16, colorTemplateF16, colorImage, colorTemplate,
                       colorImageShape, colorTemplateShape, method));
        Record(RunCase(stream, "color-u8", ACL_UINT8, colorImageU8, colorTemplateU8, colorImage, colorTemplate,
                       colorImageShape, colorTemplateShape, method));
    }

    // Every output has the same score; deterministic reduction must choose [x=0, y=0] across multiple cores.
    const std::vector<int64_t> tieImageShape = {1, 9, 10, 3};
    const std::vector<int64_t> tieTemplateShape = {1, 3, 4, 3};
    const std::vector<float> tieImage(ShapeSize(tieImageShape), 2.0f);
    const std::vector<float> tieTemplate(ShapeSize(tieTemplateShape), 2.0f);
    for (int64_t method : {0, 2, 4, 5}) {
        Record(RunCase(stream, "tie", ACL_FLOAT, tieImage, tieTemplate, tieImage, tieTemplate, tieImageShape,
                       tieTemplateShape, method));
    }

    // A 65x65 template verifies that UB caching is capacity based instead of using the old 4096-element cutoff.
    const std::vector<int64_t> fallbackImageShape = {1, 70, 70, 1};
    const std::vector<int64_t> fallbackTemplateShape = {1, 65, 65, 1};
    std::vector<float> fallbackImage(ShapeSize(fallbackImageShape));
    std::vector<float> fallbackTemplate(ShapeSize(fallbackTemplateShape));
    for (size_t i = 0; i < fallbackImage.size(); ++i) {
        fallbackImage[i] = static_cast<float>((i * 5 + 1) % 11);
    }
    for (size_t i = 0; i < fallbackTemplate.size(); ++i) {
        fallbackTemplate[i] = static_cast<float>((i * 3 + 2) % 7);
    }
    for (int64_t method : {0, 5}) {
        Record(RunCase(stream, "fallback", ACL_FLOAT, fallbackImage, fallbackTemplate, fallbackImage, fallbackTemplate,
                       fallbackImageShape, fallbackTemplateShape, method));
    }

    Record(RunInvalidMethodCase(-1));
    Record(RunInvalidMethodCase(6));

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::printf("SUMMARY: %d/%d passed\n", totalCases - failures, totalCases);
    return failures == 0 ? 0 : 1;
}
