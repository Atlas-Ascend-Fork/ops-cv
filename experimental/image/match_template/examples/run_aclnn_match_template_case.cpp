/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_match_template.h"

namespace {

struct TensorResource {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
};

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

template <typename T>
bool ReadBinary(const std::string& path, size_t count, std::vector<T>& values)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open input: %s\n", path.c_str());
        return false;
    }
    const auto bytes = input.tellg();
    const auto expected = static_cast<std::streamoff>(count * sizeof(T));
    if (bytes != expected) {
        std::fprintf(stderr, "input size mismatch for %s: expected=%ld actual=%ld\n", path.c_str(),
                     static_cast<long>(expected), static_cast<long>(bytes));
        return false;
    }
    values.resize(count);
    input.seekg(0);
    return static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), expected));
}

template <typename T>
bool WriteBinary(const std::string& path, const std::vector<T>& values)
{
    std::ofstream output(path, std::ios::binary);
    return output &&
           static_cast<bool>(output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T)));
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

template <typename T>
int RunCase(aclrtStream stream, aclDataType dtype, const std::vector<int64_t>& imageShape,
            const std::vector<int64_t>& templateShape, int64_t method, const std::string& imagePath,
            const std::string& templatePath, const std::string& outputPrefix)
{
    const int64_t outputH = imageShape[1] - templateShape[1] + 1;
    const int64_t outputW = imageShape[2] - templateShape[2] + 1;
    const std::vector<int64_t> scoreShape = {imageShape[0], outputH, outputW};
    const std::vector<int64_t> bestScoreShape = {imageShape[0]};
    const std::vector<int64_t> bestLocationShape = {imageShape[0], 2};

    std::vector<T> image;
    std::vector<T> templ;
    if (!ReadBinary(imagePath, static_cast<size_t>(ShapeSize(imageShape)), image) ||
        !ReadBinary(templatePath, static_cast<size_t>(ShapeSize(templateShape)), templ)) {
        return 1;
    }

    TensorResource imageResource;
    TensorResource templateResource;
    TensorResource scoreResource;
    TensorResource bestScoreResource;
    TensorResource bestLocationResource;
    int ret = CreateTensor(image, imageShape, dtype, imageResource);
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(templ, templateShape, dtype, templateResource);
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

    uint64_t workspaceSize = 0;
    void* workspace = nullptr;
    aclOpExecutor* executor = nullptr;
    if (ret == ACL_SUCCESS) {
        ret = aclnnMatchTemplateGetWorkspaceSize(imageResource.tensor, templateResource.tensor, method,
                                                 scoreResource.tensor, bestScoreResource.tensor,
                                                 bestLocationResource.tensor, &workspaceSize, &executor);
    }
    if (ret == ACL_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclnnMatchTemplate(workspace, workspaceSize, executor, stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }

    std::vector<float> score(static_cast<size_t>(ShapeSize(scoreShape)));
    std::vector<float> bestScore(static_cast<size_t>(ShapeSize(bestScoreShape)));
    std::vector<int64_t> bestLocation(static_cast<size_t>(ShapeSize(bestLocationShape)));
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(score.data(), score.size() * sizeof(float), scoreResource.device,
                          score.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(bestScore.data(), bestScore.size() * sizeof(float), bestScoreResource.device,
                          bestScore.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(bestLocation.data(), bestLocation.size() * sizeof(int64_t), bestLocationResource.device,
                          bestLocation.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret == ACL_SUCCESS && (!WriteBinary(outputPrefix + ".score.f32", score) ||
                               !WriteBinary(outputPrefix + ".best_score.f32", bestScore) ||
                               !WriteBinary(outputPrefix + ".best_location.i64", bestLocation))) {
        std::fprintf(stderr, "failed to write output prefix: %s\n", outputPrefix.c_str());
        ret = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    Destroy(imageResource);
    Destroy(templateResource);
    Destroy(scoreResource);
    Destroy(bestScoreResource);
    Destroy(bestLocationResource);
    if (ret == ACL_SUCCESS) {
        std::printf("runner_status=passed workspace_bytes=%lu score_elements=%zu\n", workspaceSize, score.size());
    } else {
        std::fprintf(stderr, "runner_status=failed ret=%d\n", ret);
    }
    return ret;
}

void PrintUsage(const char* program)
{
    std::fprintf(stderr,
                 "usage: %s <dtype> <N> <H> <W> <C> <h> <w> <method> <image.bin> <template.bin> "
                 "<output-prefix>\n",
                 program);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 12) {
        PrintUsage(argv[0]);
        return 2;
    }

    const std::string dtypeName = argv[1];
    int64_t batch = 0;
    int64_t imageH = 0;
    int64_t imageW = 0;
    int64_t channels = 0;
    int64_t templateH = 0;
    int64_t templateW = 0;
    int64_t method = 0;
    try {
        batch = std::stoll(argv[2]);
        imageH = std::stoll(argv[3]);
        imageW = std::stoll(argv[4]);
        channels = std::stoll(argv[5]);
        templateH = std::stoll(argv[6]);
        templateW = std::stoll(argv[7]);
        method = std::stoll(argv[8]);
    } catch (const std::exception&) {
        PrintUsage(argv[0]);
        return 2;
    }
    if (batch <= 0 || imageH <= 0 || imageW <= 0 || channels <= 0 || templateH <= 0 || templateW <= 0 ||
        templateH > imageH || templateW > imageW || method < 0 || method > 5) {
        PrintUsage(argv[0]);
        return 2;
    }

    int ret = aclInit(nullptr);
    constexpr int32_t deviceId = 0;
    if (ret == ACL_SUCCESS) {
        ret = aclrtSetDevice(deviceId);
    }
    aclrtStream stream = nullptr;
    if (ret == ACL_SUCCESS) {
        ret = aclrtCreateStream(&stream);
    }
    if (ret == ACL_SUCCESS) {
        const std::vector<int64_t> imageShape = {batch, imageH, imageW, channels};
        const std::vector<int64_t> templateShape = {1, templateH, templateW, channels};
        if (dtypeName == "float32") {
            ret = RunCase<float>(stream, ACL_FLOAT, imageShape, templateShape, method, argv[9], argv[10], argv[11]);
        } else if (dtypeName == "float16") {
            ret = RunCase<uint16_t>(stream, ACL_FLOAT16, imageShape, templateShape, method, argv[9], argv[10],
                                    argv[11]);
        } else if (dtypeName == "uint8") {
            ret = RunCase<uint8_t>(stream, ACL_UINT8, imageShape, templateShape, method, argv[9], argv[10], argv[11]);
        } else {
            PrintUsage(argv[0]);
            ret = 2;
        }
    }

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret == ACL_SUCCESS ? 0 : 1;
}
