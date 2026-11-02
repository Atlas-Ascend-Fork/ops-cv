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
#include <cstdio>
#include <cstdlib>
#include "acl/acl.h"
#include "aclnn_flood_fill.h"
#include "../flood_fill_cases.h"

namespace {
using namespace FloodFillTest;

bool Check(int status, const char* operation)
{
    if (status == ACL_SUCCESS)
        return true;
    std::fprintf(stderr, "%s failed: %d\n", operation, status);
    return false;
}

struct Resources {
    std::array<void*, 4> buffers{};
    std::array<aclTensor*, 4> tensors{};
    std::array<aclFloatArray*, 3> attributes{};
    aclIntArray* seed = nullptr;
    void* workspace = nullptr;

    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources()
    {
        for (auto tensor : tensors)
            if (tensor != nullptr)
                aclDestroyTensor(tensor);
        for (auto buffer : buffers)
            if (buffer != nullptr)
                aclrtFree(buffer);
        for (auto attribute : attributes)
            if (attribute != nullptr)
                aclDestroyFloatArray(attribute);
        if (seed != nullptr)
            aclDestroyIntArray(seed);
        if (workspace != nullptr)
            aclrtFree(workspace);
    }

    bool Create(size_t index, int64_t height, int64_t width, aclDataType type, size_t bytes, const void* input)
    {
        if (!Check(aclrtMalloc(&buffers[index], bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc"))
            return false;
        const int status = input != nullptr ?
                               aclrtMemcpy(buffers[index], bytes, input, bytes, ACL_MEMCPY_HOST_TO_DEVICE) :
                               aclrtMemset(buffers[index], bytes, 0xa5, bytes);
        if (!Check(status, "initialize tensor"))
            return false;
        const int64_t shape[] = {height, width};
        const int64_t strides[] = {width, 1};
        tensors[index] = aclCreateTensor(shape, 2, type, strides, 0, ACL_FORMAT_ND, shape, 2, buffers[index]);
        return tensors[index] != nullptr;
    }
};

template <typename T>
bool Matches(void* device, const std::vector<T>& expected, const char* name)
{
    std::vector<T> actual(expected.size());
    const size_t bytes = actual.size() * sizeof(T);
    if (!Check(aclrtMemcpy(actual.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), name))
        return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: actual=%g expected=%g\n", name, i, static_cast<double>(actual[i]),
                         static_cast<double>(expected[i]));
            return false;
        }
    }
    return true;
}

template <typename T>
bool RunCase(aclrtStream stream, aclDataType type, const Case& c)
{
    const auto f = MakeFixture<T>(c);
    const size_t bytes = f.input.size() * sizeof(T);
    Resources data;
    if (!data.Create(0, c.height, c.width, type, bytes, f.input.data()) ||
        !data.Create(1, c.height, c.width, type, bytes, nullptr) ||
        (c.maskInput && !data.Create(2, c.height + 2U, c.width + 2U, ACL_UINT8, f.mask.size(), f.mask.data())) ||
        (c.maskOutput && !data.Create(3, c.height + 2U, c.width + 2U, ACL_UINT8, f.mask.size(), nullptr)))
        return false;
    const int64_t seed[] = {c.SeedRow(), c.SeedCol()};
    const float attributes[] = {c.newValue, c.Lo(), c.Up()};
    data.seed = aclCreateIntArray(seed, 2);
    if (data.seed == nullptr)
        return false;
    for (size_t i = 0; i < data.attributes.size(); ++i) {
        data.attributes[i] = aclCreateFloatArray(&attributes[i], 1);
        if (data.attributes[i] == nullptr)
            return false;
    }
    char neighbor[] = "NEIGHBOR";
    char fixed[] = "FIXED_RANGE";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (!Check(aclnnFloodFillGetWorkspaceSize(data.tensors[0], data.tensors[2], data.seed, data.attributes[0],
                                              data.attributes[1], data.attributes[2], c.connectivity,
                                              c.fixed ? fixed : neighbor, data.tensors[1], data.tensors[3],
                                              &workspaceSize, &executor),
               "aclnnFloodFillGetWorkspaceSize"))
        return false;
    if (workspaceSize > 0 &&
        !Check(aclrtMalloc(&data.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation"))
        return false;
    const bool launched = Check(aclnnFloodFill(data.workspace, workspaceSize, executor, stream), "aclnnFloodFill");
    const bool synchronized = Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    if (!launched || !synchronized)
        return false;
    bool pass = Matches(data.buffers[1], f.expected, "output image");
    pass = Matches(data.buffers[0], f.input, "input image") && pass;
    if (c.maskOutput)
        pass = Matches(data.buffers[3], f.expectedMask, "output mask") && pass;
    if (c.maskInput)
        pass = Matches(data.buffers[2], f.mask, "input mask") && pass;
    std::printf("[%s] %s pattern=%d shape=%ux%u dtype=%d connectivity=%u fixed=%d mask_in=%d mask_out=%d filled=%zu\n",
                pass ? "PASS" : "FAIL", c.name, static_cast<int>(c.pattern), c.height, c.width, static_cast<int>(type),
                c.connectivity, c.fixed, c.maskInput, c.maskOutput, f.filled);
    std::fflush(stdout);
    return pass;
}
} // namespace

int main(int argc, char** argv)
{
    long device = 0;
    if (argc > 1) {
        char* end = nullptr;
        device = std::strtol(argv[1], &end, 10);
        if (argc != 2 || end == argv[1] || *end != '\0' || device < 0 || device > 255) {
            std::fprintf(stderr, "Usage: %s [device_id]\n", argv[0]);
            return 2;
        }
    }
    if (!Check(aclInit(nullptr), "aclInit"))
        return 1;
    if (!Check(aclrtSetDevice(device), "aclrtSetDevice")) {
        aclFinalize();
        return 1;
    }
    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(device);
        aclFinalize();
        return 1;
    }
    const auto cases = RegressionCases();
    unsigned passed = 0;
    for (const auto& c : cases) {
        passed += RunCase<uint8_t>(stream, ACL_UINT8, c);
        passed += RunCase<int32_t>(stream, ACL_INT32, c);
        passed += RunCase<float>(stream, ACL_FLOAT, c);
    }
    // Restore these five pre-reduction cases with their original shape,
    // seed, dtype, fill value, tolerance, connectivity and output-mask choice.
    passed += RunCase<uint8_t>(stream, ACL_UINT8,
                               {Pattern::LEGACY_COMB, 97, 129, 4, false, false, true, 200.0F, "span_overflow_u8"});
    passed += RunCase<float>(stream, ACL_FLOAT,
                             {Pattern::LEGACY_COMB, 97, 129, 8, true, false, true, 42.25F, "span_overflow_f32"});
    passed += RunCase<int32_t>(stream, ACL_INT32,
                               {Pattern::LEGACY_COMB, 97, 129, 4, false, false, false, -42.0F, "span_overflow_nomask"});
    passed += RunCase<uint8_t>(
        stream, ACL_UINT8, {Pattern::LEGACY_ISLANDS, 97, 129, 8, true, false, true, 200.0F, "overflow_fixed_island"});
    passed += RunCase<int32_t>(
        stream, ACL_INT32,
        {Pattern::LEGACY_ISLANDS, 97, 129, 4, true, false, false, -42.0F, "overflow_fixed_island_nomask"});
    const size_t total = cases.size() * 3U + 5U;
    std::printf("SUMMARY: %u/%zu PASS\n", passed, total);
    bool success = passed == total;
    success = Check(aclrtDestroyStream(stream), "aclrtDestroyStream") && success;
    success = Check(aclrtResetDevice(device), "aclrtResetDevice") && success;
    success = Check(aclFinalize(), "aclFinalize") && success;
    return success ? 0 : 1;
}
