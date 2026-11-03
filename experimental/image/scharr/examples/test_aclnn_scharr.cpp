/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_scharr.cpp
 * \brief Scharr aclnn 调用示例（自包含）：生成确定性输入 -> aclnn 执行 ->
 *        打印前若干输出并保存 input/output bin，便于与 CPU 参考位级核对。
 *        用法: ./test_aclnn_scharr
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_scharr.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (cond) {           \
            expr;             \
        }                     \
    } while (0)

namespace {
constexpr int64_t H = 65;
constexpr int64_t W = 64;
constexpr size_t ELEMS = static_cast<size_t>(H) * W;

uint32_t gSeed = 12345u;
uint32_t NextRand()
{
    gSeed = gSeed * 1664525u + 1013904223u;
    return gSeed;
}

void FillInput(uint8_t* data, size_t elemBytes, int mode)
{
    for (size_t i = 0; i < ELEMS; ++i) {
        uint32_t r = NextRand();
        if (mode == 0) { // uint8: 0..255
            data[i] = static_cast<uint8_t>(r & 0xFFu);
        } else if (mode == 1) { // float: -8..8 区间
            float v = static_cast<float>((static_cast<int32_t>(r % 1601u) - 800)) * 0.01f;
            (void)memcpy(data + i * elemBytes, &v, sizeof(float));
        } else { // half 位模式：小数值
            int32_t v = static_cast<int32_t>(r % 33u) - 16;
            uint16_t bits = (v >= 0) ? static_cast<uint16_t>(v << 10) :
                                       static_cast<uint16_t>(0x8000u | static_cast<uint16_t>((-v) << 10));
            (void)memcpy(data + i * elemBytes, &bits, sizeof(uint16_t));
        }
    }
}

int32_t CreateAclTensor(const std::vector<int64_t>& shape, void* deviceAddr, aclDataType dtype, aclTensor** tensor)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), deviceAddr);
    return (*tensor == nullptr) ? -1 : ACL_SUCCESS;
}

void SaveBin(const std::string& path, const void* data, size_t bytes)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (f != nullptr) {
        (void)fwrite(data, 1, bytes, f);
        fclose(f);
    }
}

int32_t RunOne(const char* dtypeName, aclDataType inDtype, size_t elemBytes, int mode, const char* direction,
               const char* border, aclrtStream stream)
{
    std::string tag = std::string(dtypeName) + "_" + direction + "_" + border;
    printf("================ %s ================\n", tag.c_str());

    const size_t inBytes = ELEMS * elemBytes;
    const size_t outBytes = ELEMS * sizeof(float);
    std::vector<uint8_t> xHost(inBytes);
    gSeed = 12345u + static_cast<uint32_t>(mode) * 777u;
    FillInput(xHost.data(), elemBytes, mode);
    SaveBin("scharr_input_" + tag + ".bin", xHost.data(), inBytes);

    void* xDev = nullptr;
    void* yDev = nullptr;
    CHECK_RET(aclrtMalloc(&xDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&yDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemcpy(xDev, inBytes, xHost.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS, return -1);

    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    std::vector<int64_t> shape = {H, W};
    CHECK_RET(CreateAclTensor(shape, xDev, inDtype, &x) != ACL_SUCCESS, return -1);
    CHECK_RET(CreateAclTensor(shape, yDev, ACL_FLOAT, &y) != ACL_SUCCESS, return -1);

    char* dirArg = const_cast<char*>(direction);
    char* borderArg = const_cast<char*>(border);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // The generated aclnn signature follows the op-def attribute order:
    // (x, border_type, direction, y).
    aclnnStatus st = aclnnScharrGetWorkspaceSize(x, borderArg, dirArg, y, &workspaceSize, &executor);
    if (st != ACL_SUCCESS) {
        printf("[ERROR] GetWorkspaceSize failed, code=%d, msg=%s\n", static_cast<int>(st), aclGetRecentErrMsg());
        return -1;
    }

    void* wsAddr = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&wsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS, return -1);
    }
    st = aclnnScharr(wsAddr, workspaceSize, executor, stream);
    if (st != ACL_SUCCESS) {
        printf("[ERROR] aclnnScharr failed, code=%d, msg=%s\n", static_cast<int>(st), aclGetRecentErrMsg());
        return -1;
    }
    CHECK_RET(aclrtSynchronizeStream(stream) != ACL_SUCCESS, return -1);

    std::vector<float> outHost(ELEMS);
    CHECK_RET(aclrtMemcpy(outHost.data(), outBytes, yDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS,
              return -1);
    SaveBin("scharr_output_" + tag + ".bin", outHost.data(), outBytes);

    printf("前10个输出: ");
    for (int i = 0; i < 10; ++i) {
        printf("%g ", outHost[static_cast<size_t>(i)]);
    }
    printf("\n");

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDev);
    aclrtFree(yDev);
    if (wsAddr != nullptr) {
        aclrtFree(wsAddr);
    }
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) != ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) != ACL_SUCCESS, {
        aclFinalize();
        return -1;
    });
    CHECK_RET(aclrtCreateStream(&stream) != ACL_SUCCESS, {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return -1;
    });

    int ret = 0;
    ret |= RunOne("uint8", ACL_UINT8, sizeof(uint8_t), 0, "X", "REFLECT_101", stream);
    ret |= RunOne("float32", ACL_FLOAT, sizeof(float), 1, "X", "REFLECT_101", stream);
    ret |= RunOne("float16", ACL_FLOAT16, sizeof(uint16_t), 2, "Y", "REPLICATE", stream);
    (void)ret;

    aclrtDestroyStream(stream);
    CHECK_RET(aclrtResetDevice(deviceId) != ACL_SUCCESS, return -1);
    CHECK_RET(aclFinalize() != ACL_SUCCESS, return -1);
    printf("example %s\n", ret == 0 ? "PASS" : "FAIL");
    return ret;
}
