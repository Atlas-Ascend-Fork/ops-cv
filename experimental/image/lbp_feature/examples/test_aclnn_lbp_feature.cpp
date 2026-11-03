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
 * \file test_aclnn_lbp_feature.cpp
 * \brief lbp_feature aclnn 真机调用验证
 *   用法: ./test_aclnn_lbp_feature [input_bin output_bin uint8|float32]
 *   缺省为 uint8。数据与 kernel UT 同源（gen_data.py 生成），
 *   输出写 bin 后由 compare_data.py 对拍。
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include "acl/acl.h"
#include "aclnn_lbp_feature.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (cond) {           \
            expr;             \
        }                     \
    } while (0)

namespace {
constexpr int64_t H = 8, W = 130;
constexpr int64_t OUT_H = H - 2, OUT_W = W - 2;
constexpr int64_t RADIUS = 1, NEIGHBORS = 8;

bool ReadBin(const std::string& path, void* buf, size_t bytes)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    size_t rd = fread(buf, 1, bytes, f);
    fclose(f);
    return rd == bytes;
}

bool WriteBin(const std::string& path, const void* buf, size_t bytes)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    size_t wr = fwrite(buf, 1, bytes, f);
    fclose(f);
    return wr == bytes;
}

int32_t CreateAclTensor(const std::vector<int64_t>& shape, void* deviceAddr, aclDataType dtype, aclTensor** tensor)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    aclTensor* t = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   shape.data(), shape.size(), deviceAddr);
    *tensor = t;
    return (t == nullptr) ? -1 : ACL_SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    // ── 参数解析：输入bin、输出bin、dtype ──
    std::string inPath = "./input_lbp_feature.bin";
    std::string outPath = "./output_npu_lbp_feature.bin";
    aclDataType inDtype = ACL_UINT8;
    if (argc >= 2) {
        inPath = argv[1];
    }
    if (argc >= 3) {
        outPath = argv[2];
    }
    if (argc >= 4 && std::string(argv[3]) == "float32") {
        inDtype = ACL_FLOAT;
    }
    const bool isFp32 = (inDtype == ACL_FLOAT);
    const size_t inElemSize = isFp32 ? sizeof(float) : sizeof(uint8_t);

    // 1. acl 初始化
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

    const size_t xBytes = static_cast<size_t>(H * W) * inElemSize;
    const size_t zBytes = static_cast<size_t>(OUT_H * OUT_W) * sizeof(int32_t);

    // 2. host 侧读入 UT 同源输入
    std::vector<uint8_t> xHostBytes(xBytes);
    if (!ReadBin(inPath, xHostBytes.data(), xBytes)) {
        std::cout << "[ERROR] 读 " << inPath << " 失败，请先用 gen_data.py 生成" << std::endl;
        return -1;
    }

    // 3. device 内存 + 搬运
    void *xDev = nullptr, *outDev = nullptr;
    CHECK_RET(aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&outDev, zBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemcpy(xDev, xBytes, xHostBytes.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS,
              return -1);

    aclTensor *x = nullptr, *out = nullptr;
    std::vector<int64_t> xShape = {H, W};
    std::vector<int64_t> outShape = {OUT_H, OUT_W};
    CHECK_RET(CreateAclTensor(xShape, xDev, inDtype, &x) != ACL_SUCCESS, return -1);
    CHECK_RET(CreateAclTensor(outShape, outDev, ACL_INT32, &out) != ACL_SUCCESS, return -1);

    // 4. 两段式调用
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_RET(aclnnLbpFeatureGetWorkspaceSize(x, RADIUS, NEIGHBORS, out, &workspaceSize, &executor) != ACL_SUCCESS, {
        std::cout << "[ERROR] aclnnLbpFeatureGetWorkspaceSize failed" << std::endl;
        return -1;
    });

    void* wsAddr = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&wsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS, return -1);
    }

    CHECK_RET(aclnnLbpFeature(wsAddr, workspaceSize, executor, stream) != ACL_SUCCESS, {
        std::cout << "[ERROR] aclnnLbpFeature failed" << std::endl;
        return -1;
    });

    // 5. 同步 + 拷回
    CHECK_RET(aclrtSynchronizeStream(stream) != ACL_SUCCESS, return -1);
    std::vector<int32_t> outHost(OUT_H * OUT_W);
    CHECK_RET(aclrtMemcpy(outHost.data(), zBytes, outDev, zBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS, return -1);

    // 6. 写输出 bin
    CHECK_RET(!WriteBin(outPath, outHost.data(), zBytes), {
        std::cout << "[ERROR] 写 " << outPath << " 失败" << std::endl;
        return -1;
    });

    std::cout << "lbp_feature NPU 执行成功[" << (isFp32 ? "float32" : "uint8") << "]，输出前10个值: ";
    for (int i = 0; i < 10 && i < OUT_H * OUT_W; ++i) {
        std::cout << outHost[i] << " ";
    }
    std::cout << std::endl;

    // 7. 释放
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    aclrtFree(xDev);
    aclrtFree(outDev);
    if (wsAddr != nullptr) {
        aclrtFree(wsAddr);
    }
    aclrtDestroyStream(stream);
    CHECK_RET(aclrtResetDevice(deviceId) != ACL_SUCCESS, return -1);
    CHECK_RET(aclFinalize() != ACL_SUCCESS, return -1);
    return 0;
}
