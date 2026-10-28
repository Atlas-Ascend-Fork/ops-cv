/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include "aclnn_clahe.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(int status, const char* stage)
{
    if (status != 0) {
        throw std::runtime_error(std::string(stage) + ": " + std::to_string(status));
    }
}
struct DeviceMemory {
    void* data = nullptr;
    explicit DeviceMemory(uint64_t bytes)
    {
        Check(aclrtMalloc(&data, std::max<uint64_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~DeviceMemory()
    {
        if (data != nullptr) {
            aclrtFree(data);
        }
    }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
};

template <typename T>
void Example(aclrtStream stream)
{
    const std::vector<int64_t> shape{1, 1, 64, 64};
    const std::vector<int64_t> strides{4096, 4096, 64, 1};
    const aclDataType dtype = sizeof(T) == 1 ? ACL_UINT8 : ACL_UINT16;
    std::vector<T> input(4096);
    std::vector<T> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<T>(i * (sizeof(T) == 1 ? 1 : 17));
    }
    const uint64_t bytes = input.size() * sizeof(T);
    DeviceMemory xMem(bytes);
    DeviceMemory yMem(bytes);
    Check(aclrtMemcpy(xMem.data, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    using TensorHandle = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
    TensorHandle x(aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                   shape.size(), xMem.data),
                   aclDestroyTensor);
    TensorHandle y(aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                   shape.size(), yMem.data),
                   aclDestroyTensor);
    if (!x || !y) {
        throw std::runtime_error("aclCreateTensor returned null");
    }
    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    // nullptr 使用注册的默认网格 [8,8]。
    Check(aclnnClaheGetWorkspaceSize(x.get(), 40.0, nullptr, y.get(), &workspaceBytes, &executor), "prepare Clahe");
    DeviceMemory workspace(workspaceBytes);
    Check(aclnnClahe(workspace.data, workspaceBytes, executor, stream), "launch Clahe");
    Check(aclrtSynchronizeStreamWithTimeout(stream, 30000), "synchronize Clahe");
    Check(aclrtMemcpy(output.data(), bytes, yMem.data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
    std::cout << "Clahe uint" << sizeof(T) * 8 << " output:";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << ' ' << static_cast<uint32_t>(output[i]);
    }
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv)
{
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "set device");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        if (argc > 1 && std::string(argv[1]) == "16") {
            Example<uint16_t>(stream);
        } else {
            Example<uint8_t>(stream);
        }
        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtResetDevice(0), "reset device");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
