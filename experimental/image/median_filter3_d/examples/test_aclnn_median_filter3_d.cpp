/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <iostream>
#include "acl/acl.h"
#include "aclnn_median_filter3_d.h"
#define CHECK(call)                                        \
    do {                                                   \
        auto rc = (call);                                  \
        if (rc != 0) {                                     \
            std::cerr << #call << ": " << rc << std::endl; \
            return 1;                                      \
        }                                                  \
    } while (0)
int main()
{
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    std::array<uint8_t, 5> input{10, 100, 5, 80, 20}, output{}, expected{10, 10, 80, 20, 20};
    void* xptr = nullptr;
    void* yptr = nullptr;
    void* workspace = nullptr;
    CHECK(aclrtMalloc(&xptr, 5, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&yptr, 5, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(xptr, 5, input.data(), 5, ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t shape[]{1, 1, 5}, strides[]{5, 5, 1}, window[]{1, 1, 3};
    auto* x = aclCreateTensor(shape, 3, ACL_UINT8, strides, 0, ACL_FORMAT_ND, shape, 3, xptr);
    auto* y = aclCreateTensor(shape, 3, ACL_UINT8, strides, 0, ACL_FORMAT_ND, shape, 3, yptr);
    auto* k = aclCreateIntArray(window, 3);
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    char mode[] = "REFLECT";
    CHECK(aclnnMedianFilter3DGetWorkspaceSize(x, k, mode, 0.0, y, &size, &executor));
    if (size > 0) {
        CHECK(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK(aclnnMedianFilter3D(workspace, size, executor, stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(output.data(), 5, yptr, 5, ACL_MEMCPY_DEVICE_TO_HOST));
    bool pass = output == expected;
    std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyIntArray(k);
    if (workspace) {
        CHECK(aclrtFree(workspace));
    }
    CHECK(aclrtFree(xptr));
    CHECK(aclrtFree(yptr));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    return pass ? 0 : 1;
}
