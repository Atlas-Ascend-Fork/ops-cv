/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <acl/acl.h>
#include "aclnn_laplacian_of_gaussian.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstring>
#include <cstdlib>
#define CHECK(e)                                    \
    do {                                            \
        auto rc = (e);                              \
        std::cout << #e << "=" << rc << std::endl;  \
        if (rc) {                                   \
            const char* msg = aclGetRecentErrMsg(); \
            if (msg)                                \
                std::cerr << msg << std::endl;      \
            return 2;                               \
        }                                           \
    } while (0)
static int RunCase(char** argv, aclrtStream stream)
{
    std::vector<int64_t> dims;
    std::stringstream ss(argv[2]);
    std::string part;
    while (std::getline(ss, part, ','))
        dims.push_back(std::stoll(part));
    std::vector<int64_t> strides(dims.size(), 1);
    uint64_t p = 1;
    for (int i = int(dims.size()) - 1; i >= 0; --i) {
        strides[i] = p;
        p *= dims[i];
    }
    auto dt = static_cast<aclDataType>(std::stoi(argv[1]));
    size_t size = dt == ACL_UINT8 ? 1 : dt == ACL_FLOAT16 ? 2 : 4;
    std::vector<char> input(p * size), after(p * size), output(p * 4), prior;
    std::ifstream f(argv[7], std::ios::binary);
    f.read(input.data(), input.size());
    if (size_t(f.gcount()) != input.size())
        return 4;
    void *x = nullptr, *y = nullptr;
    CHECK(aclrtMalloc(&x, input.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
    int canary = getenv("CANARY") ? 1 : 0;
    size_t ybytes = output.size() + (canary ? 64 : 0);
    CHECK(aclrtMalloc(&y, ybytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    if (canary) {
        std::vector<char> pat(ybytes, (char)0xAA);
        CHECK(aclrtMemcpy(y, ybytes, pat.data(), ybytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CHECK(aclrtMemcpy(x, input.size(), input.data(), input.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    aclTensor* xt = aclCreateTensor(dims.data(), dims.size(), dt, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                                    dims.size(), x);
    aclTensor* yt = aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                                    dims.size(), y);
    if (!xt || !yt)
        return 5;
    for (int repeat = 0; repeat < 2; ++repeat) {
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        CHECK(aclnnLaplacianOfGaussianGetWorkspaceSize(xt, std::stod(argv[3]), std::stoll(argv[4]), std::stod(argv[5]),
                                                       argv[6], yt, &ws, &exec));
        if (!exec)
            return 5;
        std::cout << "workspace=" << ws << " repeat=" << repeat << std::endl;
        void* work = nullptr;
        if (ws)
            CHECK(aclrtMalloc(&work, ws, ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK(aclnnLaplacianOfGaussian(work, ws, exec, stream));
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(output.data(), output.size(), y, output.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK(aclrtMemcpy(after.data(), after.size(), x, after.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        if (input != after)
            return 6;
        if (repeat && prior != output)
            return 7;
        prior = output;
        if (work)
            CHECK(aclrtFree(work));
    }
    if (canary) {
        std::vector<char> back(ybytes);
        CHECK(aclrtMemcpy(back.data(), ybytes, y, ybytes, ACL_MEMCPY_DEVICE_TO_HOST));
        int dirty = 0;
        for (size_t i = output.size(); i < ybytes; ++i)
            if (back[i] != (char)0xAA) {
                dirty = 1;
                break;
            }
        std::cout << "canary=" << (dirty ? "DIRTY" : "CLEAN") << " tail=" << (ybytes - output.size())
                  << " outb=" << output.size() << std::endl;
    }
    std::ofstream out(argv[8], std::ios::binary);
    out.write(output.data(), output.size());
    out.close();
    if (!out)
        return 8;
    std::ifstream maps("/proc/self/maps");
    std::ofstream paths(argv[9]);
    paths << maps.rdbuf();
    std::cout << "input_readonly=1 repeat_bitwise=1 output_bytes=" << output.size() << std::endl;
    CHECK(aclDestroyTensor(xt));
    CHECK(aclDestroyTensor(yt));
    CHECK(aclrtFree(x));
    CHECK(aclrtFree(y));
    return 0;
}
int main(int argc, char** argv)
{
    if (argc != 10 && !(argc == 2 && std::string(argv[1]) == "--server"))
        return 1;
    CHECK(aclInit(nullptr));
    uint32_t dc = 0;
    CHECK(aclrtGetDeviceCount(&dc));
    if (dc != 1)
        return 3;
    int32_t user = -1, phy = -1, logic = -1, back = -1;
    CHECK(aclrtGetUserDevIdByPhyDevId(4, &user));
    CHECK(aclrtGetPhyDevIdByUserDevId(user, &phy));
    CHECK(aclrtGetLogicDevIdByUserDevId(user, &logic));
    CHECK(aclrtGetUserDevIdByLogicDevId(logic, &back));
    std::cout << "mapping physical=" << phy << " user=" << user << " logic=" << logic << " back=" << back << std::endl;
    if (phy != 4 || logic != 4 || back != user)
        return 3;
    CHECK(aclrtSetDevice(user));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));

    int status = 0;
    if (argc == 2) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT")
                break;
            std::istringstream fields(line);
            std::vector<std::string> args(11);
            args[0] = "runner";
            for (int i = 1; i <= 10; ++i)
                if (!(fields >> args[i]))
                    return 9;
            std::vector<char*> pointers;
            for (auto& a : args)
                pointers.push_back(a.data());
            std::ofstream log(args[10]);
            auto original = std::cout.rdbuf(log.rdbuf());
            std::cout << "mapping physical=" << phy << " user=" << user << " logic=" << logic << " back=" << back
                      << std::endl;
            status = RunCase(pointers.data(), stream);
            std::cout.rdbuf(original);
            log.close();
            std::cout << "CASE_DONE " << status << std::endl;
            if (status)
                break;
        }
    } else
        status = RunCase(argv, stream);
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(user));
    CHECK(aclFinalize());
    return status;
}
