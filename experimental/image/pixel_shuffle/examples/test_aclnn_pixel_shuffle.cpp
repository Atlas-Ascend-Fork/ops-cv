/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <string>
#include <cstdint>
#include <cstdlib>
#include "acl/acl.h"
#include "aclnn_pixel_shuffle.h"

// float16 类型定义
#ifndef HAVE_FLOAT16_TYPE
#if defined(__GNUC__) || defined(__clang__)
using float16 = __fp16;
#else
using float16 = uint16_t;
#endif
#define HAVE_FLOAT16_TYPE
#endif

#define CHECK_RET(cond, return_expr)                                                              \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            LOG_PRINT("CHECK_RET failed: %s ret=%d %s\n", #cond, (int)ret, aclGetRecentErrMsg()); \
            return_expr;                                                                          \
        }                                                                                         \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout);                 \
    } while (0)

// 测试结果结构体
struct TestResult {
    std::string name;
    std::string dtype;
    bool passed;
    int64_t total_elements;
    int64_t correct_count;
    double accuracy;
    double max_diff;
};

std::vector<TestResult> g_test_results;

// ============ 工具函数 ============
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// ============ CPU 参考实现（模板化）============
template <typename T>
std::vector<T> cpu_pixel_shuffle(const std::vector<T>& input, const std::vector<int64_t>& input_shape,
                                 int upscale_factor)
{
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    int64_t C_out = C / (upscale_factor * upscale_factor);
    int64_t H_out = H * upscale_factor;
    int64_t W_out = W * upscale_factor;

    std::vector<T> output(N * C_out * H_out * W_out, 0);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
            for (int64_t h = 0; h < H_out; ++h) {
                for (int64_t w = 0; w < W_out; ++w) {
                    int64_t output_idx = n * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w;

                    int64_t input_c = c * upscale_factor * upscale_factor + (h % upscale_factor) * upscale_factor +
                                      (w % upscale_factor);
                    int64_t input_h = h / upscale_factor;
                    int64_t input_w = w / upscale_factor;

                    int64_t input_idx = n * C * H * W + input_c * H * W + input_h * W + input_w;

                    if (input_idx < static_cast<int64_t>(input.size())) {
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
    }

    return output;
}

// ============ CPU 参考实现：5D（模板化）============
// out[n,c,d*r+di,h*r+hi,w*r+wi] = in[n,c*r^3+(di*r^2+hi*r+wi),d,h,w]
template <typename T>
std::vector<T> cpu_pixel_shuffle_5d(const std::vector<T>& input, const std::vector<int64_t>& input_shape,
                                    int upscale_factor)
{
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];
    int64_t r = upscale_factor;
    int64_t rCube = r * r * r;
    int64_t C_out = C / rCube;
    int64_t D_out = D * r, H_out = H * r, W_out = W * r;

    std::vector<T> output(N * C_out * D_out * H_out * W_out, 0);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
            for (int64_t d = 0; d < D_out; ++d) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t di = d % r, hi = h % r, wi = w % r;
                        int64_t c_in = c * rCube + (di * r + hi) * r + wi;
                        int64_t out_idx = ((n * C_out + c) * D_out + d) * H_out * W_out + h * W_out + w;
                        int64_t in_idx = ((n * C + c_in) * D + d / r) * H * W + (h / r) * W + (w / r);
                        if (in_idx < static_cast<int64_t>(input.size())) {
                            output[out_idx] = input[in_idx];
                        }
                    }
                }
            }
        }
    }
    return output;
}

// ============ 验证函数（模板化）============
template <typename T>
TestResult verify_pixel_shuffle(const std::vector<T>& input_data, const std::vector<int64_t>& input_shape,
                                const std::vector<T>& acl_output, const std::vector<int64_t>& output_shape,
                                int upscale_factor, const std::string& test_name, const std::string& dtype_name)
{
    TestResult result;
    result.name = test_name;
    result.dtype = dtype_name;

    LOG_PRINT("\n=== 验证 Pixel Shuffle: %s (%s) ===\n", test_name.c_str(), dtype_name.c_str());

    // 计算CPU参考结果（按 rank 分派 4D/5D）
    auto cpu_output = (input_shape.size() == 5) ? cpu_pixel_shuffle_5d(input_data, input_shape, upscale_factor) :
                                                  cpu_pixel_shuffle(input_data, input_shape, upscale_factor);

    // 验证所有位置
    int64_t total_elements = GetShapeSize(output_shape);

    // 打印几个样例（输入 / ACL 输出 / CPU 参考 前若干个元素），便于人工核对
    const int64_t kSampleCount = 8;
    LOG_PRINT("\n=== 样例输出（前 %ld 个元素）===\n", kSampleCount);
    LOG_PRINT("  input[0..%ld]: ", kSampleCount - 1);
    for (int64_t i = 0; i < std::min<int64_t>(kSampleCount, static_cast<int64_t>(input_data.size())); ++i) {
        LOG_PRINT("%.2f ", static_cast<float>(input_data[i]));
    }
    LOG_PRINT("\n  acl  [0..%ld]: ", kSampleCount - 1);
    for (int64_t i = 0; i < std::min<int64_t>(kSampleCount, total_elements); ++i) {
        LOG_PRINT("%.2f ", static_cast<float>(acl_output[i]));
    }
    LOG_PRINT("\n  cpu  [0..%ld]: ", kSampleCount - 1);
    for (int64_t i = 0; i < std::min<int64_t>(kSampleCount, total_elements); ++i) {
        LOG_PRINT("%.2f ", static_cast<float>(cpu_output[i]));
    }
    LOG_PRINT("\n");

    int64_t correct_count = 0;
    double max_diff = 0.0;
    double sum_diff = 0.0;
    std::vector<int64_t> error_positions;

    // float16 用稍大的阈值
    float threshold = (sizeof(T) == 2) ? 1e-1f : 1e-4f;

    for (int64_t i = 0; i < total_elements; ++i) {
        float diff = std::abs(static_cast<float>(acl_output[i]) - static_cast<float>(cpu_output[i]));
        if (diff < threshold) {
            correct_count++;
        } else {
            error_positions.push_back(i);
            max_diff = std::max(max_diff, static_cast<double>(diff));
            sum_diff += diff;
        }
    }

    result.total_elements = total_elements;
    result.correct_count = correct_count;
    result.accuracy = static_cast<double>(correct_count) * 100.0 / total_elements;
    result.max_diff = max_diff;
    result.passed = (correct_count == total_elements);

    LOG_PRINT("\n=== 验证统计 ===\n");
    LOG_PRINT("总验证位置数: %ld\n", total_elements);
    LOG_PRINT("正确位置数: %ld\n", correct_count);
    LOG_PRINT("错误位置数: %zu\n", error_positions.size());
    LOG_PRINT("正确率: %.2f%%\n", result.accuracy);

    if (result.passed) {
        LOG_PRINT("✓ 测试通过！所有位置正确\n");
    } else {
        LOG_PRINT("✗ 测试失败！有 %zu 个错误位置\n", error_positions.size());
    }

    if (error_positions.size() > 0) {
        LOG_PRINT("平均误差: %.6f\n", sum_diff / error_positions.size());
        LOG_PRINT("最大误差: %.6f\n", max_diff);

        // 打印前5个错误位置
        LOG_PRINT("\n前5个错误位置详情:\n");
        for (size_t i = 0; i < std::min<size_t>(5, error_positions.size()); ++i) {
            int64_t idx = error_positions[i];
            LOG_PRINT("  [%ld]: ACL=%.4f, CPU=%.4f, diff=%.4f\n", idx, static_cast<float>(acl_output[idx]),
                      static_cast<float>(cpu_output[idx]),
                      std::abs(static_cast<float>(acl_output[idx] - cpu_output[idx])));
        }
    }

    // 验证数值总和
    double input_sum = 0, acl_sum = 0, cpu_sum = 0;
    for (auto v : input_data)
        input_sum += static_cast<double>(v);
    for (auto v : acl_output)
        acl_sum += static_cast<double>(v);
    for (auto v : cpu_output)
        cpu_sum += static_cast<double>(v);

    LOG_PRINT("\n=== 数值总和验证 ===\n");
    LOG_PRINT("输入总和: %.2f\n", input_sum);
    LOG_PRINT("ACL输出总和: %.2f\n", acl_sum);
    LOG_PRINT("CPU输出总和: %.2f\n", cpu_sum);

    double sum_diff_total = std::abs(input_sum - acl_sum);
    if (sum_diff_total < 1.0) {
        LOG_PRINT("✓ 数值总和一致\n");
    } else {
        LOG_PRINT("⚠ 数值总和不一致 (差: %.2f)\n", sum_diff_total);
        result.passed = false;
    }

    return result;
}

// ============ ACL Tensor 创建函数 ============
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() == 4) {
        strides[3] = 1;                     // W
        strides[2] = shape[3];              // H stride = W
        strides[1] = shape[2] * strides[2]; // C stride = H * W
        strides[0] = shape[1] * strides[1]; // N stride = C * H * W
    } else {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    std::vector<int64_t> storageDims = shape;

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageDims.data(), storageDims.size(), *deviceAddr);

    if (*tensor == nullptr) {
        LOG_PRINT("Failed to create tensor\n");
        return ACL_ERROR_FAILURE;
    }

    LOG_PRINT("Created tensor: shape=[");
    for (size_t i = 0; i < shape.size(); ++i) {
        LOG_PRINT("%ld", shape[i]);
        if (i < shape.size() - 1)
            LOG_PRINT(", ");
    }
    LOG_PRINT("], dataType=%d, deviceAddr=%p\n", dataType, *deviceAddr);

    return ACL_SUCCESS;
}

// ============ 测试用例生成函数（模板化）============
template <typename T>
std::vector<T> generate_0_15_sequence(const std::vector<int64_t>& shape)
{
    int64_t total = GetShapeSize(shape);
    std::vector<T> data(total);
    for (int64_t i = 0; i < total; ++i) {
        data[i] = static_cast<T>(i);
    }
    return data;
}

template <typename T>
std::vector<T> generate_arithmetic_sequence(const std::vector<int64_t>& shape, T start = 0.0, T step = 1.0)
{
    int64_t total = GetShapeSize(shape);
    std::vector<T> data(total);
    for (int64_t i = 0; i < total; ++i) {
        data[i] = start + static_cast<T>(i) * step;
    }
    return data;
}

template <typename T>
std::vector<T> generate_random_data(const std::vector<int64_t>& shape)
{
    int64_t total = GetShapeSize(shape);
    std::vector<T> data(total);

    // float16 用小范围，float32 用大范围
    float max_val = (sizeof(T) == 2) ? 5.0f : 100.0f;

    for (int64_t i = 0; i < total; ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        data[i] = static_cast<T>(r * max_val);
    }
    return data;
}

template <typename T>
std::vector<T> generate_channel_distinct_data(const std::vector<int64_t>& shape)
{
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    std::vector<T> data(N * C * H * W);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            T channel_base = static_cast<T>((c + 1) * 10);
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = n * C * H * W + c * H * W + h * W + w;
                    data[idx] = channel_base + static_cast<T>(h * 5 + w);
                }
            }
        }
    }
    return data;
}

// ============ 测试执行函数（模板化）============
template <typename T>
bool run_test_case(const std::vector<T>& input_data, const std::vector<int64_t>& input_shape, int upscale_factor,
                   aclrtStream stream, aclDataType acl_dtype, const std::string& test_name)
{
    std::string dtype_name = (sizeof(T) == 2) ? "float16" : "float32";

    LOG_PRINT("\n%s\n", std::string(80, '=').c_str());
    LOG_PRINT("开始测试: %s (%s)\n", test_name.c_str(), dtype_name.c_str());
    LOG_PRINT("输入形状: [%ld, %ld, %ld, %ld], upscale_factor=%d\n", input_shape[0], input_shape[1], input_shape[2],
              input_shape[3], upscale_factor);

    // 验证输入通道数
    int64_t r_squared = upscale_factor * upscale_factor;
    if (input_shape[1] % r_squared != 0) {
        LOG_PRINT("错误: 输入通道数 %ld 不能被 %d^2=%ld 整除\n", input_shape[1], upscale_factor, r_squared);
        return false;
    }

    // 计算输出形状
    std::vector<int64_t> output_shape = {input_shape[0], input_shape[1] / r_squared, input_shape[2] * upscale_factor,
                                         input_shape[3] * upscale_factor};

    // 创建输入tensor
    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    auto ret = CreateAclTensor(input_data, input_shape, &selfXDeviceAddr, acl_dtype, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 创建输出tensor
    std::vector<T> output_host_data(GetShapeSize(output_shape), 0);
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(output_host_data, output_shape, &outDeviceAddr, acl_dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 获取workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnPixelShuffleGetWorkspaceSize(selfX, upscale_factor, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnPixelShuffleGetWorkspaceSize failed: %d\n", ret);
        return false;
    }
    LOG_PRINT("workspaceSize=%lu\n", workspaceSize);

    // 执行算子
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }

    ret = aclnnPixelShuffle(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 读取输出结果
    std::vector<T> acl_output_data(GetShapeSize(output_shape), 0);
    ret = aclrtMemcpy(acl_output_data.data(), acl_output_data.size() * sizeof(T), outDeviceAddr,
                      acl_output_data.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 验证结果
    TestResult result = verify_pixel_shuffle(input_data, input_shape, acl_output_data, output_shape, upscale_factor,
                                             test_name, dtype_name);
    g_test_results.push_back(result);

    // 清理
    aclDestroyTensor(selfX);
    aclDestroyTensor(out);
    if (selfXDeviceAddr)
        aclrtFree(selfXDeviceAddr);
    if (outDeviceAddr)
        aclrtFree(outDeviceAddr);
    if (workspaceAddr)
        aclrtFree(workspaceAddr);

    LOG_PRINT("%s 测试完成\n", std::string(80, '=').c_str());
    return result.passed;
}

// ============ 测试执行函数：5D（模板化）============
template <typename T>
bool run_test_case_5d(const std::vector<T>& input_data, const std::vector<int64_t>& input_shape, int upscale_factor,
                      aclrtStream stream, aclDataType acl_dtype, const std::string& test_name)
{
    std::string dtype_name = (sizeof(T) == 2) ? "float16" : "float32";

    LOG_PRINT("\n%s\n", std::string(80, '=').c_str());
    LOG_PRINT("开始测试: %s (%s)\n", test_name.c_str(), dtype_name.c_str());
    LOG_PRINT("输入形状: [%ld, %ld, %ld, %ld, %ld], upscale_factor=%d\n", input_shape[0], input_shape[1],
              input_shape[2], input_shape[3], input_shape[4], upscale_factor);

    int64_t r = upscale_factor;
    int64_t rCube = r * r * r;
    if (input_shape[1] % rCube != 0) {
        LOG_PRINT("错误: 输入通道数 %ld 不能被 %d^3=%ld 整除\n", input_shape[1], upscale_factor, rCube);
        return false;
    }

    // 计算输出形状
    std::vector<int64_t> output_shape = {input_shape[0], input_shape[1] / rCube, input_shape[2] * upscale_factor,
                                         input_shape[3] * upscale_factor, input_shape[4] * upscale_factor};

    // 创建输入tensor
    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    auto ret = CreateAclTensor(input_data, input_shape, &selfXDeviceAddr, acl_dtype, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 创建输出tensor
    std::vector<T> output_host_data(GetShapeSize(output_shape), 0);
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(output_host_data, output_shape, &outDeviceAddr, acl_dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 获取workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnPixelShuffleGetWorkspaceSize(selfX, upscale_factor, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnPixelShuffleGetWorkspaceSize failed: %d\n", ret);
        return false;
    }
    LOG_PRINT("workspaceSize=%lu\n", workspaceSize);

    // 执行算子
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }

    ret = aclnnPixelShuffle(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 读取输出结果
    std::vector<T> acl_output_data(GetShapeSize(output_shape), 0);
    ret = aclrtMemcpy(acl_output_data.data(), acl_output_data.size() * sizeof(T), outDeviceAddr,
                      acl_output_data.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    // 验证结果
    TestResult result = verify_pixel_shuffle(input_data, input_shape, acl_output_data, output_shape, upscale_factor,
                                             test_name, dtype_name);
    g_test_results.push_back(result);

    // 清理
    aclDestroyTensor(selfX);
    aclDestroyTensor(out);
    if (selfXDeviceAddr)
        aclrtFree(selfXDeviceAddr);
    if (outDeviceAddr)
        aclrtFree(outDeviceAddr);
    if (workspaceAddr)
        aclrtFree(workspaceAddr);

    LOG_PRINT("%s 测试完成\n", std::string(80, '=').c_str());
    return result.passed;
}

// ============ 主函数 ============
int main()
{
    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("Pixel Shuffle 算子全面验证测试 (支持 float32/float16)\n");
    LOG_PRINT("%s\n\n", std::string(100, '=').c_str());

    // 清空测试结果
    g_test_results.clear();

    // 初始化随机种子
    srand(42);

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    // 初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    // ============ float32 测试 ============
    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("float32 测试\n");
    LOG_PRINT("%s\n", std::string(100, '=').c_str());

    // 测试用例1：基础0-15序列验证
    {
        std::vector<int64_t> shape = {1, 4, 2, 2};
        auto input_data = generate_0_15_sequence<float>(shape);
        run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "基础验证 (0-15序列)");
    }

    // 测试用例2：更大尺寸
    {
        std::vector<int64_t> shape = {1, 16, 4, 4};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "大尺寸测试 (C=16, H=4, W=4)");
    }

    // 测试用例3：scale=3
    {
        std::vector<int64_t> shape = {1, 9, 3, 3};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case<float>(input_data, shape, 3, stream, ACL_FLOAT, "scale=3测试");
    }

    // 测试用例4：scale=4
    {
        std::vector<int64_t> shape = {1, 16, 2, 2};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case<float>(input_data, shape, 4, stream, ACL_FLOAT, "scale=4测试");
    }

    // 测试用例5：多batch
    {
        std::vector<int64_t> shape = {2, 4, 2, 2};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "多Batch测试 (N=2)");
    }

    // 测试用例6：随机数据
    {
        std::vector<int64_t> shape = {1, 8, 8, 8};
        auto input_data = generate_random_data<float>(shape);
        run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "随机数据测试");
    }

    // ============ float16 测试 ============
    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("float16 测试\n");
    LOG_PRINT("%s\n", std::string(100, '=').c_str());

    // 测试用例1：基础0-15序列验证 (float16)
    {
        std::vector<int64_t> shape = {1, 4, 2, 2};
        auto input_data = generate_0_15_sequence<float16>(shape);
        run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "基础验证 (0-15序列)");
    }

    // 测试用例2：更大尺寸 (float16)
    {
        std::vector<int64_t> shape = {1, 16, 4, 4};
        auto input_data = generate_arithmetic_sequence<float16>(shape, float16(0.0), float16(1.0));
        run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "大尺寸测试 (C=16, H=4, W=4)");
    }

    // 测试用例3：scale=3 (float16)
    {
        std::vector<int64_t> shape = {1, 9, 3, 3};
        auto input_data = generate_arithmetic_sequence<float16>(shape, float16(0.0), float16(1.0));
        run_test_case<float16>(input_data, shape, 3, stream, ACL_FLOAT16, "scale=3测试");
    }

    // 测试用例4：scale=4 (float16)
    {
        std::vector<int64_t> shape = {1, 16, 2, 2};
        auto input_data = generate_arithmetic_sequence<float16>(shape, float16(0.0), float16(1.0));
        run_test_case<float16>(input_data, shape, 4, stream, ACL_FLOAT16, "scale=4测试");
    }

    // 测试用例5：多batch (float16)
    {
        std::vector<int64_t> shape = {2, 4, 2, 2};
        auto input_data = generate_arithmetic_sequence<float16>(shape, float16(0.0), float16(1.0));
        run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "多Batch测试 (N=2)");
    }

    // 测试用例6：随机数据 (float16)
    {
        std::vector<int64_t> shape = {1, 8, 8, 8};
        auto input_data = generate_random_data<float16>(shape);
        run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "随机数据测试");
    }

    // 测试用例7：通道区分数据 (float16)
    {
        std::vector<int64_t> shape = {1, 16, 4, 4};
        auto input_data = generate_channel_distinct_data<float16>(shape);
        run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "通道区分测试");
    }

    // ============ 大 shape 带宽测试 ============
    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("大 shape 带宽测试\n");
    LOG_PRINT("%s\n", std::string(100, '=').c_str());

    // 大 batch（多组，测试多核/多 tile 吞吐）
    {
        std::vector<int64_t> shape = {4, 32, 256, 256};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "大shape (N=4,C=32,H=W=256)");
    }

    // 单张大图（单组远超 UB，测试组内行块分片）
    {
        std::vector<int64_t> shape = {1, 64, 512, 512};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "大shape 单图 (N=1,C=64,H=W=512)");
    }

    // 大 shape fp16（用有界随机数据，避免算术序列在 fp16 下溢出为 inf）
    {
        std::vector<int64_t> shape = {4, 32, 256, 256};
        auto input_data = generate_random_data<float16>(shape);
        run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "大shape fp16 (N=4,C=32,H=W=256)");
    }

    // ============ 5D 测试（PyTorch 对齐）============
    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("5D 测试 (N,C,D,H,W)\n");
    LOG_PRINT("%s\n", std::string(100, '=').c_str());

    // 5D 基础（r=2, C=8=r^3）
    {
        std::vector<int64_t> shape = {1, 8, 2, 2, 2};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case_5d<float>(input_data, shape, 2, stream, ACL_FLOAT, "5D基础 (N=1,C=8,D=H=W=2)");
    }

    // 5D 多 batch（r=2, C=16 -> C_out=2）
    {
        std::vector<int64_t> shape = {2, 16, 2, 2, 2};
        auto input_data = generate_random_data<float>(shape);
        run_test_case_5d<float>(input_data, shape, 2, stream, ACL_FLOAT, "5D多batch (N=2,C=16,D=H=W=2)");
    }

    // 5D r=3（C=27=r^3）
    {
        std::vector<int64_t> shape = {1, 27, 2, 2, 2};
        auto input_data = generate_arithmetic_sequence<float>(shape, 0.0f, 1.0f);
        run_test_case_5d<float>(input_data, shape, 3, stream, ACL_FLOAT, "5D scale=3 (C=27)");
    }

    // 5D fp16
    {
        std::vector<int64_t> shape = {1, 8, 2, 2, 2};
        auto input_data = generate_random_data<float16>(shape);
        run_test_case_5d<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "5D fp16 (N=1,C=8,D=H=W=2)");
    }

    // ============ 大 W 分段路径测试（PS_LARGE_W=1 时启用）============
    // 修复前 A*sizeof(T) 随 W 线性膨胀，W 超过 UB 安全上限时 InitBuffer 越界。
    // 修复后 kernel 在 W 方向分段（bPerRow_ > 1），该路径需专门覆盖。
    if (getenv("PS_LARGE_W") != nullptr) {
        LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
        LOG_PRINT("大 W 分段路径测试（W 超单块上限，触发 bPerRow_>1）\n");
        LOG_PRINT("%s\n", std::string(100, '=').c_str());

        // fp32 W=3840：单块上限约 1920，bPerRow_=2
        {
            std::vector<int64_t> shape = {1, 4, 2, 3840};
            auto input_data = generate_random_data<float>(shape);
            run_test_case<float>(input_data, shape, 2, stream, ACL_FLOAT, "大W fp32 (W=3840,bPerRow=2)");
        }

        // fp16 W=7680：单块上限约 3200，bPerRow_=3，且末段 wLast_ < wPerBlock_
        {
            std::vector<int64_t> shape = {1, 4, 2, 7680};
            auto input_data = generate_random_data<float16>(shape);
            run_test_case<float16>(input_data, shape, 2, stream, ACL_FLOAT16, "大W fp16 (W=7680,bPerRow=3)");
        }
    }

    // 清理
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // ============ 打印最终测试结果汇总 ============
    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("测试结果汇总\n");
    LOG_PRINT("%s\n", std::string(100, '=').c_str());

    int total_tests = 0;
    int passed_tests = 0;
    int float32_passed = 0, float32_total = 0;
    int float16_passed = 0, float16_total = 0;

    for (const auto& result : g_test_results) {
        total_tests++;
        if (result.passed) {
            passed_tests++;
            if (result.dtype == "float32")
                float32_passed++;
            else
                float16_passed++;
        }

        if (result.dtype == "float32")
            float32_total++;
        else
            float16_total++;

        LOG_PRINT("%s [%s] %s\n", result.passed ? "✓" : "✗", result.dtype.c_str(), result.name.c_str());
        LOG_PRINT("   正确率: %.2f%% (%ld/%ld), 最大误差: %.6f\n", result.accuracy, result.correct_count,
                  result.total_elements, result.max_diff);
    }

    LOG_PRINT("\n%s\n", std::string(100, '=').c_str());
    LOG_PRINT("最终统计:\n");
    LOG_PRINT("  float32: %d/%d 通过 (%.2f%%)\n", float32_passed, float32_total,
              float32_total > 0 ? static_cast<float>(float32_passed) * 100 / float32_total : 0);
    LOG_PRINT("  float16: %d/%d 通过 (%.2f%%)\n", float16_passed, float16_total,
              float16_total > 0 ? static_cast<float>(float16_passed) * 100 / float16_total : 0);
    LOG_PRINT("  总计: %d/%d 通过 (%.2f%%)\n", passed_tests, total_tests,
              static_cast<float>(passed_tests) * 100 / total_tests);

    if (passed_tests == total_tests) {
        LOG_PRINT("\n✓ 所有测试通过！Pixel Shuffle 算子实现正确！\n");
    } else {
        LOG_PRINT("\n✗ 有 %d 个测试失败，请检查算子实现\n", total_tests - passed_tests);
    }
    LOG_PRINT("%s\n", std::string(100, '=').c_str());

    return (passed_tests == total_tests) ? 0 : -1;
}
