/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <string>
#include <cstdint>
#include <sstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_three_interpolate_backward.h"
#include "../data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void three_interpolate_backward(GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight,
                                                                 GM_ADDR grad_y, GM_ADDR workspace, GM_ADDR tiling);

class three_interpolate_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "three_interpolate_backward_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "three_interpolate_backward_test TearDown\n" << endl; }
};

namespace {
constexpr uint32_t C0 = 16;
constexpr uint32_t N0 = 16;
constexpr uint32_t TILING_ARGS_NUM = 64;

void GetInputData(uint8_t* input_grad_x, size_t grad_x_input_bytes, uint8_t* input_idx, size_t idx_input_bytes,
                  uint8_t* input_weight, size_t weight_input_bytes, string& path, string& shape_info)
{
    system("cp -r "
           "../../../../../../../ops/built-in/tests/ut/fast_op_test/three_interpolate_backward/script"
           " ./ && chmod -R 777 ./script/");

    system("cd ./script/ && rm -rf ./*bin");

    std::string cmd = "cd ./script/ && python3 gen_data.py " + shape_info;
    system(cmd.c_str());

    ReadFile(path + "/script/grad_x.bin", grad_x_input_bytes, input_grad_x, grad_x_input_bytes);
    ReadFile(path + "/script/idx.bin", idx_input_bytes, input_idx, idx_input_bytes);
    ReadFile(path + "/script/weight.bin", weight_input_bytes, input_weight, weight_input_bytes);
}

void WriteOutputData(uint8_t* output_grad_y, uint32_t grad_y_output_bytes, string& path)
{
    WriteFile(path + "/script/grad_y.bin", output_grad_y, grad_y_output_bytes);
}

bool CompareData(string& shape_info)
{
    std::string cmd = "cd ./script/ && python3 verify.py grad_y.bin golden.bin " + shape_info;
    return system(cmd.c_str()) == 0;
}

} // namespace

TEST_F(three_interpolate_backward_test, test_case_1_2_6_5_fp32)
{
    uint32_t bs = 1;
    uint32_t cs = 2;
    uint32_t ms = 6;
    uint32_t ns = 5;
    uint32_t c1 = (cs + C0 - 1) / C0;

    char* path_ = get_current_dir_name();
    string path(path_);

    stringstream shape_raw_info;
    shape_raw_info << bs << " ";
    shape_raw_info << cs << " ";
    shape_raw_info << ms << " ";
    shape_raw_info << ns << " ";
    shape_raw_info << 0 << " ";

    string shape_info = shape_raw_info.str();

    size_t grad_x_input_bytes = (bs * c1 * C0 * ns) * sizeof(float); // NC1HWC0
    size_t idx_input_bytes = (bs * ns * 3) * sizeof(int32_t);        // 精确尺寸分配（issue #1406 回归锚点）
    size_t weight_input_bytes = (bs * ns * 3) * sizeof(float);       // 精确尺寸分配（issue #1406 回归锚点）
    size_t tiling_bytes = TILING_ARGS_NUM * sizeof(uint32_t);
    size_t grad_y_output_bytes = (bs * c1 * C0 * ms) * sizeof(float); // NC1HWC0

    uint8_t* input_grad_x = (uint8_t*)AscendC::GmAlloc(grad_x_input_bytes);
    uint8_t* input_idx = (uint8_t*)AscendC::GmAlloc(idx_input_bytes);
    uint8_t* input_weight = (uint8_t*)AscendC::GmAlloc(weight_input_bytes);
    uint8_t* output_grad_y = (uint8_t*)AscendC::GmAlloc(grad_y_output_bytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_bytes);

    GetInputData(input_grad_x, grad_x_input_bytes, input_idx, idx_input_bytes, input_weight, weight_input_bytes, path,
                 shape_info);

    auto tilingData = reinterpret_cast<ThreeInterpolateBackwardTilingData*>(tiling);
    tilingData->used_core_num = 5;
    tilingData->bs = bs;
    tilingData->c1 = c1;
    tilingData->ms = ms;
    tilingData->ns = ns;
    tilingData->each_core_proc_num = 1;
    tilingData->each_core_loop_times = 1;
    tilingData->each_core_each_loop_n_cnt = 16;
    tilingData->each_core_last_loop_n_cnt = 1;
    tilingData->last_core_proc_num = 1;
    tilingData->last_core_loop_times = 1;
    tilingData->last_core_each_loop_n_cnt = 16;
    tilingData->last_core_last_loop_n_cnt = 1;
    tilingData->weight_move_block_size = 6;
    tilingData->idx_move_block_size = 6;
    tilingData->grad_x_move_block_size = 32;
    tilingData->grad_y_move_block_size = 96;
    tilingData->c_move_num = 1;
    tilingData->c_last_loop_move_num = 1;
    tilingData->c_move_loop_times = 1;
    tilingData->mulit_core_mode = 0;
    tilingData->each_core_proc_batch_num = 0;
    tilingData->core_proc_batch_padding_idx = 0;
    uint32_t block_dim = tilingData->used_core_num;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(three_interpolate_backward, tilingData->used_core_num, input_grad_x, input_idx, input_weight,
                output_grad_y, workspace, (uint8_t*)(tilingData));

    WriteOutputData(output_grad_y, grad_y_output_bytes, path);

    AscendC::GmFree((void*)input_grad_x);
    AscendC::GmFree((void*)input_idx);
    AscendC::GmFree((void*)input_weight);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)output_grad_y);
    AscendC::GmFree((void*)workspace);
    free(path_);

    EXPECT_EQ(CompareData(shape_info), true);
}

TEST_F(three_interpolate_backward_test, test_case_20_21_60_51_fp32)
{
    uint32_t bs = 20;
    uint32_t cs = 21;
    uint32_t ms = 60;
    uint32_t ns = 51;
    uint32_t c1 = (cs + C0 - 1) / C0;

    char* path_ = get_current_dir_name();
    string path(path_);

    stringstream shape_raw_info;
    shape_raw_info << bs << " ";
    shape_raw_info << cs << " ";
    shape_raw_info << ms << " ";
    shape_raw_info << ns << " ";
    shape_raw_info << 0 << " ";

    string shape_info = shape_raw_info.str();

    size_t grad_x_input_bytes = (bs * c1 * C0 * ns) * sizeof(float); // NC1HWC0
    size_t idx_input_bytes = (bs * ns * 3) * sizeof(int32_t);        // 精确尺寸分配（issue #1406 回归锚点）
    size_t weight_input_bytes = (bs * ns * 3) * sizeof(float);       // 精确尺寸分配（issue #1406 回归锚点）
    size_t tiling_bytes = TILING_ARGS_NUM * sizeof(uint32_t);
    size_t grad_y_output_bytes = (bs * c1 * C0 * ms) * sizeof(float); // NC1HWC0

    uint8_t* input_grad_x = (uint8_t*)AscendC::GmAlloc(grad_x_input_bytes);
    uint8_t* input_idx = (uint8_t*)AscendC::GmAlloc(idx_input_bytes);
    uint8_t* input_weight = (uint8_t*)AscendC::GmAlloc(weight_input_bytes);
    uint8_t* output_grad_y = (uint8_t*)AscendC::GmAlloc(grad_y_output_bytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_bytes);

    GetInputData(input_grad_x, grad_x_input_bytes, input_idx, idx_input_bytes, input_weight, weight_input_bytes, path,
                 shape_info);

    auto tilingData = reinterpret_cast<ThreeInterpolateBackwardTilingData*>(tiling);
    tilingData->used_core_num = 48;
    tilingData->bs = bs;
    tilingData->c1 = c1;
    tilingData->ms = ms;
    tilingData->ns = ns;
    tilingData->each_core_proc_num = 1;
    tilingData->each_core_loop_times = 1;
    tilingData->each_core_each_loop_n_cnt = 16;
    tilingData->each_core_last_loop_n_cnt = 1;
    tilingData->last_core_proc_num = 4;
    tilingData->last_core_loop_times = 1;
    tilingData->last_core_each_loop_n_cnt = 16;
    tilingData->last_core_last_loop_n_cnt = 4;
    tilingData->weight_move_block_size = 6;
    tilingData->idx_move_block_size = 6;
    tilingData->grad_x_move_block_size = 64;
    tilingData->grad_y_move_block_size = 192;
    tilingData->c_move_num = 2;
    tilingData->c_last_loop_move_num = 2;
    tilingData->c_move_loop_times = 1;
    tilingData->mulit_core_mode = 0;
    tilingData->each_core_proc_batch_num = 0;
    tilingData->core_proc_batch_padding_idx = 0;
    uint32_t block_dim = tilingData->used_core_num;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(three_interpolate_backward, tilingData->used_core_num, input_grad_x, input_idx, input_weight,
                output_grad_y, workspace, (uint8_t*)(tilingData));

    WriteOutputData(output_grad_y, grad_y_output_bytes, path);

    AscendC::GmFree((void*)input_grad_x);
    AscendC::GmFree((void*)input_idx);
    AscendC::GmFree((void*)input_weight);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)output_grad_y);
    AscendC::GmFree((void*)workspace);
    free(path_);

    EXPECT_EQ(CompareData(shape_info), true);
}

TEST_F(three_interpolate_backward_test, test_case_21_31_60_51_fp16)
{
    uint32_t bs = 21;
    uint32_t cs = 31;
    uint32_t ms = 60;
    uint32_t ns = 51;
    uint32_t c1 = (cs + C0 - 1) / C0;

    char* path_ = get_current_dir_name();
    string path(path_);

    stringstream shape_raw_info;
    shape_raw_info << bs << " ";
    shape_raw_info << cs << " ";
    shape_raw_info << ms << " ";
    shape_raw_info << ns << " ";
    shape_raw_info << 1 << " ";

    string shape_info = shape_raw_info.str();

    size_t grad_x_input_bytes = (bs * c1 * C0 * ns) * sizeof(half); // NC1HWC0
    size_t idx_input_bytes = (bs * ns * 3) * sizeof(int32_t);       // 精确尺寸分配（issue #1406 回归锚点）
    size_t weight_input_bytes = (bs * ns * 3) * sizeof(half);       // 精确尺寸分配（issue #1406 回归锚点）
    size_t tiling_bytes = TILING_ARGS_NUM * sizeof(uint32_t);
    size_t grad_y_output_bytes = (bs * c1 * C0 * ms) * sizeof(half); // NC1HWC0

    uint8_t* input_grad_x = (uint8_t*)AscendC::GmAlloc(grad_x_input_bytes);
    uint8_t* input_idx = (uint8_t*)AscendC::GmAlloc(idx_input_bytes);
    uint8_t* input_weight = (uint8_t*)AscendC::GmAlloc(weight_input_bytes);
    uint8_t* output_grad_y = (uint8_t*)AscendC::GmAlloc(grad_y_output_bytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_bytes);

    GetInputData(input_grad_x, grad_x_input_bytes, input_idx, idx_input_bytes, input_weight, weight_input_bytes, path,
                 shape_info);

    auto tilingData = reinterpret_cast<ThreeInterpolateBackwardTilingData*>(tiling);
    tilingData->used_core_num = 21;
    tilingData->bs = bs;
    tilingData->c1 = c1;
    tilingData->ms = ms;
    tilingData->ns = ns;
    tilingData->each_core_proc_num = 51;
    tilingData->each_core_loop_times = 4;
    tilingData->each_core_each_loop_n_cnt = 16;
    tilingData->each_core_last_loop_n_cnt = 3;
    tilingData->last_core_proc_num = 51;
    tilingData->last_core_loop_times = 4;
    tilingData->last_core_each_loop_n_cnt = 16;
    tilingData->last_core_last_loop_n_cnt = 3;
    tilingData->weight_move_block_size = 3;
    tilingData->idx_move_block_size = 6;
    tilingData->grad_x_move_block_size = 32;
    tilingData->grad_y_move_block_size = 96;
    tilingData->c_move_num = 2;
    tilingData->c_last_loop_move_num = 2;
    tilingData->c_move_loop_times = 1;
    tilingData->mulit_core_mode = 1;
    tilingData->each_core_proc_batch_num = 1;
    tilingData->core_proc_batch_padding_idx = 0;
    uint32_t block_dim = tilingData->used_core_num;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(three_interpolate_backward, tilingData->used_core_num, input_grad_x, input_idx, input_weight,
                output_grad_y, workspace, (uint8_t*)(tilingData));

    WriteOutputData(output_grad_y, grad_y_output_bytes, path);

    AscendC::GmFree((void*)input_grad_x);
    AscendC::GmFree((void*)input_idx);
    AscendC::GmFree((void*)input_weight);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)output_grad_y);
    AscendC::GmFree((void*)workspace);
    free(path_);

    EXPECT_EQ(CompareData(shape_info), true);
}

// issue #1405 回归：idx 混入越界取值（等于/大于 ms、负数）时，kernel 拦截丢弃越界贡献
// （不参与 grad_y 寻址累加），合法位置的贡献不受影响，输出仍与 golden 一致。
// shape_info 尾部追加第 6 个参数 1 使 gen_data.py 生成含越界 idx 的数据与同步丢弃的 golden
TEST_F(three_interpolate_backward_test, test_case_1_2_6_5_invalid_idx_fp32)
{
    uint32_t bs = 1;
    uint32_t cs = 2;
    uint32_t ms = 6;
    uint32_t ns = 5;
    uint32_t c1 = (cs + C0 - 1) / C0;

    char* path_ = get_current_dir_name();
    string path(path_);

    stringstream shape_raw_info;
    shape_raw_info << bs << " ";
    shape_raw_info << cs << " ";
    shape_raw_info << ms << " ";
    shape_raw_info << ns << " ";
    shape_raw_info << 0 << " ";
    shape_raw_info << 1 << " ";

    string shape_info = shape_raw_info.str();

    size_t grad_x_input_bytes = (bs * c1 * C0 * ns) * sizeof(float); // NC1HWC0
    size_t idx_input_bytes = (bs * ns * 3) * sizeof(int32_t);
    size_t weight_input_bytes = (bs * ns * 3) * sizeof(float);
    size_t tiling_bytes = TILING_ARGS_NUM * sizeof(uint32_t);
    size_t grad_y_output_bytes = (bs * c1 * C0 * ms) * sizeof(float); // NC1HWC0

    uint8_t* input_grad_x = (uint8_t*)AscendC::GmAlloc(grad_x_input_bytes);
    uint8_t* input_idx = (uint8_t*)AscendC::GmAlloc(idx_input_bytes);
    uint8_t* input_weight = (uint8_t*)AscendC::GmAlloc(weight_input_bytes);
    uint8_t* output_grad_y = (uint8_t*)AscendC::GmAlloc(grad_y_output_bytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_bytes);

    GetInputData(input_grad_x, grad_x_input_bytes, input_idx, idx_input_bytes, input_weight, weight_input_bytes, path,
                 shape_info);

    auto tilingData = reinterpret_cast<ThreeInterpolateBackwardTilingData*>(tiling);
    tilingData->used_core_num = 5;
    tilingData->bs = bs;
    tilingData->c1 = c1;
    tilingData->ms = ms;
    tilingData->ns = ns;
    tilingData->each_core_proc_num = 1;
    tilingData->each_core_loop_times = 1;
    tilingData->each_core_each_loop_n_cnt = 16;
    tilingData->each_core_last_loop_n_cnt = 1;
    tilingData->last_core_proc_num = 1;
    tilingData->last_core_loop_times = 1;
    tilingData->last_core_each_loop_n_cnt = 16;
    tilingData->last_core_last_loop_n_cnt = 1;
    tilingData->weight_move_block_size = 6;
    tilingData->idx_move_block_size = 6;
    tilingData->grad_x_move_block_size = 32;
    tilingData->grad_y_move_block_size = 96;
    tilingData->c_move_num = 1;
    tilingData->c_last_loop_move_num = 1;
    tilingData->c_move_loop_times = 1;
    tilingData->mulit_core_mode = 0;
    tilingData->each_core_proc_batch_num = 0;
    tilingData->core_proc_batch_padding_idx = 0;
    uint32_t block_dim = tilingData->used_core_num;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(three_interpolate_backward, tilingData->used_core_num, input_grad_x, input_idx, input_weight,
                output_grad_y, workspace, (uint8_t*)(tilingData));

    WriteOutputData(output_grad_y, grad_y_output_bytes, path);

    AscendC::GmFree((void*)input_grad_x);
    AscendC::GmFree((void*)input_idx);
    AscendC::GmFree((void*)input_weight);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)output_grad_y);
    AscendC::GmFree((void*)workspace);
    free(path_);

    EXPECT_EQ(CompareData(shape_info), true);
}
