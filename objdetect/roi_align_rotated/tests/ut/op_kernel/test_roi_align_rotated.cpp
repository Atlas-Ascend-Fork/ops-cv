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
 * \file test_roi_align_rotated.cpp
 * \brief arch22（vector AscendC kernel）CPU 仿真用例
 */

// ascend950（arch35/3510）构建下本用例不适用：老 kernel 为 vector AscendC 实现，
// 且 950 tiling（arch35 独立实现，TILING_DIR 选编）产出 RoiAlignRotatedArch35TilingData，
// 与老 kernel 解析的 RoiAlignRotatedTilingData 布局不符——3510 环境 TU 编译为空
// （与 test_roi_align_rotated_950.cpp 的正向守卫对称，防 ut.cmake 用例 glob 混编）。
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include <cstdint>
#include "../../../op_host/arch22/roi_align_rotated_tiling_arch22.h"

using namespace std;

extern "C" __global__ __aicore__ void roi_align_rotated(GM_ADDR input, GM_ADDR rois, GM_ADDR output, GM_ADDR workspace,
                                                        GM_ADDR tiling);

class roi_align_rotated_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "roi_align_rotated_test SetUp\n" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./roi_align_rotated_data/");
    }
    static void TearDownTestCase() { cout << "roi_align_rotated_test TearDown\n" << endl; }

private:
    const static std::string dataPath;
};
const std::string roi_align_rotated_test::
    dataPath = "../../../../objdetect/roi_align_rotated/tests/ut/op_kernel/roi_align_rotated_data";

TEST_F(roi_align_rotated_test, test_case_0)
{
    size_t inputByteSize = 1 * 8 * 8 * 8 * sizeof(float);
    size_t roisByteSize = 6 * 8 * sizeof(float);
    size_t outputByteSize = 8 * 8 * 2 * 2 * sizeof(float);
    size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* rois = (uint8_t*)AscendC::GmAlloc(roisByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    RoiAlignRotatedTilingData* tilingData = reinterpret_cast<RoiAlignRotatedTilingData*>(tiling);
    tilingData->aligned = 0;
    tilingData->clockwise = 0;
    tilingData->numBlocks = 1;
    tilingData->rois_num_per_Lcore = 8;
    tilingData->rois_num_per_Score = 0;
    tilingData->Lcore_num = 1;
    tilingData->Score_num = 63;
    tilingData->input_buffer_size = 32;
    tilingData->tileNum = 8;
    tilingData->batch_size = 1;
    tilingData->channels = 8;
    tilingData->channels_aligned = 8;
    tilingData->input_h = 8;
    tilingData->input_w = 8;
    tilingData->rois_num_aligned = 8;
    tilingData->tail_num = 0;
    tilingData->spatial_scale = 1;
    tilingData->sampling_ratio = 1;
    tilingData->pooled_height = 2;
    tilingData->pooled_width = 2;
    tilingData->ub_total_size = 262144;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_align_rotated, numBlocks, input, rois, output, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(rois);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

static void FillRoisNCHW6R(float* rois, uint32_t rois_num)
{
    for (uint32_t i = 0; i < rois_num; ++i) {
        rois[0 * rois_num + i] = 0.0f;
        rois[1 * rois_num + i] = 4.0f;
        rois[2 * rois_num + i] = 4.0f;
        rois[3 * rois_num + i] = 4.0f;
        rois[4 * rois_num + i] = 4.0f;
        rois[5 * rois_num + i] = 0.0f;
    }
}

static void FillOnes(float* ptr, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = 1.0f;
    }
}

static void RunRoiAlignRotatedKernel(uint32_t rois_num, uint32_t rois_num_aligned, uint32_t tail_num,
                                     uint32_t rois_num_per_lcore, uint32_t lcore_num, uint32_t num_blocks,
                                     uint64_t ub_total_size, int32_t pooled_h, int32_t pooled_w)
{
    const uint32_t batch = 1;
    const uint32_t channels = 8;
    const uint32_t input_h = 8;
    const uint32_t input_w = 8;
    size_t inputByteSize = batch * input_h * input_w * channels * sizeof(float);
    size_t roisByteSize = 6UL * rois_num * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(rois_num) * pooled_h * pooled_w * channels * sizeof(float);
    size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* rois = (uint8_t*)AscendC::GmAlloc(roisByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(rois, nullptr);
    ASSERT_NE(output, nullptr);

    std::memset(output, 0, outputByteSize);
    FillOnes(reinterpret_cast<float*>(input), inputByteSize / sizeof(float));
    FillRoisNCHW6R(reinterpret_cast<float*>(rois), rois_num);

    RoiAlignRotatedTilingData* tilingData = reinterpret_cast<RoiAlignRotatedTilingData*>(tiling);
    std::memset(tiling, 0, tiling_data_size);
    tilingData->aligned = 0;
    tilingData->clockwise = 0;
    tilingData->numBlocks = num_blocks;
    tilingData->rois_num_per_Lcore = rois_num_per_lcore;
    tilingData->rois_num_per_Score = 0;
    tilingData->Lcore_num = lcore_num;
    tilingData->Score_num = 0;
    tilingData->input_buffer_size = 32;
    tilingData->tileNum = 8;
    tilingData->batch_size = batch;
    tilingData->channels = channels;
    tilingData->channels_aligned = channels;
    tilingData->input_h = input_h;
    tilingData->input_w = input_w;
    tilingData->rois_num_aligned = rois_num_aligned;
    tilingData->tail_num = tail_num;
    tilingData->spatial_scale = 1;
    tilingData->sampling_ratio = 1;
    tilingData->pooled_height = pooled_h;
    tilingData->pooled_width = pooled_w;
    tilingData->ub_total_size = ub_total_size;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_align_rotated, num_blocks, input, rois, output, workspace, tiling);

    const float* out = reinterpret_cast<const float*>(output);
    const size_t per_roi = static_cast<size_t>(pooled_h) * pooled_w * channels;
    ASSERT_GT(per_roi, 0U);
    EXPECT_GT(std::fabs(out[0]), 1e-6f);
    EXPECT_GT(std::fabs(out[(rois_num - 1) * per_roi]), 1e-6f);

    AscendC::GmFree(input);
    AscendC::GmFree(rois);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(roi_align_rotated_test, test_case_rois_num_1_tight_gm)
{
    RunRoiAlignRotatedKernel(1, 8, 7, 8, 1, 1, 196608, 1, 1);
}

TEST_F(roi_align_rotated_test, test_case_rois_num_7_tight_gm)
{
    RunRoiAlignRotatedKernel(7, 8, 1, 8, 1, 1, 196608, 1, 1);
}

TEST_F(roi_align_rotated_test, test_case_rois_num_9_tight_gm)
{
    RunRoiAlignRotatedKernel(9, 16, 7, 8, 2, 2, 196608, 1, 1);
}

TEST_F(roi_align_rotated_test, test_case_multi_loop_tail) { RunRoiAlignRotatedKernel(24, 24, 0, 24, 1, 1, 3072, 1, 1); }

// 越界 batch_idx 回归（L0 随机数据 aic error 防护）：
// rois (6,R) 行 0/1/3 注入 batch_idx=1.0/1.5/2.0（N=1 截断取整后均越界 [0,1)）、行 2 合法（0.0）。
// 未防护时 kernel 以越界 batch 寻址 inputGM 触发 AICORE 越界读（aic error）；
// 修复后采样网格清 0 按空 ROI 输出 0。input 双倍分配、batch 1 平面预填哨兵 99.0
// （StackGroupPoints 前置 canary 同款手法）：未修复读到哨兵输出非 0，修复后不读输出 0。
TEST_F(roi_align_rotated_test, test_case_invalid_batch_idx)
{
    const uint32_t batch = 1;
    const uint32_t channels = 8;
    const uint32_t input_h = 8;
    const uint32_t input_w = 8;
    const uint32_t rois_num = 4;
    const int32_t pooled_h = 2;
    const int32_t pooled_w = 2;
    const size_t planeFloats = static_cast<size_t>(input_h) * input_w * channels;
    const size_t inputAllocBytes = planeFloats * 2 * sizeof(float); // 双倍：batch 1 平面为哨兵区
    const size_t roisByteSize = 6UL * rois_num * sizeof(float);
    const size_t outputByteSize = static_cast<size_t>(rois_num) * pooled_h * pooled_w * channels * sizeof(float);
    const size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputAllocBytes);
    uint8_t* rois = (uint8_t*)AscendC::GmAlloc(roisByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(rois, nullptr);
    ASSERT_NE(output, nullptr);

    // batch 0 平面填 1.0（有效数据），batch 1 平面填哨兵 99.0（越界 batch_idx=1 的读取落点）
    float* inputFloat = reinterpret_cast<float*>(input);
    for (size_t i = 0; i < planeFloats; ++i) {
        inputFloat[i] = 1.0f;
    }
    for (size_t i = planeFloats; i < planeFloats * 2; ++i) {
        inputFloat[i] = 99.0f;
    }
    // rois (6,R) 转置布局：仅 batch_idx 平面注入越界值，其余字段同 FillRoisNCHW6R
    float* roisFloat = reinterpret_cast<float*>(rois);
    for (uint32_t i = 0; i < rois_num; ++i) {
        roisFloat[0 * rois_num + i] = (i == 2) ? 0.0f : ((i == 1) ? 1.5f : ((i == 3) ? 2.0f : 1.0f));
        roisFloat[1 * rois_num + i] = 4.0f;
        roisFloat[2 * rois_num + i] = 4.0f;
        roisFloat[3 * rois_num + i] = 4.0f;
        roisFloat[4 * rois_num + i] = 4.0f;
        roisFloat[5 * rois_num + i] = 0.0f;
    }
    std::memset(output, 0, outputByteSize);

    RoiAlignRotatedTilingData* tilingData = reinterpret_cast<RoiAlignRotatedTilingData*>(tiling);
    std::memset(tiling, 0, tiling_data_size);
    tilingData->aligned = 0;
    tilingData->clockwise = 0;
    tilingData->numBlocks = 1;
    tilingData->rois_num_per_Lcore = rois_num;
    tilingData->rois_num_per_Score = 0;
    tilingData->Lcore_num = 1;
    tilingData->Score_num = 0;
    tilingData->input_buffer_size = 32;
    tilingData->tileNum = 8;
    tilingData->batch_size = batch;
    tilingData->channels = channels;
    tilingData->channels_aligned = channels;
    tilingData->input_h = input_h;
    tilingData->input_w = input_w;
    tilingData->rois_num_aligned = 8;
    tilingData->tail_num = 8 - rois_num;
    tilingData->spatial_scale = 1;
    tilingData->sampling_ratio = 1;
    tilingData->pooled_height = pooled_h;
    tilingData->pooled_width = pooled_w;
    tilingData->ub_total_size = 196608;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_align_rotated, 1, input, rois, output, workspace, tiling);

    const float* out = reinterpret_cast<const float*>(output);
    const size_t per_roi = static_cast<size_t>(pooled_h) * pooled_w * channels;
    // 行 0/1/3：越界 batch_idx → 空 ROI，输出全 0
    for (uint32_t r : {0U, 1U, 3U}) {
        for (size_t i = 0; i < per_roi; ++i) {
            EXPECT_EQ(out[static_cast<size_t>(r) * per_roi + i], 0.0f);
        }
    }
    // 行 2：合法 batch_idx=0 → 采样 batch 0 平面（全 1），输出非 0
    for (size_t i = 0; i < per_roi; ++i) {
        EXPECT_GT(std::fabs(out[2 * per_roi + i]), 1e-6f);
    }

    AscendC::GmFree(input);
    AscendC::GmFree(rois);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
#endif // !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
