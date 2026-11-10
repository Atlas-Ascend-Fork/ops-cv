/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_pooling_grad_with_arg_max.cpp
 * \brief
 */
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../../../op_kernel/arch35/roi_pooling_grad_with_arg_max_tiling_data.h"
#include "data_utils.h"
#include "gtest/gtest.h"
#include "kernel_tiling/kernel_tiling.h"
#include "objdetect/roi_pooling_grad_with_arg_max/op_kernel/roi_pooling_grad_with_arg_max.cpp"
#include "tikicpulib.h"

using namespace std;

template <uint64_t dType>
__global__ __aicore__ void roi_pooling_grad_with_arg_max(GM_ADDR grad, GM_ADDR x, GM_ADDR rois, GM_ADDR roi_actual_num,
                                                         GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class roi_pooling_grad_with_arg_max_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "roi_pooling_grad_with_arg_max SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "roi_pooling_grad_with_arg_max_test TearDown\n" << endl; }
};

namespace {
constexpr int32_t kBatch = 2;
constexpr int32_t kChannel = 1;
constexpr int32_t kHeight = 2;
constexpr int32_t kWidth = 2;
constexpr int32_t kPixelNum = kHeight * kWidth; // 4
constexpr int32_t kPooledH = 1;
constexpr int32_t kPooledW = 1;
constexpr int32_t kPadElems = 4;
constexpr float kSentinel = 7.0f;
constexpr float kGradValue = 1.5f;

struct CaseParam {
    float batchId;
    int32_t argmax;
    bool expectWrite;
    int32_t expectOffset;
};

void FillFloat(uint8_t* ptr, size_t bytes, float value)
{
    auto* data = reinterpret_cast<float*>(ptr);
    size_t n = bytes / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        data[i] = value;
    }
}

void RunOneCase(const CaseParam& param)
{
    const int32_t roisN = 1;
    const int32_t totalLength = roisN * kChannel * kPooledH * kPooledW;
    const int32_t yElems = kBatch * kChannel * kHeight * kWidth;
    const size_t gradByteSize = static_cast<size_t>(totalLength) * sizeof(float);
    const size_t xByteSize = static_cast<size_t>(yElems) * sizeof(float);
    const size_t roisByteSize = static_cast<size_t>(roisN) * 5 * sizeof(float);
    const size_t roiActualNumByteSize = static_cast<size_t>(roisN) * sizeof(int32_t);
    const size_t argmaxByteSize = static_cast<size_t>(totalLength) * sizeof(int32_t);
    const size_t yValidByteSize = static_cast<size_t>(yElems) * sizeof(float);
    const size_t yPadByteSize = static_cast<size_t>(kPadElems) * sizeof(float);
    const size_t yAllocByteSize = yPadByteSize + yValidByteSize + yPadByteSize;
    const size_t tilingDataSize = sizeof(RoiPoolingGradWithArgMaxRegBaseTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rois = (uint8_t*)AscendC::GmAlloc(roisByteSize);
    uint8_t* roiActualNum = (uint8_t*)AscendC::GmAlloc(roiActualNumByteSize);
    uint8_t* argmax = (uint8_t*)AscendC::GmAlloc(argmaxByteSize);
    uint8_t* yAlloc = (uint8_t*)AscendC::GmAlloc(yAllocByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ASSERT_NE(grad, nullptr);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(rois, nullptr);
    ASSERT_NE(roiActualNum, nullptr);
    ASSERT_NE(argmax, nullptr);
    ASSERT_NE(yAlloc, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    FillFloat(grad, gradByteSize, kGradValue);
    FillFloat(x, xByteSize, 0.0f);
    FillFloat(yAlloc, yAllocByteSize, kSentinel);
    FillFloat(workspace, 1024 * 16 * 1024, 0.0f);

    auto* roisF = reinterpret_cast<float*>(rois);
    for (int i = 0; i < 5; ++i) {
        roisF[i] = 0.0f;
    }
    roisF[0] = param.batchId;
    reinterpret_cast<int32_t*>(argmax)[0] = param.argmax;
    reinterpret_cast<int32_t*>(roiActualNum)[0] = 1;

    uint8_t* y = yAlloc + yPadByteSize;

    auto* tilingData = reinterpret_cast<RoiPoolingGradWithArgMaxRegBaseTilingData*>(tiling);
    memset(tilingData, 0, tilingDataSize);
    tilingData->yTotalCoreNum = 1;
    tilingData->yDataPerCore = yElems;
    tilingData->yDataTailCore = yElems;
    tilingData->totalLength = totalLength;
    tilingData->yTotalLength = yElems;
    tilingData->pooledH = kPooledH;
    tilingData->pooledW = kPooledW;
    tilingData->height = kHeight;
    tilingData->width = kWidth;
    tilingData->poolChannel = kChannel;
    tilingData->useCoreNum = 1;
    tilingData->batch = kBatch;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_pooling_grad_with_arg_max<0>, 1, grad, x, rois, roiActualNum, argmax, y, workspace, tiling);

    auto* yValid = reinterpret_cast<float*>(y);
    auto* yFront = reinterpret_cast<float*>(yAlloc);
    auto* yBack = reinterpret_cast<float*>(yAlloc + yPadByteSize + yValidByteSize);

    for (int i = 0; i < kPadElems; ++i) {
        EXPECT_FLOAT_EQ(yFront[i], kSentinel) << "front sentinel polluted, i=" << i;
        EXPECT_FLOAT_EQ(yBack[i], kSentinel) << "back sentinel polluted, i=" << i;
    }

    if (param.expectWrite) {
        for (int i = 0; i < yElems; ++i) {
            if (i == param.expectOffset) {
                EXPECT_FLOAT_EQ(yValid[i], kGradValue) << "valid write missing at " << i;
            } else {
                EXPECT_FLOAT_EQ(yValid[i], 0.0f) << "unexpected write at " << i;
            }
        }
    } else {
        for (int i = 0; i < yElems; ++i) {
            EXPECT_FLOAT_EQ(yValid[i], 0.0f) << "illegal index should keep y zero, i=" << i
                                             << " argmax=" << param.argmax << " batchId=" << param.batchId;
        }
    }

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(rois);
    AscendC::GmFree(roiActualNum);
    AscendC::GmFree(argmax);
    AscendC::GmFree(yAlloc);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(roi_pooling_grad_with_arg_max_test, bounds_guard_batch_id_and_argmax)
{
    const CaseParam cases[] = {
        {0.0f, 0, true, 0},
        {0.0f, kPixelNum - 1, true, kPixelNum - 1},
        {1.0f, 0, true, kChannel * kHeight * kWidth},
        {static_cast<float>(kBatch - 1), kPixelNum - 1, true,
         (kBatch - 1) * kChannel * kHeight * kWidth + (kPixelNum - 1)},
        {-1.0f, 0, false, -1},
        {static_cast<float>(kBatch), 0, false, -1},
        {static_cast<float>(kBatch + 1), 0, false, -1},
        {0.0f, -2, false, -1},
        {0.0f, -1, false, -1},
        {0.0f, kPixelNum, false, -1},
        {0.0f, std::numeric_limits<int32_t>::max(), false, -1},
    };

    for (const auto& c : cases) {
        RunOneCase(c);
    }
}
