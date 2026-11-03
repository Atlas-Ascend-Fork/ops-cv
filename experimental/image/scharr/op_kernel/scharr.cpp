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
 * \file scharr.cpp
 * \brief Scharr kernel entry. schMode selects direction/border:
 *        1=X+REFLECT_101 2=X+REPLICATE 3=X+CONSTANT
 *        4=Y+REFLECT_101 5=Y+REPLICATE 6=Y+CONSTANT
 */

#include "scharr.h"
#include "scharr_tiling_data.h"

#define RUN_SCHARR(DX, BM)                                                                                             \
    do {                                                                                                               \
        TPipe pipe;                                                                                                    \
        KernelScharrVector<DTYPE_INPUT, DX, BM> op;                                                                    \
        op.Init(&pipe, input, output, tilingData.totalLength, tilingData.batchCount, tilingData.H, tilingData.W,       \
                tilingData.usedCoreNum, tilingData.splitMode, tilingData.baseRowsPerCore,                              \
                tilingData.extraRowCoreCount, tilingData.basePixelsPerCore, tilingData.extraPixelCoreCount,            \
                tilingData.basePlanesPerCore, tilingData.extraPlaneCoreCount, tilingData.blockW, tilingData.tileH,     \
                tilingData.compRows, tilingData.compPitch, tilingData.bandW, tilingData.rawPitch,                      \
                tilingData.vectorPitch, tilingData.inputSlabPitch, tilingData.floatSlabPitch,                          \
                tilingData.outputSlabPitch, tilingData.inputBufferBytes, tilingData.outputBufferBytes,                 \
                tilingData.floatSlabBytes, tilingData.scratchBytes, tilingData.slotsOffset, tilingData.work0Offset,    \
                tilingData.work1Offset, tilingData.comp0Offset, tilingData.comp1Offset, tilingData.comp2Offset,        \
                tilingData.fallbackOffset, tilingData.maskOffset, tilingData.castHalfOffset, tilingData.offset0Offset, \
                tilingData.offset1Offset, tilingData.coefficientOffset, tilingData.tailBatchOffset,                    \
                tilingData.bandTableOffset);                                                                           \
        op.Process();                                                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void scharr(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    // Scharr uses only SIMD/Vector APIs. On the separated A2/910B architecture
    // this prevents unnecessary Cube tasks from being launched and makes
    // SetBlockDim correspond directly to the AIV task count.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ScharrTilingData);
    GET_TILING_DATA(tilingData, tiling);

    const int32_t dir = tilingData.direction;
    const int32_t bd = tilingData.border;
    if (dir == 0 && bd == 0) {
        RUN_SCHARR(true, 0);
    } else if (dir == 0 && bd == 1) {
        RUN_SCHARR(true, 1);
    } else if (dir == 0 && bd == 2) {
        RUN_SCHARR(true, 2);
    } else if (dir == 1 && bd == 0) {
        RUN_SCHARR(false, 0);
    } else if (dir == 1 && bd == 1) {
        RUN_SCHARR(false, 1);
    } else if (dir == 1 && bd == 2) {
        RUN_SCHARR(false, 2);
    }
}

#undef RUN_SCHARR
