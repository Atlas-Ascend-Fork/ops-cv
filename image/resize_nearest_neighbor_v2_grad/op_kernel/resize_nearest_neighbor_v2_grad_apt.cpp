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
 * \file resize_nearest_neighbor_v2_grad.cpp
 * \brief resize_nearest_neighbor_v2_grad
 */
#include "kernel_operator.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_simt_determine.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_simt_determine_hw.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_simt_determine_1d.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_simt.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_all_copy.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_tiling_key.h"
#include "./arch35/resize_nearest_neighbor_v2_grad_simt_hw.h"

using namespace ResizeNearestNeighborV2Grad;

template <uint64_t schId, uint64_t format, uint64_t alignCorners, uint64_t halfPixelCenters, uint64_t idxType>
__global__ __aicore__ void resize_nearest_neighbor_v2_grad(GM_ADDR grads, GM_ADDR size, GM_ADDR y, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    if (schId == TPL_SCH_ID_ALL_COPY) {
        ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradAllCopy<DTYPE_Y, DTYPE_Y> op;
        op.Init(grads, size, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (schId == TPL_SCH_ID_NOT_DETERMINE) {
        if (idxType == TPL_IDX_INT32) {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimt<DTYPE_Y, uint32_t, format, alignCorners,
                                                                         halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimt<DTYPE_Y, uint64_t, format, alignCorners,
                                                                         halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        }
    }

    if (schId == TPL_SCH_ID_NOT_DETERMINE_HW) {
        if (idxType == TPL_IDX_INT32) {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtHW<DTYPE_Y, uint32_t, format, alignCorners,
                                                                           halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtHW<DTYPE_Y, uint64_t, format, alignCorners,
                                                                           halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        }
    }

    if (schId == TPL_SCH_ID_DETERMINE) {
        if (idxType == TPL_IDX_INT32) {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtDetermine<DTYPE_Y, uint32_t, format,
                                                                                  halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtDetermine<DTYPE_Y, uint64_t, format,
                                                                                  halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        }
    }

    if (schId == TPL_SCH_ID_DETERMINE_HW) {
        if (idxType == TPL_IDX_INT32) {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtDetermineHW<DTYPE_Y, uint32_t, halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtDetermineHW<DTYPE_Y, uint64_t, halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        }
    }

    if (schId == TPL_SCH_ID_DETERMINE_1D) {
        if (idxType == TPL_IDX_INT32) {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtDetermine1D<DTYPE_Y, uint32_t, format,
                                                                                    halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        } else {
            ResizeNearestNeighborV2Grad::ResizeNearestNeighborV2GradSimtDetermine1D<DTYPE_Y, uint64_t, format,
                                                                                    halfPixelCenters>
                op;
            op.Init(grads, y, &tilingData);
            op.Process();
            return;
        }
    }
}
