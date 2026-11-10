/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2.cpp
 * \brief resize_nearest_neighbor_v2
 */
#include "./arch35/resize_nearest_neighbor_v2_base.h"
#ifdef __CCE_AICORE_SUPPORT_SIMT__
#include "./arch35/resize_nearest_neighbor_v2_simt.h"
#endif
#include "./arch35/resize_nearest_neighbor_v2_data_copy_big_c.h"
#include "./arch35/resize_nearest_neighbor_v2_data_copy_jh.h"
#include "./arch35/resize_nearest_neighbor_v2_data_copy_small_c.h"
#include "./arch35/resize_nearest_neighbor_v2_data_copy_nhwc.h"
#include "./arch35/resize_nearest_neighbor_v2_nchw_gather.h"
#include "./arch35/resize_nearest_neighbor_v2_tiling_key.h"

using namespace AscendC;

template <uint64_t schId, uint64_t format, uint64_t alignCorners, uint64_t halfPixelCenters, uint64_t idxInt32>
__global__ __aicore__ void resize_nearest_neighbor_v2(GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // idxInt32 决定 SIMD(schId 0/1/2) 输出位置 scalar 计算类型：为 1 用 int32_t，为 0 用 int64_t。
    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_SMALL_C) {
        if constexpr (idxInt32) {
            ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_S_C<DTYPE_X, int32_t> op;
            op.Init(x, size, y, userWS, &tilingData);
            op.Process();
        } else {
            ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_S_C<DTYPE_X, int64_t> op;
            op.Init(x, size, y, userWS, &tilingData);
            op.Process();
        }
        return;
    }

    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_BIG_C) {
        if constexpr (idxInt32) {
            ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_BIG_C<DTYPE_X, int32_t> op;
            op.Init(x, size, y, userWS, &tilingData);
            op.Process();
        } else {
            ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_BIG_C<DTYPE_X, int64_t> op;
            op.Init(x, size, y, userWS, &tilingData);
            op.Process();
        }
        return;
    }

    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_AGGR_C) {
        if constexpr (idxInt32) {
            ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_JH<DTYPE_X, int32_t> op;
            op.Init(x, size, y, userWS, &tilingData);
            op.Process();
        } else {
            ResizeNearestNeighborV2::TILING_KEY_DATA_COPY_NHWC_JH<DTYPE_X, int64_t> op;
            op.Init(x, size, y, userWS, &tilingData);
            op.Process();
        }
        return;
    }
    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_NOT_ALL_W_OUT) {
        // n或者h分核，输入对应的1,w,c,输出是xo,wo,c一次放不下，需循环搬出
        ResizeNearestNeighborV2::ResizeNearestNeighborV2NHWC<DTYPE_X, 0> op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    }
    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_ALL_W_OUT) {
        // n或者h分核，输入对应的1,w,c,输出是xo,wo,c可以一次放下，一次搬出
        ResizeNearestNeighborV2::ResizeNearestNeighborV2NHWC<DTYPE_X, 1> op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    }
    if constexpr (schId == TPL_SCH_MODE_DATA_COPY_CUT_NH) {
        // nh合轴分核, 2表示nh合轴切分
        ResizeNearestNeighborV2::ResizeNearestNeighborV2NHWC<DTYPE_X, 2> op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    }
    if constexpr (schId == TPL_SCH_MODE_GATHER_ALL_HW || schId == TPL_SCH_MODE_GATHER_CUT_H) {
        if constexpr (sizeof(DTYPE_X) == sizeof(float)) {
            ResizeNearestNeighborV2::ResizeGather<DTYPE_X, uint32_t, schId, alignCorners> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeNearestNeighborV2::ResizeGather<DTYPE_X, uint16_t, schId, alignCorners> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }
        return;
    }
#ifdef __CCE_AICORE_SUPPORT_SIMT__
    if constexpr (idxInt32) {
        ResizeNearestNeighborV2::ResizeNearestNeighborV2Simt<DTYPE_X, uint32_t, format, schId, alignCorners,
                                                             halfPixelCenters>
            op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    } else {
        ResizeNearestNeighborV2::ResizeNearestNeighborV2Simt<DTYPE_X, uint64_t, format, schId, alignCorners,
                                                             halfPixelCenters>
            op;
        op.Init(x, size, y, &tilingData);
        op.Process();
        return;
    }
#endif
}
