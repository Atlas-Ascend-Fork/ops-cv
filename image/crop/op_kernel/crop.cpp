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
 * \file crop.cpp
 * \brief Kernel entry for crop operator
 *
 * Template parameter:
 *   idxWidth (uint32_t): 0 = 32-bit index, 1 = 64-bit index
 *   dtype is handled by DTYPE_X macro (no tiling key enumeration).
 */

#include "arch35/crop_simt.h"
#include "arch35/crop_norm.h"
#include "arch35/crop_gather.h"

template <uint32_t idxWidth>
__global__ __aicore__ void crop(GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CropTilingData);
    GET_TILING_DATA_WITH_STRUCT(CropTilingData, tilingData, tiling);
    // Dispatch: contiguous copies use NDDMA paths, others SIMT (32/64-bit index from tiling).
    if (tilingData.useSsNddma != 0U) {
        AscendC::TPipe pipe;
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            NsCrop::CropSsNddma<int32_t> op;
            op.Init(x, y, &tilingData.ssNddma, &pipe);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            NsCrop::CropSsNddma<int64_t> op;
            op.Init(x, y, &tilingData.ssNddma, &pipe);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            NsCrop::CropSsNddma<int8_t> op;
            op.Init(x, y, &tilingData.ssNddma, &pipe);
            op.Process();
        } else {
            NsCrop::CropSsNddma<DTYPE_X> op;
            op.Init(x, y, &tilingData.ssNddma, &pipe);
            op.Process();
        }
    } else if (tilingData.useGather != 0U) {
        NsCrop::CropGatherProcess<DTYPE_X>(&tilingData, x, y);
    } else if (tilingData.useSimd != 0U) {
        NsCrop::CropSimdProcess<DTYPE_X>(&tilingData, x, y);
    } else {
        if constexpr (idxWidth == CROP_IDX_32) {
            NsCrop::CropProcess<DTYPE_X, uint32_t>(&tilingData, x, y);
        } else {
            NsCrop::CropProcess<DTYPE_X, uint64_t>(&tilingData, x, y);
        }
    }
}
