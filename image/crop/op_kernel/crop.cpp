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

template <uint32_t idxWidth>
__global__ __aicore__ void crop(GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CropTilingData);
    GET_TILING_DATA_WITH_STRUCT(CropTilingData, tilingData, tiling);
    // 32位/64位路径分发：由 Tiling 侧根据元素总量选择，32位索引节省寄存器和带宽。
    // 两条路径统一使用 1024 线程（见 crop_simt.h THREAD_NUM 说明）。
    if constexpr (idxWidth == CROP_IDX_32) {
        NsCrop::CropProcess<DTYPE_X, uint32_t>(&tilingData, x, y);
    } else {
        NsCrop::CropProcess<DTYPE_X, uint64_t>(&tilingData, x, y);
    }
}
