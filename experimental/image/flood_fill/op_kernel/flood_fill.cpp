/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill.cpp
 * \brief FloodFill kernel entry.
 */
#include "flood_fill.h"

using namespace AscendC;

template <typename T, bool HAS_MASK_INPUT>
__aicore__ inline void RunFloodFill(GM_ADDR x, GM_ADDR maskIn, GM_ADDR y, GM_ADDR maskOut, GM_ADDR workspace,
                                    const FloodFillTilingData* tiling)
{
    AscendC::TPipe pipe;
    NsFloodFill::KernelFloodFill<T, HAS_MASK_INPUT> op;
    op.Init(x, maskIn, y, maskOut, workspace, tiling, &pipe);
    op.Process();
}

template <uint64_t dType, uint64_t maskInput>
__global__ __aicore__ void flood_fill(GM_ADDR x, GM_ADDR maskIn, GM_ADDR y, GM_ADDR maskOut, GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(FloodFillTilingData);
    GET_TILING_DATA_WITH_STRUCT(FloodFillTilingData, tilingData, tiling);

    if constexpr (dType == FLOOD_FILL_TPL_UINT8) {
        RunFloodFill<uint8_t, maskInput != 0U>(x, maskIn, y, maskOut, workspace, &tilingData);
    } else if constexpr (dType == FLOOD_FILL_TPL_INT32) {
        RunFloodFill<int32_t, maskInput != 0U>(x, maskIn, y, maskOut, workspace, &tilingData);
    } else if constexpr (dType == FLOOD_FILL_TPL_FLOAT32) {
        RunFloodFill<float, maskInput != 0U>(x, maskIn, y, maskOut, workspace, &tilingData);
    }
}
