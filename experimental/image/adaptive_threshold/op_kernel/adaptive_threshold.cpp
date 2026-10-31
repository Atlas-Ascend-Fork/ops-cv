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
 * \file adaptive_threshold.cpp
 * \brief AdaptiveThreshold kernel entry.
 */
#include "adaptive_threshold.h"

using namespace AscendC;

template <uint32_t adaptiveMethod, uint64_t dType>
__global__ __aicore__ void adaptive_threshold(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AdaptiveThresholdTilingData);
    GET_TILING_DATA_WITH_STRUCT(AdaptiveThresholdTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    if constexpr (dType == ADAPTIVE_THRESHOLD_TPL_UINT8) {
        NsAdaptiveThreshold::KernelAdaptiveThreshold<adaptiveMethod, uint8_t> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (dType == ADAPTIVE_THRESHOLD_TPL_FLOAT16) {
        NsAdaptiveThreshold::KernelAdaptiveThreshold<adaptiveMethod, half> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (dType == ADAPTIVE_THRESHOLD_TPL_FLOAT32) {
        NsAdaptiveThreshold::KernelAdaptiveThreshold<adaptiveMethod, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
