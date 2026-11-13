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
 * \file yolo_apt.cpp
 * \brief Kernel entry for yolo operator
 *
 * Template parameter:
 *   schMode (uint32_t): yolo computation mode
 *     0 = YOLO_MODE_1: obj=sigmoid, classes=sigmoid
 *     1 = YOLO_MODE_2: obj=sigmoid, classes=softmax
 *     2 = YOLO_MODE_3: obj=move,    classes=sigmoid
 *     3 = YOLO_MODE_4: obj+classes combined softmax
 */

#include "arch35/yolo_simt.h"
// Ascend950PR-only body: its fast path is compiled under __NPU_ARCH__ == 3510; every other
// generation keeps the legacy NsYolo bodies below.
#include "arch35/yolo_simd_channel_permute.h"

template <uint32_t schMode>
__global__ __aicore__ void yolo(GM_ADDR x, GM_ADDR coord_data, GM_ADDR obj_prob, GM_ADDR classes_prob,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(YoloTilingData);
    GET_TILING_DATA_WITH_STRUCT(YoloTilingData, tilingData, tiling);

    if constexpr (schMode == YOLO_MODE_1) {
        NsYolo::Process<DTYPE_X, YOLO_MODE_1>(x, coord_data, obj_prob, classes_prob, &tilingData);
    } else if constexpr (schMode == YOLO_MODE_2) {
        NsYolo::Process<DTYPE_X, YOLO_MODE_2>(x, coord_data, obj_prob, classes_prob, &tilingData);
    } else if constexpr (schMode == YOLO_MODE_3) {
        NsYolo::Process<DTYPE_X, YOLO_MODE_3>(x, coord_data, obj_prob, classes_prob, &tilingData);
    } else if constexpr (schMode == YOLO_MODE_1_CHANNEL_PERMUTE) {
        // Ascend950PR path (the DMA body itself is guarded by __NPU_ARCH__ == 3510 and falls back to
        // the legacy NsYolo::Process<DTYPE_X, YOLO_MODE_1>); the host only encodes this key on that SoC.
        NsYoloChannelPermute::Process<DTYPE_X>(x, coord_data, obj_prob, classes_prob, &tilingData);
    } else {
        NsYolo::Process<DTYPE_X, YOLO_MODE_4>(x, coord_data, obj_prob, classes_prob, &tilingData);
    }
}
