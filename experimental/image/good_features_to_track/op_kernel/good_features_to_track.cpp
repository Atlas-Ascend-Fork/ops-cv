/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "good_features_to_track.h"

template <uint32_t schMode = 0>
__global__ __aicore__ void good_features_to_track(GM_ADDR image, GM_ADDR mask, GM_ADDR corners, GM_ADDR numCorners,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    // MIX task type：内核用无参 SyncAll 做跨核同步，按 !1872 口径需在同步前声明 task type；
    // KERNEL_TYPE_MIX_AIV_1_0 表示 MIX 任务但仅使用 AIV（与 image/rasterizer 等既有实现一致）。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(GoodFeaturesToTrackTilingData);
    GET_TILING_DATA_WITH_STRUCT(GoodFeaturesToTrackTilingData, tilingData, tiling);
    NsGoodFeaturesToTrack::GoodFeaturesToTrack<DTYPE_IMAGE> op;
    op.Init(image, mask, corners, numCorners, workspace, &tilingData);
    op.Process();
}
