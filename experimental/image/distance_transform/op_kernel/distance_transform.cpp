/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "distance_transform.h"

namespace {
constexpr uint32_t DISTANCE_TRANSFORM_KERNEL_REVISION = 2026081909U;
} // namespace

template <uint32_t schMode>
__global__ __aicore__ void distance_transform(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)DISTANCE_TRANSFORM_KERNEL_REVISION;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DistanceTransformTilingData);
    GET_TILING_DATA_WITH_STRUCT(DistanceTransformTilingData, tilingData, tiling);

    if constexpr (schMode == DISTANCE_TRANSFORM_SCH_MODE_DEFAULT) {
        AscendC::TPipe pipe;
        NsDistanceTransform::KernelDistanceTransform op;
        op.Init(x, y, AscendC::GetUserWorkspace(workspace), &tilingData, &pipe);
        op.Process();
    }
}
