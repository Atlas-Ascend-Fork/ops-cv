/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hit_or_miss.h"
#include "kernel_tiling/kernel_tiling.h"

extern "C" __global__ __aicore__ void hit_or_miss(GM_ADDR input, GM_ADDR kernel, GM_ADDR output, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(HitOrMissTilingData);
    GET_TILING_DATA_WITH_STRUCT(HitOrMissTilingData, tilingData, tiling);

    NsHitOrMiss::HitOrMissKernel op;
    op.Init(input, kernel, output, &tilingData);
    op.Process();
}
