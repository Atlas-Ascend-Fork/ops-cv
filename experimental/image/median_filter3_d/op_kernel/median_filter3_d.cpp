/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "median_filter3_d.h"
#include "median_filter3_d_tiling_key.h"
template <uint32_t schMode>
__global__ __aicore__ void median_filter3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MedianFilter3DTilingData);
    GET_TILING_DATA_WITH_STRUCT(MedianFilter3DTilingData, data, tiling);
    AscendC::TPipe pipe;
    NsMedianFilter3D::KernelMedianFilter3D<DTYPE_X> op;
    op.Init(x, y, &data, &pipe);
    op.Process();
}
