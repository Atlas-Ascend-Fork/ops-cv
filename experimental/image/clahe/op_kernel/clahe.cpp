/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "clahe.h"
#include "clahe_tiling_key.h"
template <uint32_t schMode>
__global__ __aicore__ void clahe(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(ClaheTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClaheTilingData, data, tiling);
    AscendC::TPipe pipe;
    NsClahe::KernelClahe<DTYPE_X> op;
    // 编译器包装层已经传入用户工作区，无需再次调用 GetUserWorkspace 或增加偏移。
    op.Init(x, y, workspace, &data, &pipe);
    op.Process();
}
