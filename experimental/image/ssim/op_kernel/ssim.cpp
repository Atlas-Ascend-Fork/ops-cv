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
 * \file ssim.cpp
 * \brief ssim kernel entry
 */
#include "ssim.h"

using namespace AscendC;
using namespace SsimKernel;

extern "C" __global__ __aicore__ void ssim(GM_ADDR x1, GM_ADDR x2, GM_ADDR yOut, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelSsim<uint8_t> op(&pipe);
        op.Init(x1, x2, yOut, AscendC::GetUserWorkspace(workspace), &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelSsim<float> op(&pipe);
        op.Init(x1, x2, yOut, AscendC::GetUserWorkspace(workspace), &tilingData);
        op.Process();
    }
}
