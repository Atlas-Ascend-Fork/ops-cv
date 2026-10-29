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
#include "kernel_tiling/kernel_tiling.h"
#include "bilateral_filter3_d_tiling_def.h"
#include "bilateral_filter3_d_impl.hpp"

using namespace AscendC;

__aicore__ inline void CopyTilingData(optiling::BilateralFilter3DTilingData& dst, GM_ADDR tiling)
{
    uint32_t* local = reinterpret_cast<uint32_t*>(&dst);
    auto global = reinterpret_cast<__gm__ uint32_t*>(tiling);
    for (uint32_t i = 0; i < sizeof(optiling::BilateralFilter3DTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
}

extern "C" __global__ __aicore__ void bilateral_filter3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace; // 纯向量实现, 无用户 workspace
    REGISTER_TILING_DEFAULT(optiling::BilateralFilter3DTilingData);
    optiling::BilateralFilter3DTilingData tilingDataIn;
    CopyTilingData(tilingDataIn, tiling);
    const auto* tilingData = &tilingDataIn;

    // TILING_KEY_IS 需要预编译期数字字面量(具名常量无法被预编译器解析):
    //   0 = fp16, 1 = fp32
    if (TILING_KEY_IS(0)) {
        NsBilateralFilter3D::BilateralFilter3DVectorKernel<half> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NsBilateralFilter3D::BilateralFilter3DVectorKernel<float> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}
