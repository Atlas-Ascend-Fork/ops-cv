/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gamma_correction_impl.hpp"
#include "gamma_correction_tiling_def.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

__aicore__ inline void CopyTilingData(optiling::GammaCorrectionTilingData& dst, GM_ADDR tiling)
{
    uint32_t* local = reinterpret_cast<uint32_t*>(&dst);
    auto global = reinterpret_cast<__gm__ uint32_t*>(tiling);
    for (uint32_t i = 0; i < sizeof(optiling::GammaCorrectionTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
}

extern "C" __global__ __aicore__ void gamma_correction(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(optiling::GammaCorrectionTilingData);
    optiling::GammaCorrectionTilingData tilingDataIn;
    CopyTilingData(tilingDataIn, tiling);
    const auto* tilingData = &tilingDataIn;

    // TILING_KEY_IS requires numeric literals in the precompilation phase; the
    // optiling::GC_TILING_KEY_* named constants cannot be resolved then (the
    // precompiler emits "can not be processed as numeric variables"), which
    // breaks per-dtype binary selection at runtime. Use the raw key values:
    //   0=fp16 1=bf16 2=fp32 3=uint8 4=int8 5=uint16 6=int16
    if (TILING_KEY_IS(0)) {
        NsGammaCorrection::GammaCorrectionKernel<half> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        NsGammaCorrection::GammaCorrectionKernel<bfloat16_t> op;
        op.Init(x, y, tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2)) {
        NsGammaCorrection::GammaCorrectionKernel<float> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        NsGammaCorrection::GammaCorrectionKernel<uint8_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        NsGammaCorrection::GammaCorrectionKernel<int8_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        NsGammaCorrection::GammaCorrectionKernel<uint16_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        NsGammaCorrection::GammaCorrectionKernel<int16_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}
