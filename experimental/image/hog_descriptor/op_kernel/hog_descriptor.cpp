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
#include "hog_descriptor_tiling_def.h"
#include "hog_descriptor_impl.hpp"

using namespace AscendC;

__aicore__ inline void CopyTilingData(optiling::HogDescriptorTilingData& dst, GM_ADDR tiling)
{
    uint32_t* local = reinterpret_cast<uint32_t*>(&dst);
    auto global = reinterpret_cast<__gm__ uint32_t*>(tiling);
    for (uint32_t i = 0; i < sizeof(optiling::HogDescriptorTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
}

extern "C" __global__ __aicore__ void hog_descriptor(GM_ADDR x, GM_ADDR descriptor, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace; // pure vector implementation, no user workspace
    REGISTER_TILING_DEFAULT(optiling::HogDescriptorTilingData);
    optiling::HogDescriptorTilingData tilingDataIn;
    CopyTilingData(tilingDataIn, tiling);
    const auto* tilingData = &tilingDataIn;

    // TILING_KEY_IS requires preprocessor numeric literals (named constants won't parse):
    //   0 = fp16, 1 = fp32. Both paths write float32 descriptors.
    if (TILING_KEY_IS(0)) {
        NsHogDescriptor::HogDescriptorVectorKernel<half> op;
        op.Init(x, descriptor, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NsHogDescriptor::HogDescriptorVectorKernel<float> op;
        op.Init(x, descriptor, tilingData);
        op.Process();
    }
}
