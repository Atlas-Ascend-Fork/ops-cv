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
#include "arch35/check_valid_kernel.h"
#include "arch35/check_valid_tiling_struct.h"
#include "arch35/check_valid_struct.h"

template <typename T>
__aicore__ inline void RunCheckValid(GM_ADDR bbox_tensor, GM_ADDR img_metas, GM_ADDR valid_tensor, GM_ADDR tiling,
                                     const CheckValidTilingData* tilingData)
{
    if (tilingData->N == 0) {
        CheckValidKernel<T, true> kernel;
        kernel.Init(bbox_tensor, img_metas, valid_tensor, tiling, tilingData);
        kernel.Process();
    } else {
        CheckValidKernel<T, false> kernel;
        kernel.Init(bbox_tensor, img_metas, valid_tensor, tiling, tilingData);
        kernel.Process();
    }
}

template <uint32_t SCH_MODE>
__global__ __aicore__ void check_valid(GM_ADDR bbox_tensor, GM_ADDR img_metas, GM_ADDR valid_tensor, GM_ADDR workspace,
                                       GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CheckValidTilingData);
    GET_TILING_DATA_WITH_STRUCT(CheckValidTilingData, tilingData, tiling);

    if constexpr (SCH_MODE == CHECK_VALID_MODE_FP32) {
        RunCheckValid<float>(bbox_tensor, img_metas, valid_tensor, tiling, &tilingData);
    } else if constexpr (SCH_MODE == CHECK_VALID_MODE_FP16) {
        RunCheckValid<half>(bbox_tensor, img_metas, valid_tensor, tiling, &tilingData);
    }

    (void)workspace;
}
