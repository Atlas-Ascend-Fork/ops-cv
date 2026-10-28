/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bicubic2d_aa_apt.cpp
 * \brief upsample_bicubic2d_aa_apt
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "./arch35/upsample_bicubic2d_aa_data_copy.h"
#include "./arch35/upsample_bicubic2d_aa_simt.h"
#include "./arch35/upsample_bicubic2d_aa_tiling_key.h"
#include "./arch35/upsample_bicubic2d_aa_tiling_data.h"

using namespace AscendC;
using namespace UpsampleBicubic2dAA;

template <uint64_t schId, uint64_t isInt32>
__global__ __aicore__ void upsample_bicubic2d_aa(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(UpsampleBicubic2dAARegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleBicubic2dAARegBaseTilingData, tilingData, tiling);

    TPipe pipe;
    if constexpr (schId == 0) {
        // 纯copy模板，输出完全等于输入
        Bicubic2dAADataCopy<DTYPE_X> op;
        op.Init(x, y, &pipe, &tilingData);
        op.Process();
        return;
    } else {
        if constexpr (isInt32 == 1) {
            Bicubic2dAASimt<DTYPE_X, uint32_t, int32_t, schId> op;
            op.Init(x, y, &tilingData);
            op.Process();
        } else {
            Bicubic2dAASimt<DTYPE_X, uint64_t, int64_t, schId> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        return;
    }
}
