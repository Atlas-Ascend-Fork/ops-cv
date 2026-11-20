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
 * \file stack_group_points_apt.cpp
 * \brief arch35 SIMT kernel entry for stack_group_points
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "./arch35/stack_group_points_kernel.h"
#include "./arch35/stack_group_points_tiling_data.h"
#include "./arch35/stack_group_points_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void stack_group_points(GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices,
                                              GM_ADDR indices_batch_cnt, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StackGroupPointsTilingData);
    GET_TILING_DATA_WITH_STRUCT(StackGroupPointsTilingData, tilingData, tiling);

    if constexpr (schMode == STACK_GROUP_POINTS_TPL_FP32) {
        NsStackGroupPoints::Process<float>(features, features_batch_cnt, indices, indices_batch_cnt, y, tilingData.m,
                                           tilingData.c, tilingData.nsample, tilingData.b, tilingData.n,
                                           tilingData.totalElements, tilingData.zeroFill);
    } else if constexpr (schMode == STACK_GROUP_POINTS_TPL_FP16) {
        NsStackGroupPoints::Process<half>(features, features_batch_cnt, indices, indices_batch_cnt, y, tilingData.m,
                                          tilingData.c, tilingData.nsample, tilingData.b, tilingData.n,
                                          tilingData.totalElements, tilingData.zeroFill);
    }
}
