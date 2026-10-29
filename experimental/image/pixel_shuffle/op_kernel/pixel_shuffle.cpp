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
 * \file pixel_shuffle.cpp
 * \brief
 */

#include "pixel_shuffle.h"

#ifndef DTYPE_X
#define DTYPE_X float
#endif

template <uint32_t schMode>
__global__ __aicore__ void pixel_shuffle(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PixelShuffleTilingData);
    GET_TILING_DATA_WITH_STRUCT(PixelShuffleTilingData, tilingData, tiling);
    NsPixelShuffle::PixelShuffle<DTYPE_X> op; // 算子kernel实例获取
    op.Init(x, y, &tilingData);               // 算子kernel实例初始化
    op.Process();                             // 算子kernel实例执行
}
