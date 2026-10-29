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
 * \file pixel_shuffle_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __PIXEL_SHUFFLE_TILLING_DATA_H__
#define __PIXEL_SHUFFLE_TILLING_DATA_H__

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>

struct PixelShuffleTilingData {
    // ND格式张量维度（4D: N,C_in,H,W；5D: N,C_in,D,H,W）
    int64_t N = 0;
    int64_t C_in = 0;
    int64_t D = 1; // 5D 深度（4D 时为 1）
    int64_t H = 0;
    int64_t W = 0;
    int64_t r = 1;
    int64_t rank = 4; // 输入维度数（4 或 5）

    // 预计算量
    int64_t totalLength = 0;
    int64_t tileNum = 1;
    uint32_t tileDataNum;
};

#endif
