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
 * \file rotated_box_decode_tiling_data.h
 * \brief Tiling data struct for rotated_box_decode operator
 */
#ifndef ROTATED_BOX_DECODE_TILING_DATA_H_
#define ROTATED_BOX_DECODE_TILING_DATA_H_

#include <cstdint>

static constexpr int64_t RBD_RANK = 3;
static constexpr int64_t RBD_CHANNELS = 5;
static constexpr int64_t RBD_WEIGHT_LEN = 5;
static constexpr int64_t RBD_UB_AXIS_N = 0;
static constexpr int64_t RBD_UB_AXIS_B = 1;
static constexpr int64_t RBD_UB_AXIS_I = 2; // N==1 SIMT branch (per-box scalar compute)
static constexpr int64_t RBD_COPY_MODE_NDDMA = 0;
// SIMT thread count per AIV block for the N==1 branch (key=512). Compile-time
// constant per the SIMT MDE rule (must match __launch_bounds__ in
// rotated_box_decode_simt.h); host mirrors it only for core-count math.
static constexpr int64_t RBD_SIMT_THREAD_NUM = 256;

struct RotatedBoxDecodeTilingData {
    int64_t rank;
    int64_t inShape[RBD_RANK];
    int64_t outShape[RBD_RANK];
    int64_t totalCount;
    int64_t perCoreCount;
    int64_t ubAxis;
    int64_t ubFactor;
    int64_t bufferSize;
    int64_t B;
    int64_t N;
    int64_t channelStride;
    float weight[RBD_WEIGHT_LEN];
    int64_t copyMode;
};

#endif // ROTATED_BOX_DECODE_TILING_DATA_H_
