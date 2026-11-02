/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill_tiling_data.h
 * \brief Tiling data definition for FloodFill.
 */
#ifndef FLOOD_FILL_TILING_DATA_H_
#define FLOOD_FILL_TILING_DATA_H_

#include <cstdint>

// diff_mode attribute values (OpenCV FLOODFILL_FIXED_RANGE mapping).
constexpr uint32_t FLOODFILL_DIFF_MODE_NEIGHBOR = 0U;
constexpr uint32_t FLOODFILL_DIFF_MODE_FIXED_RANGE = 1U;

// Bytes reserved at the beginning of workspace for cross-core control data.
constexpr uint32_t FLOOD_FILL_CONTROL_BYTES = 2112U;

// Supported dtypes: uint8 / int32 / float32.
// - Integer dtypes: tolerances and new value are quantized (llround) on
//   host; uint8 is additionally clamped to [0, 255] (uint8 pixel
//   differences never exceed 255, clamping is semantics-preserving).
// - float32: values are passed through as-is, compared in float.
struct FloodFillTilingData {
    uint32_t height;
    uint32_t width;

    uint32_t seedRow;
    uint32_t seedCol;

    // Integer dtypes (uint8 / int32).
    int32_t newValueI;
    int32_t loDiffI; // non-negative
    int32_t upDiffI; // non-negative

    // float32 dtype.
    float newValueF;
    float loDiffF; // non-negative
    float upDiffF; // non-negative

    uint32_t connectivity; // 4 or 8
    uint32_t diffMode;     // FLOODFILL_DIFF_MODE_*
};

#endif // FLOOD_FILL_TILING_DATA_H_
