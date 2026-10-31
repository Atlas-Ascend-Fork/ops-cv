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
 * \file adaptive_threshold_tiling_data.h
 * \brief Tiling data definition for AdaptiveThreshold.
 */
#ifndef ADAPTIVE_THRESHOLD_TILING_DATA_H_
#define ADAPTIVE_THRESHOLD_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE = 255U;
constexpr uint32_t ADAPTIVE_THRESHOLD_MEAN_VECTOR_MIN_WIDTH = 32U;
// Large UINT8 MEAN windows keep one input row and streaming column sums.
constexpr uint32_t ADAPTIVE_THRESHOLD_MEAN_STREAM_MIN_BLOCK_SIZE = 63U;
constexpr uint32_t ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE = 65536U;

struct AdaptiveThresholdTilingData {
    uint32_t imageCount;
    uint32_t height;
    uint32_t width;

    uint32_t totalRows;
    uint32_t baseRows;
    uint32_t extraRows;

    uint32_t tileWidth;
    uint32_t blockSize;
    uint32_t radius;

    uint32_t thresholdType;
    float outputValue;
    float thresholdDelta;

    uint32_t gaussianWeights[ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE];
};

#endif // ADAPTIVE_THRESHOLD_TILING_DATA_H_
