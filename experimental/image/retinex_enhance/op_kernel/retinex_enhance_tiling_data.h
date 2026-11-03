/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RETINEX_ENHANCE_TILING_DATA_H
#define RETINEX_ENHANCE_TILING_DATA_H

#include <cstdint>

constexpr uint32_t RETINEX_SCALE_COUNT = 3;
constexpr uint32_t RETINEX_WEIGHT_GROUP_SIZE = 32;
constexpr uint32_t RETINEX_TOTAL_WEIGHT_GROUPS = 34;

struct RetinexEnhanceTilingData {
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t tilePixels;
    uint32_t stripeCount;
    uint32_t rowElements;
    uint32_t rowElementsAligned;
    uint32_t tileElementsAligned;
    uint32_t verticalTileRows;
    uint32_t verticalTileElementsAligned;
    uint32_t maxHaloElementsAligned;
    uint32_t verticalIndexElementsAligned;
    uint32_t radii[RETINEX_SCALE_COUNT];
    uint32_t groupOffsets[RETINEX_SCALE_COUNT];
    uint32_t groupCounts[RETINEX_SCALE_COUNT];
    float ratioSteps[RETINEX_SCALE_COUNT];
    float anchorWeights[RETINEX_TOTAL_WEIGHT_GROUPS];
    float anchorRatios[RETINEX_TOTAL_WEIGHT_GROUPS];
};

#endif
