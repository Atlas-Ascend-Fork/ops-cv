/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DISTANCE_TRANSFORM_TILING_DATA_H__
#define __DISTANCE_TRANSFORM_TILING_DATA_H__

#include <cstdint>

struct DistanceTransformTilingData {
    uint32_t imageCount;
    uint32_t height;
    uint32_t width;
    uint32_t imageStride;
    uint32_t distanceType; // 0: L1, 1: L2, 2: CHESSBOARD
    uint32_t maskSize;     // 0: precise, 3: 3x3 chamfer, 5: 5x5 chamfer
    uint32_t usedCoreNum;
    uint32_t tileRows;
    uint32_t vectorTileElems;
    uint32_t useRowBuffer;     // 0: GM, 1: row UB, 2: L1/chess vector, 3: chamfer vector
    uint32_t exactTileColumns; // precise-L2 column tile width; 0 selects the scalar-GM fallback
};

#endif
