/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef LAPLACIAN_OF_GAUSSIAN_TILING_DATA_H
#define LAPLACIAN_OF_GAUSSIAN_TILING_DATA_H
#include <cstdint>
// Dynamic g and d arrays follow this aligned fixed-size header, each padded to 32B.
// reserved packs (rowChunk << 1) | haloPath so the 64-byte ABI stays intact.
struct LaplacianOfGaussianTilingData {
    uint64_t n, h, w, tiles;
    uint32_t radius, k, border, blocks;
    uint32_t weightOffset, weightStride, tileWidth, reserved;
};
static_assert(sizeof(LaplacianOfGaussianTilingData) == 64, "tiling ABI");
// Match the native host registration's default 2048-byte tiling budget.
// The fixed header ABI remains 64 bytes; aligned FP32 g/d arrays occupy its tail.
struct LaplacianOfGaussianTilingStorage {
    LaplacianOfGaussianTilingData header;
    uint8_t weights[2048 - sizeof(LaplacianOfGaussianTilingData)];
};
static_assert(sizeof(LaplacianOfGaussianTilingStorage) == 2048, "tiling storage ABI");
#endif
