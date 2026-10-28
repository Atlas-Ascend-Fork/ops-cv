/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CLAHE_TILING_DATA_H
#define CLAHE_TILING_DATA_H
#include <cstdint>
struct ClaheTilingData {
    uint64_t imageElements;
    uint64_t lutElements;
    uint64_t regionCount;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t tilesX;
    uint32_t tilesY;
    uint32_t tileWidth;
    uint32_t tileHeight;
    uint32_t bins;
    uint32_t clipCount;
    uint32_t coreNum;
    uint32_t wideHistogram;
    uint32_t reserved;
    float lutScale;
    float invTileWidth;
    float invTileHeight;
    uint32_t padding;
};
#endif
