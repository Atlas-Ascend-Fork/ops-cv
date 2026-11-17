/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEDIAN_FILTER3_D_TILING_DATA_H
#define MEDIAN_FILTER3_D_TILING_DATA_H
#include <cstdint>
struct MedianFilter3DTilingData {
    uint64_t rows;
    uint64_t tasks;
    uint32_t d, h, w;
    uint32_t kd, kh, kw;
    uint32_t count, padded, tile, tilesPerRow, cores, mode;
    float cval;
    uint32_t reserved;
};
#endif
