/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file match_template_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __MATCH_TEMPLATE_TILLING_DATA_H__
#define __MATCH_TEMPLATE_TILLING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

struct MatchTemplateTilingData {
    int64_t batch = 0;
    int64_t imageH = 0;
    int64_t imageW = 0;
    int64_t channels = 0;
    int64_t templateH = 0;
    int64_t templateW = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int64_t method = 0;
    int64_t tileH = 0;
    int64_t tileW = 0;
    int64_t tileRows = 0;
    int64_t tileCols = 0;
    int64_t tilesPerBatch = 0;
    int64_t totalTiles = 0;
    int64_t usedCores = 0;
    int64_t useHaloTile = 0;
    // Dedicated C3/16x16 vector schedule.  This is intentionally separate
    // from the generic halo schedule so the optimization can be reverted by
    // tiling without changing any other shape or dtype path.
    int64_t useC3VectorTile = 0;
    int64_t templateElements = 0;
    int64_t templateBufferBytes = 0;
    int64_t haloBufferBytes = 0;
    int64_t templateFloatBufferBytes = 0;
    int64_t haloFloatBufferBytes = 0;
    int64_t templateStatsBufferBytes = 0;
    // Offsets from the user workspace; local best scores start at offset zero.
    int64_t workspaceIndexOffset = 0;
    int64_t workspacePackedImageOffset = 0;
    int64_t packedImageBytes = 0;
    int64_t prepackScratchBytes = 0;
};
#endif
