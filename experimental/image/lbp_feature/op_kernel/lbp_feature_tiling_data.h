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
 * \file lbp_feature_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __LBP_FEATURE_TILING_DATA_H__
#define __LBP_FEATURE_TILING_DATA_H__

#include <cstdint>

struct LbpFeatureTilingData {
    int32_t radius = 0;
    int32_t neighbors = 0;
    int32_t totalLength = 0;
    int32_t outputLength = 0;
    int32_t outH = 0;
    int32_t outW = 0;
    int32_t H = 0;
    int32_t W = 0;
    int32_t batchCount = 0;
    uint32_t totalRows = 0;
    uint32_t smallCoreTileNum = 0;
    uint32_t bigCoreTileNum = 0;
    uint32_t bigCoreCount = 0;
    uint32_t smallCoreRows = 0;
    uint32_t bigCoreRows = 0;
    uint32_t bigRowCoreCount = 0;
    uint32_t tileRows = 0;
    uint32_t tileCols = 0;
    uint32_t rowStride = 0;
    uint32_t tileColCount = 0;
    uint32_t imageTileCount = 0;
    int32_t offset1_ndarray[31] = {};
    int32_t offset2_ndarray[31] = {};
    int32_t offset3_ndarray[31] = {};
    int32_t offset4_ndarray[31] = {};
    float w1_ndarray[31] = {};
    float w2_ndarray[31] = {};
    float w3_ndarray[31] = {};
    float w4_ndarray[31] = {};
    uint32_t groupSlabMode = 0;
    uint32_t colOffsetCount = 0;
    int32_t slabRowSpan = 0;
    int32_t slabRowMin = 0;
    int32_t colOffsets[31] = {};
    int32_t fxSlabIdx[31] = {};
    int32_t cxSlabIdx[31] = {};
    int32_t fyRowOff[31] = {};
    int32_t cyRowOff[31] = {};
};

#endif // __LBP_FEATURE_TILING_DATA_H__
