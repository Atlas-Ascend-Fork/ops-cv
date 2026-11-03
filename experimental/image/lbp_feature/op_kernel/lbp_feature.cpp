/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lbp_feature_tiling_data.h"
#include "kernel_operator.h"

#include "lbp_feature.h"

using namespace AscendC;

enum class LbpFeatureTilingKey : uint32_t {
    TILING_KEY_ROW_SPLIT = 1,
    TILING_KEY_TILE_2D = 2,
};

// 1. 改为 uint32_t schMode，接收编译器传入的 tilingkey
template <uint32_t schMode>
__global__ __aicore__ void lbp_feature(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // Pure Vector kernel: restrict the task to AIV cores so that
    // SetBlockDim maps directly to the AIV task count.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(LbpFeatureTilingData);
    // 2. 解析 tiling 数据
    GET_TILING_DATA(tilingData, tiling);
    // GET_TILING_DATA(LbpFeatureTilingData, tilingData, tiling);

    TPipe pipe;
    // 使用编译期注入的数据类型
    NsLbpFeature::LbpFeature<DTYPE_X, DTYPE_Y> op;

    // 3. 使用 if constexpr 进行编译期分支选择
    if constexpr (schMode == static_cast<uint32_t>(LbpFeatureTilingKey::TILING_KEY_ROW_SPLIT)) {
        // 分支 1：ROW_SPLIT
        op.Init(x, y, tilingData.totalLength, tilingData.radius, tilingData.neighbors, tilingData.outputLength,
                tilingData.offset1_ndarray, tilingData.offset2_ndarray, tilingData.offset3_ndarray,
                tilingData.offset4_ndarray, tilingData.w1_ndarray, tilingData.w2_ndarray, tilingData.w3_ndarray,
                tilingData.w4_ndarray, tilingData.outH, tilingData.outW, tilingData.H, tilingData.W,
                tilingData.totalRows, tilingData.tileRows, tilingData.tileCols, tilingData.rowStride,
                tilingData.tileColCount, tilingData.imageTileCount, tilingData.smallCoreTileNum,
                tilingData.bigCoreTileNum, tilingData.bigCoreCount, tilingData.smallCoreRows, tilingData.bigCoreRows,
                tilingData.bigRowCoreCount, tilingData.groupSlabMode, tilingData.colOffsetCount, tilingData.slabRowSpan,
                tilingData.slabRowMin, tilingData.colOffsets, tilingData.fxSlabIdx, tilingData.cxSlabIdx,
                tilingData.fyRowOff, tilingData.cyRowOff, &pipe);

        op.ProcessRowSplit();
    } else if constexpr (schMode == static_cast<uint32_t>(LbpFeatureTilingKey::TILING_KEY_TILE_2D)) {
        // 分支 2：TILE_2D
        op.Init(x, y, tilingData.totalLength, tilingData.radius, tilingData.neighbors, tilingData.outputLength,
                tilingData.offset1_ndarray, tilingData.offset2_ndarray, tilingData.offset3_ndarray,
                tilingData.offset4_ndarray, tilingData.w1_ndarray, tilingData.w2_ndarray, tilingData.w3_ndarray,
                tilingData.w4_ndarray, tilingData.outH, tilingData.outW, tilingData.H, tilingData.W,
                tilingData.totalRows, tilingData.tileRows, tilingData.tileCols, tilingData.rowStride,
                tilingData.tileColCount, tilingData.imageTileCount, tilingData.smallCoreTileNum,
                tilingData.bigCoreTileNum, tilingData.bigCoreCount, tilingData.smallCoreRows, tilingData.bigCoreRows,
                tilingData.bigRowCoreCount, tilingData.groupSlabMode, tilingData.colOffsetCount, tilingData.slabRowSpan,
                tilingData.slabRowMin, tilingData.colOffsets, tilingData.fxSlabIdx, tilingData.cxSlabIdx,
                tilingData.fyRowOff, tilingData.cyRowOff, &pipe);

        op.ProcessTile2D();
    }
}
