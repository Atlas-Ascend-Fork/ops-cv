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
 * \file lbp_feature.h
 * \brief LbpFeature kernel implementation (multi-core vectorized elbp, OpenCV-compatible).
 */

#ifndef __LBP_FEATURE_H__
#define __LBP_FEATURE_H__

#include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
#include "lbp_feature_tiling_data.h"
#include "lbp_feature_tiling_key.h"
#include <type_traits>

namespace NsLbpFeature {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float FLOAT_EPSILON = 1.1920928955078125e-7f;

template <typename TYPE_X, typename TYPE_Y>
class LbpFeature {
private:
    struct TileInfo {
        uint32_t rowStart;
        uint32_t colStart;
        uint32_t validRows;
        uint32_t validCols;
        int32_t inputBase;
        int32_t outputBase;
        int32_t centerRowStart;
        int32_t centerColStart;
        int32_t outputOffset;
    };

public:
    __aicore__ inline LbpFeature() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t totalLength, int32_t radius, int32_t neighbors,
                                int32_t outputLength, int32_t* offset1Ndarray, int32_t* offset2Ndarray,
                                int32_t* offset3Ndarray, int32_t* offset4Ndarray, float* w1Ndarray, float* w2Ndarray,
                                float* w3Ndarray, float* w4Ndarray, int32_t outH, int32_t outW, int32_t H, int32_t W,
                                uint32_t totalRows, uint32_t tileRows, uint32_t tileCols, uint32_t rowStride,
                                uint32_t tileColCount, uint32_t imageTileCount, uint32_t smallCoreTileNum,
                                uint32_t bigCoreTileNum, uint32_t bigCoreCount, uint32_t smallCoreRows,
                                uint32_t bigCoreRows, uint32_t bigRowCoreCount, uint32_t groupSlabMode,
                                uint32_t colOffsetCount, int32_t slabRowSpan, int32_t slabRowMin, int32_t* colOffsets,
                                int32_t* fxSlabIdx, int32_t* cxSlabIdx, int32_t* fyRowOff, int32_t* cyRowOff,
                                TPipe* pipeIn)
    {
        this->radius = radius;
        this->neighbors = neighbors;
        this->outH = outH;
        this->outW = outW;
        this->H = H;
        this->W = W;
        this->totalRows = totalRows;
        this->tileRows = tileRows;
        this->tileCols = tileCols;
        this->rowStride = rowStride;
        constexpr uint32_t VECTOR_COMPUTE_ALIGN_ELEMENTS = 64U;
        const uint32_t rawTileElements = tileRows * rowStride;
        this->tileCapacity = ((rawTileElements + VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) / VECTOR_COMPUTE_ALIGN_ELEMENTS) *
                             VECTOR_COMPUTE_ALIGN_ELEMENTS;
        this->tileColCount = tileColCount;
        this->imageTileCount = imageTileCount;
        this->smallCoreTileNum = smallCoreTileNum;
        this->bigCoreTileNum = bigCoreTileNum;
        this->bigCoreCount = bigCoreCount;
        this->smallCoreRows = smallCoreRows;
        this->bigCoreRows = bigCoreRows;
        this->bigRowCoreCount = bigRowCoreCount;
        this->groupSlabMode = groupSlabMode;
        this->colOffsetCount = colOffsetCount;
        this->slabRowSpan = slabRowSpan;
        this->slabRowMin = slabRowMin;
        this->colOffsets = colOffsets;
        this->fxSlabIdx = fxSlabIdx;
        this->cxSlabIdx = cxSlabIdx;
        this->fyRowOff = fyRowOff;
        this->cyRowOff = cyRowOff;
        this->offset1Ndarray = offset1Ndarray;
        this->offset2Ndarray = offset2Ndarray;
        this->offset3Ndarray = offset3Ndarray;
        this->offset4Ndarray = offset4Ndarray;
        this->w1Ndarray = w1Ndarray;
        this->w2Ndarray = w2Ndarray;
        this->w3Ndarray = w3Ndarray;
        this->w4Ndarray = w4Ndarray;

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y, outputLength);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        const uint32_t inputBytes = tileCapacity * sizeof(TYPE_X);
        const uint32_t outputBytes = tileCapacity * sizeof(TYPE_Y);
        const uint32_t floatBytes = tileCapacity * sizeof(float);
        const uint32_t halfBytes = tileCapacity * sizeof(half);
        const uint32_t uint8Bytes = tileCapacity * sizeof(uint8_t);
        const uint32_t int32Bytes = tileCapacity * sizeof(int32_t);

        // Slabs carry one halo row beyond tileRows so that the second corner
        // row window (offset +1 row) is always resident.  The extra 64
        // elements keep the 64-aligned vector read count from running past
        // the buffer.
        const uint32_t slabElements = ((rawTileElements + rowStride + 2U * VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) /
                                       VECTOR_COMPUTE_ALIGN_ELEMENTS) *
                                      VECTOR_COMPUTE_ALIGN_ELEMENTS;
        const uint32_t slabBytes = slabElements * sizeof(TYPE_X);

        // v4: one slab per distinct column offset, all row windows are views.
        const uint32_t slabSetElements = (((colOffsetCount * (tileRows + static_cast<uint32_t>(slabRowSpan)) *
                                            rowStride) +
                                           VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) /
                                          VECTOR_COMPUTE_ALIGN_ELEMENTS) *
                                             VECTOR_COMPUTE_ALIGN_ELEMENTS +
                                         VECTOR_COMPUTE_ALIGN_ELEMENTS;

        pipeIn->InitBuffer(centerQueue, BUFFER_NUM, inputBytes);
        if (groupSlabMode == 1U) {
            pipeIn->InitBuffer(slabSetQueue, BUFFER_NUM, slabSetElements * sizeof(TYPE_X));
        } else {
            pipeIn->InitBuffer(neighborAQueue, BUFFER_NUM, slabBytes);
            pipeIn->InitBuffer(neighborBQueue, BUFFER_NUM, slabBytes);
        }
        pipeIn->InitBuffer(outQueue, BUFFER_NUM, outputBytes);
        pipeIn->InitBuffer(interpBuf, floatBytes);
        pipeIn->InitBuffer(conditionBuf, halfBytes);
        pipeIn->InitBuffer(oneHalfBuf, halfBytes);
        pipeIn->InitBuffer(compareBitsBuf, uint8Bytes);
        pipeIn->InitBuffer(bitValueBuf, int32Bytes);

        if constexpr (std::is_same_v<TYPE_X, uint8_t>) {
            pipeIn->InitBuffer(centerFloatBuf, floatBytes);
            pipeIn->InitBuffer(castHalfBuf, halfBytes);
            pipeIn->InitBuffer(workBuf, floatBytes);
        }

        oneHalf = oneHalfBuf.Get<half>();
        Duplicate(oneHalf, half(1), tileCapacity);
    }

    __aicore__ inline void ProcessTile2D()
    {
        if (outH <= 0 || outW <= 0 || tileRows == 0U || tileCols == 0U || rowStride == 0U) {
            return;
        }
        const int32_t inputImageSize = H * W;
        const int32_t outputImageSize = outH * outW;
        const uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());
        uint32_t coreTileNum = smallCoreTileNum;
        uint32_t startTile = 0U;
        if (coreId < bigCoreCount) {
            coreTileNum = bigCoreTileNum;
            startTile = coreId * bigCoreTileNum;
        } else {
            startTile = bigCoreCount * bigCoreTileNum + (coreId - bigCoreCount) * smallCoreTileNum;
        }

        TileInfo current;
        InitTileInfo(current, startTile, imageTileCount, tileColCount, inputImageSize, outputImageSize);
        CopyInCenter(current);
        if (groupSlabMode == 1U) {
            CopyInSlabSetTile(current);
        }

        for (uint32_t i = 0U; i < coreTileNum; ++i) {
            const bool hasNext = (i + 1U < coreTileNum);
            TileInfo next;
            if (hasNext) {
                next = current;
                AdvanceTileInfo(next, inputImageSize, outputImageSize);
                CopyInCenter(next);
                if (groupSlabMode == 1U) {
                    CopyInSlabSetTile(next);
                }
            }
            ComputeTile2D(current.inputBase, current.centerRowStart, current.centerColStart, current.validRows,
                          current.validCols);
            CopyOut2D(current.outputOffset, current.validRows, current.validCols);
            if (hasNext) {
                current = next;
            }
        }
    }

    __aicore__ inline void ProcessRowSplit()
    {
        if (outH <= 0 || outW <= 0 || tileRows == 0U || rowStride == 0U) {
            return;
        }
        const int32_t inputImageSize = H * W;
        const int32_t outputImageSize = outH * outW;
        const uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());
        uint32_t coreRowNum = smallCoreRows;
        uint32_t startGlobalRow = 0U;
        if (coreId < bigRowCoreCount) {
            coreRowNum = bigCoreRows;
            startGlobalRow = coreId * bigCoreRows;
        } else {
            startGlobalRow = bigRowCoreCount * bigCoreRows + (coreId - bigRowCoreCount) * smallCoreRows;
        }
        if (coreRowNum == 0U || startGlobalRow >= totalRows) {
            return;
        }
        const uint32_t availableRows = totalRows - startGlobalRow;
        coreRowNum = (coreRowNum < availableRows) ? coreRowNum : availableRows;

        TileInfo current;
        InitRowTileInfo(current, startGlobalRow, coreRowNum, inputImageSize, outputImageSize);
        CopyInCenter(current);
        if (groupSlabMode == 1U) {
            CopyInSlabSetTile(current);
        }
        uint32_t remainingRows = coreRowNum;
        while (remainingRows > 0U) {
            const uint32_t currentRows = current.validRows;
            const bool hasNext = remainingRows > currentRows;
            TileInfo next;
            if (hasNext) {
                next = current;
                AdvanceRowTileInfo(next, currentRows, remainingRows - currentRows, inputImageSize, outputImageSize);
                CopyInCenter(next);
                if (groupSlabMode == 1U) {
                    CopyInSlabSetTile(next);
                }
            }
            ComputeTile2D(current.inputBase, current.centerRowStart, current.centerColStart, current.validRows,
                          current.validCols);
            CopyOut2D(current.outputOffset, current.validRows, current.validCols);
            remainingRows -= currentRows;
            if (hasNext) {
                current = next;
            }
        }
    }

private:
    __aicore__ inline void RefreshRowTileInfo(TileInfo& tile, uint32_t remainingRows)
    {
        const uint32_t imageRows = static_cast<uint32_t>(outH) - tile.rowStart;
        tile.validRows = (remainingRows < tileRows) ? remainingRows : tileRows;
        tile.validRows = (tile.validRows < imageRows) ? tile.validRows : imageRows;
        tile.validCols = static_cast<uint32_t>(outW);
        tile.centerRowStart = static_cast<int32_t>(tile.rowStart) + radius;
        tile.centerColStart = radius;
        tile.outputOffset = tile.outputBase + static_cast<int32_t>(tile.rowStart) * outW;
    }
    __aicore__ inline void InitRowTileInfo(TileInfo& tile, uint32_t globalRowIndex, uint32_t remainingRows,
                                           int32_t inputImageSize, int32_t outputImageSize)
    {
        const uint32_t batch = globalRowIndex / static_cast<uint32_t>(outH);
        tile.rowStart = globalRowIndex % static_cast<uint32_t>(outH);
        tile.colStart = 0U;
        tile.inputBase = static_cast<int32_t>(batch) * inputImageSize;
        tile.outputBase = static_cast<int32_t>(batch) * outputImageSize;
        RefreshRowTileInfo(tile, remainingRows);
    }
    __aicore__ inline void AdvanceRowTileInfo(TileInfo& tile, uint32_t completedRows, uint32_t remainingRows,
                                              int32_t inputImageSize, int32_t outputImageSize)
    {
        tile.rowStart += completedRows;
        if (tile.rowStart >= static_cast<uint32_t>(outH)) {
            tile.rowStart = 0U;
            tile.inputBase += inputImageSize;
            tile.outputBase += outputImageSize;
        }
        RefreshRowTileInfo(tile, remainingRows);
    }
    __aicore__ inline void RefreshTileInfo(TileInfo& tile)
    {
        const uint32_t rowRemain = static_cast<uint32_t>(outH) - tile.rowStart;
        const uint32_t colRemain = static_cast<uint32_t>(outW) - tile.colStart;
        tile.validRows = (rowRemain < tileRows) ? rowRemain : tileRows;
        tile.validCols = (colRemain < tileCols) ? colRemain : tileCols;
        tile.centerRowStart = static_cast<int32_t>(tile.rowStart) + radius;
        tile.centerColStart = static_cast<int32_t>(tile.colStart) + radius;
        tile.outputOffset = tile.outputBase + static_cast<int32_t>(tile.rowStart) * outW +
                            static_cast<int32_t>(tile.colStart);
    }
    __aicore__ inline void InitTileInfo(TileInfo& tile, uint32_t globalTileIndex, uint32_t imageTileCount,
                                        uint32_t tileColCount, int32_t inputImageSize, int32_t outputImageSize)
    {
        const uint32_t batch = globalTileIndex / imageTileCount;
        const uint32_t tileIndex = globalTileIndex % imageTileCount;
        const uint32_t tileRowIndex = tileIndex / tileColCount;
        const uint32_t tileColIndex = tileIndex % tileColCount;
        tile.rowStart = tileRowIndex * tileRows;
        tile.colStart = tileColIndex * tileCols;
        tile.inputBase = static_cast<int32_t>(batch) * inputImageSize;
        tile.outputBase = static_cast<int32_t>(batch) * outputImageSize;
        RefreshTileInfo(tile);
    }
    __aicore__ inline void AdvanceTileInfo(TileInfo& tile, int32_t inputImageSize, int32_t outputImageSize)
    {
        tile.colStart += tileCols;
        if (tile.colStart >= static_cast<uint32_t>(outW)) {
            tile.colStart = 0U;
            tile.rowStart += tileRows;
            if (tile.rowStart >= static_cast<uint32_t>(outH)) {
                tile.rowStart = 0U;
                tile.inputBase += inputImageSize;
                tile.outputBase += outputImageSize;
            }
        }
        RefreshTileInfo(tile);
    }
    __aicore__ inline void CopyInCenter(const TileInfo& tile)
    {
        const int32_t centerOffset = tile.inputBase + tile.centerRowStart * W + tile.centerColStart;
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<TYPE_X> padParams;
        BuildCopyInParams(tile.validRows, tile.validCols, copyParams, padParams);
        CopyIn2D(centerQueue, centerOffset, copyParams, padParams);
    }
    __aicore__ inline void BuildCopyInParams(uint32_t validRows, uint32_t validCols, DataCopyExtParams& copyParams,
                                             DataCopyPadExtParams<TYPE_X>& padParams)
    {
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        const uint32_t copyAlignElements = DATA_BLOCK_BYTES / sizeof(TYPE_X);
        const uint32_t blockLenBytes = validCols * static_cast<uint32_t>(sizeof(TYPE_X));
        const uint32_t srcStrideBytes = (static_cast<uint32_t>(W) - validCols) * static_cast<uint32_t>(sizeof(TYPE_X));
        const uint32_t alignedValidCols = ((validCols + copyAlignElements - 1U) / copyAlignElements) *
                                          copyAlignElements;
        const uint32_t dstGapElements = (rowStride > alignedValidCols) ? (rowStride - alignedValidCols) : 0U;
        const uint32_t dstStrideBlocks = (dstGapElements * static_cast<uint32_t>(sizeof(TYPE_X))) / DATA_BLOCK_BYTES;
        const uint32_t rightPadding = alignedValidCols - validCols;
        copyParams.blockCount = static_cast<uint16_t>(validRows);
        copyParams.blockLen = blockLenBytes;
        copyParams.srcStride = srcStrideBytes;
        copyParams.dstStride = dstStrideBlocks;
        copyParams.rsv = 0;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(rightPadding);
        padParams.paddingValue = static_cast<TYPE_X>(0);
    }
    __aicore__ inline void CopyIn2D(TQue<QuePosition::VECIN, 2>& queue, int32_t gmOffset,
                                    const DataCopyExtParams& copyParams, const DataCopyPadExtParams<TYPE_X>& padParams)
    {
        LocalTensor<TYPE_X> local = queue.AllocTensor<TYPE_X>();
        DataCopyPad(local, xGm[gmOffset], copyParams, padParams);
        queue.EnQue(local);
    }
    __aicore__ inline void CopyInSlabs(int32_t centerOffset, int32_t neighbor, const DataCopyExtParams& baseParams,
                                       const DataCopyPadExtParams<TYPE_X>& padParams)
    {
        // Corner rows: offset1/offset2 live on row fy, offset3/offset4 on row
        // cy = fy + dy with dy in {0, 1}.  Loading the column windows of
        // offset1/offset2 with dy extra rows lets the lower corner row pair be
        // read back as a 32B aligned row-offset view instead of a second DMA.
        const int32_t idx1 = centerOffset + offset1Ndarray[neighbor];
        const int32_t idx2 = centerOffset + offset2Ndarray[neighbor];
        const int32_t dy = (offset3Ndarray[neighbor] - offset1Ndarray[neighbor]) / W;
        DataCopyExtParams slabParams = baseParams;
        slabParams.blockCount = static_cast<uint16_t>(static_cast<uint32_t>(baseParams.blockCount) +
                                                      static_cast<uint32_t>(dy));
        CopyIn2D(neighborAQueue, idx1, slabParams, padParams);
        CopyIn2D(neighborBQueue, idx2, slabParams, padParams);
    }
    __aicore__ inline void CopyInSlabSetTile(const TileInfo& tile)
    {
        const uint32_t slabRows = tile.validRows + static_cast<uint32_t>(slabRowSpan);
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<TYPE_X> padParams;
        BuildCopyInParams(slabRows, tile.validCols, copyParams, padParams);
        LocalTensor<TYPE_X> slabSet = slabSetQueue.AllocTensor<TYPE_X>();
        const uint32_t slabStrideElems = slabRows * rowStride;
        for (uint32_t k = 0U; k < colOffsetCount; ++k) {
            const int32_t gmOffset = tile.inputBase + (tile.centerRowStart + slabRowMin) * W +
                                     (tile.centerColStart + colOffsets[k]);
            DataCopyPad(slabSet[k * slabStrideElems], xGm[gmOffset], copyParams, padParams);
        }
        slabSetQueue.EnQue(slabSet);
    }
    __aicore__ inline void InterpolateFirstPair(const LocalTensor<TYPE_X>& slab0, const LocalTensor<TYPE_X>& slab1,
                                                int32_t neighbor, uint32_t count)
    {
        LocalTensor<float> interp = interpBuf.Get<float>();
        if constexpr (std::is_same_v<TYPE_X, uint8_t>) {
            LocalTensor<half> castHalf = castHalfBuf.Get<half>();
            LocalTensor<float> work = workBuf.Get<float>();
            Cast(castHalf, slab0, RoundMode::CAST_NONE, count);
            Cast(work, castHalf, RoundMode::CAST_NONE, count);
            Muls(interp, work, w1Ndarray[neighbor], count);
            Cast(castHalf, slab1, RoundMode::CAST_NONE, count);
            Cast(work, castHalf, RoundMode::CAST_NONE, count);
            Axpy(interp, work, w2Ndarray[neighbor], count);
        } else {
            Muls(interp, slab0, w1Ndarray[neighbor], count);
            Axpy(interp, slab1, w2Ndarray[neighbor], count);
        }
    }
    __aicore__ inline void InterpolateSecondPair(const LocalTensor<TYPE_X>& slab0, const LocalTensor<TYPE_X>& slab1,
                                                 uint32_t rowOffsetElements, int32_t neighbor, uint32_t count)
    {
        LocalTensor<float> interp = interpBuf.Get<float>();
        LocalTensor<TYPE_X> lower0 = slab0[rowOffsetElements];
        LocalTensor<TYPE_X> lower1 = slab1[rowOffsetElements];
        if constexpr (std::is_same_v<TYPE_X, uint8_t>) {
            LocalTensor<float> work = workBuf.Get<float>();
            LocalTensor<half> castHalf = castHalfBuf.Get<half>();
            Cast(castHalf, lower0, RoundMode::CAST_NONE, count);
            Cast(work, castHalf, RoundMode::CAST_NONE, count);
            Axpy(interp, work, w3Ndarray[neighbor], count);
            Cast(castHalf, lower1, RoundMode::CAST_NONE, count);
            Cast(work, castHalf, RoundMode::CAST_NONE, count);
            Axpy(interp, work, w4Ndarray[neighbor], count);
        } else {
            Axpy(interp, lower0, w3Ndarray[neighbor], count);
            Axpy(interp, lower1, w4Ndarray[neighbor], count);
        }
    }
    __aicore__ inline void AccumulateNeighbor(const LocalTensor<float>& center, LocalTensor<TYPE_Y>& output,
                                              int32_t neighbor, uint32_t count)
    {
        LocalTensor<float> interp = interpBuf.Get<float>();
        LocalTensor<uint8_t> compareBits = compareBitsBuf.Get<uint8_t>();
        LocalTensor<half> condition = conditionBuf.Get<half>();
        Sub(interp, interp, center, count);
        CompareScalar(compareBits, interp, -FLOAT_EPSILON, CMPMODE::GT, count);
        Select(condition, compareBits, oneHalf, half(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        if (neighbor == 0) {
            Cast(output, condition, RoundMode::CAST_ROUND, count);
        } else {
            LocalTensor<int32_t> bitValue = bitValueBuf.Get<int32_t>();
            Cast(bitValue, condition, RoundMode::CAST_ROUND, count);
            ShiftLeft(bitValue, bitValue, neighbor, count);
            Add(output, output, bitValue, count);
        }
    }
    __aicore__ inline void ComputeNeighbors2D(const LocalTensor<float>& center, LocalTensor<TYPE_Y>& output,
                                              int32_t inputBase, int32_t centerRowStart, int32_t centerColStart,
                                              uint32_t validRows, uint32_t validCols, uint32_t computeCount)
    {
        if (groupSlabMode == 1U) {
            LocalTensor<TYPE_X> slabSet = slabSetQueue.DeQue<TYPE_X>();
            const uint32_t slabStrideElems = (validRows + static_cast<uint32_t>(slabRowSpan)) * rowStride;
            for (int32_t neighbor = 0; neighbor < neighbors; ++neighbor) {
                const uint32_t fxBase = static_cast<uint32_t>(fxSlabIdx[neighbor]) * slabStrideElems +
                                        static_cast<uint32_t>(fyRowOff[neighbor]) * rowStride;
                const uint32_t cxBase = static_cast<uint32_t>(cxSlabIdx[neighbor]) * slabStrideElems +
                                        static_cast<uint32_t>(fyRowOff[neighbor]) * rowStride;
                const uint32_t fxBaseLo = static_cast<uint32_t>(fxSlabIdx[neighbor]) * slabStrideElems +
                                          static_cast<uint32_t>(cyRowOff[neighbor]) * rowStride;
                const uint32_t cxBaseLo = static_cast<uint32_t>(cxSlabIdx[neighbor]) * slabStrideElems +
                                          static_cast<uint32_t>(cyRowOff[neighbor]) * rowStride;
                InterpolateFirstPair(slabSet[fxBase], slabSet[cxBase], neighbor, computeCount);
                InterpolateSecondPair(slabSet[fxBaseLo], slabSet[cxBaseLo], 0U, neighbor, computeCount);
                AccumulateNeighbor(center, output, neighbor, computeCount);
            }
            slabSetQueue.FreeTensor(slabSet);
            return;
        }
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<TYPE_X> padParams;
        BuildCopyInParams(validRows, validCols, copyParams, padParams);
        const int32_t centerOffset = inputBase + centerRowStart * W + centerColStart;
        // Prefetch neighbor 0 and then stay one neighbor pair ahead so that
        // MTE2 copies of the next neighbor overlap the vector math of the
        // current one.
        CopyInSlabs(centerOffset, 0, copyParams, padParams);
        for (int32_t neighbor = 0; neighbor < neighbors; ++neighbor) {
            if (neighbor + 1 < neighbors) {
                CopyInSlabs(centerOffset, neighbor + 1, copyParams, padParams);
            }
            LocalTensor<TYPE_X> slab0 = neighborAQueue.DeQue<TYPE_X>();
            LocalTensor<TYPE_X> slab1 = neighborBQueue.DeQue<TYPE_X>();
            const int32_t dy = (offset3Ndarray[neighbor] - offset1Ndarray[neighbor]) / W;
            const uint32_t lowerRowOffset = static_cast<uint32_t>(dy) * rowStride;
            InterpolateFirstPair(slab0, slab1, neighbor, computeCount);
            InterpolateSecondPair(slab0, slab1, lowerRowOffset, neighbor, computeCount);
            AccumulateNeighbor(center, output, neighbor, computeCount);
            neighborAQueue.FreeTensor(slab0);
            neighborBQueue.FreeTensor(slab1);
        }
    }
    __aicore__ inline void ComputeTile2D(int32_t inputBase, int32_t centerRowStart, int32_t centerColStart,
                                         uint32_t validRows, uint32_t validCols)
    {
        LocalTensor<TYPE_X> center = centerQueue.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> output = outQueue.AllocTensor<TYPE_Y>();
        constexpr uint32_t VECTOR_COMPUTE_ALIGN_ELEMENTS = 64U;
        const uint32_t validStorageElements = validRows * rowStride;
        const uint32_t computeCount = ((validStorageElements + VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) /
                                       VECTOR_COMPUTE_ALIGN_ELEMENTS) *
                                      VECTOR_COMPUTE_ALIGN_ELEMENTS;
        if constexpr (std::is_same_v<TYPE_X, uint8_t>) {
            LocalTensor<float> centerFloat = centerFloatBuf.Get<float>();
            LocalTensor<half> castHalf = castHalfBuf.Get<half>();
            Cast(castHalf, center, RoundMode::CAST_NONE, computeCount);
            Cast(centerFloat, castHalf, RoundMode::CAST_NONE, computeCount);
            ComputeNeighbors2D(centerFloat, output, inputBase, centerRowStart, centerColStart, validRows, validCols,
                               computeCount);
        } else {
            ComputeNeighbors2D(center, output, inputBase, centerRowStart, centerColStart, validRows, validCols,
                               computeCount);
        }
        centerQueue.FreeTensor(center);
        outQueue.EnQue(output);
    }
    __aicore__ inline void CopyOut2D(int32_t gmOffset, uint32_t validRows, uint32_t validCols)
    {
        LocalTensor<TYPE_Y> output = outQueue.DeQue<TYPE_Y>();
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        const uint32_t elementBytes = sizeof(TYPE_Y);
        const uint32_t blockLenBytes = validCols * elementBytes;
        const uint32_t alignedBlockBytes = ((blockLenBytes + DATA_BLOCK_BYTES - 1U) / DATA_BLOCK_BYTES) *
                                           DATA_BLOCK_BYTES;
        const uint32_t srcRowBytes = rowStride * elementBytes;
        const uint32_t srcStrideBlocks = (srcRowBytes > alignedBlockBytes) ?
                                             ((srcRowBytes - alignedBlockBytes) / DATA_BLOCK_BYTES) :
                                             0U;
        const uint32_t dstStrideBytes = (static_cast<uint32_t>(outW) - validCols) * elementBytes;
        DataCopyExtParams copyParams{static_cast<uint16_t>(validRows), blockLenBytes, srcStrideBlocks, dstStrideBytes,
                                     0};
        DataCopyPad(yGm[gmOffset], output, copyParams);
        outQueue.FreeTensor(output);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 2> centerQueue;
    TQue<QuePosition::VECIN, 2> neighborAQueue;
    TQue<QuePosition::VECIN, 2> neighborBQueue;
    TQue<QuePosition::VECIN, 2> slabSetQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> interpBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> centerFloatBuf;
    TBuf<QuePosition::VECCALC> conditionBuf;
    TBuf<QuePosition::VECCALC> oneHalfBuf;
    TBuf<QuePosition::VECCALC> castHalfBuf;
    TBuf<QuePosition::VECCALC> compareBitsBuf;
    TBuf<QuePosition::VECCALC> bitValueBuf;
    LocalTensor<half> oneHalf;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    int32_t radius;
    int32_t neighbors;
    int32_t outH;
    int32_t outW;
    int32_t H;
    int32_t W;
    uint32_t totalRows;
    uint32_t smallCoreTileNum;
    uint32_t bigCoreTileNum;
    uint32_t bigCoreCount;
    uint32_t smallCoreRows;
    uint32_t bigCoreRows;
    uint32_t bigRowCoreCount;

    uint32_t groupSlabMode;
    uint32_t colOffsetCount;
    int32_t slabRowSpan;
    int32_t slabRowMin;
    int32_t* colOffsets;
    int32_t* fxSlabIdx;
    int32_t* cxSlabIdx;
    int32_t* fyRowOff;
    int32_t* cyRowOff;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t rowStride;
    uint32_t tileCapacity;
    uint32_t tileColCount;
    uint32_t imageTileCount;
    int32_t* offset1Ndarray;
    int32_t* offset2Ndarray;
    int32_t* offset3Ndarray;
    int32_t* offset4Ndarray;
    float* w1Ndarray;
    float* w2Ndarray;
    float* w3Ndarray;
    float* w4Ndarray;
};

} // namespace NsLbpFeature
#endif // LBP_FEATURE_H

// #ifndef __LBP_FEATURE_H__
// #define __LBP_FEATURE_H__

// #include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
// #include "lbp_feature_tiling_data.h"
// #include "lbp_feature_tiling_key.h"

// namespace NsLbpFeature {

// using namespace AscendC;

// constexpr int32_t BUFFER_NUM = 2;

// template <typename T>
// class LbpFeature {
// public:
//     __aicore__ inline LbpFeature(){};

//     __aicore__ inline void Init(/*参数列表*/);
//     __aicore__ inline void Process(/*参数列表*/);

// private:
//     __aicore__ inline void CopyIn(/*参数列表*/);
//     __aicore__ inline void CopyOut(/*参数列表*/);
//     __aicore__ inline void Compute(/*参数列表*/);

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> XXX;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> YYY;
// };

// template <typename T>
// __aicore__ inline void LbpFeature<T>::Init(/*参数列表*/)
// {
// }

// template <typename T>
// __aicore__ inline void LbpFeature<T>::CopyIn(/*参数列表*/)
// {
// }

// template <typename T>
// __aicore__ inline void LbpFeature<T>::CopyOut(/*参数列表*/)
// {
// }

// template <typename T>
// __aicore__ inline void LbpFeature<T>::Compute(/*参数列表*/)
// {
// }

// template <typename T>
// __aicore__ inline void LbpFeature<T>::Process()
// {
//     for (int32_t i = 0; i < /*循环次数*/; i++) {
//         CopyIn(/*参数列表*/);
//         Compute(/*参数列表*/);
//         CopyOut(/*参数列表*/);
//     }
// }

// } // namespace NsLbpFeature
// #endif // LBP_FEATURE_H
