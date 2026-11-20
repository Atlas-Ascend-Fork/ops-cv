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
 * \file crop_norm.h
 * \brief NDDMA kernel implementation for crop operator
 *
 * The output sub-region is merged into effective dims; inner dims form one strided NDDMA
 * copy (GM -> UB, compact), and DataCopyPad writes the contiguous UB data back to GM.
 * MTE handles the copy, so no per-element coordinate math is needed.
 */

#ifndef CROP_NORM_H_
#define CROP_NORM_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "crop_tiling_data.h"
#include "crop_tiling_key.h"

namespace NsCrop {
using namespace AscendC;

constexpr uint8_t SIMD_NDDMA_DIMS = 5;
constexpr int32_t SIMD_BUFFER_NUM = 2;
static constexpr NdDmaConfig CROP_NDDMA_CONFIG = {false, 0, 0, false};

template <typename T>
__aicore__ inline void FillNdDmaParams(NdDmaParams<T, SIMD_NDDMA_DIMS>& params, const CropTilingData* tilingData,
                                       int64_t overrideDimElems)
{
    for (int32_t i = 0; i < SIMD_NDDMA_DIMS; i++) {
        params.loopInfo.loopSize[i] = (i < static_cast<int32_t>(tilingData->copyDimNum)) ? tilingData->copySize[i] :
                                                                                           static_cast<uint32_t>(1);
        params.loopInfo.loopSrcStride[i] = (i < static_cast<int32_t>(tilingData->copyDimNum)) ?
                                               tilingData->copySrcStride[i] :
                                               static_cast<uint64_t>(0);
        params.loopInfo.loopDstStride[i] = (i < static_cast<int32_t>(tilingData->copyDimNum)) ?
                                               tilingData->copyDstStride[i] :
                                               static_cast<uint32_t>(0);
        params.loopInfo.loopLpSize[i] = 0;
        params.loopInfo.loopRpSize[i] = 0;
    }
    // overrideDimElems >= 0 overrides the chunk dim loopSize with the tail length
    if (overrideDimElems >= 0) {
        params.loopInfo.loopSize[tilingData->chunkDim] = static_cast<uint32_t>(overrideDimElems);
    }
    params.constantValue = 0;
}

// Row partition across cores with per-core chunking
template <typename T>
__aicore__ inline void CropSimdRowProcess(const CropTilingData* tilingData, GM_ADDR x, GM_ADDR y)
{
    int64_t coreNum = static_cast<int64_t>(tilingData->simdCoreNum);
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= coreNum || coreNum <= 0) {
        return;
    }
    int64_t perCore = (tilingData->rowTotal + coreNum - 1) / coreNum;
    int64_t rowStart = blockIdx * perCore;
    int64_t rowEnd = rowStart + perCore;
    if (rowEnd > tilingData->rowTotal) {
        rowEnd = tilingData->rowTotal;
    }
    if (rowStart >= rowEnd) {
        return;
    }

    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue;
    pipe.InitBuffer(vecQue, 2, tilingData->simdUbBufferBytes);

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;
    inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + tilingData->baseOffset);
    outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    NdDmaParams<T, SIMD_NDDMA_DIMS> params;
    FillNdDmaParams<T>(params, tilingData, -1);

    int64_t chunk = tilingData->chunkElems;
    int64_t innerElems = tilingData->baseElems / chunk;
    int32_t cd = static_cast<int32_t>(tilingData->chunkDim);
    int64_t srcStride = static_cast<int64_t>(tilingData->copySrcStride[cd]);

    for (int64_t row = rowStart; row < rowEnd; row += chunk) {
        int64_t cur = (rowEnd - row < chunk) ? (rowEnd - row) : chunk;
        params.loopInfo.loopSize[cd] = static_cast<uint32_t>(cur);
        int64_t srcOffset = row * srcStride;
        int64_t dstOffset = row * innerElems;
        int64_t elems = cur * innerElems;

        LocalTensor<T> local = vecQue.AllocTensor<T>();
        DataCopy<T, SIMD_NDDMA_DIMS, CROP_NDDMA_CONFIG>(local, inputGM[srcOffset], params);
        vecQue.EnQue(local);
        local = vecQue.DeQue<T>();

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(elems * static_cast<int64_t>(sizeof(T)));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(outputGM[dstOffset], local, outParams);
        vecQue.FreeTensor(local);
    }
}

template <typename T>
__aicore__ inline void CropSimdProcess(const CropTilingData* tilingData, GM_ADDR x, GM_ADDR y)
{
    if (tilingData->rowTotal > 0) {
        CropSimdRowProcess<T>(tilingData, x, y);
        return;
    }
    if (tilingData->totalIters <= 0 || tilingData->simdCoreNum == 0) {
        return;
    }
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(tilingData->simdCoreNum)) {
        return;
    }

    // Even split across cores; the first rem cores take one extra iteration
    int64_t perCore = tilingData->totalIters / static_cast<int64_t>(tilingData->simdCoreNum);
    int64_t rem = tilingData->totalIters % static_cast<int64_t>(tilingData->simdCoreNum);
    int64_t itBegin = perCore * blockIdx + (blockIdx < rem ? blockIdx : rem);
    int64_t itCount = perCore + (blockIdx < rem ? 1 : 0);
    if (itCount <= 0) {
        return;
    }

    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue;
    pipe.InitBuffer(vecQue, SIMD_BUFFER_NUM, tilingData->simdUbBufferBytes);

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;
    inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + tilingData->baseOffset);
    outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    NdDmaParams<T, SIMD_NDDMA_DIMS> mainParams;
    NdDmaParams<T, SIMD_NDDMA_DIMS> tailParams;
    FillNdDmaParams<T>(mainParams, tilingData, -1);
    FillNdDmaParams<T>(tailParams, tilingData, -1);
    if (tilingData->tailElems > 0) {
        int64_t innerElems = tilingData->baseElems / tilingData->chunkElems;
        FillNdDmaParams<T>(tailParams, tilingData, tilingData->tailElems / innerElems);
    }

    int64_t perIterElems = tilingData->baseElems;
    int64_t innerElems = (tilingData->chunkElems > 0) ? (tilingData->baseElems / tilingData->chunkElems) : 1;

    // Output span per outer row = chunked dim length * inner elements (tail chunks are shorter)
    int64_t perRowElems = tilingData->chunkDimSize * innerElems;
    for (int64_t k = 0; k < itCount; k++) {
        int64_t iter = itBegin + k;
        int64_t outerIter = iter;
        int64_t chunkIdx = 0;
        int64_t srcOffset = 0;
        if (tilingData->chunkElems > 0) {
            chunkIdx = iter % tilingData->numChunks;
            outerIter = iter / tilingData->numChunks;
            srcOffset = chunkIdx * tilingData->chunkElems *
                        static_cast<int64_t>(tilingData->copySrcStride[tilingData->chunkDim]);
        }
        int64_t outerRow = outerIter;
        // Enumerate outer dims row-major (innermost varies fastest)
        for (int32_t j = static_cast<int32_t>(tilingData->outerDimNum) - 1; j >= 0; j--) {
            int64_t coord = outerIter % tilingData->outerSize[j];
            outerIter /= tilingData->outerSize[j];
            srcOffset += coord * tilingData->outerSrcStride[j];
        }

        // The tail is the last chunk of each outer row, not the global last iteration
        bool isTail = (tilingData->tailElems > 0) && (tilingData->chunkElems > 0) &&
                      (chunkIdx == tilingData->numChunks - 1);
        int64_t curElems = isTail ? tilingData->tailElems : perIterElems;
        int64_t dstOffset = outerRow * perRowElems + chunkIdx * perIterElems;

        LocalTensor<T> local = vecQue.AllocTensor<T>();
        DataCopy<T, SIMD_NDDMA_DIMS, CROP_NDDMA_CONFIG>(local, inputGM[srcOffset], isTail ? tailParams : mainParams);
        vecQue.EnQue(local);
        local = vecQue.DeQue<T>();

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(curElems * static_cast<int64_t>(sizeof(T)));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(outputGM[dstOffset], local, outParams);
        vecQue.FreeTensor(local);
    }
}
// NDDMA row-partition template: split rows across cores, chunk within each core, double buffered.
__aicore__ inline int64_t CeilDivInt(int64_t a, int64_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

template <typename T>
class CropSsNddma {
public:
    __aicore__ inline CropSsNddma() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CropSsNddmaTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const CropSsNddmaTilingData* tilingData, int64_t blockIdx);
    __aicore__ inline int64_t GetInputGmAddr(int64_t rowIdx) const;
    __aicore__ inline int64_t GetOutputGmAddr(int64_t rowIdx) const;
    __aicore__ inline void GetProcessRowsOffset(int64_t& rowsOffset, int64_t blockIdx) const;
    __aicore__ inline void CalcProcessLoopsNum(int64_t& curCoreLoopsNum, int64_t& ubSplitLoopNum) const;
    __aicore__ inline void SetLoopInfo(NdDmaLoopInfo<SIMD_NDDMA_DIMS>& loopInfo,
                                       NdDmaLoopInfo<SIMD_NDDMA_DIMS>& loopInfoTail) const;
    __aicore__ inline void SetCopyOutParams(DataCopyExtParams& copyOutParamsMain,
                                            DataCopyExtParams& copyOutParamsTail) const;
    __aicore__ inline void ProcessPerBlock();

    TPipe* pipe_ = nullptr;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int64_t blockIdx_ = 0;

    // Parsed tiling data
    int64_t ubSize_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t inputDims_ = 0;
    int64_t outputShape_[MAX_NDIM] = {0};
    int64_t begin_[MAX_NDIM] = {0};
    int64_t strides_[MAX_NDIM] = {0};
    int64_t rowsOffsetSteps_[MAX_NDIM] = {0};
    int64_t inputSteps_[MAX_NDIM] = {0};

    int64_t ubIndex_ = 0;
    int64_t ubFactor_ = 0;
    int64_t ubTailFactor_ = 0;
    int64_t blkIndex_ = 0;
    int64_t blkFactor_ = 0;
    int64_t blkTailFactor_ = 0;
    int64_t blkSplitOutNum_ = 0;

    int64_t ubOutLoopSteps_ = 0;
    int64_t ubInLoopSteps_ = 0;
    int64_t nddmaTotalNum_ = 0;
    int64_t nddmaLoopSize_[CROP_SS_NDDMA_DIMS] = {0};
    int64_t nddmaLoopSrcStride_[CROP_SS_NDDMA_DIMS] = {0};
    int64_t nddmaLoopDstStride_[CROP_SS_NDDMA_DIMS] = {0};

    // Runtime temporaries
    int64_t ubSplitLoopsNum_ = 0;
    int64_t curCoreLoopsNum_ = 0;
    int64_t curCoreRowsOffset_ = 0;
};

template <typename T>
__aicore__ inline void CropSsNddma<T>::Init(GM_ADDR x, GM_ADDR y, const CropSsNddmaTilingData* tilingData,
                                            TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    ParseTilingData(tilingData, blockIdx_);
    pipe_->InitBuffer(vecQue_, SIMD_BUFFER_NUM, ubSize_ / SIMD_BUFFER_NUM);
}

template <typename T>
__aicore__ inline void CropSsNddma<T>::ParseTilingData(const CropSsNddmaTilingData* tilingData, int64_t blockIdx)
{
    ubSize_ = tilingData->base.ubSize;
    realCoreNum_ = tilingData->base.realCoreNum;
    inputDims_ = tilingData->base.inputDims;
    for (int64_t i = 0; i < inputDims_; i++) {
        outputShape_[i] = tilingData->base.outputShape[i];
        begin_[i] = tilingData->base.begin[i];
        strides_[i] = tilingData->base.strides[i];
        rowsOffsetSteps_[i] = tilingData->base.rowsOffsetSteps[i];
        inputSteps_[i] = tilingData->base.inputSteps[i];
    }
    blkIndex_ = tilingData->base.blkIndex;
    blkFactor_ = tilingData->base.blkFactor;
    blkTailFactor_ = tilingData->base.blkTailFactor;
    blkSplitOutNum_ = CeilDivInt(outputShape_[blkIndex_], blkFactor_);

    ubIndex_ = tilingData->base.ubIndex;
    ubFactor_ = tilingData->base.ubFactor;
    ubTailFactor_ = tilingData->base.ubTailFactor;
    if ((blockIdx % blkSplitOutNum_ == blkSplitOutNum_ - 1) && (blkIndex_ == ubIndex_) && (blkTailFactor_ != 0)) {
        ubTailFactor_ = tilingData->base.ubTailTailFactor;
    }
    ubInLoopSteps_ = tilingData->base.ubInLoopSteps;
    ubOutLoopSteps_ = tilingData->base.ubOutLoopSteps;

    for (int32_t i = 0; i < CROP_SS_NDDMA_DIMS; i++) {
        nddmaLoopSize_[i] = tilingData->nddmaLoopSize[i];
        nddmaLoopSrcStride_[i] = tilingData->nddmaLoopSrcStride[i];
        nddmaLoopDstStride_[i] = tilingData->nddmaLoopDstStride[i];
    }
    nddmaTotalNum_ = tilingData->nddmaTotalNum;
}

template <typename T>
__aicore__ inline void CropSsNddma<T>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    CalcProcessLoopsNum(curCoreLoopsNum_, ubSplitLoopsNum_);
    GetProcessRowsOffset(curCoreRowsOffset_, blockIdx_);
    ProcessPerBlock();
}

// Input offset of row rowIdx (one innermost-dim element group)
template <typename T>
__aicore__ inline int64_t CropSsNddma<T>::GetInputGmAddr(int64_t rowIdx) const
{
    int64_t inputGmAddr = begin_[inputDims_ - 1];
    int64_t curDim = 0;
    int64_t tmpBegin = 0;
    for (int64_t i = inputDims_ - 1; i > 0; i--) {
        curDim = outputShape_[i - 1];
        tmpBegin = begin_[i - 1] + rowIdx % curDim * strides_[i - 1];
        inputGmAddr = inputGmAddr + inputSteps_[i] * tmpBegin;
        rowIdx = rowIdx / curDim;
    }
    return inputGmAddr;
}

template <typename T>
__aicore__ inline int64_t CropSsNddma<T>::GetOutputGmAddr(int64_t rowIdx) const
{
    return rowIdx * outputShape_[inputDims_ - 1];
}

// First row handled by this core (split by blkFactor; last core uses blkTailFactor)
template <typename T>
__aicore__ inline void CropSsNddma<T>::GetProcessRowsOffset(int64_t& rowsOffset, int64_t blockIdx) const
{
    rowsOffset = blockIdx / blkSplitOutNum_ * rowsOffsetSteps_[blkIndex_];
    rowsOffset = rowsOffset + blockIdx % blkSplitOutNum_ * blkFactor_ * rowsOffsetSteps_[blkIndex_ + 1];
}

template <typename T>
__aicore__ inline void CropSsNddma<T>::CalcProcessLoopsNum(int64_t& curCoreLoopsNum, int64_t& ubSplitLoopNum) const
{
    if (((blkSplitOutNum_ > 0) && (blockIdx_ % blkSplitOutNum_ == blkSplitOutNum_ - 1)) && (blkTailFactor_ != 0)) {
        curCoreLoopsNum = blkTailFactor_;
    } else {
        curCoreLoopsNum = blkFactor_;
    }

    if (blkIndex_ == ubIndex_) {
        // When blk and ub split the same axis, tiling ensures blkFactor >= ubFactor
        ubSplitLoopNum = curCoreLoopsNum / ubFactor_;
        curCoreLoopsNum = 1;
        return;
    }
    for (int64_t i = blkIndex_ + 1; i < ubIndex_; i++) {
        curCoreLoopsNum = curCoreLoopsNum * outputShape_[i];
    }
    ubSplitLoopNum = outputShape_[ubIndex_] / ubFactor_;
}

template <typename T>
__aicore__ inline void CropSsNddma<T>::SetLoopInfo(NdDmaLoopInfo<SIMD_NDDMA_DIMS>& loopInfo,
                                                   NdDmaLoopInfo<SIMD_NDDMA_DIMS>& loopInfoTail) const
{
    int64_t inUbDims = inputDims_ - ubIndex_;
    for (int64_t i = 0; i < inUbDims; i++) {
        loopInfo.loopSize[i] = nddmaLoopSize_[SIMD_NDDMA_DIMS - 1 - i];
        loopInfo.loopSrcStride[i] = nddmaLoopSrcStride_[SIMD_NDDMA_DIMS - 1 - i];
        loopInfo.loopDstStride[i] = nddmaLoopDstStride_[SIMD_NDDMA_DIMS - 1 - i];
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;

        loopInfoTail.loopSize[i] = nddmaLoopSize_[SIMD_NDDMA_DIMS - 1 - i];
        loopInfoTail.loopSrcStride[i] = nddmaLoopSrcStride_[SIMD_NDDMA_DIMS - 1 - i];
        loopInfoTail.loopDstStride[i] = nddmaLoopDstStride_[SIMD_NDDMA_DIMS - 1 - i];
        loopInfoTail.loopLpSize[i] = 0;
        loopInfoTail.loopRpSize[i] = 0;
    }
    loopInfoTail.loopSize[inUbDims - 1] = ubTailFactor_;

    for (int64_t i = inUbDims; i < SIMD_NDDMA_DIMS; i++) {
        loopInfo.loopSize[i] = 1;
        loopInfo.loopSrcStride[i] = 0;
        loopInfo.loopDstStride[i] = 0;
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
        loopInfoTail.loopSize[i] = 1;
        loopInfoTail.loopSrcStride[i] = 0;
        loopInfoTail.loopDstStride[i] = 0;
        loopInfoTail.loopLpSize[i] = 0;
        loopInfoTail.loopRpSize[i] = 0;
    }
}

template <typename T>
__aicore__ inline void CropSsNddma<T>::SetCopyOutParams(DataCopyExtParams& copyOutParamsMain,
                                                        DataCopyExtParams& copyOutParamsTail) const
{
    copyOutParamsMain.blockCount = 1;
    copyOutParamsMain.blockLen = nddmaTotalNum_ * sizeof(T);
    copyOutParamsMain.dstStride = 0;
    copyOutParamsMain.srcStride = 0;

    if (ubTailFactor_ > 0) {
        copyOutParamsTail.blockCount = 1;
        copyOutParamsTail.blockLen = nddmaTotalNum_ / ubFactor_ * ubTailFactor_ * sizeof(T);
        copyOutParamsTail.dstStride = 0;
        copyOutParamsTail.srcStride = 0;
    }
}

template <typename T>
__aicore__ inline void CropSsNddma<T>::ProcessPerBlock()
{
    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;

    NdDmaParams<T, SIMD_NDDMA_DIMS> paramsMain;
    NdDmaParams<T, SIMD_NDDMA_DIMS> paramsTail;
    paramsMain.constantValue = 0;
    paramsTail.constantValue = 0;
    SetLoopInfo(paramsMain.loopInfo, paramsTail.loopInfo);

    DataCopyExtParams copyOutParamsMain;
    DataCopyExtParams copyOutParamsTail;
    SetCopyOutParams(copyOutParamsMain, copyOutParamsTail);

    for (int64_t idx = 0; idx < curCoreLoopsNum_; idx++) {
        inputGmAddr = GetInputGmAddr(curCoreRowsOffset_ + idx * rowsOffsetSteps_[ubIndex_]);
        outputGmAddr = GetOutputGmAddr(curCoreRowsOffset_ + idx * rowsOffsetSteps_[ubIndex_]);
        for (int64_t loops = 0; loops < ubSplitLoopsNum_; loops++) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopy<T, SIMD_NDDMA_DIMS, CROP_NDDMA_CONFIG>(inputLocal, inputGM_[inputGmAddr + loops * ubInLoopSteps_],
                                                            paramsMain);
            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + loops * ubOutLoopSteps_], inputLocal, copyOutParamsMain);
            vecQue_.FreeTensor(inputLocal);
        }
        if (ubTailFactor_ > 0) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopy<T, SIMD_NDDMA_DIMS, CROP_NDDMA_CONFIG>(
                inputLocal, inputGM_[inputGmAddr + ubSplitLoopsNum_ * ubInLoopSteps_], paramsTail);
            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + ubSplitLoopsNum_ * ubOutLoopSteps_], inputLocal, copyOutParamsTail);
            vecQue_.FreeTensor(inputLocal);
        }
    }
}

} // namespace NsCrop
#endif // CROP_NORM_H_
