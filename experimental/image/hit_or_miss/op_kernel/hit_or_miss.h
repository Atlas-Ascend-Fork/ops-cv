/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HIT_OR_MISS_H
#define HIT_OR_MISS_H

#include "kernel_operator.h"
#include "hit_or_miss_tiling_data.h"

namespace NsHitOrMiss {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

constexpr uint32_t KERNEL_BUFFER_BYTES = 4096;

// A2/A3 的 CompareScalar 不支持 uint8 输入，快速路径需先转换为 half 再进行零值比较。
constexpr uint32_t FAST_TILE_ELEMS = 4096;
// 128 个 half 为 256 字节，作为当前向量比较和掩码处理的基本粒度。
constexpr uint32_t FAST_VECTOR_GRANULARITY = 128;
constexpr uint32_t FAST_MASK_BYTES = FAST_TILE_ELEMS / 8;

// 稀疏路径最多缓存 25 个有效约束，超过该数量后使用稠密遍历。
constexpr int32_t FAST_COMPACT_ACTIVE_CAPACITY = 25;
constexpr uint32_t FAST_ACTIVE_LIST_BYTES = 64;
constexpr uint16_t FAST_ACTIVE_COORD_MASK = 0x3FU;
constexpr uint16_t FAST_ACTIVE_COL_SHIFT = 6U;
constexpr uint16_t FAST_ACTIVE_NEGATIVE_FLAG = 0x1000U;

constexpr uint32_t GENERIC_OUTPUT_TILE_BYTES = 16384;

class HitOrMissKernel {
public:
    __aicore__ inline HitOrMissKernel() = default;

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR kernel, GM_ADDR output, const HitOrMissTilingData* tilingData)
    {
        tilingData_ = tilingData;
        inputGm_.SetGlobalBuffer((__gm__ uint8_t*)input, tilingData_->totalLength);
        kernelGm_.SetGlobalBuffer((__gm__ int8_t*)kernel,
                                  static_cast<int64_t>(tilingData_->kernelH) * tilingData_->kernelW);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t*)output, tilingData_->totalLength);

        pipe_.InitBuffer(kernelQueue_, BUFFER_NUM, KERNEL_BUFFER_BYTES);

        pipe_.InitBuffer(fastInputBuf_, FAST_TILE_ELEMS * sizeof(uint8_t));

        pipe_.InitBuffer(fastOutputQueue_, BUFFER_NUM, FAST_TILE_ELEMS * sizeof(uint8_t));
        pipe_.InitBuffer(fastHalfBuf_, FAST_TILE_ELEMS * sizeof(half));
        pipe_.InitBuffer(fastMaskAccumBuf_, FAST_MASK_BYTES);
        pipe_.InitBuffer(fastMaskTempBuf_, FAST_MASK_BYTES);
        pipe_.InitBuffer(fastActiveListBuf_, FAST_ACTIVE_LIST_BYTES);

        pipe_.InitBuffer(genericOutputQueue_, BUFFER_NUM, GENERIC_OUTPUT_TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());

        if (blockIdx >= tilingData_->usedCoreNum || tilingData_->totalLength <= 0) {
            return;
        }

        LocalTensor<int8_t> kernelLocal = CopyKernelToUb();

        LocalTensor<uint16_t> activeList = fastActiveListBuf_.Get<uint16_t>();

        const int32_t activeKernelCount = BuildActiveKernelList(kernelLocal, activeList);

        const bool hasActiveKernel = activeKernelCount > 0;

        PipeBarrier<PIPE_ALL>();

        if (CanUseFast2DPath()) {
            ProcessFast2D(blockIdx, hasActiveKernel, activeKernelCount, activeList, kernelLocal);
        } else {
            ProcessGeneric(blockIdx, hasActiveKernel, kernelLocal);
        }

        kernelQueue_.FreeTensor(kernelLocal);
    }

    // 快速二维路径仅用于最后两轴连续存储的布局。
    __aicore__ inline bool CanUseFast2DPath() const
    {
        return tilingData_->axis1Stride == 1 && tilingData_->axis0Stride == tilingData_->axis1Dim &&
               tilingData_->axis0Dim > 0 && tilingData_->axis1Dim > 0;
    }

    // 复用输入缓冲前，等待此前的向量读取完成。
    __aicore__ inline void SyncFastInputBeforeReuse()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline LocalTensor<int8_t> CopyKernelToUb()
    {
        LocalTensor<int8_t> kernelLocal = kernelQueue_.AllocTensor<int8_t>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(tilingData_->kernelH * tilingData_->kernelW * sizeof(int8_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(kernelLocal, kernelGm_[0], copyParams, {false, 0, 0, 0});
        kernelQueue_.EnQue(kernelLocal);
        return kernelQueue_.DeQue<int8_t>();
    }

    __aicore__ inline int32_t BuildActiveKernelList(const LocalTensor<int8_t>& kernelLocal,
                                                    const LocalTensor<uint16_t>& activeList)
    {
        int32_t activeCount = 0;
        int32_t kernelIndex = 0;

        for (int32_t kernelRow = 0; kernelRow < tilingData_->kernelH; ++kernelRow) {
            for (int32_t kernelCol = 0; kernelCol < tilingData_->kernelW; ++kernelCol, ++kernelIndex) {
                const int8_t value = kernelLocal.GetValue(kernelIndex);

                if (value != static_cast<int8_t>(1) && value != static_cast<int8_t>(-1)) {
                    continue;
                }

                if (activeCount < FAST_COMPACT_ACTIVE_CAPACITY) {
                    uint16_t encoded = static_cast<uint16_t>(kernelRow) |
                                       static_cast<uint16_t>(static_cast<uint16_t>(kernelCol) << FAST_ACTIVE_COL_SHIFT);

                    if (value == static_cast<int8_t>(-1)) {
                        encoded |= FAST_ACTIVE_NEGATIVE_FLAG;
                    }

                    activeList.SetValue(activeCount, encoded);
                }

                ++activeCount;
            }
        }

        return activeCount;
    }

    __aicore__ inline void ProcessFast2D(const int32_t blockIdx, const bool hasActiveKernel,
                                         const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
                                         const LocalTensor<int8_t>& kernelLocal)
    {
        const int64_t height = tilingData_->axis0Dim;
        const int64_t width = tilingData_->axis1Dim;
        const int64_t totalRows = tilingData_->totalLength / width;

        const int64_t rowsPerCore = (totalRows + static_cast<int64_t>(tilingData_->usedCoreNum) - 1) /
                                    static_cast<int64_t>(tilingData_->usedCoreNum);

        const int64_t rowBegin = static_cast<int64_t>(blockIdx) * rowsPerCore;

        int64_t rowEnd = rowBegin + rowsPerCore;

        if (rowEnd > totalRows) {
            rowEnd = totalRows;
        }

        if (rowBegin >= rowEnd) {
            return;
        }

        LocalTensor<half> inputHalf = fastHalfBuf_.Get<half>();
        LocalTensor<uint8_t> maskAccum = fastMaskAccumBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> maskTemp = fastMaskTempBuf_.Get<uint8_t>();

        for (int64_t globalRow = rowBegin; globalRow < rowEnd; ++globalRow) {
            const int64_t rowInPlane = globalRow % height;
            const int64_t rowBase = globalRow * width;

            for (int64_t tileCol = 0; tileCol < width; tileCol += static_cast<int64_t>(FAST_TILE_ELEMS)) {
                int64_t currentNum = width - tileCol;

                if (currentNum > static_cast<int64_t>(FAST_TILE_ELEMS)) {
                    currentNum = static_cast<int64_t>(FAST_TILE_ELEMS);
                }

                LocalTensor<uint8_t> outputLocal = fastOutputQueue_.AllocTensor<uint8_t>();

                if (!hasActiveKernel) {
                    ComputeAllZeroFast(rowBase, tileCol, currentNum, outputLocal, inputHalf, maskTemp);
                } else {
                    ComputeHitMissFastTile(rowBase, rowInPlane, tileCol, currentNum, activeKernelCount, activeList,
                                           outputLocal, inputHalf, maskAccum, maskTemp, kernelLocal);
                }

                fastOutputQueue_.EnQue(outputLocal);

                CopyFastOutput(rowBase + tileCol, currentNum);
            }
        }
    }

    __aicore__ inline void ComputeAllZeroFast(const int64_t rowBase, const int64_t tileCol, const int64_t currentNum,
                                              const LocalTensor<uint8_t>& outputLocal,
                                              const LocalTensor<half>& inputHalf, const LocalTensor<uint8_t>& maskTemp)
    {
        const int64_t vectorCount = (currentNum / static_cast<int64_t>(FAST_VECTOR_GRANULARITY)) *
                                    static_cast<int64_t>(FAST_VECTOR_GRANULARITY);

        if (vectorCount > 0) {
            CompareInputRangeToZero(rowBase + tileCol, static_cast<uint32_t>(vectorCount), CMPMODE::NE, inputHalf,
                                    maskTemp);

            PipeBarrier<PIPE_ALL>();

            UnpackMaskToOutput(maskTemp, 0, static_cast<uint32_t>(vectorCount), outputLocal);
        }

        const int64_t tailCount = currentNum - vectorCount;

        if (tailCount > 0) {
            CompareInputRangeToZeroPadded128(rowBase + tileCol + vectorCount, static_cast<uint32_t>(tailCount),
                                             CMPMODE::NE, inputHalf, maskTemp);

            PipeBarrier<PIPE_ALL>();

            UnpackMaskToOutput(maskTemp, vectorCount, FAST_VECTOR_GRANULARITY, outputLocal);
        }
    }

    __aicore__ inline void ComputeHitMissFastTile(
        const int64_t rowBase, const int64_t rowInPlane, const int64_t tileCol, const int64_t currentNum,
        const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
        const LocalTensor<uint8_t>& outputLocal, const LocalTensor<half>& inputHalf,
        const LocalTensor<uint8_t>& maskAccum, const LocalTensor<uint8_t>& maskTemp,
        const LocalTensor<int8_t>& kernelLocal)
    {
        const int64_t height = tilingData_->axis0Dim;

        const int64_t width = tilingData_->axis1Dim;

        const int64_t planeSize = height * width;

        const int64_t planeIndex = rowBase / planeSize;

        const int64_t planeBase = planeIndex * planeSize;

        const int64_t safeColBegin = tilingData_->anchorW;

        const int64_t safeColEnd = width - (static_cast<int64_t>(tilingData_->kernelW) - 1 - tilingData_->anchorW);

        const bool verticalInterior = rowInPlane >= tilingData_->anchorH &&
                                      rowInPlane < height - (static_cast<int64_t>(tilingData_->kernelH) - 1 -
                                                             tilingData_->anchorH);

        const int64_t tileBegin = tileCol;

        const int64_t tileEnd = tileCol + currentNum;

        if (!verticalInterior) {
            ProcessBorderAwareRange128(rowBase, rowInPlane, planeBase, tileBegin, tileEnd, tileBegin, false,
                                       activeKernelCount, activeList, outputLocal, inputHalf, maskAccum, maskTemp,
                                       kernelLocal);
            return;
        }

        constexpr int64_t WIDE_ROW_MAX_SIDE_PAD = 32;

        const int64_t wideRowRightRadius = static_cast<int64_t>(tilingData_->kernelW) - 1 - tilingData_->anchorW;

        if (activeKernelCount > 0 && activeKernelCount <= FAST_COMPACT_ACTIVE_CAPACITY &&
            width > static_cast<int64_t>(FAST_VECTOR_GRANULARITY) && tileBegin == 0 && currentNum == width &&
            tilingData_->anchorW >= 0 && tilingData_->anchorW <= WIDE_ROW_MAX_SIDE_PAD && wideRowRightRadius >= 0 &&
            wideRowRightRadius <= WIDE_ROW_MAX_SIDE_PAD) {
            if (ComputeWideSparseInteriorRow(planeBase, rowInPlane, width, activeKernelCount, activeList, outputLocal,
                                             inputHalf, maskAccum, maskTemp)) {
                return;
            }
        }

        int64_t safeBeginInTile = tileBegin > safeColBegin ? tileBegin : safeColBegin;

        int64_t safeEndInTile = tileEnd < safeColEnd ? tileEnd : safeColEnd;

        if (safeEndInTile <= safeBeginInTile) {
            ProcessBorderAwareRange128(rowBase, rowInPlane, planeBase, tileBegin, tileEnd, tileBegin, true,
                                       activeKernelCount, activeList, outputLocal, inputHalf, maskAccum, maskTemp,
                                       kernelLocal);
            return;
        }

        constexpr int64_t BLOCK = static_cast<int64_t>(FAST_VECTOR_GRANULARITY);

        const int64_t mainBegin = ((safeBeginInTile + BLOCK - 1) / BLOCK) * BLOCK;

        const int64_t mainEnd = (safeEndInTile / BLOCK) * BLOCK;

        if (mainEnd <= mainBegin) {
            ProcessBorderAwareRange128(rowBase, rowInPlane, planeBase, tileBegin, tileEnd, tileBegin, true,
                                       activeKernelCount, activeList, outputLocal, inputHalf, maskAccum, maskTemp,
                                       kernelLocal);
            return;
        }

        if (tileBegin < mainBegin) {
            ProcessBorderAwareRange128(rowBase, rowInPlane, planeBase, tileBegin, mainBegin, tileBegin, true,
                                       activeKernelCount, activeList, outputLocal, inputHalf, maskAccum, maskTemp,
                                       kernelLocal);
        }

        ComputeVectorInterior(planeBase, rowInPlane, mainBegin, static_cast<uint32_t>(mainEnd - mainBegin),
                              mainBegin - tileBegin, activeKernelCount, activeList, outputLocal, inputHalf, maskAccum,
                              maskTemp, kernelLocal);

        if (mainEnd < tileEnd) {
            ProcessBorderAwareRange128(rowBase, rowInPlane, planeBase, mainEnd, tileEnd, tileBegin, true,
                                       activeKernelCount, activeList, outputLocal, inputHalf, maskAccum, maskTemp,
                                       kernelLocal);
        }
    }

    __aicore__ inline bool CompareInputRangeToZeroBorderAwareWide(
        const int64_t planeBase, const int64_t srcRow, const int64_t width, const int64_t colDelta,
        const uint32_t realCount, const uint32_t executionCount, const bool isNegative, const bool borderMatch,
        const LocalTensor<half>& inputHalf, const LocalTensor<uint8_t>& maskOut)
    {
        constexpr int64_t MAX_SIDE_PAD_ELEMS = 32;

        const int64_t rawBegin = colDelta;

        int64_t leftPadding = rawBegin < 0 ? -rawBegin : 0;

        if (leftPadding > static_cast<int64_t>(realCount)) {
            leftPadding = static_cast<int64_t>(realCount);
        }

        const int64_t validBegin = rawBegin + leftPadding;

        const int64_t remainingAfterLeft = static_cast<int64_t>(realCount) - leftPadding;

        int64_t validCount = 0;

        if (remainingAfterLeft > 0 && validBegin >= 0 && validBegin < width) {
            validCount = width - validBegin;

            if (validCount > remainingAfterLeft) {
                validCount = remainingAfterLeft;
            }
        }

        const int64_t rightPadding = static_cast<int64_t>(realCount) - leftPadding - validCount;

        if (leftPadding > MAX_SIDE_PAD_ELEMS || rightPadding > MAX_SIDE_PAD_ELEMS || validCount <= 0 ||
            executionCount < realCount || executionCount > FAST_TILE_ELEMS) {
            return false;
        }

        SyncFastInputBeforeReuse();
        LocalTensor<uint8_t> inputU8 = fastInputBuf_.Get<uint8_t>();

        if (executionCount > realCount) {
            LocalTensor<half> zeroHalf = inputU8.ReinterpretCast<half>();

            Duplicate(zeroHalf, static_cast<half>(0.0f),
                      static_cast<int32_t>(executionCount / static_cast<uint32_t>(sizeof(half))));

            PipeBarrier<PIPE_ALL>();
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(validCount * static_cast<int64_t>(sizeof(uint8_t)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        const uint8_t padRaw = isNegative ? (borderMatch ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1)) :
                                            (borderMatch ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));

        DataCopyPadExtParams<uint8_t> padParams;

        padParams.isPad = leftPadding > 0 || rightPadding > 0;

        padParams.leftPadding = static_cast<uint8_t>(leftPadding);

        padParams.rightPadding = static_cast<uint8_t>(rightPadding);

        padParams.paddingValue = padRaw;

        const int64_t sourceOffset = planeBase + srcRow * width + validBegin;

        DataCopyPad(inputU8, inputGm_[sourceOffset], copyParams, padParams);

        event_t fastInputMte2VEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);
        WaitFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);

        LocalTensor<uint8_t> inputReady = inputU8;

        PipeBarrier<PIPE_ALL>();

        Cast(inputHalf, inputReady, RoundMode::CAST_NONE, executionCount);

        PipeBarrier<PIPE_V>();

        CompareScalar(maskOut, inputHalf, static_cast<half>(0.0f), isNegative ? CMPMODE::EQ : CMPMODE::NE,
                      executionCount);

        return true;
    }

    __aicore__ inline bool ComputeWideSparseInteriorRow(
        const int64_t planeBase, const int64_t rowInPlane, const int64_t width, const int32_t activeKernelCount,
        const LocalTensor<uint16_t>& activeList, const LocalTensor<uint8_t>& outputLocal,
        const LocalTensor<half>& inputHalf, const LocalTensor<uint8_t>& maskAccum, const LocalTensor<uint8_t>& maskTemp)
    {
        if (width <= 0 || width > static_cast<int64_t>(FAST_TILE_ELEMS) || activeKernelCount <= 0 ||
            activeKernelCount > FAST_COMPACT_ACTIVE_CAPACITY) {
            return false;
        }

        const uint32_t realCount = static_cast<uint32_t>(width);

        const uint32_t executionCount = ((realCount + FAST_VECTOR_GRANULARITY - 1U) / FAST_VECTOR_GRANULARITY) *
                                        FAST_VECTOR_GRANULARITY;

        if (executionCount > FAST_TILE_ELEMS) {
            return false;
        }

        const uint32_t maskU16Count = executionCount / 16U;

        LocalTensor<uint16_t> accumU16 = maskAccum.ReinterpretCast<uint16_t>();

        LocalTensor<uint16_t> tempU16 = maskTemp.ReinterpretCast<uint16_t>();

        const bool borderMatch = tilingData_->borderValue != 0;

        bool hasAccum = false;

        for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
            const uint16_t encoded = activeList.GetValue(activeIndex);

            const int32_t kernelRow = static_cast<int32_t>(encoded & FAST_ACTIVE_COORD_MASK);

            const int32_t kernelCol = static_cast<int32_t>((encoded >> FAST_ACTIVE_COL_SHIFT) & FAST_ACTIVE_COORD_MASK);

            const bool isNegative = (encoded & FAST_ACTIVE_NEGATIVE_FLAG) != 0;

            const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

            const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

            LocalTensor<uint8_t> targetMask = hasAccum ? maskTemp : maskAccum;

            const bool ok = CompareInputRangeToZeroBorderAwareWide(planeBase, srcRow, width, colDelta, realCount,
                                                                   executionCount, isNegative, borderMatch, inputHalf,
                                                                   targetMask);

            if (!ok) {
                return false;
            }

            PipeBarrier<PIPE_V>();

            if (!hasAccum) {
                hasAccum = true;
            } else {
                And(accumU16, accumU16, tempU16, static_cast<int32_t>(maskU16Count));

                PipeBarrier<PIPE_V>();
            }
        }

        if (!hasAccum) {
            return false;
        }

        PipeBarrier<PIPE_ALL>();

        UnpackMaskToOutput(maskAccum, 0, executionCount, outputLocal);

        return true;
    }

    __aicore__ inline void ProcessBorderAwareRange128(
        const int64_t rowBase, const int64_t rowInPlane, const int64_t planeBase, const int64_t rangeBegin,
        const int64_t rangeEnd, const int64_t tileBegin, const bool verticalInterior, const int32_t activeKernelCount,
        const LocalTensor<uint16_t>& activeList, const LocalTensor<uint8_t>& outputLocal,
        const LocalTensor<half>& inputHalf, const LocalTensor<uint8_t>& maskAccum, const LocalTensor<uint8_t>& maskTemp,
        const LocalTensor<int8_t>& kernelLocal)
    {
        constexpr int64_t BLOCK = static_cast<int64_t>(FAST_VECTOR_GRANULARITY);

        for (int64_t blockBegin = rangeBegin; blockBegin < rangeEnd; blockBegin += BLOCK) {
            int64_t remaining = rangeEnd - blockBegin;

            uint32_t realCount = static_cast<uint32_t>(remaining < BLOCK ? remaining : BLOCK);

            ComputeBorderAwareBlock128(rowBase, rowInPlane, planeBase, blockBegin, realCount, blockBegin - tileBegin,
                                       verticalInterior, activeKernelCount, activeList, outputLocal, inputHalf,
                                       maskAccum, maskTemp, kernelLocal);
        }
    }

    __aicore__ inline bool HasVerticalOutOfBoundsConstraint(const int64_t rowInPlane, const int32_t activeKernelCount,
                                                            const LocalTensor<uint16_t>& activeList,
                                                            const LocalTensor<int8_t>& kernelLocal)
    {
        const int64_t height = tilingData_->axis0Dim;

        if (activeKernelCount <= FAST_COMPACT_ACTIVE_CAPACITY) {
            for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
                const uint16_t encoded = activeList.GetValue(activeIndex);

                const int32_t kernelRow = static_cast<int32_t>(encoded & FAST_ACTIVE_COORD_MASK);

                const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

                if (srcRow < 0 || srcRow >= height) {
                    return true;
                }
            }

            return false;
        }

        for (int32_t kernelRow = 0; kernelRow < tilingData_->kernelH; ++kernelRow) {
            const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

            if (srcRow >= 0 && srcRow < height) {
                continue;
            }

            for (int32_t kernelCol = 0; kernelCol < tilingData_->kernelW; ++kernelCol) {
                const int32_t kernelIndex = kernelRow * tilingData_->kernelW + kernelCol;

                const int8_t kernelValue = kernelLocal.GetValue(kernelIndex);

                if (kernelValue == static_cast<int8_t>(1) || kernelValue == static_cast<int8_t>(-1)) {
                    return true;
                }
            }
        }

        return false;
    }

    __aicore__ inline void FillBorderAwareConstant128(const bool value, const int64_t outputLocalOffset,
                                                      const LocalTensor<uint8_t>& outputLocal,
                                                      const LocalTensor<uint8_t>& maskTemp)
    {
        constexpr uint32_t MASK_BYTES = FAST_VECTOR_GRANULARITY / 8U;

        const uint8_t byteValue = value ? static_cast<uint8_t>(0xFF) : static_cast<uint8_t>(0x00);

        for (uint32_t i = 0; i < MASK_BYTES; ++i) {
            maskTemp.SetValue(i, byteValue);
        }

        PipeBarrier<PIPE_ALL>();

        UnpackMaskToOutput(maskTemp, outputLocalOffset, FAST_VECTOR_GRANULARITY, outputLocal);
    }

    __aicore__ inline void ComputeScalarFastBlockFallback(
        const int64_t rowBase, const int64_t rowInPlane, const int64_t outputColBegin, const uint32_t realCount,
        const int64_t outputLocalOffset, const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
        const LocalTensor<uint8_t>& outputLocal, const LocalTensor<int8_t>& kernelLocal)
    {
        for (uint32_t i = 0; i < realCount; ++i) {
            const int64_t col = outputColBegin + static_cast<int64_t>(i);

            const bool value = MatchAtFast2D(rowBase, rowInPlane, col, activeKernelCount, activeList, kernelLocal);

            outputLocal.SetValue(outputLocalOffset + static_cast<int64_t>(i),
                                 value ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
        }
    }

    __aicore__ inline bool TryCompactBatchStage128(
        const int64_t planeBase, const int64_t rowInPlane, const int64_t outputColBegin, const uint32_t realCount,
        const int64_t outputLocalOffset, const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
        const LocalTensor<uint8_t>& outputLocal, const LocalTensor<half>& inputHalf,
        const LocalTensor<uint8_t>& maskAccum, const LocalTensor<uint8_t>& maskTemp)
    {
        constexpr uint32_t PADDED_COUNT = FAST_VECTOR_GRANULARITY;
        constexpr uint32_t SLOT_STRIDE = FAST_VECTOR_GRANULARITY;
        constexpr int64_t MAX_SIDE_PAD_ELEMS = 32;
        constexpr int32_t MASK_U16_COUNT = static_cast<int32_t>((FAST_VECTOR_GRANULARITY / 8U) / sizeof(uint16_t));

        if (activeKernelCount <= 0 || activeKernelCount > FAST_COMPACT_ACTIVE_CAPACITY ||
            static_cast<uint32_t>(activeKernelCount) * SLOT_STRIDE > FAST_TILE_ELEMS) {
            return false;
        }

        const int64_t width = tilingData_->axis1Dim;

        const bool borderMatch = tilingData_->borderValue != 0;

        for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
            const uint16_t encoded = activeList.GetValue(activeIndex);

            const int32_t kernelCol = static_cast<int32_t>((encoded >> FAST_ACTIVE_COL_SHIFT) & FAST_ACTIVE_COORD_MASK);

            const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

            const int64_t rawBegin = outputColBegin + colDelta;

            int64_t leftPadding = rawBegin < 0 ? -rawBegin : 0;

            if (leftPadding > static_cast<int64_t>(realCount)) {
                leftPadding = static_cast<int64_t>(realCount);
            }

            const int64_t validBegin = rawBegin + leftPadding;

            const int64_t remaining = static_cast<int64_t>(realCount) - leftPadding;

            int64_t validCount = 0;

            if (remaining > 0 && validBegin >= 0 && validBegin < width) {
                validCount = width - validBegin;

                if (validCount > remaining) {
                    validCount = remaining;
                }
            }

            const int64_t rightPadding = static_cast<int64_t>(realCount) - leftPadding - validCount;

            if (leftPadding > MAX_SIDE_PAD_ELEMS || rightPadding > MAX_SIDE_PAD_ELEMS || validCount <= 0) {
                return false;
            }
        }

        SyncFastInputBeforeReuse();
        LocalTensor<uint8_t> inputU8 = fastInputBuf_.Get<uint8_t>();

        if (realCount < PADDED_COUNT) {
            LocalTensor<half> zeroHalf = inputU8.ReinterpretCast<half>();

            Duplicate(zeroHalf, static_cast<half>(0.0f),
                      activeKernelCount * static_cast<int32_t>(SLOT_STRIDE / sizeof(half)));

            PipeBarrier<PIPE_ALL>();
        }

        for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
            const uint16_t encoded = activeList.GetValue(activeIndex);

            const int32_t kernelRow = static_cast<int32_t>(encoded & FAST_ACTIVE_COORD_MASK);

            const int32_t kernelCol = static_cast<int32_t>((encoded >> FAST_ACTIVE_COL_SHIFT) & FAST_ACTIVE_COORD_MASK);

            const bool isNegative = (encoded & FAST_ACTIVE_NEGATIVE_FLAG) != 0;

            const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

            const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

            const int64_t rawBegin = outputColBegin + colDelta;

            int64_t leftPadding = rawBegin < 0 ? -rawBegin : 0;

            if (leftPadding > static_cast<int64_t>(realCount)) {
                leftPadding = static_cast<int64_t>(realCount);
            }

            const int64_t validBegin = rawBegin + leftPadding;

            const int64_t remaining = static_cast<int64_t>(realCount) - leftPadding;

            int64_t validCount = 0;

            if (remaining > 0 && validBegin >= 0 && validBegin < width) {
                validCount = width - validBegin;

                if (validCount > remaining) {
                    validCount = remaining;
                }
            }

            const int64_t rightPadding = static_cast<int64_t>(realCount) - leftPadding - validCount;

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(validCount * sizeof(uint8_t));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            const uint8_t padRaw = isNegative ? (borderMatch ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1)) :
                                                (borderMatch ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));

            DataCopyPadExtParams<uint8_t> padParams;

            padParams.isPad = leftPadding > 0 || rightPadding > 0;

            padParams.leftPadding = static_cast<uint8_t>(leftPadding);

            padParams.rightPadding = static_cast<uint8_t>(rightPadding);

            padParams.paddingValue = padRaw;

            const int64_t sourceOffset = planeBase + srcRow * width + validBegin;

            const uint32_t slotOffset = static_cast<uint32_t>(activeIndex) * SLOT_STRIDE;

            DataCopyPad(inputU8[slotOffset], inputGm_[sourceOffset], copyParams, padParams);
        }

        event_t batchMte2VEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        SetFlag<HardEvent::MTE2_V>(batchMte2VEvent);
        WaitFlag<HardEvent::MTE2_V>(batchMte2VEvent);

        PipeBarrier<PIPE_ALL>();

        LocalTensor<uint16_t> accumU16 = maskAccum.ReinterpretCast<uint16_t>();

        LocalTensor<uint16_t> tempU16 = maskTemp.ReinterpretCast<uint16_t>();

        for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
            const uint16_t encoded = activeList.GetValue(activeIndex);

            const bool isNegative = (encoded & FAST_ACTIVE_NEGATIVE_FLAG) != 0;

            const uint32_t slotOffset = static_cast<uint32_t>(activeIndex) * SLOT_STRIDE;

            LocalTensor<uint8_t> inputReady = inputU8[slotOffset];

            Cast(inputHalf, inputReady, RoundMode::CAST_NONE, PADDED_COUNT);

            PipeBarrier<PIPE_V>();

            LocalTensor<uint8_t> targetMask = activeIndex == 0 ? maskAccum : maskTemp;

            CompareScalar(targetMask, inputHalf, static_cast<half>(0.0f), isNegative ? CMPMODE::EQ : CMPMODE::NE,
                          PADDED_COUNT);

            PipeBarrier<PIPE_V>();

            if (activeIndex > 0) {
                And(accumU16, accumU16, tempU16, MASK_U16_COUNT);

                PipeBarrier<PIPE_V>();
            }
        }

        UnpackMaskToOutput(maskAccum, outputLocalOffset, realCount, outputLocal);

        return true;
    }

    __aicore__ inline void ComputeBorderAwareBlock128(
        const int64_t rowBase, const int64_t rowInPlane, const int64_t planeBase, const int64_t outputColBegin,
        const uint32_t realCount, const int64_t outputLocalOffset, const bool verticalInterior,
        const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
        const LocalTensor<uint8_t>& outputLocal, const LocalTensor<half>& inputHalf,
        const LocalTensor<uint8_t>& maskAccum, const LocalTensor<uint8_t>& maskTemp,
        const LocalTensor<int8_t>& kernelLocal)
    {
        constexpr uint32_t PADDED_MASK_U16_COUNT = (FAST_VECTOR_GRANULARITY / 8U) /
                                                   static_cast<uint32_t>(sizeof(uint16_t));

        const int64_t height = tilingData_->axis0Dim;

        const int64_t width = tilingData_->axis1Dim;

        const bool borderMatch = tilingData_->borderValue != 0;

        if (!verticalInterior && !borderMatch &&
            HasVerticalOutOfBoundsConstraint(rowInPlane, activeKernelCount, activeList, kernelLocal)) {
            FillBorderAwareConstant128(false, outputLocalOffset, outputLocal, maskTemp);
            return;
        }

        if (verticalInterior && activeKernelCount <= FAST_COMPACT_ACTIVE_CAPACITY &&
            TryCompactBatchStage128(planeBase, rowInPlane, outputColBegin, realCount, outputLocalOffset,
                                    activeKernelCount, activeList, outputLocal, inputHalf, maskAccum, maskTemp)) {
            return;
        }

        LocalTensor<uint16_t> accumU16 = maskAccum.ReinterpretCast<uint16_t>();

        LocalTensor<uint16_t> tempU16 = maskTemp.ReinterpretCast<uint16_t>();

        bool hasAccum = false;

        if (activeKernelCount <= FAST_COMPACT_ACTIVE_CAPACITY) {
            for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
                const uint16_t encoded = activeList.GetValue(activeIndex);

                const int32_t kernelRow = static_cast<int32_t>(encoded & FAST_ACTIVE_COORD_MASK);

                const int32_t kernelCol = static_cast<int32_t>((encoded >> FAST_ACTIVE_COL_SHIFT) &
                                                               FAST_ACTIVE_COORD_MASK);

                const bool isNegative = (encoded & FAST_ACTIVE_NEGATIVE_FLAG) != 0;

                const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

                if (!verticalInterior && (srcRow < 0 || srcRow >= height)) {
                    if (borderMatch) {
                        continue;
                    }

                    FillBorderAwareConstant128(false, outputLocalOffset, outputLocal, maskTemp);
                    return;
                }

                const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

                LocalTensor<uint8_t> targetMask = hasAccum ? maskTemp : maskAccum;

                const bool vectorizable = CompareInputRangeToZeroBorderAware128(planeBase, srcRow, outputColBegin,
                                                                                colDelta, realCount, isNegative,
                                                                                borderMatch, inputHalf, targetMask);

                if (!vectorizable) {
                    ComputeScalarFastBlockFallback(rowBase, rowInPlane, outputColBegin, realCount, outputLocalOffset,
                                                   activeKernelCount, activeList, outputLocal, kernelLocal);
                    return;
                }

                PipeBarrier<PIPE_V>();

                if (!hasAccum) {
                    hasAccum = true;
                } else {
                    And(accumU16, accumU16, tempU16, static_cast<int32_t>(PADDED_MASK_U16_COUNT));

                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            for (int32_t kernelRow = 0; kernelRow < tilingData_->kernelH; ++kernelRow) {
                const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

                const bool rowInside = srcRow >= 0 && srcRow < height;

                for (int32_t kernelCol = 0; kernelCol < tilingData_->kernelW; ++kernelCol) {
                    const int32_t kernelIndex = kernelRow * tilingData_->kernelW + kernelCol;

                    const int8_t kernelValue = kernelLocal.GetValue(kernelIndex);

                    if (kernelValue != static_cast<int8_t>(1) && kernelValue != static_cast<int8_t>(-1)) {
                        continue;
                    }

                    if (!verticalInterior && !rowInside) {
                        if (borderMatch) {
                            continue;
                        }

                        FillBorderAwareConstant128(false, outputLocalOffset, outputLocal, maskTemp);
                        return;
                    }

                    const bool isNegative = kernelValue == static_cast<int8_t>(-1);

                    const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

                    LocalTensor<uint8_t> targetMask = hasAccum ? maskTemp : maskAccum;

                    const bool vectorizable = CompareInputRangeToZeroBorderAware128(planeBase, srcRow, outputColBegin,
                                                                                    colDelta, realCount, isNegative,
                                                                                    borderMatch, inputHalf, targetMask);

                    if (!vectorizable) {
                        ComputeScalarFastBlockFallback(rowBase, rowInPlane, outputColBegin, realCount,
                                                       outputLocalOffset, activeKernelCount, activeList, outputLocal,
                                                       kernelLocal);
                        return;
                    }

                    PipeBarrier<PIPE_V>();

                    if (!hasAccum) {
                        hasAccum = true;
                    } else {
                        And(accumU16, accumU16, tempU16, static_cast<int32_t>(PADDED_MASK_U16_COUNT));

                        PipeBarrier<PIPE_V>();
                    }
                }
            }
        }

        if (!hasAccum) {
            FillBorderAwareConstant128(true, outputLocalOffset, outputLocal, maskTemp);
            return;
        }

        PipeBarrier<PIPE_ALL>();

        UnpackMaskToOutput(maskAccum, outputLocalOffset, FAST_VECTOR_GRANULARITY, outputLocal);
    }

    // DataCopyPad 单侧显式填充最多 32 字节，超过该范围时必须回退以保证正确性。
    __aicore__ inline bool CompareInputRangeToZeroBorderAware128(const int64_t planeBase, const int64_t srcRow,
                                                                 const int64_t outputColBegin, const int64_t colDelta,
                                                                 const uint32_t realCount, const bool isNegative,
                                                                 const bool borderMatch,
                                                                 const LocalTensor<half>& inputHalf,
                                                                 const LocalTensor<uint8_t>& maskOut)
    {
        constexpr uint32_t PADDED_COUNT = FAST_VECTOR_GRANULARITY;

        constexpr int64_t MAX_SIDE_PAD_ELEMS = 32;

        const int64_t width = tilingData_->axis1Dim;

        const int64_t rawBegin = outputColBegin + colDelta;

        int64_t leftPadding = rawBegin < 0 ? -rawBegin : 0;

        if (leftPadding > static_cast<int64_t>(realCount)) {
            leftPadding = static_cast<int64_t>(realCount);
        }

        const int64_t validBegin = rawBegin + leftPadding;

        const int64_t remainingAfterLeft = static_cast<int64_t>(realCount) - leftPadding;

        int64_t validCount = 0;

        if (remainingAfterLeft > 0 && validBegin >= 0 && validBegin < width) {
            validCount = width - validBegin;

            if (validCount > remainingAfterLeft) {
                validCount = remainingAfterLeft;
            }
        }

        const int64_t rightPadding = static_cast<int64_t>(realCount) - leftPadding - validCount;

        if (leftPadding > MAX_SIDE_PAD_ELEMS || rightPadding > MAX_SIDE_PAD_ELEMS || validCount <= 0) {
            return false;
        }

        SyncFastInputBeforeReuse();
        LocalTensor<uint8_t> inputU8 = fastInputBuf_.Get<uint8_t>();

        if (realCount < PADDED_COUNT) {
            constexpr int32_t PADDED_HALF_COUNT = static_cast<int32_t>(PADDED_COUNT * sizeof(uint8_t) / sizeof(half));

            LocalTensor<half> zeroHalf = inputU8.ReinterpretCast<half>();

            Duplicate(zeroHalf, static_cast<half>(0.0f), PADDED_HALF_COUNT);

            PipeBarrier<PIPE_ALL>();
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(validCount * static_cast<int64_t>(sizeof(uint8_t)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        const uint8_t padRaw = isNegative ? (borderMatch ? static_cast<uint8_t>(0) : static_cast<uint8_t>(1)) :
                                            (borderMatch ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));

        DataCopyPadExtParams<uint8_t> padParams;

        padParams.isPad = leftPadding > 0 || rightPadding > 0;

        padParams.leftPadding = static_cast<uint8_t>(leftPadding);

        padParams.rightPadding = static_cast<uint8_t>(rightPadding);

        padParams.paddingValue = padRaw;

        const int64_t sourceOffset = planeBase + srcRow * width + validBegin;

        DataCopyPad(inputU8, inputGm_[sourceOffset], copyParams, padParams);

        event_t fastInputMte2VEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);
        WaitFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);

        LocalTensor<uint8_t> inputReady = inputU8;

        PipeBarrier<PIPE_ALL>();

        Cast(inputHalf, inputReady, RoundMode::CAST_NONE, PADDED_COUNT);

        PipeBarrier<PIPE_V>();

        CompareScalar(maskOut, inputHalf, static_cast<half>(0.0f), isNegative ? CMPMODE::EQ : CMPMODE::NE,
                      PADDED_COUNT);

        return true;
    }

    __aicore__ inline void ComputeVectorInterior(
        const int64_t planeBase, const int64_t rowInPlane, const int64_t outputColBegin, const uint32_t count,
        const int64_t outputLocalOffset, const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
        const LocalTensor<uint8_t>& outputLocal, const LocalTensor<half>& inputHalf,
        const LocalTensor<uint8_t>& maskAccum, const LocalTensor<uint8_t>& maskTemp,
        const LocalTensor<int8_t>& kernelLocal)
    {
        const uint32_t maskU16Count = count / 16U;

        LocalTensor<uint16_t> accumU16 = maskAccum.ReinterpretCast<uint16_t>();

        LocalTensor<uint16_t> tempU16 = maskTemp.ReinterpretCast<uint16_t>();

        const int64_t width = tilingData_->axis1Dim;

        if (activeKernelCount > 0 && activeKernelCount <= FAST_COMPACT_ACTIVE_CAPACITY) {
            for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
                const uint16_t encoded = activeList.GetValue(activeIndex);

                const int32_t kernelRow = static_cast<int32_t>(encoded & FAST_ACTIVE_COORD_MASK);

                const int32_t kernelCol = static_cast<int32_t>((encoded >> FAST_ACTIVE_COL_SHIFT) &
                                                               FAST_ACTIVE_COORD_MASK);

                const bool isNegative = (encoded & FAST_ACTIVE_NEGATIVE_FLAG) != 0;

                const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

                const int64_t srcCol = outputColBegin + static_cast<int64_t>(kernelCol - tilingData_->anchorW);

                const int64_t sourceOffset = planeBase + srcRow * width + srcCol;

                if (activeIndex == 0) {
                    CompareInputRangeToZero(sourceOffset, count, isNegative ? CMPMODE::EQ : CMPMODE::NE, inputHalf,
                                            maskAccum);

                    PipeBarrier<PIPE_V>();
                } else {
                    CompareInputRangeToZero(sourceOffset, count, isNegative ? CMPMODE::EQ : CMPMODE::NE, inputHalf,
                                            maskTemp);

                    PipeBarrier<PIPE_V>();

                    And(accumU16, accumU16, tempU16, static_cast<int32_t>(maskU16Count));

                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            bool hasAccum = false;

            for (int32_t kernelRow = 0; kernelRow < tilingData_->kernelH; ++kernelRow) {
                const int64_t srcRow = rowInPlane + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

                for (int32_t kernelCol = 0; kernelCol < tilingData_->kernelW; ++kernelCol) {
                    const int32_t kernelIndex = kernelRow * tilingData_->kernelW + kernelCol;

                    const int8_t kernelValue = kernelLocal.GetValue(kernelIndex);

                    if (kernelValue != static_cast<int8_t>(1) && kernelValue != static_cast<int8_t>(-1)) {
                        continue;
                    }

                    const int64_t srcCol = outputColBegin + static_cast<int64_t>(kernelCol - tilingData_->anchorW);

                    const int64_t sourceOffset = planeBase + srcRow * width + srcCol;

                    if (!hasAccum) {
                        CompareInputRangeToZero(sourceOffset, count,
                                                kernelValue == static_cast<int8_t>(1) ? CMPMODE::NE : CMPMODE::EQ,
                                                inputHalf, maskAccum);

                        PipeBarrier<PIPE_V>();
                        hasAccum = true;
                    } else {
                        CompareInputRangeToZero(sourceOffset, count,
                                                kernelValue == static_cast<int8_t>(1) ? CMPMODE::NE : CMPMODE::EQ,
                                                inputHalf, maskTemp);

                        PipeBarrier<PIPE_V>();

                        And(accumU16, accumU16, tempU16, static_cast<int32_t>(maskU16Count));

                        PipeBarrier<PIPE_V>();
                    }
                }
            }
        }

        PipeBarrier<PIPE_ALL>();

        UnpackMaskToOutput(maskAccum, outputLocalOffset, count, outputLocal);
    }

    // 不足 128 个元素的尾块补齐后按 128 个元素执行向量运算，最终只写回真实长度。
    __aicore__ inline void CompareInputRangeToZeroPadded128(const int64_t sourceOffset, const uint32_t realCount,
                                                            const CMPMODE mode, const LocalTensor<half>& inputHalf,
                                                            const LocalTensor<uint8_t>& maskOut)
    {
        constexpr uint32_t PADDED_COUNT = FAST_VECTOR_GRANULARITY;

        SyncFastInputBeforeReuse();
        LocalTensor<uint8_t> inputU8 = fastInputBuf_.Get<uint8_t>();

        constexpr int32_t PADDED_HALF_COUNT = static_cast<int32_t>(PADDED_COUNT * sizeof(uint8_t) / sizeof(half));

        LocalTensor<half> zeroHalf = inputU8.ReinterpretCast<half>();

        Duplicate(zeroHalf, static_cast<half>(0.0f), PADDED_HALF_COUNT);

        PipeBarrier<PIPE_ALL>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = realCount * static_cast<uint32_t>(sizeof(uint8_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(inputU8, inputGm_[sourceOffset], copyParams, {false, 0, 0, 0});

        event_t fastInputMte2VEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);
        WaitFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);
        LocalTensor<uint8_t> inputReady = inputU8;

        PipeBarrier<PIPE_ALL>();

        Cast(inputHalf, inputReady, RoundMode::CAST_NONE, PADDED_COUNT);
        PipeBarrier<PIPE_V>();

        CompareScalar(maskOut, inputHalf, static_cast<half>(0.0f), mode, PADDED_COUNT);
    }

    __aicore__ inline void CompareInputRangeToZero(const int64_t sourceOffset, const uint32_t count, const CMPMODE mode,
                                                   const LocalTensor<half>& inputHalf,
                                                   const LocalTensor<uint8_t>& maskOut)
    {
        SyncFastInputBeforeReuse();
        LocalTensor<uint8_t> inputU8 = fastInputBuf_.Get<uint8_t>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(uint8_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(inputU8, inputGm_[sourceOffset], copyParams, {false, 0, 0, 0});
        event_t fastInputMte2VEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);
        WaitFlag<HardEvent::MTE2_V>(fastInputMte2VEvent);
        LocalTensor<uint8_t> inputReady = inputU8;

        Cast(inputHalf, inputReady, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskOut, inputHalf, static_cast<half>(0.0f), mode, count);
    }

    __aicore__ inline void UnpackMaskToOutput(const LocalTensor<uint8_t>& mask, const int64_t outputOffset,
                                              const uint32_t count, const LocalTensor<uint8_t>& outputLocal)
    {
        LocalTensor<half> expandedHalf = fastHalfBuf_.Get<half>();

        Duplicate(expandedHalf, static_cast<half>(1.0f), static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();

        Select(expandedHalf, mask, expandedHalf, static_cast<half>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();

        Cast(outputLocal[outputOffset], expandedHalf, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyFastOutput(const int64_t globalOffset, const int64_t currentNum)
    {
        LocalTensor<uint8_t> outputLocal = fastOutputQueue_.DeQue<uint8_t>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(uint8_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(outputGm_[globalOffset], outputLocal, copyParams);
        fastOutputQueue_.FreeTensor(outputLocal);
    }

    // 非连续轴布局使用通用路径，以保证任意合法 axes 的计算正确性。
    __aicore__ inline void ProcessGeneric(const int32_t blockIdx, const bool hasActiveKernel,
                                          const LocalTensor<int8_t>& kernelLocal)
    {
        const int64_t start = static_cast<int64_t>(blockIdx) * tilingData_->blockLength;
        if (start >= tilingData_->totalLength) {
            return;
        }

        int64_t end = start + tilingData_->blockLength;
        if (end > tilingData_->totalLength) {
            end = tilingData_->totalLength;
        }

        const int64_t localLength = end - start;
        const int64_t loopCount = (localLength + static_cast<int64_t>(GENERIC_OUTPUT_TILE_BYTES) - 1) /
                                  static_cast<int64_t>(GENERIC_OUTPUT_TILE_BYTES);

        for (int64_t progress = 0; progress < loopCount; ++progress) {
            const int64_t globalOffset = start + progress * static_cast<int64_t>(GENERIC_OUTPUT_TILE_BYTES);
            int64_t currentNum = end - globalOffset;
            if (currentNum > static_cast<int64_t>(GENERIC_OUTPUT_TILE_BYTES)) {
                currentNum = static_cast<int64_t>(GENERIC_OUTPUT_TILE_BYTES);
            }

            ComputeGenericTile(globalOffset, currentNum, hasActiveKernel, kernelLocal);
            CopyGenericOutput(globalOffset, currentNum);
        }
    }

    __aicore__ inline void ComputeGenericTile(const int64_t globalOffset, const int64_t currentNum,
                                              const bool hasActiveKernel, const LocalTensor<int8_t>& kernelLocal)
    {
        LocalTensor<uint8_t> outputLocal = genericOutputQueue_.AllocTensor<uint8_t>();

        for (int64_t localIndex = 0; localIndex < currentNum; ++localIndex) {
            const int64_t outputIndex = globalOffset + localIndex;

            bool value = false;
            if (!hasActiveKernel) {
                value = inputGm_.GetValue(outputIndex) != 0;
            } else {
                value = MatchAt(outputIndex, kernelLocal);
            }

            outputLocal.SetValue(localIndex, value ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
        }

        genericOutputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyGenericOutput(const int64_t globalOffset, const int64_t currentNum)
    {
        LocalTensor<uint8_t> outputLocal = genericOutputQueue_.DeQue<uint8_t>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(uint8_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(outputGm_[globalOffset], outputLocal, copyParams);
        genericOutputQueue_.FreeTensor(outputLocal);
    }

    // 越界位置直接使用 borderValue 作为匹配结果，kernel 为 -1 时也不对该值取反。
    __aicore__ inline bool MatchAtFast2D(const int64_t rowBase, const int64_t rowInPlane, const int64_t col,
                                         const int32_t activeKernelCount, const LocalTensor<uint16_t>& activeList,
                                         const LocalTensor<int8_t>& kernelLocal)
    {
        const int64_t width = tilingData_->axis1Dim;
        const int64_t height = tilingData_->axis0Dim;
        const int64_t outputIndex = rowBase + col;

        const bool borderMatch = tilingData_->borderValue != 0;

        if (activeKernelCount <= FAST_COMPACT_ACTIVE_CAPACITY) {
            for (int32_t activeIndex = 0; activeIndex < activeKernelCount; ++activeIndex) {
                const uint16_t encoded = activeList.GetValue(activeIndex);

                const int32_t kernelRow = static_cast<int32_t>(encoded & FAST_ACTIVE_COORD_MASK);

                const int32_t kernelCol = static_cast<int32_t>((encoded >> FAST_ACTIVE_COL_SHIFT) &
                                                               FAST_ACTIVE_COORD_MASK);

                const bool isNegative = (encoded & FAST_ACTIVE_NEGATIVE_FLAG) != 0;

                const int64_t rowDelta = static_cast<int64_t>(kernelRow - tilingData_->anchorH);

                const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

                const int64_t inputRow = rowInPlane + rowDelta;

                const int64_t inputCol = col + colDelta;

                bool currentMatch = borderMatch;

                const bool inside = inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width;

                if (inside) {
                    const int64_t inputIndex = outputIndex + rowDelta * width + colDelta;

                    const bool inputValue = inputGm_.GetValue(inputIndex) != 0;

                    currentMatch = isNegative ? !inputValue : inputValue;
                }

                if (!currentMatch) {
                    return false;
                }
            }

            return true;
        }

        for (int32_t kernelRow = 0; kernelRow < tilingData_->kernelH; ++kernelRow) {
            const int64_t rowDelta = static_cast<int64_t>(kernelRow - tilingData_->anchorH);

            const int64_t inputRow = rowInPlane + rowDelta;

            const bool rowInside = inputRow >= 0 && inputRow < height;

            const int64_t rowOffset = rowDelta * width;

            for (int32_t kernelCol = 0; kernelCol < tilingData_->kernelW; ++kernelCol) {
                const int32_t kernelIndex = kernelRow * tilingData_->kernelW + kernelCol;

                const int8_t kernelValue = kernelLocal.GetValue(kernelIndex);

                if (kernelValue != static_cast<int8_t>(1) && kernelValue != static_cast<int8_t>(-1)) {
                    continue;
                }

                const int64_t colDelta = static_cast<int64_t>(kernelCol - tilingData_->anchorW);

                const int64_t inputCol = col + colDelta;

                bool currentMatch = borderMatch;

                const bool inside = rowInside && inputCol >= 0 && inputCol < width;

                if (inside) {
                    const int64_t inputIndex = outputIndex + rowOffset + colDelta;

                    const bool inputValue = inputGm_.GetValue(inputIndex) != 0;

                    currentMatch = kernelValue == static_cast<int8_t>(1) ? inputValue : !inputValue;
                }

                if (!currentMatch) {
                    return false;
                }
            }
        }

        return true;
    }

    __aicore__ inline bool MatchAt(const int64_t outputIndex, const LocalTensor<int8_t>& kernelLocal)
    {
        const int64_t axis0Coord = (outputIndex / tilingData_->axis0Stride) % tilingData_->axis0Dim;
        const int64_t axis1Coord = (outputIndex / tilingData_->axis1Stride) % tilingData_->axis1Dim;

        for (int32_t kernelRow = 0; kernelRow < tilingData_->kernelH; ++kernelRow) {
            const int64_t inputAxis0 = axis0Coord + static_cast<int64_t>(kernelRow - tilingData_->anchorH);

            for (int32_t kernelCol = 0; kernelCol < tilingData_->kernelW; ++kernelCol) {
                const int32_t kernelIndex = kernelRow * tilingData_->kernelW + kernelCol;
                const int8_t kernelValue = kernelLocal.GetValue(kernelIndex);

                if (kernelValue == static_cast<int8_t>(0)) {
                    continue;
                }

                if (kernelValue != static_cast<int8_t>(1) && kernelValue != static_cast<int8_t>(-1)) {
                    continue;
                }

                const int64_t inputAxis1 = axis1Coord + static_cast<int64_t>(kernelCol - tilingData_->anchorW);
                bool currentMatch = tilingData_->borderValue != 0;

                const bool inside = inputAxis0 >= 0 && inputAxis0 < tilingData_->axis0Dim && inputAxis1 >= 0 &&
                                    inputAxis1 < tilingData_->axis1Dim;
                if (inside) {
                    const int64_t inputIndex = outputIndex + (inputAxis0 - axis0Coord) * tilingData_->axis0Stride +
                                               (inputAxis1 - axis1Coord) * tilingData_->axis1Stride;
                    const bool inputValue = inputGm_.GetValue(inputIndex) != 0;
                    currentMatch = kernelValue == static_cast<int8_t>(1) ? inputValue : !inputValue;
                }

                if (!currentMatch) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    TPipe pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM> kernelQueue_;

    TBuf<TPosition::VECCALC> fastInputBuf_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> fastOutputQueue_;
    TBuf<TPosition::VECCALC> fastHalfBuf_;
    TBuf<TPosition::VECCALC> fastMaskAccumBuf_;
    TBuf<TPosition::VECCALC> fastMaskTempBuf_;
    TBuf<TPosition::VECCALC> fastActiveListBuf_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> genericOutputQueue_;

    GlobalTensor<uint8_t> inputGm_;
    GlobalTensor<int8_t> kernelGm_;
    GlobalTensor<uint8_t> outputGm_;
    const HitOrMissTilingData* tilingData_ = nullptr;
};

} // namespace NsHitOrMiss

#endif
