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
 * \file adaptive_threshold.h
 * \brief AdaptiveThreshold Ascend C kernel.
 */
#ifndef ADAPTIVE_THRESHOLD_H_
#define ADAPTIVE_THRESHOLD_H_

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_threshold_tiling_data.h"
#include "adaptive_threshold_tiling_key.h"
#include "adaptive_threshold_mean_vector.h"

namespace NsAdaptiveThreshold {

using namespace AscendC;

constexpr uint32_t OUTPUT_BUFFER_NUM = 1;
constexpr uint32_t DMA_ALIGN_BYTES = 32;
constexpr uint32_t BINARY_THRESHOLD = 0;

template <uint32_t adaptiveMethod, typename T>
class KernelAdaptiveThreshold {
public:
    __aicore__ inline KernelAdaptiveThreshold() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AdaptiveThresholdTilingData* tilingData, TPipe* pipePtr)
    {
        pipe = pipePtr;

        height = tilingData->height;
        width = tilingData->width;

        totalRows = tilingData->totalRows;
        tileWidth = tilingData->tileWidth;
        blockSize = tilingData->blockSize;
        radius = tilingData->radius;

        thresholdType = tilingData->thresholdType;
        outputValue = tilingData->outputValue;
        thresholdDelta = tilingData->thresholdDelta;
        gaussianWeights = tilingData->gaussianWeights;

        const uint32_t blockId = GetBlockIdx();
        const uint32_t baseRows = tilingData->baseRows;
        const uint32_t extraRows = tilingData->extraRows;

        myRows = baseRows + (blockId < extraRows ? 1U : 0U);
        startRow = blockId * baseRows + (blockId < extraRows ? blockId : extraRows);

        // Empty tensors and idle cores need no GM binding or UB allocation.
        if (myRows == 0 || totalRows == 0 || width == 0 || height == 0) {
            return;
        }

        if constexpr (adaptiveMethod == ADAPTIVE_THRESHOLD_METHOD_MEAN && std::is_same<T, uint8_t>::value) {
            if (width >= ADAPTIVE_THRESHOLD_MEAN_VECTOR_MIN_WIDTH) {
                vectorMean.Init(x, y, tilingData, pipe);
                return;
            }
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        const uint32_t elementSize = static_cast<uint32_t>(sizeof(T));
        const uint32_t maxWindowWidth = tileWidth + 2U * radius;
        const uint32_t windowBytes = AlignBytes((UseStreamingMean() ? 1U : blockSize) * maxWindowWidth * elementSize);
        const uint32_t outputBytes = AlignBytes(tileWidth * elementSize);
        const uint32_t colSumBytes = AlignBytes(maxWindowWidth * sizeof(int64_t));

        pipe->InitBuffer(inputWindowBuf, windowBytes);
        pipe->InitBuffer(colSumBuf, colSumBytes);
        pipe->InitBuffer(outputQueue, OUTPUT_BUFFER_NUM, outputBytes);
    }

    __aicore__ inline void Process()
    {
        if (myRows == 0 || totalRows == 0 || width == 0 || height == 0) {
            return;
        }

        if constexpr (adaptiveMethod == ADAPTIVE_THRESHOLD_METHOD_MEAN && std::is_same<T, uint8_t>::value) {
            if (width >= ADAPTIVE_THRESHOLD_MEAN_VECTOR_MIN_WIDTH) {
                vectorMean.Process();
                return;
            }
        }

        if constexpr (adaptiveMethod == ADAPTIVE_THRESHOLD_METHOD_MEAN) {
            for (uint32_t tileStart = 0; tileStart < width; tileStart += tileWidth) {
                const uint32_t currentWidth = MinU32(tileWidth, width - tileStart);

                ProcessColumnTileMean(tileStart, currentWidth);
            }
        } else {
            for (uint32_t tileStart = 0; tileStart < width; tileStart += tileWidth) {
                const uint32_t currentWidth = MinU32(tileWidth, width - tileStart);

                ProcessColumnTileGaussian(tileStart, currentWidth);
            }
        }
    }

private:
    __aicore__ inline static uint32_t AlignBytes(uint32_t byteCount)
    {
        return ((byteCount + DMA_ALIGN_BYTES - 1U) / DMA_ALIGN_BYTES) * DMA_ALIGN_BYTES;
    }

    __aicore__ inline static uint32_t MinU32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline static uint32_t ClampCoordinate(int64_t coordinate, uint32_t limit)
    {
        if (coordinate < 0) {
            return 0;
        }

        const uint64_t unsignedCoordinate = static_cast<uint64_t>(coordinate);

        if (unsignedCoordinate >= limit) {
            return limit - 1U;
        }

        return static_cast<uint32_t>(unsignedCoordinate);
    }

    __aicore__ inline static float ToFloat(T value)
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            return static_cast<float>(static_cast<int32_t>(value));
        } else {
            return static_cast<float>(value);
        }
    }

    __aicore__ inline static T FromFloat(float value)
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            return static_cast<T>(static_cast<int32_t>(value));
        } else {
            return static_cast<T>(value);
        }
    }
    // Stream only when every neighborhood covers all image rows. Taller images
    // keep the ring buffer to avoid additional GM reads on each vertical update.
    __aicore__ inline bool UseStreamingMean() const
    {
        return adaptiveMethod == ADAPTIVE_THRESHOLD_METHOD_MEAN && std::is_same<T, uint8_t>::value &&
               blockSize >= ADAPTIVE_THRESHOLD_MEAN_STREAM_MIN_BLOCK_SIZE && radius >= height - 1U;
    }

    __aicore__ inline void InitStreamingColSum(uint32_t imageIndex, uint32_t rowInImage, uint32_t tileStart,
                                               uint32_t windowWidth, LocalTensor<float>& colSum)
    {
        const int64_t firstCandidate = static_cast<int64_t>(rowInImage) - radius;
        const int64_t lastCandidate = static_cast<int64_t>(rowInImage) + radius;
        const uint32_t firstRow = ClampCoordinate(firstCandidate, height);
        const uint32_t lastRow = ClampCoordinate(lastCandidate, height);
        const uint32_t topCopies = firstCandidate < 0 ? static_cast<uint32_t>(-firstCandidate) : 0U;
        const uint32_t bottomCopies = lastCandidate >= height ? static_cast<uint32_t>(lastCandidate - height + 1) : 0U;
        const uint64_t imageBase = static_cast<uint64_t>(imageIndex) * height * width;

        // UINT8 column sums stay exact (<= 255 * 255). Visit GM in row order,
        // count replicated border rows once, and retain all halo columns.
        for (uint32_t col = 0; col < windowWidth; ++col) {
            colSum.SetValue(col, 0.0F);
        }
        for (uint32_t row = firstRow; row <= lastRow; ++row) {
            const uint64_t rowBase = imageBase + static_cast<uint64_t>(row) * width;
            const int32_t copies = static_cast<int32_t>(1U + (row == 0 ? topCopies : 0U) +
                                                        (row == height - 1U ? bottomCopies : 0U));
            for (uint32_t col = 0; col < windowWidth; ++col) {
                const uint32_t sourceColumn = ClampCoordinate(static_cast<int64_t>(tileStart) + col - radius, width);
                const int32_t contribution = copies * static_cast<int32_t>(xGm.GetValue(rowBase + sourceColumn));
                colSum.SetValue(col, colSum.GetValue(col) + static_cast<float>(contribution));
            }
        }
    }

    __aicore__ inline void UpdateStreamingColSum(uint32_t imageIndex, uint32_t rowInImage, uint32_t tileStart,
                                                 uint32_t windowWidth, LocalTensor<float>& colSum)
    {
        const uint32_t leavingRow = ClampCoordinate(static_cast<int64_t>(rowInImage) - radius - 1, height);
        const uint32_t enteringRow = ClampCoordinate(static_cast<int64_t>(rowInImage) + radius, height);
        const uint64_t imageBase = static_cast<uint64_t>(imageIndex) * height * width;
        const uint64_t leavingBase = imageBase + static_cast<uint64_t>(leavingRow) * width;
        const uint64_t enteringBase = imageBase + static_cast<uint64_t>(enteringRow) * width;
        for (uint32_t col = 0; col < windowWidth; ++col) {
            const uint32_t sourceColumn = ClampCoordinate(static_cast<int64_t>(tileStart) + col - radius, width);
            const float delta = ToFloat(xGm.GetValue(enteringBase + sourceColumn)) -
                                ToFloat(xGm.GetValue(leavingBase + sourceColumn));
            colSum.SetValue(col, colSum.GetValue(col) + delta);
        }
    }

    __aicore__ inline void ProcessColumnTileMeanStreaming(uint32_t tileStart, uint32_t currentWidth)
    {
        const uint32_t windowWidth = currentWidth + 2U * radius;
        LocalTensor<T> inputRow = inputWindowBuf.template Get<T>();
        LocalTensor<float> colSum = colSumBuf.template Get<float>();
        uint32_t prevImageIndex = 0xFFFFFFFFU;

        for (uint32_t localRow = 0; localRow < myRows; ++localRow) {
            const uint32_t flattenedRow = startRow + localRow;
            const uint32_t imageIndex = flattenedRow / height;
            const uint32_t rowInImage = flattenedRow % height;
            if (localRow == 0 || imageIndex != prevImageIndex) {
                InitStreamingColSum(imageIndex, rowInImage, tileStart, windowWidth, colSum);
            } else {
                UpdateStreamingColSum(imageIndex, rowInImage, tileStart, windowWidth, colSum);
            }
            LoadRow(imageIndex, rowInImage, tileStart, windowWidth, 0, inputRow);
            LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
            ProcessRowMean(inputRow, colSum, outputLocal, windowWidth, currentWidth, 0);
            outputQueue.EnQue(outputLocal);
            CopyOut(imageIndex, rowInImage, tileStart, currentWidth);
            prevImageIndex = imageIndex;
        }
    }

    __aicore__ inline void ProcessColumnTileMean(uint32_t tileStart, uint32_t currentWidth)
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            if (UseStreamingMean()) {
                ProcessColumnTileMeanStreaming(tileStart, currentWidth);
                return;
            }
        }

        const uint32_t windowWidth = currentWidth + 2U * radius;

        LocalTensor<T> inputWindow = inputWindowBuf.template Get<T>();
        LocalTensor<float> colSum = colSumBuf.template Get<float>();

        uint32_t rowOffset = 0;
        uint32_t prevImageIndex = 0xFFFFFFFFu;

        for (uint32_t localRow = 0; localRow < myRows; ++localRow) {
            const uint32_t flattenedRow = startRow + localRow;
            const uint32_t imageIndex = flattenedRow / height;
            const uint32_t rowInImage = flattenedRow % height;

            if (localRow == 0 || imageIndex != prevImageIndex) {
                LoadInputWindow(imageIndex, rowInImage, tileStart, windowWidth, currentWidth, inputWindow);

                ComputeColSum(inputWindow, colSum, windowWidth);
                rowOffset = 0;
            } else {
                const int64_t enteringRow = static_cast<int64_t>(rowInImage) + static_cast<int64_t>(radius);

                LoadAndUpdateRow(imageIndex, enteringRow, tileStart, windowWidth, rowOffset, inputWindow, colSum);

                rowOffset = (rowOffset + 1U) % blockSize;
            }

            LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

            const uint32_t centerRow = (rowOffset + radius) % blockSize;

            ProcessRowMean(inputWindow, colSum, outputLocal, windowWidth, currentWidth, centerRow);

            outputQueue.EnQue(outputLocal);
            CopyOut(imageIndex, rowInImage, tileStart, currentWidth);

            prevImageIndex = imageIndex;
        }
    }

    __aicore__ inline void ProcessColumnTileGaussian(uint32_t tileStart, uint32_t currentWidth)
    {
        const uint32_t windowWidth = currentWidth + 2U * radius;

        LocalTensor<T> inputWindow = inputWindowBuf.template Get<T>();

        uint32_t rowOffset = 0;
        uint32_t prevImageIndex = 0xFFFFFFFFu;

        for (uint32_t localRow = 0; localRow < myRows; ++localRow) {
            const uint32_t flattenedRow = startRow + localRow;
            const uint32_t imageIndex = flattenedRow / height;
            const uint32_t rowInImage = flattenedRow % height;

            if (localRow == 0 || imageIndex != prevImageIndex) {
                LoadInputWindow(imageIndex, rowInImage, tileStart, windowWidth, currentWidth, inputWindow);

                ComputeWeightedColSum(inputWindow, windowWidth, 0);
                rowOffset = 0;
            } else {
                const int64_t enteringRow = static_cast<int64_t>(rowInImage) + static_cast<int64_t>(radius);

                LoadRow(imageIndex, enteringRow, tileStart, windowWidth, rowOffset, inputWindow);

                rowOffset = (rowOffset + 1U) % blockSize;

                ComputeWeightedColSum(inputWindow, windowWidth, rowOffset);
            }

            LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

            const uint32_t centerRow = (rowOffset + radius) % blockSize;

            ProcessGaussianRow(inputWindow, outputLocal, windowWidth, currentWidth, centerRow);

            outputQueue.EnQue(outputLocal);
            CopyOut(imageIndex, rowInImage, tileStart, currentWidth);

            prevImageIndex = imageIndex;
        }
    }

    __aicore__ inline void ComputeWeightedColSum(const LocalTensor<T>& inputWindow, uint32_t windowWidth,
                                                 uint32_t weightBase)
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            LocalTensor<int64_t> wcolSum = colSumBuf.template Get<int64_t>();
            ComputeWeightedColSumUint8(inputWindow, wcolSum, windowWidth, weightBase);
        } else {
            LocalTensor<float> wcolSum = colSumBuf.template Get<float>();
            ComputeWeightedColSumFloat(inputWindow, wcolSum, windowWidth, weightBase);
        }
    }

    __aicore__ inline void ProcessGaussianRow(const LocalTensor<T>& inputWindow, LocalTensor<T>& outputLocal,
                                              uint32_t windowWidth, uint32_t currentWidth, uint32_t centerRow)
    {
        if constexpr (std::is_same<T, uint8_t>::value) {
            LocalTensor<int64_t> wcolSum = colSumBuf.template Get<int64_t>();

            constexpr uint64_t denominator = static_cast<uint64_t>(ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE) *
                                             ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE;

            for (uint32_t outputColumn = 0; outputColumn < currentWidth; ++outputColumn) {
                // Symmetric folding of the horizontal weighted sum:
                // kernel columns k and blockSize-1-k share one weight,
                // so their column sums are added before the multiply.
                // Bit-identical to the unfolded integer computation.
                uint64_t weightedSum = static_cast<uint64_t>(gaussianWeights[radius]) *
                                       static_cast<uint64_t>(wcolSum.GetValue(outputColumn + radius));

                uint32_t kernelColumn = 0;
                for (; kernelColumn + 1U < radius; kernelColumn += 2U) {
                    weightedSum += static_cast<uint64_t>(gaussianWeights[kernelColumn]) *
                                   static_cast<uint64_t>(
                                       wcolSum.GetValue(outputColumn + kernelColumn) +
                                       wcolSum.GetValue(outputColumn + blockSize - 1U - kernelColumn));
                    weightedSum += static_cast<uint64_t>(gaussianWeights[kernelColumn + 1U]) *
                                   static_cast<uint64_t>(
                                       wcolSum.GetValue(outputColumn + kernelColumn + 1U) +
                                       wcolSum.GetValue(outputColumn + blockSize - 2U - kernelColumn));
                }
                for (; kernelColumn < radius; ++kernelColumn) {
                    weightedSum += static_cast<uint64_t>(gaussianWeights[kernelColumn]) *
                                   static_cast<uint64_t>(
                                       wcolSum.GetValue(outputColumn + kernelColumn) +
                                       wcolSum.GetValue(outputColumn + blockSize - 1U - kernelColumn));
                }

                const float localMean = static_cast<float>(
                    static_cast<int32_t>((weightedSum + denominator / 2ULL) / denominator));

                WriteOutput(inputWindow, outputLocal, windowWidth, outputColumn, centerRow, localMean);
            }
        } else {
            LocalTensor<float> wcolSum = colSumBuf.template Get<float>();

            constexpr float weightScale = 65536.0F;

            for (uint32_t outputColumn = 0; outputColumn < currentWidth; ++outputColumn) {
                // Symmetric folding of the horizontal weighted sum for
                // the float path (see the uint8 branch above).
                float weightedSum = static_cast<float>(static_cast<int32_t>(gaussianWeights[radius])) / weightScale *
                                    wcolSum.GetValue(outputColumn + radius);

                uint32_t kernelColumn = 0;
                for (; kernelColumn + 1U < radius; kernelColumn += 2U) {
                    const float columnWeight = static_cast<float>(static_cast<int32_t>(gaussianWeights[kernelColumn])) /
                                               weightScale;
                    weightedSum += columnWeight * (wcolSum.GetValue(outputColumn + kernelColumn) +
                                                   wcolSum.GetValue(outputColumn + blockSize - 1U - kernelColumn));

                    const float columnWeight1 = static_cast<float>(
                                                    static_cast<int32_t>(gaussianWeights[kernelColumn + 1U])) /
                                                weightScale;
                    weightedSum += columnWeight1 * (wcolSum.GetValue(outputColumn + kernelColumn + 1U) +
                                                    wcolSum.GetValue(outputColumn + blockSize - 2U - kernelColumn));
                }
                for (; kernelColumn < radius; ++kernelColumn) {
                    const float columnWeight = static_cast<float>(static_cast<int32_t>(gaussianWeights[kernelColumn])) /
                                               weightScale;
                    weightedSum += columnWeight * (wcolSum.GetValue(outputColumn + kernelColumn) +
                                                   wcolSum.GetValue(outputColumn + blockSize - 1U - kernelColumn));
                }

                WriteOutput(inputWindow, outputLocal, windowWidth, outputColumn, centerRow, weightedSum);
            }
        }
    }

    __aicore__ inline void ProcessRowMean(const LocalTensor<T>& inputWindow, const LocalTensor<float>& colSum,
                                          LocalTensor<T>& outputLocal, uint32_t windowWidth, uint32_t currentWidth,
                                          uint32_t centerRow)
    {
        const float elementCount = static_cast<float>(static_cast<int32_t>(blockSize) *
                                                      static_cast<int32_t>(blockSize));

        // LoadInputWindow already clamps halo coordinates to the image boundary.
        // Include those real neighboring columns at internal tile boundaries.
        float windowSum = 0.0F;
        for (uint32_t col = 0; col < blockSize; ++col) {
            windowSum += colSum.GetValue(col);
        }

        WriteMeanPixel(inputWindow, outputLocal, windowWidth, centerRow, 0, windowSum, elementCount);
        for (uint32_t oc = 1; oc < currentWidth; ++oc) {
            windowSum -= colSum.GetValue(oc - 1U);
            windowSum += colSum.GetValue(oc + blockSize - 1U);
            WriteMeanPixel(inputWindow, outputLocal, windowWidth, centerRow, oc, windowSum, elementCount);
        }
    }

    __aicore__ inline void WriteMeanPixel(const LocalTensor<T>& inputWindow, LocalTensor<T>& outputLocal,
                                          uint32_t windowWidth, uint32_t centerRow, uint32_t outputColumn,
                                          float windowSum, float elementCount)
    {
        float localMean = 0.0F;
        if constexpr (std::is_same<T, uint8_t>::value) {
            // UINT8 window sums are exact in float; round using integers to preserve half-area boundaries.
            const int32_t integerSum = static_cast<int32_t>(windowSum);
            const int32_t area = static_cast<int32_t>(blockSize * blockSize);
            localMean = static_cast<float>((integerSum + area / 2) / area);
        } else {
            localMean = windowSum / elementCount;
        }

        const uint32_t centerIndex = centerRow * windowWidth + outputColumn + radius;

        const float sourceValue = ToFloat(inputWindow.GetValue(centerIndex));

        const bool aboveThreshold = sourceValue > localMean - thresholdDelta;

        T result = static_cast<T>(0);
        if (thresholdType == BINARY_THRESHOLD) {
            result = aboveThreshold ? FromFloat(outputValue) : static_cast<T>(0);
        } else {
            result = aboveThreshold ? static_cast<T>(0) : FromFloat(outputValue);
        }

        outputLocal.SetValue(outputColumn, result);
    }

    __aicore__ inline void WriteOutput(const LocalTensor<T>& inputWindow, LocalTensor<T>& outputLocal,
                                       uint32_t windowWidth, uint32_t outputColumn, uint32_t centerRow, float localMean)
    {
        const uint32_t centerIndex = centerRow * windowWidth + outputColumn + radius;

        const float sourceValue = ToFloat(inputWindow.GetValue(centerIndex));

        const bool aboveThreshold = sourceValue > localMean - thresholdDelta;

        T result = static_cast<T>(0);
        if (thresholdType == BINARY_THRESHOLD) {
            result = aboveThreshold ? FromFloat(outputValue) : static_cast<T>(0);
        } else {
            result = aboveThreshold ? static_cast<T>(0) : FromFloat(outputValue);
        }

        outputLocal.SetValue(outputColumn, result);
    }

    __aicore__ inline void ComputeWeightedColSumUint8(const LocalTensor<T>& inputWindow, LocalTensor<int64_t>& wcolSum,
                                                      uint32_t windowWidth, uint32_t weightBase)
    {
        // Symmetric weight folding: gaussianWeights[j] ==
        // gaussianWeights[blockSize-1-j], so the rows holding weight
        // index j and blockSize-1-j share one weight. Grouping their
        // pixel values halves the multiply count. The integer sum is
        // bit-identical to the unfolded computation. The pair loop is
        // then unrolled over 4 columns per iteration to expose ILP.
        uint32_t centerRow = weightBase + radius;
        if (centerRow >= blockSize) {
            centerRow -= blockSize;
        }

        const uint32_t columns4 = windowWidth / 4U;
        const uint32_t remainderStart = columns4 * 4U;
        const uint32_t centerRowBase = centerRow * windowWidth;
        const int64_t centerWeight = static_cast<int64_t>(gaussianWeights[radius]);

        for (uint32_t colGroup = 0; colGroup < columns4; ++colGroup) {
            const uint32_t baseCol = colGroup * 4U;

            int64_t sum0 = centerWeight * static_cast<int64_t>(inputWindow.GetValue(centerRowBase + baseCol));
            int64_t sum1 = centerWeight * static_cast<int64_t>(inputWindow.GetValue(centerRowBase + baseCol + 1U));
            int64_t sum2 = centerWeight * static_cast<int64_t>(inputWindow.GetValue(centerRowBase + baseCol + 2U));
            int64_t sum3 = centerWeight * static_cast<int64_t>(inputWindow.GetValue(centerRowBase + baseCol + 3U));

            for (uint32_t j = 0; j < radius; ++j) {
                uint32_t rowLow = weightBase + j;
                if (rowLow >= blockSize) {
                    rowLow -= blockSize;
                }

                uint32_t rowHigh = weightBase + blockSize - 1U - j;
                if (rowHigh >= blockSize) {
                    rowHigh -= blockSize;
                }

                const int64_t weight = static_cast<int64_t>(gaussianWeights[j]);
                const uint32_t lowBase = rowLow * windowWidth;
                const uint32_t highBase = rowHigh * windowWidth;

                sum0 += weight * (static_cast<int64_t>(inputWindow.GetValue(lowBase + baseCol)) +
                                  static_cast<int64_t>(inputWindow.GetValue(highBase + baseCol)));
                sum1 += weight * (static_cast<int64_t>(inputWindow.GetValue(lowBase + baseCol + 1U)) +
                                  static_cast<int64_t>(inputWindow.GetValue(highBase + baseCol + 1U)));
                sum2 += weight * (static_cast<int64_t>(inputWindow.GetValue(lowBase + baseCol + 2U)) +
                                  static_cast<int64_t>(inputWindow.GetValue(highBase + baseCol + 2U)));
                sum3 += weight * (static_cast<int64_t>(inputWindow.GetValue(lowBase + baseCol + 3U)) +
                                  static_cast<int64_t>(inputWindow.GetValue(highBase + baseCol + 3U)));
            }

            wcolSum.SetValue(baseCol, sum0);
            wcolSum.SetValue(baseCol + 1U, sum1);
            wcolSum.SetValue(baseCol + 2U, sum2);
            wcolSum.SetValue(baseCol + 3U, sum3);
        }

        for (uint32_t col = remainderStart; col < windowWidth; ++col) {
            int64_t sum = centerWeight * static_cast<int64_t>(inputWindow.GetValue(centerRowBase + col));

            for (uint32_t j = 0; j < radius; ++j) {
                uint32_t rowLow = weightBase + j;
                if (rowLow >= blockSize) {
                    rowLow -= blockSize;
                }

                uint32_t rowHigh = weightBase + blockSize - 1U - j;
                if (rowHigh >= blockSize) {
                    rowHigh -= blockSize;
                }

                const int64_t pair = static_cast<int64_t>(inputWindow.GetValue(rowLow * windowWidth + col)) +
                                     static_cast<int64_t>(inputWindow.GetValue(rowHigh * windowWidth + col));

                sum += static_cast<int64_t>(gaussianWeights[j]) * pair;
            }
            wcolSum.SetValue(col, sum);
        }
    }

    __aicore__ inline void ComputeWeightedColSumFloat(const LocalTensor<T>& inputWindow, LocalTensor<float>& wcolSum,
                                                      uint32_t windowWidth, uint32_t weightBase)
    {
        constexpr float weightScale = 65536.0F;

        uint32_t centerRow = weightBase + radius;
        if (centerRow >= blockSize) {
            centerRow -= blockSize;
        }

        const uint32_t columns4 = windowWidth / 4U;
        const uint32_t remainderStart = columns4 * 4U;
        const uint32_t centerRowBase = centerRow * windowWidth;
        const float centerWeight = static_cast<float>(static_cast<int32_t>(gaussianWeights[radius])) / weightScale;

        for (uint32_t colGroup = 0; colGroup < columns4; ++colGroup) {
            const uint32_t baseCol = colGroup * 4U;

            float sum0 = centerWeight * static_cast<float>(inputWindow.GetValue(centerRowBase + baseCol));
            float sum1 = centerWeight * static_cast<float>(inputWindow.GetValue(centerRowBase + baseCol + 1U));
            float sum2 = centerWeight * static_cast<float>(inputWindow.GetValue(centerRowBase + baseCol + 2U));
            float sum3 = centerWeight * static_cast<float>(inputWindow.GetValue(centerRowBase + baseCol + 3U));

            for (uint32_t j = 0; j < radius; ++j) {
                uint32_t rowLow = weightBase + j;
                if (rowLow >= blockSize) {
                    rowLow -= blockSize;
                }

                uint32_t rowHigh = weightBase + blockSize - 1U - j;
                if (rowHigh >= blockSize) {
                    rowHigh -= blockSize;
                }

                const float weight = static_cast<float>(static_cast<int32_t>(gaussianWeights[j])) / weightScale;
                const uint32_t lowBase = rowLow * windowWidth;
                const uint32_t highBase = rowHigh * windowWidth;

                sum0 += weight * (static_cast<float>(inputWindow.GetValue(lowBase + baseCol)) +
                                  static_cast<float>(inputWindow.GetValue(highBase + baseCol)));
                sum1 += weight * (static_cast<float>(inputWindow.GetValue(lowBase + baseCol + 1U)) +
                                  static_cast<float>(inputWindow.GetValue(highBase + baseCol + 1U)));
                sum2 += weight * (static_cast<float>(inputWindow.GetValue(lowBase + baseCol + 2U)) +
                                  static_cast<float>(inputWindow.GetValue(highBase + baseCol + 2U)));
                sum3 += weight * (static_cast<float>(inputWindow.GetValue(lowBase + baseCol + 3U)) +
                                  static_cast<float>(inputWindow.GetValue(highBase + baseCol + 3U)));
            }

            wcolSum.SetValue(baseCol, sum0);
            wcolSum.SetValue(baseCol + 1U, sum1);
            wcolSum.SetValue(baseCol + 2U, sum2);
            wcolSum.SetValue(baseCol + 3U, sum3);
        }

        for (uint32_t col = remainderStart; col < windowWidth; ++col) {
            float sum = centerWeight * static_cast<float>(inputWindow.GetValue(centerRowBase + col));

            for (uint32_t j = 0; j < radius; ++j) {
                uint32_t rowLow = weightBase + j;
                if (rowLow >= blockSize) {
                    rowLow -= blockSize;
                }

                uint32_t rowHigh = weightBase + blockSize - 1U - j;
                if (rowHigh >= blockSize) {
                    rowHigh -= blockSize;
                }

                const float pair = static_cast<float>(inputWindow.GetValue(rowLow * windowWidth + col)) +
                                   static_cast<float>(inputWindow.GetValue(rowHigh * windowWidth + col));

                sum += static_cast<float>(static_cast<int32_t>(gaussianWeights[j])) / weightScale * pair;
            }
            wcolSum.SetValue(col, sum);
        }
    }

    __aicore__ inline void ComputeColSum(const LocalTensor<T>& inputWindow, LocalTensor<float>& colSum,
                                         uint32_t windowWidth)
    {
        const uint32_t columns4 = windowWidth / 4U;
        const uint32_t remainderStart = columns4 * 4U;

        for (uint32_t col = 0; col < columns4; ++col) {
            const uint32_t baseCol = col * 4U;
            float sum0 = 0.0F;
            float sum1 = 0.0F;
            float sum2 = 0.0F;
            float sum3 = 0.0F;
            for (uint32_t row = 0; row < blockSize; ++row) {
                const uint32_t rowOffset = row * windowWidth;
                sum0 += ToFloat(inputWindow.GetValue(rowOffset + baseCol));
                sum1 += ToFloat(inputWindow.GetValue(rowOffset + baseCol + 1U));
                sum2 += ToFloat(inputWindow.GetValue(rowOffset + baseCol + 2U));
                sum3 += ToFloat(inputWindow.GetValue(rowOffset + baseCol + 3U));
            }
            colSum.SetValue(baseCol, sum0);
            colSum.SetValue(baseCol + 1U, sum1);
            colSum.SetValue(baseCol + 2U, sum2);
            colSum.SetValue(baseCol + 3U, sum3);
        }

        for (uint32_t col = remainderStart; col < windowWidth; ++col) {
            float sum = 0.0F;
            for (uint32_t row = 0; row < blockSize; ++row) {
                sum += ToFloat(inputWindow.GetValue(row * windowWidth + col));
            }
            colSum.SetValue(col, sum);
        }
    }

    __aicore__ inline void LoadAndUpdateRow(uint32_t imageIndex, int64_t sourceRowInImage, uint32_t tileStart,
                                            uint32_t windowWidth, uint32_t bufferRow, const LocalTensor<T>& inputWindow,
                                            LocalTensor<float>& colSum)
    {
        const uint64_t imageStride = static_cast<uint64_t>(height) * width;
        const uint64_t imageBase = static_cast<uint64_t>(imageIndex) * imageStride;

        const uint32_t sourceRow = ClampCoordinate(sourceRowInImage, height);

        const uint64_t rowBase = imageBase + static_cast<uint64_t>(sourceRow) * width;

        for (uint32_t windowColumn = 0; windowColumn < windowWidth; ++windowColumn) {
            const int64_t sourceColumnCandidate = static_cast<int64_t>(tileStart) + static_cast<int64_t>(windowColumn) -
                                                  static_cast<int64_t>(radius);

            const uint32_t sourceColumn = ClampCoordinate(sourceColumnCandidate, width);

            const T newValue = xGm.GetValue(rowBase + sourceColumn);

            const uint32_t bufferIndex = bufferRow * windowWidth + windowColumn;

            const float delta = ToFloat(newValue) - ToFloat(inputWindow.GetValue(bufferIndex));

            inputWindow.SetValue(bufferIndex, newValue);

            colSum.SetValue(windowColumn, colSum.GetValue(windowColumn) + delta);
        }
    }

    __aicore__ inline void LoadRow(uint32_t imageIndex, int64_t sourceRowInImage, uint32_t tileStart,
                                   uint32_t windowWidth, uint32_t bufferRow, const LocalTensor<T>& inputWindow)
    {
        const uint64_t imageStride = static_cast<uint64_t>(height) * width;
        const uint64_t imageBase = static_cast<uint64_t>(imageIndex) * imageStride;

        const uint32_t sourceRow = ClampCoordinate(sourceRowInImage, height);

        const uint64_t rowBase = imageBase + static_cast<uint64_t>(sourceRow) * width;

        for (uint32_t windowColumn = 0; windowColumn < windowWidth; ++windowColumn) {
            const int64_t sourceColumnCandidate = static_cast<int64_t>(tileStart) + static_cast<int64_t>(windowColumn) -
                                                  static_cast<int64_t>(radius);

            const uint32_t sourceColumn = ClampCoordinate(sourceColumnCandidate, width);

            const T value = xGm.GetValue(rowBase + sourceColumn);

            inputWindow.SetValue(bufferRow * windowWidth + windowColumn, value);
        }
    }

    __aicore__ inline void LoadInputWindow(uint32_t imageIndex, uint32_t rowInImage, uint32_t tileStart,
                                           uint32_t windowWidth, uint32_t currentWidth,
                                           const LocalTensor<T>& inputWindow)
    {
        const uint64_t imageStride = static_cast<uint64_t>(height) * width;
        const uint64_t imageBase = static_cast<uint64_t>(imageIndex) * imageStride;

        for (uint32_t kernelRow = 0; kernelRow < blockSize; ++kernelRow) {
            const int64_t sourceRowCandidate = static_cast<int64_t>(rowInImage) + static_cast<int64_t>(kernelRow) -
                                               static_cast<int64_t>(radius);

            const uint32_t sourceRow = ClampCoordinate(sourceRowCandidate, height);

            const uint64_t rowBase = imageBase + static_cast<uint64_t>(sourceRow) * width;

            for (uint32_t windowColumn = 0; windowColumn < windowWidth; ++windowColumn) {
                const int64_t sourceColumnCandidate = static_cast<int64_t>(tileStart) +
                                                      static_cast<int64_t>(windowColumn) - static_cast<int64_t>(radius);

                const uint32_t sourceColumn = ClampCoordinate(sourceColumnCandidate, width);

                const T value = xGm.GetValue(rowBase + sourceColumn);

                inputWindow.SetValue(kernelRow * windowWidth + windowColumn, value);
            }
        }
    }

    __aicore__ inline void CopyOut(uint32_t imageIndex, uint32_t rowInImage, uint32_t tileStart, uint32_t currentWidth)
    {
        LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();

        const uint64_t outputOffset = (static_cast<uint64_t>(imageIndex) * height + rowInImage) * width + tileStart;

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentWidth * sizeof(T)), 0, 0, 0};

        DataCopyPad(yGm[outputOffset], outputLocal, copyParams);

        outputQueue.FreeTensor(outputLocal);
    }

private:
    TPipe* pipe = nullptr;
    KernelAdaptiveThresholdMeanVector vectorMean;

    TBuf<TPosition::VECCALC> inputWindowBuf;
    TBuf<TPosition::VECCALC> colSumBuf;
    TQue<TPosition::VECOUT, OUTPUT_BUFFER_NUM> outputQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    const uint32_t* gaussianWeights = nullptr;

    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t totalRows = 0;

    uint32_t startRow = 0;
    uint32_t myRows = 0;

    uint32_t tileWidth = 0;
    uint32_t blockSize = 0;
    uint32_t radius = 0;

    uint32_t thresholdType = 0;
    float outputValue = 0.0F;
    float thresholdDelta = 0.0F;
};

} // namespace NsAdaptiveThreshold

#endif // ADAPTIVE_THRESHOLD_H_
