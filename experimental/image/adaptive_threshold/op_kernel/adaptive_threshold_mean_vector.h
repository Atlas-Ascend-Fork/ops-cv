/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTIVE_THRESHOLD_MEAN_VECTOR_H_
#define ADAPTIVE_THRESHOLD_MEAN_VECTOR_H_

#include "kernel_operator.h"
#include "adaptive_threshold_tiling_data.h"

namespace NsAdaptiveThreshold {

// UINT8 sums are exact in float: even the full 255x255 window is < 2^24.
// Keep column sums in UB, and stream entering/leaving rows through DMA.
class KernelAdaptiveThresholdMeanVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AdaptiveThresholdTilingData* data, AscendC::TPipe* p)
    {
        pipe = p;
        height = data->height;
        width = data->width;
        tileWidth = data->tileWidth;
        blockSize = data->blockSize;
        radius = data->radius;
        thresholdType = data->thresholdType;
        outputValue = static_cast<uint8_t>(static_cast<int32_t>(data->outputValue));
        delta = data->thresholdDelta;
        const uint32_t core = AscendC::GetBlockIdx();
        rows = data->baseRows + (core < data->extraRows ? 1U : 0U);
        firstRow = core * data->baseRows + (core < data->extraRows ? core : data->extraRows);
        const uint32_t count = Align64(tileWidth + 2U * radius + 31U);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe->InitBuffer(rowBuf, count);
        pipe->InitBuffer(halfBuf, count * sizeof(half));
        pipe->InitBuffer(floatBuf, count * sizeof(float));
        pipe->InitBuffer(sumBuf, count * sizeof(float));
        pipe->InitBuffer(outputQueue, 1, (tileWidth + 31U) / 32U * 32U);
    }

    __aicore__ inline void Process()
    {
        using namespace AscendC;
        auto sum = sumBuf.Get<float>();
        for (uint32_t tile = 0; tile < width; tile += tileWidth) {
            const uint32_t currentWidth = Min(tileWidth, width - tile);
            const uint32_t windowWidth = currentWidth + 2U * radius;
            const uint32_t leftCopies = tile < radius ? radius - tile : 0U;
            const uint32_t alignedLeft = (leftCopies + 31U) / 32U * 32U;
            const uint32_t shift = alignedLeft - leftCopies;
            const uint32_t count = Align64(windowWidth + 31U);
            const uint32_t firstColumn = tile < radius ? 0U : tile - radius;
            const uint32_t realColumns = Min(windowWidth - leftCopies, width - firstColumn);
            uint32_t previousImage = 0xFFFFFFFFU;
            for (uint32_t row = firstRow; row < firstRow + rows; ++row) {
                const uint32_t image = row / height;
                const uint32_t y = row % height;
                if (image != previousImage) {
                    // Scalar horizontal readers from the previous image must finish first.
                    Sync<HardEvent::S_V>();
                    Duplicate(sum, 0.0F, count);
                    PipeBarrier<PIPE_V>();
                    const int64_t top = static_cast<int64_t>(y) - radius;
                    const int64_t bottom = static_cast<int64_t>(y) + radius;
                    const uint32_t begin = Clamp(top);
                    const uint32_t end = Clamp(bottom);
                    for (uint32_t source = begin; source <= end; ++source) {
                        const uint32_t copies = 1U + (source == 0U && top < 0 ? static_cast<uint32_t>(-top) : 0U) +
                                                (source == height - 1U && bottom >= height ?
                                                     static_cast<uint32_t>(bottom - height + 1) :
                                                     0U);
                        Accumulate(image, source, firstColumn, realColumns, alignedLeft, count,
                                   static_cast<float>(static_cast<int32_t>(copies)));
                    }
                } else {
                    const uint32_t leaving = Clamp(static_cast<int64_t>(y) - radius - 1);
                    const uint32_t entering = Clamp(static_cast<int64_t>(y) + radius);
                    Sync<HardEvent::S_V>();
                    if (leaving != entering) {
                        Accumulate(image, leaving, firstColumn, realColumns, alignedLeft, count, -1.0F);
                        Accumulate(image, entering, firstColumn, realColumns, alignedLeft, count, 1.0F);
                    }
                }
                LoadRow(image, y, firstColumn, realColumns, alignedLeft, count);
                Sync<HardEvent::V_S>();
                auto input = rowBuf.Get<uint8_t>();
                auto output = outputQueue.AllocTensor<uint8_t>();
                float windowSum = 0.0F;
                for (uint32_t col = 0; col < blockSize; ++col) {
                    windowSum += sum.GetValue(shift + col);
                }
                const int32_t area = static_cast<int32_t>(blockSize * blockSize);
                for (uint32_t col = 0; col < currentWidth; ++col) {
                    if (col != 0) {
                        windowSum -= sum.GetValue(shift + col - 1U);
                        windowSum += sum.GetValue(shift + col + blockSize - 1U);
                    }
                    // The sum is exact in float, but adding a fractional half-area may round it up.
                    // For UINT8 and blockSize <= 255, the integer numerator fits safely in int32_t.
                    const int32_t integerSum = static_cast<int32_t>(windowSum);
                    const float mean = static_cast<float>((integerSum + area / 2) / area);
                    const float value = static_cast<float>(static_cast<int32_t>(input.GetValue(shift + col + radius)));
                    const bool above = value > mean - delta;
                    output.SetValue(col, (above != (thresholdType != 0U)) ? outputValue : 0U);
                }
                outputQueue.EnQue(output);
                output = outputQueue.DeQue<uint8_t>();
                const uint64_t offset = static_cast<uint64_t>(row) * width + tile;
                DataCopyExtParams params{1, currentWidth, 0, 0, 0};
                DataCopyPad(yGm[offset], output, params);
                outputQueue.FreeTensor(output);
                previousImage = image;
            }
        }
    }

private:
    __aicore__ inline static uint32_t Align64(uint32_t count) { return (count + 63U) / 64U * 64U; }
    __aicore__ inline static uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
    __aicore__ inline uint32_t Clamp(int64_t row) const
    {
        return row < 0 ? 0U : (static_cast<uint64_t>(row) >= height ? height - 1U : static_cast<uint32_t>(row));
    }

    template <AscendC::HardEvent event>
    __aicore__ inline void Sync()
    {
        const auto id = static_cast<event_t>(pipe->FetchEventID(event));
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }

    __aicore__ inline void LoadRow(uint32_t image, uint32_t row, uint32_t firstColumn, uint32_t realColumns,
                                   uint32_t alignedLeft, uint32_t count)
    {
        using namespace AscendC;
        auto input = rowBuf.Get<uint8_t>();
        Sync<HardEvent::V_MTE2>();
        Sync<HardEvent::S_MTE2>();
        const uint64_t offset = (static_cast<uint64_t>(image) * height + row) * width + firstColumn;
        DataCopyExtParams params{1, realColumns, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
        DataCopyPad(input[alignedLeft], xGm[offset], params, padding);
        Sync<HardEvent::MTE2_S>();
        const uint8_t first = input.GetValue(alignedLeft);
        const uint8_t last = input.GetValue(alignedLeft + realColumns - 1U);
        for (uint32_t col = 0; col < alignedLeft; ++col) {
            input.SetValue(col, first);
        }
        for (uint32_t col = alignedLeft + realColumns; col < count; ++col) {
            input.SetValue(col, last);
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Accumulate(uint32_t image, uint32_t row, uint32_t firstColumn, uint32_t realColumns,
                                      uint32_t alignedLeft, uint32_t count, float weight)
    {
        using namespace AscendC;
        LoadRow(image, row, firstColumn, realColumns, alignedLeft, count);
        auto input = rowBuf.Get<uint8_t>();
        auto halfRow = halfBuf.Get<half>();
        auto floatRow = floatBuf.Get<float>();
        auto sum = sumBuf.Get<float>();
        Cast(halfRow, input, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(floatRow, halfRow, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(floatRow, floatRow, weight, count);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, floatRow, count);
        PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf, halfBuf, floatBuf, sumBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::GlobalTensor<uint8_t> xGm, yGm;
    uint32_t height = 0, width = 0, tileWidth = 0, blockSize = 0, radius = 0;
    uint32_t rows = 0, firstRow = 0, thresholdType = 0;
    uint8_t outputValue = 0;
    float delta = 0.0F;
};

} // namespace NsAdaptiveThreshold
#endif
