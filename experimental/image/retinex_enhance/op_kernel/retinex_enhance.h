/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RETINEX_ENHANCE_H
#define RETINEX_ENHANCE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "retinex_enhance_tiling_data.h"

namespace NsRetinexEnhance {
using namespace AscendC;

constexpr uint32_t GRAYSCALE_CHANNELS = 1;
constexpr uint32_t RGB_CHANNELS = 3;
constexpr uint32_t RGBA_CHANNELS = 4;
constexpr uint32_t MAX_WEIGHT_ELEMENTS = 751;
constexpr uint32_t MAX_WEIGHT_ELEMENTS_ALIGNED = 752;
constexpr float INPUT_MIN = 0.0f;
constexpr float INPUT_MAX = 255.0f;
constexpr float INPUT_OFFSET = 1.0f;
constexpr float SCALE_WEIGHT = 0.3333333333333333f;
constexpr float COLOR_ALPHA = 125.0f;
constexpr float OUTPUT_GAIN = 30.0f;
constexpr float OUTPUT_OFFSET = 6.0f;

class KernelRetinexEnhance {
public:
    __aicore__ inline KernelRetinexEnhance() {}

    __aicore__ inline void Init(GM_ADDR image, GM_ADDR enhanced, GM_ADDR workspace,
                                const RetinexEnhanceTilingData* tilingData, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tiling = tilingData;
        const uint64_t totalElements = static_cast<uint64_t>(tiling->height) * tiling->width * tiling->channels;
        imageGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(image), totalElements);
        enhancedGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(enhanced), totalElements);
        horizontalGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), totalElements);

        pipe->InitBuffer(inputQueue, 1, tiling->rowElementsAligned * sizeof(float));
        pipe->InitBuffer(verticalQueue, 1, tiling->tileElementsAligned * sizeof(float));
        pipe->InitBuffer(outputQueue, 1, tiling->tileElementsAligned * sizeof(float));
        pipe->InitBuffer(haloBuf, tiling->maxHaloElementsAligned * sizeof(float));
        pipe->InitBuffer(indexBuf, tiling->maxHaloElementsAligned * sizeof(uint32_t));
        pipe->InitBuffer(verticalIndexBuf, tiling->verticalIndexElementsAligned * sizeof(uint32_t));
        pipe->InitBuffer(windowIndexBuf, tiling->tileElementsAligned * sizeof(uint32_t));
        pipe->InitBuffer(tempBuf, tiling->tileElementsAligned * sizeof(float));
        pipe->InitBuffer(sumBuf, tiling->tileElementsAligned * sizeof(float));
        pipe->InitBuffer(verticalAccumBuf, tiling->verticalTileElementsAligned * sizeof(float));
        pipe->InitBuffer(weightBuf, MAX_WEIGHT_ELEMENTS_ALIGNED * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockId = GetBlockIdx();
        const uint32_t blockCount = GetBlockNum();
        for (uint32_t scale = 0; scale < RETINEX_SCALE_COUNT; ++scale) {
            GenerateWeights(scale);
            BuildVerticalIndices(scale);
            for (uint32_t stripe = blockId; stripe < tiling->stripeCount; stripe += blockCount) {
                const uint32_t xStart = stripe * tiling->tilePixels;
                const uint32_t remaining = tiling->width - xStart;
                const uint32_t pixelCount = remaining < tiling->tilePixels ? remaining : tiling->tilePixels;
                HorizontalConvolution(scale, xStart, pixelCount);
                GmStoreToLoadBarrier();
                VerticalConvolution(scale, xStart, pixelCount);
            }
        }
        GmStoreToLoadBarrier();
        for (uint32_t stripe = blockId; stripe < tiling->stripeCount; stripe += blockCount) {
            const uint32_t xStart = stripe * tiling->tilePixels;
            const uint32_t remaining = tiling->width - xStart;
            const uint32_t pixelCount = remaining < tiling->tilePixels ? remaining : tiling->tilePixels;
            Finalize(xStart, pixelCount);
        }
    }

private:
    __aicore__ inline uint32_t ReflectIndex(int64_t index, uint32_t length) const
    {
        const int64_t period = static_cast<int64_t>(length) * 2;
        int64_t mapped = index % period;
        if (mapped < 0) {
            mapped += period;
        }
        if (mapped >= static_cast<int64_t>(length)) {
            mapped = period - 1 - mapped;
        }
        return static_cast<uint32_t>(mapped);
    }

    __aicore__ inline void ScalarToVectorBarrier()
    {
        const event_t eventId = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void VectorToScalarBarrier()
    {
        const event_t eventId = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void GmStoreToLoadBarrier()
    {
        const event_t eventId = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void GenerateWeights(uint32_t scale)
    {
        LocalTensor<float> weights = weightBuf.Get<float>();
        const uint32_t radius = tiling->radii[scale];
        const uint32_t anchorOffset = tiling->groupOffsets[scale];
        const float ratioStep = tiling->ratioSteps[scale];
        float weight = 0.0f;
        float ratio = 0.0f;
        for (uint32_t tap = 0; tap <= radius; ++tap) {
            if (tap % RETINEX_WEIGHT_GROUP_SIZE == 0) {
                const uint32_t group = tap / RETINEX_WEIGHT_GROUP_SIZE;
                weight = tiling->anchorWeights[anchorOffset + group];
                ratio = tiling->anchorRatios[anchorOffset + group];
            }
            weights.SetValue(tap, weight);
            weight *= ratio;
            ratio *= ratioStep;
        }
    }

    __aicore__ inline uint32_t BuildHaloIndices(uint32_t xStart, uint32_t pixelCount, uint32_t radius)
    {
        const uint32_t channels = tiling->channels;
        const uint32_t haloPixels = pixelCount + 2 * radius;
        const uint32_t haloElements = haloPixels * channels;
        LocalTensor<uint32_t> indices = indexBuf.Get<uint32_t>();
        for (uint32_t element = haloElements; element < tiling->maxHaloElementsAligned; ++element) {
            indices.SetValue(element, 0U);
        }
        for (uint32_t haloX = 0; haloX < haloPixels; ++haloX) {
            const int64_t logicalX = static_cast<int64_t>(xStart) + haloX - radius;
            const uint32_t sourceX = ReflectIndex(logicalX, tiling->width);
            for (uint32_t channel = 0; channel < channels; ++channel) {
                indices.SetValue(haloX * channels + channel, (sourceX * channels + channel) * sizeof(float));
            }
        }
        ScalarToVectorBarrier();
        return haloElements;
    }

    __aicore__ inline void BuildVerticalIndices(uint32_t scale)
    {
        const uint32_t radius = tiling->radii[scale];
        const uint32_t indexCount = tiling->height + 2 * radius;
        LocalTensor<uint32_t> rowIndices = verticalIndexBuf.Get<uint32_t>();
        for (uint32_t index = 0; index < indexCount; ++index) {
            const int64_t logicalRow = static_cast<int64_t>(index) - radius;
            rowIndices.SetValue(index, ReflectIndex(logicalRow, tiling->height));
        }
    }

    __aicore__ inline void ClampAndOffset(LocalTensor<float> tensor, uint32_t count)
    {
        Maxs(tensor, tensor, INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
        Mins(tensor, tensor, INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Adds(tensor, tensor, INPUT_OFFSET, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> LoadGmTile(GlobalTensor<float> source, uint64_t offset, uint32_t count)
    {
        LocalTensor<float> local = inputQueue.AllocTensor<float>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(local, source[offset], params, padParams);
        inputQueue.EnQue(local);
        return inputQueue.DeQue<float>();
    }

    __aicore__ inline void StoreGmTile(GlobalTensor<float> target, uint64_t offset, LocalTensor<float> local,
                                       uint32_t count)
    {
        outputQueue.EnQue(local);
        LocalTensor<float> ready = outputQueue.DeQue<float>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(target[offset], ready, params);
        outputQueue.FreeTensor(ready);
    }

    __aicore__ inline LocalTensor<float> LoadVerticalTile(uint64_t offset, uint32_t count)
    {
        LocalTensor<float> local = verticalQueue.AllocTensor<float>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(local, horizontalGm[offset], params, padParams);
        verticalQueue.EnQue(local);
        return verticalQueue.DeQue<float>();
    }

    __aicore__ inline void HorizontalConvolution(uint32_t scale, uint32_t xStart, uint32_t pixelCount)
    {
        const uint32_t channels = tiling->channels;
        const uint32_t radius = tiling->radii[scale];
        const uint32_t haloElements = BuildHaloIndices(xStart, pixelCount, radius);
        const uint32_t tileElements = pixelCount * channels;
        LocalTensor<uint32_t> indices = indexBuf.Get<uint32_t>();
        LocalTensor<uint32_t> windowIndices = windowIndexBuf.Get<uint32_t>();
        LocalTensor<float> halo = haloBuf.Get<float>();
        LocalTensor<float> temp = tempBuf.Get<float>();
        LocalTensor<float> weights = weightBuf.Get<float>();
        LocalTensor<int32_t> signedIndices = windowIndices.ReinterpretCast<int32_t>();

        // The center-window byte offsets are invariant across rows. Initialize them once;
        // each row resets the vector after the final tap below.
        VectorToScalarBarrier();
        for (uint32_t element = tileElements; element < tiling->tileElementsAligned; ++element) {
            windowIndices.SetValue(element, 0U);
        }
        for (uint32_t element = 0; element < tileElements; ++element) {
            windowIndices.SetValue(element, (radius * channels + element) * sizeof(float));
        }
        ScalarToVectorBarrier();

        for (uint32_t row = 0; row < tiling->height; ++row) {
            const uint64_t rowOffset = static_cast<uint64_t>(row) * tiling->rowElements;
            LocalTensor<float> rowLocal = LoadGmTile(imageGm, rowOffset, tiling->rowElements);
            ClampAndOffset(rowLocal, tiling->rowElements);
            Gather(halo, rowLocal, indices, static_cast<uint32_t>(0), haloElements);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> pair = sumBuf.Get<float>();

            LocalTensor<float> accumulator = outputQueue.AllocTensor<float>();
            Gather(temp, halo, windowIndices, static_cast<uint32_t>(0), tileElements);
            PipeBarrier<PIPE_V>();
            Muls(accumulator, temp, weights.GetValue(0), tileElements);
            PipeBarrier<PIPE_V>();
            if (radius > 0) {
                Adds(signedIndices, signedIndices, -static_cast<int32_t>(channels * sizeof(float)), tileElements);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t distance = 1; distance <= radius; ++distance) {
                Gather(pair, halo, windowIndices, static_cast<uint32_t>(0), tileElements);
                PipeBarrier<PIPE_V>();
                Adds(signedIndices, signedIndices, static_cast<int32_t>(2 * distance * channels * sizeof(float)),
                     tileElements);
                PipeBarrier<PIPE_V>();
                Gather(temp, halo, windowIndices, static_cast<uint32_t>(0), tileElements);
                PipeBarrier<PIPE_V>();
                Add(temp, temp, pair, tileElements);
                PipeBarrier<PIPE_V>();
                const float coefficient = weights.GetValue(distance);
                Axpy(accumulator, temp, coefficient, tileElements);
                PipeBarrier<PIPE_V>();
                if (distance < radius) {
                    Adds(signedIndices, signedIndices,
                         -static_cast<int32_t>((2 * distance + 1) * channels * sizeof(float)), tileElements);
                    PipeBarrier<PIPE_V>();
                }
            }
            if (radius > 0) {
                Adds(signedIndices, signedIndices, -static_cast<int32_t>(radius * channels * sizeof(float)),
                     tileElements);
                PipeBarrier<PIPE_V>();
            }
            inputQueue.FreeTensor(rowLocal);
            const uint64_t outputOffset = rowOffset + static_cast<uint64_t>(xStart) * channels;
            StoreGmTile(horizontalGm, outputOffset, accumulator, tileElements);
        }
    }

    __aicore__ inline void VerticalConvolution(uint32_t scale, uint32_t xStart, uint32_t pixelCount)
    {
        const uint32_t channels = tiling->channels;
        const uint32_t radius = tiling->radii[scale];
        const uint32_t tileElements = pixelCount * channels;
        LocalTensor<float> weights = weightBuf.Get<float>();
        LocalTensor<uint32_t> rowIndices = verticalIndexBuf.Get<uint32_t>();
        LocalTensor<float> accumulators = verticalAccumBuf.Get<float>();

        for (uint32_t rowStart = 0; rowStart < tiling->height; rowStart += tiling->verticalTileRows) {
            const uint32_t remainingRows = tiling->height - rowStart;
            const uint32_t rowCount = remainingRows < tiling->verticalTileRows ? remainingRows :
                                                                                 tiling->verticalTileRows;
            const int64_t logicalFirst = static_cast<int64_t>(rowStart) - radius;
            const int64_t logicalLast = static_cast<int64_t>(rowStart) + rowCount - 1 + radius;

            // Reuse each horizontally filtered row for every output row in this vertical UB tile.
            for (int64_t logicalRow = logicalFirst; logicalRow <= logicalLast; ++logicalRow) {
                const uint32_t sourceRow = rowIndices.GetValue(static_cast<uint32_t>(logicalRow + radius));
                const uint64_t sourceOffset = (static_cast<uint64_t>(sourceRow) * tiling->width + xStart) * channels;
                LocalTensor<float> source = LoadVerticalTile(sourceOffset, tileElements);
                for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                    const int64_t outputRow = static_cast<int64_t>(rowStart + localRow);
                    const int64_t delta = logicalRow - outputRow;
                    if (delta < -static_cast<int64_t>(radius) || delta > static_cast<int64_t>(radius)) {
                        continue;
                    }
                    const uint32_t distance = static_cast<uint32_t>(delta < 0 ? -delta : delta);
                    LocalTensor<float> accumulator = accumulators[localRow * tiling->tileElementsAligned];
                    const float coefficient = weights.GetValue(distance);
                    if (logicalRow == outputRow - radius) {
                        Muls(accumulator, source, coefficient, tileElements);
                    } else {
                        Axpy(accumulator, source, coefficient, tileElements);
                    }
                    PipeBarrier<PIPE_V>();
                }
                verticalQueue.FreeTensor(source);
            }

            for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                const uint32_t row = rowStart + localRow;
                LocalTensor<float> accumulator = accumulators[localRow * tiling->tileElementsAligned];
                LocalTensor<float> result = outputQueue.AllocTensor<float>();
                const uint64_t outputOffset = (static_cast<uint64_t>(row) * tiling->width + xStart) * channels;
                LocalTensor<float> input = LoadGmTile(imageGm, outputOffset, tileElements);
                ClampAndOffset(input, tileElements);
                Ln(input, input, tileElements);
                PipeBarrier<PIPE_V>();
                Ln(accumulator, accumulator, tileElements);
                PipeBarrier<PIPE_V>();
                Sub(result, input, accumulator, tileElements);
                PipeBarrier<PIPE_V>();
                Muls(result, result, SCALE_WEIGHT, tileElements);
                PipeBarrier<PIPE_V>();
                inputQueue.FreeTensor(input);

                if (scale > 0) {
                    LocalTensor<float> previous = LoadGmTile(enhancedGm, outputOffset, tileElements);
                    Add(result, result, previous, tileElements);
                    PipeBarrier<PIPE_V>();
                    inputQueue.FreeTensor(previous);
                }
                StoreGmTile(enhancedGm, outputOffset, result, tileElements);
            }
        }
    }

    __aicore__ inline void Finalize(uint32_t xStart, uint32_t pixelCount)
    {
        const uint32_t channels = tiling->channels;
        const uint32_t tileElements = pixelCount * channels;
        LocalTensor<float> expandedSum = sumBuf.Get<float>();
        LocalTensor<float> preservedAlpha = tempBuf.Get<float>();
        for (uint32_t row = 0; row < tiling->height; ++row) {
            const uint64_t offset = (static_cast<uint64_t>(row) * tiling->width + xStart) * channels;
            if (channels == GRAYSCALE_CHANNELS) {
                LocalTensor<float> msr = LoadGmTile(enhancedGm, offset, tileElements);
                LocalTensor<float> result = outputQueue.AllocTensor<float>();
                Muls(result, msr, 1.0f, tileElements);
                PipeBarrier<PIPE_V>();
                inputQueue.FreeTensor(msr);
                Adds(result, result, OUTPUT_OFFSET, tileElements);
                PipeBarrier<PIPE_V>();
                Muls(result, result, OUTPUT_GAIN, tileElements);
                PipeBarrier<PIPE_V>();
                Maxs(result, result, INPUT_MIN, tileElements);
                PipeBarrier<PIPE_V>();
                Mins(result, result, INPUT_MAX, tileElements);
                PipeBarrier<PIPE_V>();
                StoreGmTile(enhancedGm, offset, result, tileElements);
                continue;
            }

            LocalTensor<float> input = LoadGmTile(imageGm, offset, tileElements);
            ClampAndOffset(input, tileElements);
            VectorToScalarBarrier();
            for (uint32_t pixel = 0; pixel < pixelCount; ++pixel) {
                const uint32_t base = pixel * channels;
                const float channelSum = input.GetValue(base) + input.GetValue(base + 1) + input.GetValue(base + 2);
                for (uint32_t channel = 0; channel < channels; ++channel) {
                    expandedSum.SetValue(base + channel, channelSum);
                }
                if (channels == RGBA_CHANNELS) {
                    preservedAlpha.SetValue(base + RGB_CHANNELS, input.GetValue(base + RGB_CHANNELS) - INPUT_OFFSET);
                }
            }
            ScalarToVectorBarrier();

            LocalTensor<float> result = outputQueue.AllocTensor<float>();
            Muls(result, input, COLOR_ALPHA, tileElements);
            PipeBarrier<PIPE_V>();
            Ln(result, result, tileElements);
            PipeBarrier<PIPE_V>();
            Ln(expandedSum, expandedSum, tileElements);
            PipeBarrier<PIPE_V>();
            Sub(result, result, expandedSum, tileElements);
            PipeBarrier<PIPE_V>();
            inputQueue.FreeTensor(input);

            LocalTensor<float> msr = LoadGmTile(enhancedGm, offset, tileElements);
            Mul(result, result, msr, tileElements);
            PipeBarrier<PIPE_V>();
            Adds(result, result, OUTPUT_OFFSET, tileElements);
            PipeBarrier<PIPE_V>();
            Muls(result, result, OUTPUT_GAIN, tileElements);
            PipeBarrier<PIPE_V>();
            Maxs(result, result, INPUT_MIN, tileElements);
            PipeBarrier<PIPE_V>();
            Mins(result, result, INPUT_MAX, tileElements);
            PipeBarrier<PIPE_V>();
            inputQueue.FreeTensor(msr);

            if (channels == RGBA_CHANNELS) {
                VectorToScalarBarrier();
                for (uint32_t pixel = 0; pixel < pixelCount; ++pixel) {
                    const uint32_t alphaOffset = pixel * channels + RGB_CHANNELS;
                    result.SetValue(alphaOffset, preservedAlpha.GetValue(alphaOffset));
                }
                ScalarToVectorBarrier();
                Muls(result, result, 1.0f, tileElements);
                PipeBarrier<PIPE_V>();
            }
            StoreGmTile(enhancedGm, offset, result, tileElements);
        }
    }

private:
    TPipe* pipe = nullptr;
    const RetinexEnhanceTilingData* tiling = nullptr;
    GlobalTensor<float> imageGm;
    GlobalTensor<float> enhancedGm;
    GlobalTensor<float> horizontalGm;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECIN, 1> verticalQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> haloBuf;
    TBuf<QuePosition::VECCALC> indexBuf;
    TBuf<QuePosition::VECCALC> verticalIndexBuf;
    TBuf<QuePosition::VECCALC> windowIndexBuf;
    TBuf<QuePosition::VECCALC> tempBuf;
    TBuf<QuePosition::VECCALC> sumBuf;
    TBuf<QuePosition::VECCALC> verticalAccumBuf;
    TBuf<QuePosition::VECCALC> weightBuf;
};
} // namespace NsRetinexEnhance

#endif
