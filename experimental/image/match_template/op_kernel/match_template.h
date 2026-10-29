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
 * \file match_template.h
 * \brief Multi-core halo-tile implementation for MatchTemplate.
 */
#ifndef __MATCH_TEMPLATE_H__
#define __MATCH_TEMPLATE_H__

#include "kernel_operator.h"
#include "lib/std/cmath.h"
#include "match_template_tiling_data.h"

namespace NsMatchTemplate {

constexpr float NORMALIZE_EPSILON = 1e-12f;
constexpr int64_t DATA_BLOCK_BYTES = 32;
constexpr int64_t VECTOR_FP32_ELEMENTS = 512;
constexpr int64_t VECTOR_SCRATCH_SLOTS = 9;
constexpr int64_t VECTOR_PLANAR_CHANNELS = 3;
// Keep one aligned vector segment per channel and leave headroom for a 512-wide
// output tile plus the largest directed 16-wide template halo.
constexpr int64_t VECTOR_PLANAR_STRIDE_ELEMENTS = 576;
constexpr int64_t C3_VECTOR_DEFAULT_TILE_H = 4;
constexpr int64_t C3_VECTOR_CCORR_TILE_H = 16;
constexpr int64_t C3_VECTOR_UINT8_CCORR_TILE_H = 16;
// A 512-wide tile fills one FP32 Vector instruction and removes the
// half-width tail that dominated the C3 medium-window path.
constexpr int64_t C3_VECTOR_TILE_W = 512;

template <typename T, uint32_t METHOD>
class KernelMatchTemplate {
public:
    static constexpr int64_t C3_VECTOR_TILE_H = METHOD == 2 ? (AscendC::IsSameType<T, uint8_t>::value ?
                                                                   C3_VECTOR_UINT8_CCORR_TILE_H :
                                                                   C3_VECTOR_CCORR_TILE_H) :
                                                              C3_VECTOR_DEFAULT_TILE_H;
    static constexpr int64_t C3_VECTOR_ACC_ELEMENTS = C3_VECTOR_TILE_H * C3_VECTOR_TILE_W;
    static constexpr int64_t C3_VECTOR_CHANNEL_SUM_OFFSET = 2 * C3_VECTOR_ACC_ELEMENTS;
    static constexpr int64_t C3_VECTOR_ACCUMULATOR_PLANES = METHOD == 2 ? 1 : 2 + VECTOR_PLANAR_CHANNELS;

    __aicore__ inline void Init(GM_ADDR image, GM_ADDR templateImage, GM_ADDR score, GM_ADDR bestScore,
                                GM_ADDR bestLocation, GM_ADDR workspace, const MatchTemplateTilingData* tiling)
    {
        imageGm_.SetGlobalBuffer((__gm__ T*)image);
        templateGm_.SetGlobalBuffer((__gm__ T*)templateImage);
        scoreGm_.SetGlobalBuffer((__gm__ float*)score);
        bestScoreGm_.SetGlobalBuffer((__gm__ float*)bestScore);
        bestLocationGm_.SetGlobalBuffer((__gm__ int64_t*)bestLocation);
        const uint64_t workspaceEntries = static_cast<uint64_t>(tiling->batch * tiling->usedCores);
        localBestScoreGm_.SetGlobalBuffer((__gm__ float*)workspace, workspaceEntries);
        localBestIndexGm_.SetGlobalBuffer((__gm__ int64_t*)(workspace + tiling->workspaceIndexOffset),
                                          workspaceEntries);
        packedImageGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace + tiling->workspacePackedImageOffset),
                                       static_cast<uint64_t>(tiling->packedImageBytes) / sizeof(half));

        batch_ = tiling->batch;
        imageH_ = tiling->imageH;
        imageW_ = tiling->imageW;
        channels_ = tiling->channels;
        templateH_ = tiling->templateH;
        templateW_ = tiling->templateW;
        outputH_ = tiling->outputH;
        outputW_ = tiling->outputW;
        tileH_ = tiling->tileH;
        tileW_ = tiling->tileW;
        tileCols_ = tiling->tileCols;
        tilesPerBatch_ = tiling->tilesPerBatch;
        totalTiles_ = tiling->totalTiles;
        usedCores_ = tiling->usedCores;
        useHaloTile_ = tiling->useHaloTile != 0;
        useC3VectorTile_ = tiling->useC3VectorTile != 0;
        packedImageBytes_ = tiling->packedImageBytes;
        prepackScratchBytes_ = tiling->prepackScratchBytes;
        useCompactUint8C3_ = AscendC::IsSameType<T, uint8_t>::value && (METHOD == 2 || METHOD == 5) &&
                             useC3VectorTile_ && packedImageBytes_ > 0;
        templateElements_ = tiling->templateElements;
        templateBufferBytes_ = tiling->templateBufferBytes;
        haloBufferBytes_ = tiling->haloBufferBytes;
        templateFloatBufferBytes_ = tiling->templateFloatBufferBytes;
        haloFloatBufferBytes_ = tiling->haloFloatBufferBytes;
        templateStatsBufferBytes_ = tiling->templateStatsBufferBytes;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (useHaloTile_) {
            pipe_.InitBuffer(templateBuf_, static_cast<uint32_t>(templateBufferBytes_));
            pipe_.InitBuffer(imageBuf_, static_cast<uint32_t>(haloBufferBytes_));
            templateLocal_ = templateBuf_.template Get<T>();
            imageLocal_ = imageBuf_.template Get<T>();
            if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
                pipe_.InitBuffer(templateFloatBuf_, static_cast<uint32_t>(templateFloatBufferBytes_));
                templateFloatLocal_ = templateFloatBuf_.template Get<float>();
                if (haloFloatBufferBytes_ > 0) {
                    pipe_.InitBuffer(imageFloatBuf_, static_cast<uint32_t>(haloFloatBufferBytes_));
                    if (useCompactUint8C3_) {
                        imageHalfLocal_ = imageFloatBuf_.template Get<half>();
                    } else {
                        imageFloatLocal_ = imageFloatBuf_.template Get<float>();
                    }
                }
            }
        }
        if (templateStatsBufferBytes_ > 0) {
            pipe_.InitBuffer(templateStatsBuf_, static_cast<uint32_t>(templateStatsBufferBytes_));
            templateChannelSumLocal_ = templateStatsBuf_.template Get<float>();
        }
        if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value ||
                      AscendC::IsSameType<T, uint8_t>::value) {
            if (useHaloTile_) {
                mte2ToVectorEvent_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
                vectorToScalarEvent_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::V_S));
                scalarToVectorEvent_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::S_V));
                if (useCompactUint8C3_) {
                    vectorToMte3Event_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
                }
                pipe_.InitBuffer(vectorBuf_, VECTOR_SCRATCH_SLOTS * VECTOR_FP32_ELEMENTS * sizeof(float));
                AscendC::LocalTensor<float> vectorLocal = vectorBuf_.template Get<float>();
                vectorWorkLocal_ = vectorLocal;
                vectorAcc0Local_ = vectorLocal[VECTOR_FP32_ELEMENTS];
                vectorAcc1Local_ = vectorLocal[2 * VECTOR_FP32_ELEMENTS];
                vectorAcc2Local_ = vectorLocal[3 * VECTOR_FP32_ELEMENTS];
                vectorInputLocal_ = vectorLocal[4 * VECTOR_FP32_ELEMENTS];
                vectorOffsetLocal_ = vectorLocal[5 * VECTOR_FP32_ELEMENTS].template ReinterpretCast<uint32_t>();
                vectorRawLocal_ = vectorLocal[6 * VECTOR_FP32_ELEMENTS].template ReinterpretCast<T>();
                vectorMaskLocal_ = vectorLocal[6 * VECTOR_FP32_ELEMENTS].template ReinterpretCast<uint8_t>();
                vectorHalfLocal_ = vectorLocal[7 * VECTOR_FP32_ELEMENTS].template ReinterpretCast<half>();
                vectorPlanarOffsetLocal_ = vectorLocal[8 * VECTOR_FP32_ELEMENTS].template ReinterpretCast<uint32_t>();
                uint32_t inputElementBytes = sizeof(T);
                if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
                    // Compact UINT8 C3 schedules gather from the packed FP16 halo.
                    inputElementBytes = useCompactUint8C3_ ? sizeof(half) : sizeof(float);
                }
                for (uint32_t index = 0; index < VECTOR_FP32_ELEMENTS; ++index) {
                    vectorOffsetLocal_.SetValue(index, index * channels_ * inputElementBytes);
                    vectorPlanarOffsetLocal_.SetValue(index, index * sizeof(float));
                }
                if (channels_ == VECTOR_PLANAR_CHANNELS) {
                    pipe_.InitBuffer(vectorPlaneBuf_,
                                     VECTOR_PLANAR_CHANNELS * VECTOR_PLANAR_STRIDE_ELEMENTS * sizeof(float));
                    vectorPlaneLocal_ = vectorPlaneBuf_.template Get<float>();
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVectorEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVectorEvent_);
            }
            if (useC3VectorTile_) {
                pipe_.InitBuffer(c3VectorBuf_, C3_VECTOR_ACCUMULATOR_PLANES * C3_VECTOR_ACC_ELEMENTS * sizeof(float));
                c3VectorAcc0Local_ = c3VectorBuf_.template Get<float>();
            }
        }
        tileScoreRowStride_ = AlignUp(tileW_ * static_cast<int64_t>(sizeof(float)), DATA_BLOCK_BYTES) /
                              static_cast<int64_t>(sizeof(float));
        // Non-compact CCORR tail tiles use ProcessVectorRow, which writes the
        // generic score buffer even when full tiles use the C3 accumulators.
        const bool needsTileScoreBuffer = !useC3VectorTile_ || (METHOD == 2 && !useCompactUint8C3_);
        const uint32_t tileScoreBufferBytes = needsTileScoreBuffer ?
                                                  static_cast<uint32_t>(tileH_ * tileScoreRowStride_ * sizeof(float)) :
                                                  static_cast<uint32_t>(DATA_BLOCK_BYTES);
        pipe_.InitBuffer(tileScoreBuf_, tileScoreBufferBytes);
        pipe_.InitBuffer(workspaceScoreBuf_, DATA_BLOCK_BYTES);
        pipe_.InitBuffer(workspaceIndexBuf_, DATA_BLOCK_BYTES);
        tileScoreLocal_ = tileScoreBuf_.template Get<float>();
        workspaceScoreLocal_ = workspaceScoreBuf_.template Get<float>();
        workspaceIndexLocal_ = workspaceIndexBuf_.template Get<int64_t>();
    }

    __aicore__ inline void Process()
    {
        InitializeWorkspace();
        if (useCompactUint8C3_) {
            PrepackUint8Image();
        }
        if (useHaloTile_ && blockIdx_ < totalTiles_) {
            LoadTemplate();
            if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
                ConvertUint8TemplateToFloat();
            }
        }
        PrecomputeTemplateStats();

        int64_t currentBatch = -1;
        float localBestValue = 0.0f;
        int64_t localBestIndex = -1;
        for (int64_t globalTile = blockIdx_; globalTile < totalTiles_; globalTile += usedCores_) {
            const int64_t n = globalTile / tilesPerBatch_;
            if (currentBatch != n) {
                FlushLocalBest(currentBatch, localBestValue, localBestIndex);
                currentBatch = n;
                localBestIndex = -1;
            }
            const int64_t tileIndex = globalTile - n * tilesPerBatch_;
            ProcessTile(n, tileIndex, localBestValue, localBestIndex);
        }
        FlushLocalBest(currentBatch, localBestValue, localBestIndex);

        AscendC::PipeBarrier<PIPE_ALL>();
        if (usedCores_ > 1) {
            AscendC::SyncAll<true>();
        }
        if (blockIdx_ == 0) {
            ReduceBestResults();
        }
    }

private:
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment) const
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline int64_t Min(int64_t lhs, int64_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline float Clamp(float value, float lower, float upper) const
    {
        value = value < lower ? lower : value;
        return value > upper ? upper : value;
    }

    __aicore__ inline float Normalize(float numerator, float denominatorProduct, float lower, float upper,
                                      float zeroDenominatorValue) const
    {
        if (denominatorProduct <= 0.0f) {
            return zeroDenominatorValue;
        }
        const float denominator = AscendC::Std::sqrt(denominatorProduct);
        if (denominator <= NORMALIZE_EPSILON) {
            return zeroDenominatorValue;
        }
        return Clamp(numerator / denominator, lower, upper);
    }

    __aicore__ inline float ToFloat(T value) const
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            const int32_t promoted = static_cast<int32_t>(value);
            return static_cast<float>(promoted);
        } else {
            return static_cast<float>(value);
        }
    }

    __aicore__ inline float GetTemplateValue(int64_t offset) const
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            return useHaloTile_ ? templateFloatLocal_.GetValue(offset) : ToFloat(templateGm_.GetValue(offset));
        } else {
            return useHaloTile_ ? ToFloat(templateLocal_.GetValue(offset)) : ToFloat(templateGm_.GetValue(offset));
        }
    }

    __aicore__ inline void ConvertUint8RangeToFloat(AscendC::LocalTensor<float> destination,
                                                    AscendC::LocalTensor<T> source, int64_t count)
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            for (int64_t offset = 0; offset < count; offset += VECTOR_FP32_ELEMENTS) {
                const uint32_t current = static_cast<uint32_t>(Min(VECTOR_FP32_ELEMENTS, count - offset));
                AscendC::Cast(vectorHalfLocal_, source[offset], AscendC::RoundMode::CAST_NONE, current);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(destination[offset], vectorHalfLocal_, AscendC::RoundMode::CAST_NONE, current);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ConvertUint8TemplateToFloat()
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            ConvertUint8RangeToFloat(templateFloatLocal_, templateLocal_, templateElements_);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent_);
        }
    }

    __aicore__ inline void ConvertUint8HaloToFloat(int64_t haloRows, int64_t rowStrideElements)
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            for (int64_t row = 0; row < haloRows; ++row) {
                const int64_t rowOffset = row * rowStrideElements;
                // Convert the zero-padded tail as well: Gather aligns to a float block and may read it.
                ConvertUint8RangeToFloat(imageFloatLocal_[rowOffset], imageLocal_[rowOffset], rowStrideElements);
            }
        }
    }

    __aicore__ inline void ConvertUint8HaloToHalf(int64_t haloRows, int64_t rowStrideElements)
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            for (int64_t row = 0; row < haloRows; ++row) {
                const int64_t rowOffset = row * rowStrideElements;
                for (int64_t offset = 0; offset < rowStrideElements; offset += VECTOR_FP32_ELEMENTS) {
                    const uint32_t current = static_cast<uint32_t>(
                        Min(VECTOR_FP32_ELEMENTS, rowStrideElements - offset));
                    AscendC::Cast(imageHalfLocal_[rowOffset + offset], imageLocal_[rowOffset + offset],
                                  AscendC::RoundMode::CAST_NONE, current);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepackUint8Image()
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            const int64_t totalElements = batch_ * imageH_ * imageW_ * channels_;
            const int64_t chunkElements = prepackScratchBytes_;
            if (chunkElements <= 0 || packedImageBytes_ <= 0) {
                return;
            }
            for (int64_t base = blockIdx_ * chunkElements; base < totalElements; base += usedCores_ * chunkElements) {
                const int64_t current = Min(chunkElements, totalElements - base);
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(current), 0, 0, 0};
                AscendC::DataCopyPadExtParams<uint8_t> padParams{
                    true, 0, static_cast<uint8_t>(AlignUp(current, DATA_BLOCK_BYTES) - current), 0};
                AscendC::DataCopyPad(imageLocal_, imageGm_[base], copyParams, padParams);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int64_t offset = 0; offset < current; offset += VECTOR_FP32_ELEMENTS) {
                    const uint32_t count = static_cast<uint32_t>(Min(VECTOR_FP32_ELEMENTS, current - offset));
                    AscendC::Cast(imageHalfLocal_[offset], imageLocal_[offset], AscendC::RoundMode::CAST_NONE, count);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3Event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3Event_);
                AscendC::DataCopy(packedImageGm_[base], imageHalfLocal_, static_cast<uint32_t>(current));
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            if (usedCores_ > 1) {
                AscendC::SyncAll<true>();
            }
        }
    }

    __aicore__ inline bool IsBetter(float value, int64_t index, float bestValue, int64_t bestIndex) const
    {
        if (bestIndex < 0) {
            return true;
        }
        constexpr bool minimize = METHOD == 0 || METHOD == 1;
        if (minimize) {
            return value < bestValue || (value == bestValue && index < bestIndex);
        }
        return value > bestValue || (value == bestValue && index < bestIndex);
    }

    __aicore__ inline void InitializeWorkspace()
    {
        workspaceScoreLocal_.SetValue(0, 0.0f);
        workspaceIndexLocal_.SetValue(0, -1);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams scoreCopy{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyExtParams indexCopy{1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        for (int64_t n = 0; n < batch_; ++n) {
            const int64_t workspaceIndex = n * usedCores_ + blockIdx_;
            AscendC::DataCopyPad(localBestScoreGm_[workspaceIndex], workspaceScoreLocal_, scoreCopy);
            AscendC::DataCopyPad(localBestIndexGm_[workspaceIndex], workspaceIndexLocal_, indexCopy);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadTemplate()
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(templateElements_ * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            AscendC::DataCopyPadExtParams<T> padParams{
                true, 0, static_cast<uint8_t>(templateBufferBytes_ - templateElements_), static_cast<T>(0)};
            AscendC::DataCopyPad(templateLocal_, templateGm_, copyParams, padParams);
        } else {
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(templateLocal_, templateGm_, copyParams, padParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrecomputeTemplateStats()
    {
        if constexpr (METHOD == 0 || METHOD == 2) {
            return;
        }

        const float spatialSize = static_cast<float>(templateH_ * templateW_);
        templateSquareSum_ = 0.0f;
        templateVariance_ = 0.0f;
        for (int64_t c = 0; c < channels_; ++c) {
            float channelSum = 0.0f;
            float channelSquare = 0.0f;
            for (int64_t ty = 0; ty < templateH_; ++ty) {
                const int64_t templateRow = ty * templateW_ * channels_;
                for (int64_t tx = 0; tx < templateW_; ++tx) {
                    const int64_t offset = templateRow + tx * channels_ + c;
                    const float value = GetTemplateValue(offset);
                    channelSum += value;
                    channelSquare += value * value;
                }
            }
            templateSquareSum_ += channelSquare;
            if constexpr (METHOD == 4 || METHOD == 5) {
                templateChannelSumLocal_.SetValue(c, channelSum);
                templateVariance_ += channelSquare - channelSum * channelSum / spatialSize;
            }
        }
        templateVariance_ = templateVariance_ < 0.0f ? 0.0f : templateVariance_;
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int64_t LoadHalo(int64_t n, int64_t tileStartY, int64_t tileStartX, int64_t currentTileH,
                                       int64_t currentTileW)
    {
        const int64_t haloRows = templateH_ + currentTileH - 1;
        const int64_t haloWidth = templateW_ + currentTileW - 1;
        const int64_t rowBytes = haloWidth * channels_ * static_cast<int64_t>(sizeof(T));
        const int64_t rowStrideElements = AlignUp(rowBytes, DATA_BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(rowBytes);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        for (int64_t row = 0; row < haloRows; ++row) {
            const int64_t imageOffset = ((n * imageH_ + tileStartY + row) * imageW_ + tileStartX) * channels_;
            if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
                AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rowStrideElements - rowBytes),
                                                           static_cast<T>(0)};
                AscendC::DataCopyPad(imageLocal_[row * rowStrideElements], imageGm_[imageOffset], copyParams,
                                     padParams);
            } else {
                AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(imageLocal_[row * rowStrideElements], imageGm_[imageOffset], copyParams,
                                     padParams);
            }
        }
        if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVectorEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVectorEvent_);
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        return rowStrideElements;
    }

    __aicore__ inline int64_t LoadPackedHalo(int64_t n, int64_t tileStartY, int64_t tileStartX, int64_t currentTileH,
                                             int64_t currentTileW)
    {
        const int64_t haloRows = templateH_ + currentTileH - 1;
        const int64_t haloWidth = templateW_ + currentTileW - 1;
        const int64_t rowBytes = haloWidth * channels_ * static_cast<int64_t>(sizeof(half));
        const int64_t rowStrideElements = AlignUp(rowBytes, DATA_BLOCK_BYTES) / static_cast<int64_t>(sizeof(half));
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams{
            true, 0, static_cast<uint8_t>(rowStrideElements - rowBytes / sizeof(half)), static_cast<half>(0)};
        for (int64_t row = 0; row < haloRows; ++row) {
            const int64_t imageOffset = ((n * imageH_ + tileStartY + row) * imageW_ + tileStartX) * channels_;
            AscendC::DataCopyPad(imageHalfLocal_[row * rowStrideElements], packedImageGm_[imageOffset], copyParams,
                                 padParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return rowStrideElements;
    }

    __aicore__ inline float FinishVectorScore(int64_t localX) const
    {
        const float primary = vectorAcc0Local_.GetValue(localX);
        if constexpr (METHOD == 0 || METHOD == 2) {
            return primary;
        } else if constexpr (METHOD == 1) {
            const float imageSquare = vectorAcc1Local_.GetValue(localX);
            const float zeroValue = primary <= NORMALIZE_EPSILON ? 0.0f : 1.0f;
            return Normalize(primary, imageSquare * templateSquareSum_, 0.0f, 1.0f, zeroValue);
        } else if constexpr (METHOD == 3) {
            const float imageSquare = vectorAcc1Local_.GetValue(localX);
            return Normalize(primary, imageSquare * templateSquareSum_, -1.0f, 1.0f, 0.0f);
        } else {
            const float spatialSize = static_cast<float>(templateH_ * templateW_);
            if constexpr (METHOD == 4) {
                if (channels_ > 1) {
                    return primary;
                }
                const float templateSum = templateChannelSumLocal_.GetValue(0);
                const float imageSum = vectorAcc1Local_.GetValue(localX);
                return primary - imageSum * templateSum / spatialSize;
            } else {
                if constexpr (METHOD == 5) {
                    return primary;
                }
                const float imageSquare = vectorAcc1Local_.GetValue(localX);
                float coefficient = primary;
                float imageVariance = imageSquare;
                if (channels_ == 1) {
                    const float templateSum = templateChannelSumLocal_.GetValue(0);
                    const float imageSum = vectorAcc2Local_.GetValue(localX);
                    coefficient -= imageSum * templateSum / spatialSize;
                    imageVariance -= imageSum * imageSum / spatialSize;
                }
                imageVariance = imageVariance < 0.0f ? 0.0f : imageVariance;
                return Normalize(coefficient, imageVariance * templateVariance_, -1.0f, 1.0f, 0.0f);
            }
        }
    }

    __aicore__ inline void NormalizeVectorScores(uint32_t count)
    {
        if constexpr (METHOD == 5) {
            constexpr float denominatorProductEpsilon = NORMALIZE_EPSILON * NORMALIZE_EPSILON;
            const uint32_t alignedCount = (count + VECTOR_FP32_ELEMENTS - 1) / VECTOR_FP32_ELEMENTS *
                                          VECTOR_FP32_ELEMENTS;
            AscendC::Muls(vectorWorkLocal_, vectorAcc1Local_, templateVariance_, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(vectorMaskLocal_, vectorWorkLocal_, denominatorProductEpsilon, AscendC::CMPMODE::GT,
                                   alignedCount);
            AscendC::Maxs(vectorWorkLocal_, vectorWorkLocal_, denominatorProductEpsilon, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(vectorWorkLocal_, vectorWorkLocal_, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(vectorAcc0Local_, vectorAcc0Local_, vectorWorkLocal_, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(vectorAcc0Local_, vectorAcc0Local_, -1.0f, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(vectorAcc0Local_, vectorAcc0Local_, 1.0f, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(vectorAcc0Local_, vectorMaskLocal_, vectorAcc0Local_, 0.0f,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Normalization for the batched C3 tile.  Unlike FinishVectorScore, this
    // keeps the denominator and clamp operations on Vector and leaves only the
    // final best-location scan to Scalar.  METHOD=3 and METHOD=5 share the
    // same image-square accumulator but use different template constants.
    __aicore__ inline void NormalizeC3VectorScores(AscendC::LocalTensor<float> primary,
                                                   AscendC::LocalTensor<float> imageSquare, uint32_t count)
    {
        if constexpr (METHOD == 3 || METHOD == 5) {
            const uint32_t alignedCount = (count + C3_VECTOR_TILE_W - 1) / C3_VECTOR_TILE_W * C3_VECTOR_TILE_W;
            const float scale = METHOD == 3 ? templateSquareSum_ : templateVariance_;
            constexpr float lower = -1.0f;
            constexpr float upper = 1.0f;
            AscendC::Muls(vectorWorkLocal_, imageSquare, scale, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            constexpr float denominatorProductEpsilon = NORMALIZE_EPSILON * NORMALIZE_EPSILON;
            AscendC::CompareScalar(vectorMaskLocal_, vectorWorkLocal_, denominatorProductEpsilon, AscendC::CMPMODE::GT,
                                   alignedCount);
            AscendC::Maxs(vectorWorkLocal_, vectorWorkLocal_, denominatorProductEpsilon, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sqrt(vectorWorkLocal_, vectorWorkLocal_, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(primary, primary, vectorWorkLocal_, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(primary, primary, lower, alignedCount);
            AscendC::Mins(primary, primary, upper, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(primary, vectorMaskLocal_, primary, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void GatherVectorInputTo(AscendC::LocalTensor<float> destination, int64_t imageElementOffset,
                                               uint32_t count)
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            constexpr int64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(float);
            const int64_t alignedImageElementOffset = imageElementOffset / elementsPerBlock * elementsPerBlock;
            const uint32_t imageBaseOffset = static_cast<uint32_t>((imageElementOffset - alignedImageElementOffset) *
                                                                   sizeof(float));
            AscendC::Gather(destination, imageLocal_[alignedImageElementOffset], vectorOffsetLocal_, imageBaseOffset,
                            count);
        } else if constexpr (AscendC::IsSameType<T, half>::value) {
            constexpr int64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(half);
            const int64_t alignedImageElementOffset = imageElementOffset / elementsPerBlock * elementsPerBlock;
            const uint32_t imageBaseOffset = static_cast<uint32_t>((imageElementOffset - alignedImageElementOffset) *
                                                                   sizeof(half));
            AscendC::Gather(vectorRawLocal_, imageLocal_[alignedImageElementOffset], vectorOffsetLocal_,
                            imageBaseOffset, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(destination, vectorRawLocal_, AscendC::RoundMode::CAST_NONE, count);
        } else {
            constexpr int64_t floatElementsPerBlock = DATA_BLOCK_BYTES / sizeof(float);
            const int64_t alignedFloatOffset = imageElementOffset / floatElementsPerBlock * floatElementsPerBlock;
            const uint32_t floatBaseOffset = static_cast<uint32_t>((imageElementOffset - alignedFloatOffset) *
                                                                   sizeof(float));
            AscendC::Gather(destination, imageFloatLocal_[alignedFloatOffset], vectorOffsetLocal_, floatBaseOffset,
                            count);
        }
    }

    __aicore__ inline void GatherVectorInput(int64_t imageElementOffset, uint32_t count)
    {
        GatherVectorInputTo(vectorInputLocal_, imageElementOffset, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherVectorPlaneTo(AscendC::LocalTensor<float> destination, int64_t imageElementOffset,
                                               uint32_t count)
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            constexpr uint32_t maxGatherElements = VECTOR_FP32_ELEMENTS;
            constexpr int64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(float);
            for (uint32_t chunk = 0; chunk < count; chunk += maxGatherElements) {
                const uint32_t chunkCount = static_cast<uint32_t>(Min(maxGatherElements, count - chunk));
                const int64_t chunkOffset = imageElementOffset + static_cast<int64_t>(chunk) * channels_;
                const int64_t alignedOffset = chunkOffset / elementsPerBlock * elementsPerBlock;
                const uint32_t baseOffset = static_cast<uint32_t>((chunkOffset - alignedOffset) * sizeof(float));
                AscendC::Gather(destination[chunk], imageLocal_[alignedOffset], vectorOffsetLocal_, baseOffset,
                                chunkCount);
            }
        } else if constexpr (AscendC::IsSameType<T, half>::value) {
            // The C3 FP16 halo is 527 elements wide (512 output columns plus
            // the 16-column template overlap).  Gather metadata and the raw
            // scratch vector are 512 elements, so split the request into
            // bounded chunks and advance the interleaved source by channels.
            constexpr uint32_t maxGatherElements = VECTOR_FP32_ELEMENTS;
            constexpr int64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(half);
            for (uint32_t chunk = 0; chunk < count; chunk += maxGatherElements) {
                const uint32_t chunkCount = static_cast<uint32_t>(Min(maxGatherElements, count - chunk));
                const int64_t chunkOffset = imageElementOffset + static_cast<int64_t>(chunk) * channels_;
                const int64_t alignedOffset = chunkOffset / elementsPerBlock * elementsPerBlock;
                const uint32_t baseOffset = static_cast<uint32_t>((chunkOffset - alignedOffset) * sizeof(half));
                AscendC::Gather(vectorRawLocal_, imageLocal_[alignedOffset], vectorOffsetLocal_, baseOffset,
                                chunkCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(destination[chunk], vectorRawLocal_, AscendC::RoundMode::CAST_NONE, chunkCount);
                if constexpr (METHOD != 2) {
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        } else {
            constexpr uint32_t maxGatherElements = VECTOR_FP32_ELEMENTS;
            constexpr int64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(float);
            for (uint32_t chunk = 0; chunk < count; chunk += maxGatherElements) {
                const uint32_t chunkCount = static_cast<uint32_t>(Min(maxGatherElements, count - chunk));
                const int64_t chunkOffset = imageElementOffset + static_cast<int64_t>(chunk) * channels_;
                const int64_t alignedOffset = chunkOffset / elementsPerBlock * elementsPerBlock;
                const uint32_t baseOffset = static_cast<uint32_t>((chunkOffset - alignedOffset) * sizeof(float));
                AscendC::Gather(destination[chunk], imageFloatLocal_[alignedOffset], vectorOffsetLocal_, baseOffset,
                                chunkCount);
            }
        }
    }

    __aicore__ inline void GatherPackedUint8PlaneTo(AscendC::LocalTensor<float> destination, int64_t imageElementOffset,
                                                    uint32_t count)
    {
        if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
            constexpr uint32_t maxGatherElements = VECTOR_FP32_ELEMENTS;
            constexpr int64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(half);
            for (uint32_t chunk = 0; chunk < count; chunk += maxGatherElements) {
                const uint32_t chunkCount = static_cast<uint32_t>(Min(maxGatherElements, count - chunk));
                const int64_t chunkOffset = imageElementOffset + static_cast<int64_t>(chunk) * channels_;
                const int64_t alignedOffset = chunkOffset / elementsPerBlock * elementsPerBlock;
                const uint32_t baseOffset = static_cast<uint32_t>((chunkOffset - alignedOffset) * sizeof(half));
                AscendC::Gather(vectorHalfLocal_, imageHalfLocal_[alignedOffset], vectorOffsetLocal_, baseOffset,
                                chunkCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(destination[chunk], vectorHalfLocal_, AscendC::RoundMode::CAST_NONE, chunkCount);
            }
        }
    }

    // Load one contiguous C3 plane segment into an aligned vector buffer.  The
    // generic path has to gather this segment once for every output row and
    // template row.  The C3 schedule calls this helper once per source halo
    // row and reuses the resulting vector for all output rows that overlap it.
    __aicore__ inline void LoadC3VectorInput(int64_t channel, int64_t templateX, uint32_t count, uint32_t alignedCount)
    {
        const int64_t planeOffset = channel * VECTOR_PLANAR_STRIDE_ELEMENTS;
        const uint32_t planeCount = count + static_cast<uint32_t>(templateW_ - 1);
        const bool alignedCopyInRange = (templateX % (DATA_BLOCK_BYTES / sizeof(float))) == 0 &&
                                        static_cast<uint32_t>(templateX) + alignedCount <= planeCount;
        if (alignedCopyInRange) {
            AscendC::DataCopy(vectorInputLocal_, vectorPlaneLocal_[planeOffset + templateX], alignedCount);
        } else {
            AscendC::Gather(vectorInputLocal_, vectorPlaneLocal_[planeOffset], vectorPlanarOffsetLocal_,
                            static_cast<uint32_t>(templateX * sizeof(float)), count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateVectorInput(AscendC::LocalTensor<float> input, float templateValue, uint32_t count)
    {
        if constexpr (METHOD == 0 || METHOD == 1) {
            AscendC::Adds(vectorWorkLocal_, input, -templateValue, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(vectorAcc0Local_, vectorWorkLocal_, vectorWorkLocal_, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (METHOD == 2 || METHOD == 3 || METHOD == 4 || METHOD == 5) {
            AscendC::Axpy(vectorAcc0Local_, input, templateValue, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (METHOD == 1 || METHOD == 3 || METHOD == 5) {
            AscendC::MulAddDst(vectorAcc1Local_, input, input, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (METHOD == 4) {
            if (channels_ == 1) {
                AscendC::Add(vectorAcc1Local_, vectorAcc1Local_, input, count);
            } else {
                AscendC::Add(vectorAcc2Local_, vectorAcc2Local_, input, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (METHOD == 5) {
            AscendC::Add(vectorAcc2Local_, vectorAcc2Local_, input, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessVectorRow(int64_t localY, int64_t tileStartY, int64_t tileStartX,
                                            int64_t currentTileW, int64_t haloRowStride, float& localBestValue,
                                            int64_t& localBestIndex)
    {
        const uint32_t count = static_cast<uint32_t>(currentTileW);
        const uint32_t alignedCount = (count + VECTOR_FP32_ELEMENTS - 1) / VECTOR_FP32_ELEMENTS * VECTOR_FP32_ELEMENTS;
        AscendC::Duplicate(vectorAcc0Local_, 0.0f, alignedCount);
        if constexpr (METHOD == 1 || METHOD == 3 || METHOD == 4 || METHOD == 5) {
            AscendC::Duplicate(vectorAcc1Local_, 0.0f, alignedCount);
        }
        if constexpr (METHOD == 5) {
            AscendC::Duplicate(vectorAcc2Local_, 0.0f, alignedCount);
        }
        if constexpr (AscendC::IsSameType<T, half>::value) {
            AscendC::Duplicate(vectorRawLocal_, static_cast<half>(0.0f), alignedCount);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (int64_t c = 0; c < channels_; ++c) {
            if constexpr (METHOD == 4 || METHOD == 5) {
                if (channels_ > 1) {
                    AscendC::Duplicate(vectorAcc2Local_, 0.0f, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            for (int64_t ty = 0; ty < templateH_; ++ty) {
                const int64_t imageRow = (localY + ty) * haloRowStride;
                const int64_t templateRow = ty * templateW_ * channels_;
                if (channels_ == VECTOR_PLANAR_CHANNELS) {
                    const uint32_t planeCount = static_cast<uint32_t>(currentTileW + templateW_ - 1);
                    for (int64_t planeChannel = 0; planeChannel < VECTOR_PLANAR_CHANNELS; ++planeChannel) {
                        GatherVectorPlaneTo(vectorPlaneLocal_[planeChannel * VECTOR_PLANAR_STRIDE_ELEMENTS],
                                            imageRow + planeChannel, planeCount);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (int64_t tx = 0; tx < templateW_; ++tx) {
                    const float templateValue = GetTemplateValue(templateRow + tx * channels_ + c);
                    if (channels_ == VECTOR_PLANAR_CHANNELS) {
                        // Gather from an aligned planar base with a byte base offset.  A
                        // LocalTensor slice at +tx can be non-32-byte aligned on Ascend 910B,
                        // which is not a valid source address for vector arithmetic or UB DMA.
                        if ((tx % (DATA_BLOCK_BYTES / sizeof(float))) == 0) {
                            AscendC::DataCopy(vectorInputLocal_,
                                              vectorPlaneLocal_[c * VECTOR_PLANAR_STRIDE_ELEMENTS + tx], alignedCount);
                            AscendC::PipeBarrier<PIPE_ALL>();
                        } else {
                            AscendC::Gather(vectorInputLocal_, vectorPlaneLocal_[c * VECTOR_PLANAR_STRIDE_ELEMENTS],
                                            vectorPlanarOffsetLocal_, static_cast<uint32_t>(tx * sizeof(float)), count);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        AccumulateVectorInput(vectorInputLocal_, templateValue, count);
                    } else {
                        const int64_t imageElementOffset = imageRow + tx * channels_ + c;
                        GatherVectorInput(imageElementOffset, count);
                        AccumulateVectorInput(vectorInputLocal_, templateValue, count);
                    }
                }
            }
            if constexpr (METHOD == 4 || METHOD == 5) {
                if (channels_ > 1) {
                    const float spatialSize = static_cast<float>(templateH_ * templateW_);
                    const float templateMeanScale = -templateChannelSumLocal_.GetValue(c) / spatialSize;
                    AscendC::Axpy(vectorAcc0Local_, vectorAcc2Local_, templateMeanScale, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (METHOD == 5) {
                        AscendC::Mul(vectorWorkLocal_, vectorAcc2Local_, vectorAcc2Local_, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Axpy(vectorAcc1Local_, vectorWorkLocal_, -1.0f / spatialSize, count);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }
        }
        if constexpr (METHOD == 5) {
            if (channels_ == 1) {
                const float spatialSize = static_cast<float>(templateH_ * templateW_);
                const float templateMeanScale = -templateChannelSumLocal_.GetValue(0) / spatialSize;
                AscendC::Axpy(vectorAcc0Local_, vectorAcc2Local_, templateMeanScale, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(vectorWorkLocal_, vectorAcc2Local_, vectorAcc2Local_, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Axpy(vectorAcc1Local_, vectorWorkLocal_, -1.0f / spatialSize, count);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        NormalizeVectorScores(count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent_);

        const int64_t outY = tileStartY + localY;
        for (int64_t localX = 0; localX < currentTileW; ++localX) {
            const int64_t outX = tileStartX + localX;
            const int64_t outputIndex = outY * outputW_ + outX;
            const float value = FinishVectorScore(localX);
            tileScoreLocal_.SetValue(localY * tileScoreRowStride_ + localX, value);
            if (IsBetter(value, outputIndex, localBestValue, localBestIndex)) {
                localBestValue = value;
                localBestIndex = outputIndex;
            }
        }
    }

    template <bool USE_UB>
    __aicore__ inline float CalculateScoreScalar(int64_t n, int64_t outY, int64_t outX, int64_t localOutY,
                                                 int64_t localOutX, int64_t haloRowStride) const
    {
        float crossSum = 0.0f;
        float imageSquareSum = 0.0f;
        float squareDifferenceSum = 0.0f;
        float coefficient = 0.0f;
        float imageVariance = 0.0f;
        const float spatialSize = static_cast<float>(templateH_ * templateW_);

        for (int64_t c = 0; c < channels_; ++c) {
            float channelCross = 0.0f;
            float channelImageSum = 0.0f;
            float channelImageSquare = 0.0f;
            for (int64_t ty = 0; ty < templateH_; ++ty) {
                const int64_t imageRow = ((n * imageH_ + outY + ty) * imageW_ + outX) * channels_;
                const int64_t templateRow = ty * templateW_ * channels_;
                const int64_t localImageRow = (localOutY + ty) * haloRowStride + localOutX * channels_;
                for (int64_t tx = 0; tx < templateW_; ++tx) {
                    const int64_t templateOffset = templateRow + tx * channels_ + c;
                    float imageValue = 0.0f;
                    float templateValue = 0.0f;
                    if constexpr (USE_UB) {
                        const int64_t imageOffset = localImageRow + tx * channels_ + c;
                        imageValue = ToFloat(imageLocal_.GetValue(imageOffset));
                        templateValue = ToFloat(templateLocal_.GetValue(templateOffset));
                    } else {
                        const int64_t imageOffset = imageRow + tx * channels_ + c;
                        imageValue = ToFloat(imageGm_.GetValue(imageOffset));
                        templateValue = ToFloat(templateGm_.GetValue(templateOffset));
                    }
                    if constexpr (METHOD == 0 || METHOD == 1) {
                        const float difference = imageValue - templateValue;
                        squareDifferenceSum += difference * difference;
                    }
                    if constexpr (METHOD == 2 || METHOD == 3 || METHOD == 4 || METHOD == 5) {
                        channelCross += imageValue * templateValue;
                    }
                    if constexpr (METHOD == 1 || METHOD == 3 || METHOD == 5) {
                        channelImageSquare += imageValue * imageValue;
                    }
                    if constexpr (METHOD == 4 || METHOD == 5) {
                        channelImageSum += imageValue;
                    }
                }
            }
            if constexpr (METHOD == 2 || METHOD == 3) {
                crossSum += channelCross;
            }
            if constexpr (METHOD == 1 || METHOD == 3) {
                imageSquareSum += channelImageSquare;
            }
            if constexpr (METHOD == 4 || METHOD == 5) {
                const float templateChannelSum = templateChannelSumLocal_.GetValue(c);
                coefficient += channelCross - channelImageSum * templateChannelSum / spatialSize;
                if constexpr (METHOD == 5) {
                    imageVariance += channelImageSquare - channelImageSum * channelImageSum / spatialSize;
                }
            }
        }

        if constexpr (METHOD == 0) {
            return squareDifferenceSum;
        } else if constexpr (METHOD == 1) {
            const float zeroValue = squareDifferenceSum <= NORMALIZE_EPSILON ? 0.0f : 1.0f;
            return Normalize(squareDifferenceSum, imageSquareSum * templateSquareSum_, 0.0f, 1.0f, zeroValue);
        } else if constexpr (METHOD == 2) {
            return crossSum;
        } else if constexpr (METHOD == 3) {
            return Normalize(crossSum, imageSquareSum * templateSquareSum_, -1.0f, 1.0f, 0.0f);
        } else if constexpr (METHOD == 4) {
            return coefficient;
        } else {
            imageVariance = imageVariance < 0.0f ? 0.0f : imageVariance;
            return Normalize(coefficient, imageVariance * templateVariance_, -1.0f, 1.0f, 0.0f);
        }
    }

    template <bool USE_UB>
    __aicore__ inline float CalculateScore(int64_t n, int64_t outY, int64_t outX, int64_t localOutY, int64_t localOutX,
                                           int64_t haloRowStride)
    {
        return CalculateScoreScalar<USE_UB>(n, outY, outX, localOutY, localOutX, haloRowStride);
    }

    // C3/16x16 medium-window path.  The generic implementation walks
    // output-row -> template-row -> template-column and gathers the same
    // interleaved source row once for every output row.  Here the loop is
    // changed to channel -> source-row -> template-column: one gathered plane
    // row is reused by every output row whose halo overlaps it.  The numerical
    // accumulation order for each output remains channel-major and then
    // template-row-major, matching the generic Vector path.
    __aicore__ inline void ProcessC3VectorTile(int64_t tileStartY, int64_t tileStartX, int64_t currentTileH,
                                               int64_t currentTileW, int64_t haloRowStride, float& localBestValue,
                                               int64_t& localBestIndex)
    {
        const uint32_t count = static_cast<uint32_t>(currentTileW);
        // FP16 uses the full 512-wide Vector tile; UINT8 keeps the 256-wide
        // tile that fits together with its FP32 halo conversion buffer.
        const uint32_t alignedCount = currentTileW > 256 ? 512 : 256;
        const int64_t accumulatorStride = C3_VECTOR_TILE_W;
        const int64_t haloRows = templateH_ + currentTileH - 1;
        const uint32_t planeCount = static_cast<uint32_t>(currentTileW + templateW_ - 1);
        const float spatialSize = static_cast<float>(templateH_ * templateW_);

        for (int64_t localY = 0; localY < currentTileH; ++localY) {
            AscendC::Duplicate(c3VectorAcc0Local_[localY * accumulatorStride], 0.0f, alignedCount);
            if constexpr (METHOD == 3 || METHOD == 5) {
                AscendC::Duplicate(c3VectorAcc0Local_[C3_VECTOR_ACC_ELEMENTS + localY * accumulatorStride], 0.0f,
                                   alignedCount);
            }
            if constexpr (METHOD == 4 || METHOD == 5) {
                for (int64_t channel = 0; channel < VECTOR_PLANAR_CHANNELS; ++channel) {
                    AscendC::Duplicate(
                        c3VectorAcc0Local_[C3_VECTOR_CHANNEL_SUM_OFFSET + channel * C3_VECTOR_ACC_ELEMENTS +
                                           localY * accumulatorStride],
                        0.0f, alignedCount);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (int64_t channel = 0; channel < VECTOR_PLANAR_CHANNELS; ++channel) {
            for (int64_t sourceRow = 0; sourceRow < haloRows; ++sourceRow) {
                const int64_t sourceOffset = sourceRow * haloRowStride + channel;
                if constexpr (AscendC::IsSameType<T, uint8_t>::value && (METHOD == 2 || METHOD == 5)) {
                    GatherPackedUint8PlaneTo(vectorPlaneLocal_[channel * VECTOR_PLANAR_STRIDE_ELEMENTS], sourceOffset,
                                             planeCount);
                } else {
                    GatherVectorPlaneTo(vectorPlaneLocal_[channel * VECTOR_PLANAR_STRIDE_ELEMENTS], sourceOffset,
                                        planeCount);
                }
                if constexpr (!AscendC::IsSameType<T, half>::value || METHOD != 2) {
                    AscendC::PipeBarrier<PIPE_V>();
                }

                const int64_t firstOutputRow = sourceRow >= templateH_ - 1 ? sourceRow - (templateH_ - 1) : 0;
                const int64_t lastOutputRow = Min(currentTileH - 1, sourceRow);
                if (firstOutputRow > lastOutputRow) {
                    continue;
                }
                if constexpr (METHOD == 3 && AscendC::IsSameType<T, half>::value) {
                    // The same source row contributes to every overlapping
                    // output row.  Form its horizontal 16-column square sum
                    // once, then reuse that vector below instead of squaring
                    // the identical input separately for each output row.
                    AscendC::Duplicate(vectorWorkLocal_, 0.0f, alignedCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (int64_t templateX = 0; templateX < templateW_; ++templateX) {
                    LoadC3VectorInput(channel, templateX, count, alignedCount);
                    for (int64_t localY = firstOutputRow; localY <= lastOutputRow; ++localY) {
                        const int64_t templateY = sourceRow - localY;
                        const int64_t templateOffset = templateY * templateW_ * channels_ + templateX * channels_ +
                                                       channel;
                        const float templateValue = GetTemplateValue(templateOffset);
                        // All updates below consume the same input vector. Keep
                        // each accumulator family in the Vector pipe and
                        // synchronize once per family rather than once per
                        // output row. The accumulator dependency is ordered by
                        // the Vector pipe itself.
                        AscendC::Axpy(c3VectorAcc0Local_[localY * accumulatorStride], vectorInputLocal_, templateValue,
                                      count);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (METHOD == 3 && AscendC::IsSameType<T, half>::value) {
                        AscendC::MulAddDst(vectorWorkLocal_, vectorInputLocal_, vectorInputLocal_, count);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else if constexpr (METHOD == 3 || METHOD == 5) {
                        for (int64_t localY = firstOutputRow; localY <= lastOutputRow; ++localY) {
                            AscendC::MulAddDst(c3VectorAcc0Local_[C3_VECTOR_ACC_ELEMENTS + localY * accumulatorStride],
                                               vectorInputLocal_, vectorInputLocal_, count);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    if constexpr (METHOD == 4 || METHOD == 5) {
                        for (int64_t localY = firstOutputRow; localY <= lastOutputRow; ++localY) {
                            AscendC::Add(
                                c3VectorAcc0Local_[C3_VECTOR_CHANNEL_SUM_OFFSET + channel * C3_VECTOR_ACC_ELEMENTS +
                                                   localY * accumulatorStride],
                                vectorInputLocal_,
                                c3VectorAcc0Local_[C3_VECTOR_CHANNEL_SUM_OFFSET + channel * C3_VECTOR_ACC_ELEMENTS +
                                                   localY * accumulatorStride],
                                count);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                if constexpr (METHOD == 3 && AscendC::IsSameType<T, half>::value) {
                    for (int64_t localY = firstOutputRow; localY <= lastOutputRow; ++localY) {
                        AscendC::Add(c3VectorAcc0Local_[C3_VECTOR_ACC_ELEMENTS + localY * accumulatorStride],
                                     vectorWorkLocal_,
                                     c3VectorAcc0Local_[C3_VECTOR_ACC_ELEMENTS + localY * accumulatorStride], count);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }

        for (int64_t localY = 0; localY < currentTileH; ++localY) {
            AscendC::LocalTensor<float> primary = c3VectorAcc0Local_[localY * accumulatorStride];
            if constexpr (METHOD == 4) {
                for (int64_t channel = 0; channel < VECTOR_PLANAR_CHANNELS; ++channel) {
                    AscendC::LocalTensor<float> channelSum = c3VectorAcc0Local_[C3_VECTOR_CHANNEL_SUM_OFFSET +
                                                                                channel * C3_VECTOR_ACC_ELEMENTS +
                                                                                localY * accumulatorStride];
                    const float templateMeanScale = -templateChannelSumLocal_.GetValue(channel) / spatialSize;
                    AscendC::Axpy(primary, channelSum, templateMeanScale, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            } else if constexpr (METHOD == 5) {
                for (int64_t channel = 0; channel < VECTOR_PLANAR_CHANNELS; ++channel) {
                    AscendC::LocalTensor<float> channelSum = c3VectorAcc0Local_[C3_VECTOR_CHANNEL_SUM_OFFSET +
                                                                                channel * C3_VECTOR_ACC_ELEMENTS +
                                                                                localY * accumulatorStride];
                    const float templateMeanScale = -templateChannelSumLocal_.GetValue(channel) / spatialSize;
                    AscendC::Axpy(primary, channelSum, templateMeanScale, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (METHOD == 5) {
                        AscendC::Mul(vectorWorkLocal_, channelSum, channelSum, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Axpy(c3VectorAcc0Local_[C3_VECTOR_ACC_ELEMENTS + localY * accumulatorStride],
                                      vectorWorkLocal_, -1.0f / spatialSize, count);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }
            if constexpr (METHOD == 3 || METHOD == 5) {
                NormalizeC3VectorScores(primary,
                                        c3VectorAcc0Local_[C3_VECTOR_ACC_ELEMENTS + localY * accumulatorStride], count);
            }

            const int64_t outY = tileStartY + localY;
            AscendC::ReduceMax(vectorWorkLocal_, primary, vectorInputLocal_, static_cast<int32_t>(count), true);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float rowBestValue = vectorWorkLocal_.GetValue(0);
            const int32_t rowBestX = vectorWorkLocal_.template ReinterpretCast<int32_t>().GetValue(1);
            const int64_t outputIndex = outY * outputW_ + tileStartX + rowBestX;
            if (IsBetter(rowBestValue, outputIndex, localBestValue, localBestIndex)) {
                localBestValue = rowBestValue;
                localBestIndex = outputIndex;
            }
        }
    }

    __aicore__ inline void ProcessTile(int64_t n, int64_t tileIndex, float& localBestValue, int64_t& localBestIndex)
    {
        const int64_t tileRow = tileIndex / tileCols_;
        const int64_t tileCol = tileIndex - tileRow * tileCols_;
        const int64_t tileStartY = tileRow * tileH_;
        const int64_t tileStartX = tileCol * tileW_;
        const int64_t currentTileH = Min(tileH_, outputH_ - tileStartY);
        const int64_t currentTileW = Min(tileW_, outputW_ - tileStartX);
        int64_t haloRowStride = 0;
        if (useHaloTile_) {
            if (useCompactUint8C3_) {
                haloRowStride = LoadPackedHalo(n, tileStartY, tileStartX, currentTileH, currentTileW);
            } else {
                haloRowStride = LoadHalo(n, tileStartY, tileStartX, currentTileH, currentTileW);
            }
            if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
                if (haloFloatBufferBytes_ > 0 && !useCompactUint8C3_) {
                    const int64_t haloRows = templateH_ + currentTileH - 1;
                    ConvertUint8HaloToFloat(haloRows, haloRowStride);
                }
            }
        }

        if (useC3VectorTile_ && (currentTileH == C3_VECTOR_TILE_H || METHOD != 2 || useCompactUint8C3_)) {
            ProcessC3VectorTile(tileStartY, tileStartX, currentTileH, currentTileW, haloRowStride, localBestValue,
                                localBestIndex);
            AscendC::PipeBarrier<PIPE_ALL>();
            const int64_t outputsPerBatch = outputH_ * outputW_;
            AscendC::DataCopyExtParams scoreCopy{1, static_cast<uint32_t>(currentTileW * sizeof(float)), 0, 0, 0};
            for (int64_t localY = 0; localY < currentTileH; ++localY) {
                const int64_t scoreOffset = n * outputsPerBatch + (tileStartY + localY) * outputW_ + tileStartX;
                AscendC::DataCopyPad(scoreGm_[scoreOffset], c3VectorAcc0Local_[localY * C3_VECTOR_TILE_W], scoreCopy);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        bool vectorProcessed = false;
        if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value ||
                      AscendC::IsSameType<T, uint8_t>::value) {
            if (useHaloTile_) {
                for (int64_t localY = 0; localY < currentTileH; ++localY) {
                    ProcessVectorRow(localY, tileStartY, tileStartX, currentTileW, haloRowStride, localBestValue,
                                     localBestIndex);
                }
                vectorProcessed = true;
            }
        }

        const int64_t outputsPerBatch = outputH_ * outputW_;
        if (!vectorProcessed) {
            for (int64_t localY = 0; localY < currentTileH; ++localY) {
                for (int64_t localX = 0; localX < currentTileW; ++localX) {
                    const int64_t outY = tileStartY + localY;
                    const int64_t outX = tileStartX + localX;
                    const int64_t outputIndex = outY * outputW_ + outX;
                    const float value = useHaloTile_ ?
                                            CalculateScore<true>(n, outY, outX, localY, localX, haloRowStride) :
                                            CalculateScore<false>(n, outY, outX, 0, 0, 0);
                    tileScoreLocal_.SetValue(localY * tileScoreRowStride_ + localX, value);
                    if (IsBetter(value, outputIndex, localBestValue, localBestIndex)) {
                        localBestValue = value;
                        localBestIndex = outputIndex;
                    }
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams scoreCopy{1, static_cast<uint32_t>(currentTileW * sizeof(float)), 0, 0, 0};
        for (int64_t localY = 0; localY < currentTileH; ++localY) {
            const int64_t scoreOffset = n * outputsPerBatch + (tileStartY + localY) * outputW_ + tileStartX;
            AscendC::DataCopyPad(scoreGm_[scoreOffset], tileScoreLocal_[localY * tileScoreRowStride_], scoreCopy);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void FlushLocalBest(int64_t n, float value, int64_t index)
    {
        if (n < 0 || index < 0) {
            return;
        }
        const int64_t workspaceIndex = n * usedCores_ + blockIdx_;
        workspaceScoreLocal_.SetValue(0, value);
        workspaceIndexLocal_.SetValue(0, index);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams scoreCopy{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyExtParams indexCopy{1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPad(localBestScoreGm_[workspaceIndex], workspaceScoreLocal_, scoreCopy);
        AscendC::DataCopyPad(localBestIndexGm_[workspaceIndex], workspaceIndexLocal_, indexCopy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReduceBestResults()
    {
        for (int64_t n = 0; n < batch_; ++n) {
            float bestValue = 0.0f;
            int64_t bestIndex = -1;
            for (int64_t core = 0; core < usedCores_; ++core) {
                const int64_t workspaceIndex = n * usedCores_ + core;
                const int64_t candidateIndex = localBestIndexGm_.GetValue(workspaceIndex);
                if (candidateIndex < 0) {
                    continue;
                }
                const float candidateValue = localBestScoreGm_.GetValue(workspaceIndex);
                if (IsBetter(candidateValue, candidateIndex, bestValue, bestIndex)) {
                    bestValue = candidateValue;
                    bestIndex = candidateIndex;
                }
            }
            const int64_t bestY = bestIndex / outputW_;
            const int64_t bestX = bestIndex - bestY * outputW_;
            workspaceScoreLocal_.SetValue(0, bestValue);
            workspaceIndexLocal_.SetValue(0, bestX);
            workspaceIndexLocal_.SetValue(1, bestY);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams scoreCopy{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyExtParams locationCopy{1, static_cast<uint32_t>(2 * sizeof(int64_t)), 0, 0, 0};
            AscendC::DataCopyPad(bestScoreGm_[n], workspaceScoreLocal_, scoreCopy);
            AscendC::DataCopyPad(bestLocationGm_[n * 2], workspaceIndexLocal_, locationCopy);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECIN> templateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> imageBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> templateStatsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vectorBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vectorPlaneBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> templateFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> imageFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> c3VectorBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tileScoreBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workspaceScoreBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workspaceIndexBuf_;
    AscendC::LocalTensor<T> templateLocal_;
    AscendC::LocalTensor<T> imageLocal_;
    AscendC::LocalTensor<float> templateChannelSumLocal_;
    AscendC::LocalTensor<float> vectorWorkLocal_;
    AscendC::LocalTensor<float> vectorAcc0Local_;
    AscendC::LocalTensor<float> vectorAcc1Local_;
    AscendC::LocalTensor<float> vectorAcc2Local_;
    AscendC::LocalTensor<float> vectorInputLocal_;
    AscendC::LocalTensor<float> vectorPlaneLocal_;
    AscendC::LocalTensor<uint32_t> vectorOffsetLocal_;
    AscendC::LocalTensor<uint32_t> vectorPlanarOffsetLocal_;
    AscendC::LocalTensor<uint8_t> vectorMaskLocal_;
    AscendC::LocalTensor<T> vectorRawLocal_;
    AscendC::LocalTensor<half> vectorHalfLocal_;
    AscendC::LocalTensor<float> templateFloatLocal_;
    AscendC::LocalTensor<float> imageFloatLocal_;
    AscendC::LocalTensor<half> imageHalfLocal_;
    AscendC::LocalTensor<float> c3VectorAcc0Local_;
    AscendC::LocalTensor<float> tileScoreLocal_;
    AscendC::LocalTensor<float> workspaceScoreLocal_;
    AscendC::LocalTensor<int64_t> workspaceIndexLocal_;
    AscendC::GlobalTensor<T> imageGm_;
    AscendC::GlobalTensor<T> templateGm_;
    AscendC::GlobalTensor<float> scoreGm_;
    AscendC::GlobalTensor<float> bestScoreGm_;
    AscendC::GlobalTensor<int64_t> bestLocationGm_;
    AscendC::GlobalTensor<float> localBestScoreGm_;
    AscendC::GlobalTensor<int64_t> localBestIndexGm_;
    AscendC::GlobalTensor<half> packedImageGm_;
    event_t mte2ToVectorEvent_;
    event_t vectorToScalarEvent_;
    event_t scalarToVectorEvent_;
    event_t vectorToMte3Event_;
    int64_t batch_ = 0;
    int64_t imageH_ = 0;
    int64_t imageW_ = 0;
    int64_t channels_ = 0;
    int64_t templateH_ = 0;
    int64_t templateW_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t tileH_ = 0;
    int64_t tileW_ = 0;
    int64_t tileCols_ = 0;
    int64_t tilesPerBatch_ = 0;
    int64_t totalTiles_ = 0;
    int64_t usedCores_ = 0;
    int64_t templateElements_ = 0;
    int64_t templateBufferBytes_ = 0;
    int64_t haloBufferBytes_ = 0;
    int64_t templateFloatBufferBytes_ = 0;
    int64_t haloFloatBufferBytes_ = 0;
    int64_t packedImageBytes_ = 0;
    int64_t prepackScratchBytes_ = 0;
    int64_t templateStatsBufferBytes_ = 0;
    int64_t blockIdx_ = 0;
    int64_t tileScoreRowStride_ = 0;
    float templateSquareSum_ = 0.0f;
    float templateVariance_ = 0.0f;
    bool useHaloTile_ = false;
    bool useC3VectorTile_ = false;
    bool useCompactUint8C3_ = false;
};

} // namespace NsMatchTemplate
#endif // MATCH_TEMPLATE_H
