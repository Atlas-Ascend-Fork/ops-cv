/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_CV_EXPERIMENTAL_IMAGE_DISTANCE_TRANSFORM_H_
#define OPS_CV_EXPERIMENTAL_IMAGE_DISTANCE_TRANSFORM_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "distance_transform_tiling_data.h"
#include "distance_transform_tiling_key.h"

namespace NsDistanceTransform {
using namespace AscendC;

constexpr uint32_t DISTANCE_L1 = 0U;
constexpr uint32_t DISTANCE_L2 = 1U;
constexpr uint32_t DISTANCE_CHESSBOARD = 2U;
constexpr uint32_t DMA_ALIGN_BYTES = 32U;
constexpr uint32_t MASK_PRECISE = 0U;
constexpr uint32_t MASK_3 = 3U;
constexpr uint32_t MASK_5 = 5U;
constexpr uint32_t PACKED_INPUT_BYTES = 4U;
constexpr uint32_t L1_HORIZONTAL_GROUP_ROWS = 3U;
constexpr uint32_t VECTOR_ROW_BUFFER = 2U;
constexpr uint32_t CHAMFER_VECTOR_ROW_BUFFER = 3U;
constexpr uint32_t MASK3_GUARD_BLOCK_ELEMS = 32U;
constexpr uint32_t MASK3_VECTOR_PROPAGATION_LIMIT = 2U * MASK3_GUARD_BLOCK_ELEMS;
constexpr uint32_t FLOAT_BLOCK_ELEMS = DMA_ALIGN_BYTES / sizeof(float);
constexpr uint32_t FLOAT_VECTOR_REPEAT_ELEMS = 64U;
constexpr float L2_MASK_3_AXIAL = 0.955F;
constexpr float L2_MASK_3_DIAGONAL = 1.3693F;
constexpr float L2_MASK_5_KNIGHT = 2.1969F;
constexpr float L2_MASK_5_DIAGONAL = 1.4F;

class KernelDistanceTransform {
public:
    __aicore__ inline KernelDistanceTransform() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const DistanceTransformTilingData* tilingData,
                                TPipe* pipe)
    {
        imageCount_ = tilingData->imageCount;
        height_ = tilingData->height;
        width_ = tilingData->width;
        imageStride_ = tilingData->imageStride;
        distanceType_ = tilingData->distanceType;
        maskSize_ = tilingData->maskSize;
        usedCoreNum_ = tilingData->usedCoreNum;
        tileRows_ = tilingData->tileRows;
        vectorTileElems_ = tilingData->vectorTileElems;
        useRowBuffer_ = tilingData->useRowBuffer;
        exactTileColumns_ = tilingData->exactTileColumns;
        blockId_ = GetBlockIdx();

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x), imageCount_ * imageStride_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), imageCount_ * imageStride_);

        (void)workspace;
        if (distanceType_ == DISTANCE_L2 && maskSize_ == MASK_PRECISE) {
            const uint32_t indexBytes = AlignUp32(height_ * sizeof(int32_t));
            const uint32_t breakBytes = AlignUp32((height_ + 1U) * sizeof(float));
            const uint32_t valueBytes = AlignUp32(height_ * sizeof(float));
            pipe->InitBuffer(envelopeIndexBuf_, indexBytes);
            pipe->InitBuffer(envelopeBreakBuf_, breakBytes);
            pipe->InitBuffer(envelopeValueBuf_, valueBytes);
            if (exactTileColumns_ != 0U) {
                pipe->InitBuffer(inputRowBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
                pipe->InitBuffer(inputRowBuf1_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
                pipe->InitBuffer(inputRowBuf2_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
                pipe->InitBuffer(rowBuf0_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf1_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf2_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf3_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf4_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf5_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf6_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(inputHalfBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(half))));
                pipe->InitBuffer(vectorIndexBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
                pipe->InitBuffer(exactTileBuf_, height_ * exactTileColumns_ * sizeof(float));
            } else {
                pipe->InitBuffer(sqrtQueue_, 2U, vectorTileElems_ * sizeof(float));
            }
        } else if ((distanceType_ == DISTANCE_L1 || distanceType_ == DISTANCE_CHESSBOARD) && useRowBuffer_ != 0U) {
            pipe->InitBuffer(inputRowBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
            pipe->InitBuffer(inputRowBuf1_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
            pipe->InitBuffer(inputRowBuf2_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
            pipe->InitBuffer(rowBuf0_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf1_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf2_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            if (useRowBuffer_ == VECTOR_ROW_BUFFER) {
                pipe->InitBuffer(rowBuf3_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf4_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf5_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(rowBuf6_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
                pipe->InitBuffer(inputHalfBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(half))));
                pipe->InitBuffer(vectorIndexBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
                if (distanceType_ == DISTANCE_CHESSBOARD) {
                    pipe->InitBuffer(chessIndexRightBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
                }
            } else if (distanceType_ == DISTANCE_CHESSBOARD) {
                pipe->InitBuffer(vectorIndexBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
                pipe->InitBuffer(chessIndexRightBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
            }
        } else if (distanceType_ == DISTANCE_L2 && useRowBuffer_ == CHAMFER_VECTOR_ROW_BUFFER) {
            pipe->InitBuffer(inputRowBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
            pipe->InitBuffer(rowBuf0_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf1_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf2_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf3_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf4_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf5_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf6_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(inputHalfBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(half))));
            pipe->InitBuffer(vectorIndexBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
            pipe->InitBuffer(chessIndexRightBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
            pipe->InitBuffer(chamferPrefixIndexBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(int32_t))));
        } else if (useRowBuffer_ != 0U) {
            pipe->InitBuffer(inputRowBuf_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(uint8_t))));
            pipe->InitBuffer(rowBuf0_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf1_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
            pipe->InitBuffer(rowBuf2_, AlignUp32(static_cast<uint32_t>(width_ * sizeof(float))));
        }
    }

    __aicore__ inline void Process()
    {
        if (distanceType_ == DISTANCE_L2 && maskSize_ == MASK_PRECISE && exactTileColumns_ != 0U) {
            ProcessExactL2Tasks();
            return;
        }
        const uint32_t baseImages = imageCount_ / usedCoreNum_;
        const uint32_t extraImages = imageCount_ % usedCoreNum_;
        const uint32_t myImageCount = baseImages + (blockId_ < extraImages ? 1U : 0U);
        const uint32_t firstImage = blockId_ * baseImages + (blockId_ < extraImages ? blockId_ : extraImages);
        for (uint32_t localImage = 0; localImage < myImageCount; ++localImage) {
            const uint32_t imageOffset = (firstImage + localImage) * imageStride_;
            if (distanceType_ == DISTANCE_L2 && maskSize_ == MASK_PRECISE) {
                ProcessL2(imageOffset);
            } else if ((distanceType_ == DISTANCE_L1 || distanceType_ == DISTANCE_CHESSBOARD) &&
                       useRowBuffer_ == VECTOR_ROW_BUFFER) {
                ProcessVectorSeparable(imageOffset, distanceType_ == DISTANCE_CHESSBOARD);
            } else if ((distanceType_ == DISTANCE_L1 || distanceType_ == DISTANCE_CHESSBOARD) && useRowBuffer_ != 0U) {
                ProcessBufferedSeparable(imageOffset, distanceType_ == DISTANCE_CHESSBOARD);
            } else if (distanceType_ == DISTANCE_L2 && useRowBuffer_ == CHAMFER_VECTOR_ROW_BUFFER) {
                ProcessL2ChamferVector(imageOffset);
            } else if (useRowBuffer_ != 0U) {
                ProcessBufferedTwoPass(imageOffset);
            } else {
                ProcessGmTwoPass(imageOffset);
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t bytes) const
    {
        return (bytes + DMA_ALIGN_BYTES - 1U) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
    }
    __aicore__ inline float Min(float lhs, float rhs) const { return lhs < rhs ? lhs : rhs; }
    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }
    __aicore__ inline float UintToFloat(uint32_t value) const
    {
        return static_cast<float>(static_cast<int32_t>(value));
    }

    __aicore__ inline bool UsesDiagonal() const { return distanceType_ != DISTANCE_L1; }
    __aicore__ inline bool UsesKnight() const { return distanceType_ == DISTANCE_L2 && maskSize_ == MASK_5; }
    __aicore__ inline float AxialCost() const
    {
        return distanceType_ == DISTANCE_L2 && maskSize_ == MASK_3 ? L2_MASK_3_AXIAL : 1.0F;
    }
    __aicore__ inline float DiagonalCost() const
    {
        if (distanceType_ != DISTANCE_L2) {
            return 1.0F;
        }
        return maskSize_ == MASK_3 ? L2_MASK_3_DIAGONAL : L2_MASK_5_DIAGONAL;
    }
    __aicore__ inline float DistanceInfinity() const
    {
        return distanceType_ == DISTANCE_CHESSBOARD ? UintToFloat(height_ > width_ ? height_ : width_) :
                                                      UintToFloat(height_) + UintToFloat(width_);
    }

    __aicore__ inline LocalTensor<float> GetBufferedRow(uint32_t slot)
    {
        if (slot == 0U) {
            return rowBuf0_.Get<float>();
        }
        if (slot == 1U) {
            return rowBuf1_.Get<float>();
        }
        return rowBuf2_.Get<float>();
    }

    __aicore__ inline LocalTensor<uint8_t> GetBufferedInputRow(uint32_t slot)
    {
        if (slot == 0U) {
            return inputRowBuf_.Get<uint8_t>();
        }
        if (slot == 1U) {
            return inputRowBuf1_.Get<uint8_t>();
        }
        return inputRowBuf2_.Get<uint8_t>();
    }

    __aicore__ inline LocalTensor<float> GetL1VectorRow(uint32_t bank, uint32_t lane)
    {
        const uint32_t slot = bank * L1_HORIZONTAL_GROUP_ROWS + lane;
        if (slot == 0U) {
            return rowBuf0_.Get<float>();
        }
        if (slot == 1U) {
            return rowBuf1_.Get<float>();
        }
        if (slot == 2U) {
            return rowBuf2_.Get<float>();
        }
        if (slot == 3U) {
            return rowBuf3_.Get<float>();
        }
        if (slot == 4U) {
            return rowBuf4_.Get<float>();
        }
        return rowBuf5_.Get<float>();
    }

    __aicore__ inline void LoadInputRow(uint32_t rowOffset, const LocalTensor<uint8_t>& inputRow)
    {
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(width_ * sizeof(uint8_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(inputRow, xGm_[rowOffset], copyParams, padParams);
    }

    __aicore__ inline void LoadDistanceRow(uint32_t rowOffset, const LocalTensor<float>& distanceRow)
    {
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(width_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
        DataCopyPad(distanceRow, yGm_[rowOffset], copyParams, padParams);
    }

    __aicore__ inline void StoreDistanceRow(uint32_t rowOffset, const LocalTensor<float>& distanceRow)
    {
        PipeBarrier<PIPE_ALL>();
        StoreDistanceRowAsync(rowOffset, distanceRow);
        // Complete MTE3 before scalar or vector code reuses this UB storage.
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreDistanceRowAsync(uint32_t rowOffset, const LocalTensor<float>& distanceRow)
    {
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(width_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(yGm_[rowOffset], distanceRow, copyParams);
    }

    __aicore__ inline void EmitL1Background(const LocalTensor<float>& distanceRow, uint32_t col,
                                            uint32_t& lastBackground)
    {
        if (lastBackground == width_) {
            for (uint32_t pos = 0U; pos <= col; ++pos) {
                distanceRow.SetValue(pos, UintToFloat(col - pos));
            }
        } else {
            const uint32_t gap = col - lastBackground;
            for (uint32_t step = 1U; step <= gap; ++step) {
                const uint32_t reverseStep = gap - step;
                distanceRow.SetValue(lastBackground + step, UintToFloat(step < reverseStep ? step : reverseStep));
            }
        }
        lastBackground = col;
    }

    __aicore__ inline void FinishHorizontalL1Row(const LocalTensor<float>& distanceRow, uint32_t lastBackground,
                                                 float infinity)
    {
        if (lastBackground == width_) {
            Duplicate(distanceRow, infinity, width_);
            return;
        }
        for (uint32_t pos = lastBackground + 1U; pos < width_; ++pos) {
            distanceRow.SetValue(pos, UintToFloat(pos - lastBackground));
        }
    }

    __aicore__ inline void HorizontalL1Row(const LocalTensor<uint8_t>& inputRow, const LocalTensor<float>& distanceRow,
                                           float infinity)
    {
        // Emit each output exactly once. The previous implementation first
        // wrote the left distance and then read/updated the row in a reverse
        // pass. Scanning background positions and filling the intervals
        // removes that complete UB read/update pass.
        uint32_t lastBackground = width_;
        LocalTensor<uint32_t> packedInput = inputRow.ReinterpretCast<uint32_t>();
        const uint32_t packedCount = width_ / PACKED_INPUT_BYTES;
        for (uint32_t packedIndex = 0U; packedIndex < packedCount; ++packedIndex) {
            const uint32_t value = packedInput.GetValue(packedIndex);
            const uint32_t baseCol = packedIndex * PACKED_INPUT_BYTES;
            if ((value & 0x000000FFU) == 0U) {
                EmitL1Background(distanceRow, baseCol, lastBackground);
            }
            if ((value & 0x0000FF00U) == 0U) {
                EmitL1Background(distanceRow, baseCol + 1U, lastBackground);
            }
            if ((value & 0x00FF0000U) == 0U) {
                EmitL1Background(distanceRow, baseCol + 2U, lastBackground);
            }
            if ((value & 0xFF000000U) == 0U) {
                EmitL1Background(distanceRow, baseCol + 3U, lastBackground);
            }
        }
        for (uint32_t col = packedCount * PACKED_INPUT_BYTES; col < width_; ++col) {
            if (inputRow.GetValue(col) == 0U) {
                EmitL1Background(distanceRow, col, lastBackground);
            }
        }

        FinishHorizontalL1Row(distanceRow, lastBackground, infinity);
    }

    __aicore__ inline void HorizontalL1Rows3(const LocalTensor<uint8_t>& input0, const LocalTensor<uint8_t>& input1,
                                             const LocalTensor<uint8_t>& input2, const LocalTensor<float>& output0,
                                             const LocalTensor<float>& output1, const LocalTensor<float>& output2,
                                             float infinity)
    {
        LocalTensor<uint32_t> packed0 = input0.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> packed1 = input1.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> packed2 = input2.ReinterpretCast<uint32_t>();
        uint32_t last0 = width_;
        uint32_t last1 = width_;
        uint32_t last2 = width_;
        const uint32_t packedCount = width_ / PACKED_INPUT_BYTES;

        // Load and inspect three independent rows together. The separate
        // background chains give the scalar pipeline useful instruction-level
        // parallelism instead of waiting on one row's dependent scan.
        for (uint32_t packedIndex = 0U; packedIndex < packedCount; ++packedIndex) {
            const uint32_t value0 = packed0.GetValue(packedIndex);
            const uint32_t value1 = packed1.GetValue(packedIndex);
            const uint32_t value2 = packed2.GetValue(packedIndex);
            const uint32_t baseCol = packedIndex * PACKED_INPUT_BYTES;

            if ((value0 & 0x000000FFU) == 0U) {
                EmitL1Background(output0, baseCol, last0);
            }
            if ((value1 & 0x000000FFU) == 0U) {
                EmitL1Background(output1, baseCol, last1);
            }
            if ((value2 & 0x000000FFU) == 0U) {
                EmitL1Background(output2, baseCol, last2);
            }
            if ((value0 & 0x0000FF00U) == 0U) {
                EmitL1Background(output0, baseCol + 1U, last0);
            }
            if ((value1 & 0x0000FF00U) == 0U) {
                EmitL1Background(output1, baseCol + 1U, last1);
            }
            if ((value2 & 0x0000FF00U) == 0U) {
                EmitL1Background(output2, baseCol + 1U, last2);
            }
            if ((value0 & 0x00FF0000U) == 0U) {
                EmitL1Background(output0, baseCol + 2U, last0);
            }
            if ((value1 & 0x00FF0000U) == 0U) {
                EmitL1Background(output1, baseCol + 2U, last1);
            }
            if ((value2 & 0x00FF0000U) == 0U) {
                EmitL1Background(output2, baseCol + 2U, last2);
            }
            if ((value0 & 0xFF000000U) == 0U) {
                EmitL1Background(output0, baseCol + 3U, last0);
            }
            if ((value1 & 0xFF000000U) == 0U) {
                EmitL1Background(output1, baseCol + 3U, last1);
            }
            if ((value2 & 0xFF000000U) == 0U) {
                EmitL1Background(output2, baseCol + 3U, last2);
            }
        }

        for (uint32_t col = packedCount * PACKED_INPUT_BYTES; col < width_; ++col) {
            if (input0.GetValue(col) == 0U) {
                EmitL1Background(output0, col, last0);
            }
            if (input1.GetValue(col) == 0U) {
                EmitL1Background(output1, col, last1);
            }
            if (input2.GetValue(col) == 0U) {
                EmitL1Background(output2, col, last2);
            }
        }
        FinishHorizontalL1Row(output0, last0, infinity);
        FinishHorizontalL1Row(output1, last1, infinity);
        FinishHorizontalL1Row(output2, last2, infinity);
    }

    __aicore__ inline void ProcessHorizontalL1(uint32_t imageOffset, float infinity)
    {
        // Solve three independent rows together so their scalar dependency
        // chains can overlap and one set of pipeline barriers covers all rows.
        for (uint32_t groupStart = 0U; groupStart < height_; groupStart += L1_HORIZONTAL_GROUP_ROWS) {
            const uint32_t groupRows = MinU32(L1_HORIZONTAL_GROUP_ROWS, height_ - groupStart);
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                const uint32_t row = groupStart + lane;
                LoadInputRow(imageOffset + row * width_, GetBufferedInputRow(lane));
            }
            PipeBarrier<PIPE_ALL>();
            if (groupRows == L1_HORIZONTAL_GROUP_ROWS) {
                HorizontalL1Rows3(GetBufferedInputRow(0U), GetBufferedInputRow(1U), GetBufferedInputRow(2U),
                                  GetBufferedRow(0U), GetBufferedRow(1U), GetBufferedRow(2U), infinity);
            } else {
                for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                    HorizontalL1Row(GetBufferedInputRow(lane), GetBufferedRow(lane), infinity);
                }
            }
            PipeBarrier<PIPE_ALL>();
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                const uint32_t row = groupStart + lane;
                StoreDistanceRowAsync(imageOffset + row * width_, GetBufferedRow(lane));
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessBufferedSeparable(uint32_t imageOffset, bool useDiagonal)
    {
        ProcessHorizontalL1(imageOffset, DistanceInfinity());
        ProcessL1Vertical(imageOffset, useDiagonal);
    }

    __aicore__ inline uint32_t ComputeVectorHorizontalRows(uint32_t groupRows, float infinity)
    {
        LocalTensor<half> inputHalf = inputHalfBuf_.Get<half>();
        LocalTensor<float> scratch = rowBuf6_.Get<float>();
        LocalTensor<int32_t> gatherIndex = vectorIndexBuf_.Get<int32_t>();
        LocalTensor<uint32_t> gatherIndexU32 = gatherIndex.ReinterpretCast<uint32_t>();
        for (uint32_t lane = 0U; lane < groupRows; ++lane) {
            LocalTensor<float> initial = GetL1VectorRow(0U, lane);
            Cast(inputHalf, GetBufferedInputRow(lane), RoundMode::CAST_NONE, width_);
            Cast(initial, inputHalf, RoundMode::CAST_NONE, width_);
            Muls(initial, initial, infinity, width_);
        }

        uint32_t sourceBank = 0U;
        for (uint32_t jump = 1U; jump < width_; jump <<= 1U) {
            const uint32_t targetBank = sourceBank ^ 1U;
            const uint32_t shiftedCount = width_ - jump;
            const float jumpValue = UintToFloat(jump);
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                LocalTensor<float> source = GetL1VectorRow(sourceBank, lane);
                LocalTensor<float> target = GetL1VectorRow(targetBank, lane);
                Adds(target, source, 0.0F, width_);
            }
            if (jump < DMA_ALIGN_BYTES / sizeof(float)) {
                const int32_t jumpBytes = static_cast<int32_t>(jump * sizeof(float));
                const int32_t maxBytes = static_cast<int32_t>((width_ - 1U) * sizeof(float));
                ArithProgression(gatherIndex, -jumpBytes, static_cast<int32_t>(sizeof(float)), width_);
                Maxs(gatherIndex, gatherIndex, static_cast<int32_t>(0), width_);
                for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                    LocalTensor<float> source = GetL1VectorRow(sourceBank, lane);
                    LocalTensor<float> target = GetL1VectorRow(targetBank, lane);
                    Gather(scratch, source, gatherIndexU32, 0U, width_);
                    Adds(scratch, scratch, jumpValue, width_);
                    AscendC::Min(target, target, scratch, width_);
                }
                ArithProgression(gatherIndex, jumpBytes, static_cast<int32_t>(sizeof(float)), width_);
                Mins(gatherIndex, gatherIndex, maxBytes, width_);
                for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                    LocalTensor<float> source = GetL1VectorRow(sourceBank, lane);
                    LocalTensor<float> target = GetL1VectorRow(targetBank, lane);
                    Gather(scratch, source, gatherIndexU32, 0U, width_);
                    Adds(scratch, scratch, jumpValue, width_);
                    AscendC::Min(target, target, scratch, width_);
                }
            } else {
                for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                    LocalTensor<float> source = GetL1VectorRow(sourceBank, lane);
                    LocalTensor<float> target = GetL1VectorRow(targetBank, lane);
                    Adds(scratch, source, jumpValue, shiftedCount);
                    AscendC::Min(target[jump], target[jump], scratch, shiftedCount);
                    Adds(scratch, source[jump], jumpValue, shiftedCount);
                    AscendC::Min(target, target, scratch, shiftedCount);
                }
            }
            sourceBank = targetBank;
        }
        return sourceBank;
    }

    __aicore__ inline void ProcessVectorSeparable(uint32_t imageOffset, bool useDiagonal)
    {
        const float infinity = DistanceInfinity();
        event_t mte2ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        for (uint32_t groupStart = 0U; groupStart < height_; groupStart += L1_HORIZONTAL_GROUP_ROWS) {
            const uint32_t groupRows = MinU32(L1_HORIZONTAL_GROUP_ROWS, height_ - groupStart);
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                LoadInputRow(imageOffset + (groupStart + lane) * width_, GetBufferedInputRow(lane));
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
            const uint32_t sourceBank = ComputeVectorHorizontalRows(groupRows, infinity);

            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                const uint32_t row = groupStart + lane;
                StoreDistanceRowAsync(imageOffset + row * width_, GetL1VectorRow(sourceBank, lane));
            }
            // Both vector banks are reused by the next group. Complete the
            // current output drain before PIPE_V starts overwriting them.
            SetFlag<HardEvent::MTE3_V>(mte3ToVector);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
        }

        // The vertical pass reads the horizontal result back from GM through
        // MTE2; close the final MTE3->MTE2 dependency once per image.
        PipeBarrier<PIPE_ALL>();
        ProcessL1Vertical(imageOffset, useDiagonal);
    }

    __aicore__ inline void UpdateVerticalRow(const LocalTensor<float>& current, const LocalTensor<float>& neighbor,
                                             const LocalTensor<float>& scratch, bool useDiagonal,
                                             const LocalTensor<uint32_t>& leftIndex,
                                             const LocalTensor<uint32_t>& rightIndex)
    {
        Adds(scratch, neighbor, 1.0F, width_);
        AscendC::Min(current, current, scratch, width_);
        if (useDiagonal) {
            Gather(scratch, neighbor, leftIndex, 0U, width_);
            Adds(scratch, scratch, 1.0F, width_);
            AscendC::Min(current, current, scratch, width_);
            Gather(scratch, neighbor, rightIndex, 0U, width_);
            Adds(scratch, scratch, 1.0F, width_);
            AscendC::Min(current, current, scratch, width_);
        }
    }

    __aicore__ inline void ProcessL1Vertical(uint32_t imageOffset, bool useDiagonal = false)
    {
        if (height_ <= 1U) {
            return;
        }

        // In the vertical stage every column is independent. Update a complete
        // row with vector Adds + Min instead of issuing scalar UB accesses for
        // every pixel. Two row slots preserve the directional dependency; the
        // third slot is vector scratch storage.
        LocalTensor<float> scratch = rowBuf2_.Get<float>();
        LocalTensor<float> firstRow = GetBufferedRow(0U);
        LocalTensor<uint32_t> leftIndex;
        LocalTensor<uint32_t> rightIndex;
        if (useDiagonal) {
            LocalTensor<int32_t> leftIndexI32 = vectorIndexBuf_.Get<int32_t>();
            LocalTensor<int32_t> rightIndexI32 = chessIndexRightBuf_.Get<int32_t>();
            leftIndex = leftIndexI32.ReinterpretCast<uint32_t>();
            rightIndex = rightIndexI32.ReinterpretCast<uint32_t>();
            const int32_t elementBytes = static_cast<int32_t>(sizeof(float));
            const int32_t maxBytes = static_cast<int32_t>((width_ - 1U) * sizeof(float));
            ArithProgression(leftIndexI32, -elementBytes, elementBytes, width_);
            Maxs(leftIndexI32, leftIndexI32, static_cast<int32_t>(0), width_);
            ArithProgression(rightIndexI32, elementBytes, elementBytes, width_);
            Mins(rightIndexI32, rightIndexI32, maxBytes, width_);
            PipeBarrier<PIPE_V>();
        }
        event_t mte2ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t vectorToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const event_t storeDone0 = EVENT_ID0;
        const event_t storeDone1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(storeDone0);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone1);
        LoadDistanceRow(imageOffset, firstRow);
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
        for (uint32_t row = 1U; row < height_; ++row) {
            const uint32_t currentSlot = row % 2U;
            const uint32_t previousSlot = (row - 1U) % 2U;
            const event_t storeDone = currentSlot == 0U ? storeDone0 : storeDone1;
            LocalTensor<float> current = GetBufferedRow(currentSlot);
            LocalTensor<float> previous = GetBufferedRow(previousSlot);
            const uint32_t rowOffset = imageOffset + row * width_;
            // Wait only when MTE2 is about to overwrite this bank. The
            // previous iteration's MTE3 remains free to overlap until here.
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
            LoadDistanceRow(rowOffset, current);
            SetFlag<HardEvent::MTE2_V>(mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
            UpdateVerticalRow(current, previous, scratch, useDiagonal, leftIndex, rightIndex);
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            StoreDistanceRowAsync(rowOffset, current);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            SetFlag<HardEvent::V_MTE2>(vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        }

        // Both directional passes communicate through GM. Drain only the
        // last outstanding store of each bank at this necessary pass boundary.
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone1);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone0);
        PipeBarrier<PIPE_MTE2>();

        // Reset both bank tokens for the reverse directional pass.
        SetFlag<HardEvent::MTE3_MTE2>(storeDone0);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone1);

        const uint32_t lastRow = height_ - 1U;
        // Reload the reverse boundary explicitly. Relying on the final forward
        // row remaining resident is fragile in CPU Debug and after pipeline
        // scheduling changes.
        LocalTensor<float> lastDistanceRow = GetBufferedRow(lastRow % 2U);
        LoadDistanceRow(imageOffset + lastRow * width_, lastDistanceRow);
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
        uint32_t row = lastRow;
        while (row > 0U) {
            --row;
            const uint32_t currentSlot = row % 2U;
            const uint32_t nextSlot = (row + 1U) % 2U;
            const event_t storeDone = currentSlot == 0U ? storeDone0 : storeDone1;
            LocalTensor<float> current = GetBufferedRow(currentSlot);
            LocalTensor<float> next = GetBufferedRow(nextSlot);
            const uint32_t rowOffset = imageOffset + row * width_;
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
            LoadDistanceRow(rowOffset, current);
            SetFlag<HardEvent::MTE2_V>(mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
            UpdateVerticalRow(current, next, scratch, useDiagonal, leftIndex, rightIndex);
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            StoreDistanceRowAsync(rowOffset, current);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            SetFlag<HardEvent::V_MTE2>(vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone0);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone1);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void PrepareChamferIndices()
    {
        LocalTensor<int32_t> left1 = vectorIndexBuf_.Get<int32_t>();
        LocalTensor<int32_t> right1 = chessIndexRightBuf_.Get<int32_t>();
        LocalTensor<int32_t> left2 = rowBuf5_.Get<int32_t>();
        LocalTensor<int32_t> right2 = rowBuf6_.Get<int32_t>();
        const int32_t elementBytes = static_cast<int32_t>(sizeof(float));
        const int32_t maxBytes = static_cast<int32_t>((width_ - 1U) * sizeof(float));
        ArithProgression(left1, -elementBytes, elementBytes, width_);
        Maxs(left1, left1, static_cast<int32_t>(0), width_);
        ArithProgression(right1, elementBytes, elementBytes, width_);
        Mins(right1, right1, maxBytes, width_);
        ArithProgression(left2, -2 * elementBytes, elementBytes, width_);
        Maxs(left2, left2, static_cast<int32_t>(0), width_);
        ArithProgression(right2, 2 * elementBytes, elementBytes, width_);
        Mins(right2, right2, maxBytes, width_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateChamferSame(const LocalTensor<float>& current, const LocalTensor<float>& neighbor,
                                             const LocalTensor<float>& scratch, float cost)
    {
        Adds(scratch, neighbor, cost, width_);
        AscendC::Min(current, current, scratch, width_);
    }

    __aicore__ inline void UpdateChamferShift(const LocalTensor<float>& current, const LocalTensor<float>& neighbor,
                                              const LocalTensor<float>& scratch, const LocalTensor<uint32_t>& index,
                                              float cost)
    {
        Gather(scratch, neighbor, index, 0U, width_);
        Adds(scratch, scratch, cost, width_);
        AscendC::Min(current, current, scratch, width_);
    }

    __aicore__ inline void ApplyChamferHorizontalJump(const LocalTensor<float>& source,
                                                      const LocalTensor<float>& target,
                                                      const LocalTensor<float>& scratch, LocalTensor<int32_t> index,
                                                      uint32_t jump, float axialCost, bool forward)
    {
        Adds(target, source, 0.0F, width_);
        const uint32_t shiftedCount = width_ - jump;
        const float jumpCost = UintToFloat(jump) * axialCost;
        if (jump < DMA_ALIGN_BYTES / sizeof(float)) {
            const int32_t jumpBytes = static_cast<int32_t>(jump * sizeof(float));
            const int32_t maxBytes = static_cast<int32_t>((width_ - 1U) * sizeof(float));
            ArithProgression(index, forward ? -jumpBytes : jumpBytes, static_cast<int32_t>(sizeof(float)), width_);
            if (forward) {
                Maxs(index, index, static_cast<int32_t>(0), width_);
            } else {
                Mins(index, index, maxBytes, width_);
            }
            LocalTensor<uint32_t> gatherIndex = index.ReinterpretCast<uint32_t>();
            UpdateChamferShift(target, source, scratch, gatherIndex, jumpCost);
        } else if (forward) {
            Adds(scratch, source, jumpCost, shiftedCount);
            AscendC::Min(target[jump], target[jump], scratch, shiftedCount);
        } else {
            Adds(scratch, source[jump], jumpCost, shiftedCount);
            AscendC::Min(target, target, scratch, shiftedCount);
        }
    }

    __aicore__ inline void PropagateChamferHorizontal(const LocalTensor<float>& current,
                                                      const LocalTensor<float>& auxiliary,
                                                      const LocalTensor<float>& scratch, float axialCost, bool forward,
                                                      uint32_t propagationLimit)
    {
        LocalTensor<int32_t> prefixIndex = chamferPrefixIndexBuf_.Get<int32_t>();
        bool sourceIsCurrent = true;
        const uint32_t jumpLimit = MinU32(width_, propagationLimit);
        for (uint32_t jump = 1U; jump < jumpLimit; jump <<= 1U) {
            if (sourceIsCurrent) {
                ApplyChamferHorizontalJump(current, auxiliary, scratch, prefixIndex, jump, axialCost, forward);
            } else {
                ApplyChamferHorizontalJump(auxiliary, current, scratch, prefixIndex, jump, axialCost, forward);
            }
            sourceIsCurrent = !sourceIsCurrent;
        }
        if (!sourceIsCurrent) {
            Adds(current, auxiliary, 0.0F, width_);
        }
    }

    __aicore__ inline void PropagateMask3Horizontal(const LocalTensor<float>& current, float axialCost, bool forward,
                                                    event_t vectorToScalar, event_t scalarToVector)
    {
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        if (forward) {
            for (uint32_t col = 1U; col < width_; ++col) {
                current.SetValue(col, Min(current.GetValue(col), current.GetValue(col - 1U) + axialCost));
            }
        } else {
            uint32_t col = width_ - 1U;
            while (col > 0U) {
                --col;
                current.SetValue(col, Min(current.GetValue(col), current.GetValue(col + 1U) + axialCost));
            }
        }
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        Adds(current, current, 0.0F, width_);
    }

    __aicore__ inline bool NeedsMask3ScalarHorizontal(const LocalTensor<uint8_t>& input,
                                                      const LocalTensor<float>& binaryInput,
                                                      const LocalTensor<float>& reducedBlocks,
                                                      bool propagationHasBackground, bool& rowHasBackground,
                                                      event_t vectorToScalar)
    {
        const uint32_t fullBlockCount = width_ / MASK3_GUARD_BLOCK_ELEMS;
        WholeReduceMin(reducedBlocks, binaryInput, MASK3_GUARD_BLOCK_ELEMS, fullBlockCount, 1U, 1U,
                       MASK3_GUARD_BLOCK_ELEMS / FLOAT_BLOCK_ELEMS, ReduceOrder::ORDER_ONLY_VALUE);
        const bool useVectorSummary = fullBlockCount <= FLOAT_VECTOR_REPEAT_ELEMS;
        uint32_t summaryOffset = 0U;
        if (useVectorSummary) {
            PipeBarrier<PIPE_V>();
            summaryOffset = AlignUp32(fullBlockCount * sizeof(float)) / sizeof(float);
            WholeReduceMin(reducedBlocks[summaryOffset], reducedBlocks, fullBlockCount, 1U, 1U, 1U, FLOAT_BLOCK_ELEMS,
                           ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMax(reducedBlocks[summaryOffset + FLOAT_BLOCK_ELEMS], reducedBlocks, fullBlockCount, 1U, 1U, 1U,
                           FLOAT_BLOCK_ELEMS, ReduceOrder::ORDER_ONLY_VALUE);
        }
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);

        bool hasEmptyFullBlock = false;
        rowHasBackground = false;
        if (useVectorSummary) {
            rowHasBackground = reducedBlocks.GetValue(summaryOffset) < 0.5F;
            hasEmptyFullBlock = reducedBlocks.GetValue(summaryOffset + FLOAT_BLOCK_ELEMS) > 0.5F;
        } else {
            for (uint32_t block = 0U; block < fullBlockCount; ++block) {
                const bool blockHasBackground = reducedBlocks.GetValue(block) < 0.5F;
                rowHasBackground = rowHasBackground || blockHasBackground;
                hasEmptyFullBlock = hasEmptyFullBlock || !blockHasBackground;
            }
        }
        const uint32_t tailStart = fullBlockCount * MASK3_GUARD_BLOCK_ELEMS;
        for (uint32_t col = tailStart; col < width_; ++col) {
            rowHasBackground = rowHasBackground || input.GetValue(col) == 0U;
        }
        return rowHasBackground ? hasEmptyFullBlock : propagationHasBackground;
    }

    __aicore__ inline void FinishChamferHorizontal(const LocalTensor<float>& current,
                                                   const LocalTensor<float>& auxiliary,
                                                   const LocalTensor<float>& scratch, float axialCost, bool forward,
                                                   bool useScalarHorizontal, bool rowHasBackground,
                                                   event_t vectorToScalar, event_t scalarToVector)
    {
        if (useScalarHorizontal) {
            PropagateMask3Horizontal(current, axialCost, forward, vectorToScalar, scalarToVector);
            return;
        }
        if (maskSize_ == MASK_3 && !rowHasBackground) {
            return;
        }
        const uint32_t propagationLimit = maskSize_ == MASK_3 ? MASK3_VECTOR_PROPAGATION_LIMIT : width_;
        PropagateChamferHorizontal(current, auxiliary, scratch, axialCost, forward, propagationLimit);
    }

    __aicore__ inline void UpdateChamferRows(const LocalTensor<float>& current, uint32_t row, bool forward,
                                             bool useKnight, float axialCost, float diagonalCost)
    {
        LocalTensor<float> scratch = rowBuf4_.Get<float>();
        LocalTensor<int32_t> left1I32 = vectorIndexBuf_.Get<int32_t>();
        LocalTensor<int32_t> right1I32 = chessIndexRightBuf_.Get<int32_t>();
        LocalTensor<int32_t> left2I32 = rowBuf5_.Get<int32_t>();
        LocalTensor<int32_t> right2I32 = rowBuf6_.Get<int32_t>();
        LocalTensor<uint32_t> left1 = left1I32.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> right1 = right1I32.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> left2 = left2I32.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> right2 = right2I32.ReinterpretCast<uint32_t>();
        const bool hasNeighbor1 = forward ? row > 0U : row + 1U < height_;
        const bool hasNeighbor2 = forward ? row > 1U : row + 2U < height_;
        if (hasNeighbor1) {
            const uint32_t neighborRow = forward ? row - 1U : row + 1U;
            LocalTensor<float> neighbor = GetBufferedRow(neighborRow % 3U);
            UpdateChamferSame(current, neighbor, scratch, axialCost);
            UpdateChamferShift(current, neighbor, scratch, left1, diagonalCost);
            UpdateChamferShift(current, neighbor, scratch, right1, diagonalCost);
            if (useKnight) {
                UpdateChamferShift(current, neighbor, scratch, left2, L2_MASK_5_KNIGHT);
                UpdateChamferShift(current, neighbor, scratch, right2, L2_MASK_5_KNIGHT);
            }
        }
        if (useKnight && hasNeighbor2) {
            const uint32_t neighborRow = forward ? row - 2U : row + 2U;
            LocalTensor<float> neighbor = GetBufferedRow(neighborRow % 3U);
            UpdateChamferShift(current, neighbor, scratch, left1, L2_MASK_5_KNIGHT);
            UpdateChamferShift(current, neighbor, scratch, right1, L2_MASK_5_KNIGHT);
        }
    }

    __aicore__ inline void ProcessL2ChamferVector(uint32_t imageOffset)
    {
        const float axialCost = AxialCost();
        const float diagonalCost = DiagonalCost();
        const bool useKnight = UsesKnight();
        const float infinity = DistanceInfinity();
        LocalTensor<uint8_t> input = inputRowBuf_.Get<uint8_t>();
        LocalTensor<half> inputHalf = inputHalfBuf_.Get<half>();
        LocalTensor<float> auxiliary = rowBuf3_.Get<float>();
        LocalTensor<float> scratch = rowBuf4_.Get<float>();
        event_t mte2ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        PrepareChamferIndices();

        bool forwardHasBackground = false;
        for (uint32_t row = 0U; row < height_; ++row) {
            const uint32_t rowOffset = imageOffset + row * width_;
            LocalTensor<float> current = GetBufferedRow(row % 3U);
            LoadInputRow(rowOffset, input);
            SetFlag<HardEvent::MTE2_V>(mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
            Cast(inputHalf, input, RoundMode::CAST_NONE, width_);
            Cast(current, inputHalf, RoundMode::CAST_NONE, width_);
            bool rowHasBackground = false;
            bool useScalarHorizontal = false;
            if (maskSize_ == MASK_3) {
                useScalarHorizontal = NeedsMask3ScalarHorizontal(input, current, scratch, forwardHasBackground,
                                                                 rowHasBackground, vectorToScalar);
            }
            forwardHasBackground = forwardHasBackground || rowHasBackground;
            Muls(current, current, infinity, width_);
            UpdateChamferRows(current, row, true, useKnight, axialCost, diagonalCost);
            FinishChamferHorizontal(current, auxiliary, scratch, axialCost, true, useScalarHorizontal, rowHasBackground,
                                    vectorToScalar, scalarToVector);
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            StoreDistanceRowAsync(rowOffset, current);
            SetFlag<HardEvent::MTE3_V>(mte3ToVector);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
        }

        // Reverse rows may already contain finite values produced by the forward pass,
        // even before the reverse scan reaches the source background row.
        bool reverseHasBackground = forwardHasBackground;
        uint32_t row = height_;
        while (row > 0U) {
            --row;
            const uint32_t rowOffset = imageOffset + row * width_;
            LocalTensor<float> current = GetBufferedRow(row % 3U);
            if (maskSize_ == MASK_3) {
                LoadInputRow(rowOffset, input);
            }
            LoadDistanceRow(rowOffset, current);
            SetFlag<HardEvent::MTE2_V>(mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
            if (maskSize_ == MASK_3) {
                Cast(inputHalf, input, RoundMode::CAST_NONE, width_);
                Cast(auxiliary, inputHalf, RoundMode::CAST_NONE, width_);
            }
            bool rowHasBackground = false;
            const bool useScalarHorizontal = maskSize_ == MASK_3 &&
                                             NeedsMask3ScalarHorizontal(input, auxiliary, scratch, reverseHasBackground,
                                                                        rowHasBackground, vectorToScalar);
            reverseHasBackground = reverseHasBackground || rowHasBackground;
            UpdateChamferRows(current, row, false, useKnight, axialCost, diagonalCost);
            FinishChamferHorizontal(current, auxiliary, scratch, axialCost, false, useScalarHorizontal,
                                    rowHasBackground, vectorToScalar, scalarToVector);
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            StoreDistanceRowAsync(rowOffset, current);
            SetFlag<HardEvent::MTE3_V>(mte3ToVector);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
        }
    }

    __aicore__ inline void ProcessBufferedTwoPass(uint32_t imageOffset)
    {
        const bool useDiagonal = UsesDiagonal();
        const bool useKnight = UsesKnight();
        const float axialCost = AxialCost();
        const float diagonalCost = DiagonalCost();
        const float infinity = DistanceInfinity();
        LocalTensor<uint8_t> inputRow = inputRowBuf_.Get<uint8_t>();

        // Keep complete rows in a three-slot ring. This preserves OpenCV's
        // directional recurrence while replacing per-neighbor GM transactions
        // with UB scalar accesses and one bulk DMA per row.
        for (uint32_t row = 0U; row < height_; ++row) {
            const uint32_t rowOffset = imageOffset + row * width_;
            LocalTensor<float> current = GetBufferedRow(row % 3U);
            LoadInputRow(rowOffset, inputRow);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t col = 0U; col < width_; ++col) {
                if (inputRow.GetValue(col) == 0U) {
                    current.SetValue(col, 0.0F);
                    continue;
                }
                float best = infinity;
                if (col > 0U) {
                    best = Min(best, current.GetValue(col - 1U) + axialCost);
                }
                if (row > 0U) {
                    LocalTensor<float> previous = GetBufferedRow((row - 1U) % 3U);
                    best = Min(best, previous.GetValue(col) + axialCost);
                    if (useDiagonal && col > 0U) {
                        best = Min(best, previous.GetValue(col - 1U) + diagonalCost);
                    }
                    if (useDiagonal && col + 1U < width_) {
                        best = Min(best, previous.GetValue(col + 1U) + diagonalCost);
                    }
                    if (useKnight && col > 1U) {
                        best = Min(best, previous.GetValue(col - 2U) + L2_MASK_5_KNIGHT);
                    }
                    if (useKnight && col + 2U < width_) {
                        best = Min(best, previous.GetValue(col + 2U) + L2_MASK_5_KNIGHT);
                    }
                }
                if (useKnight && row > 1U) {
                    LocalTensor<float> previous2 = GetBufferedRow((row - 2U) % 3U);
                    if (col > 0U) {
                        best = Min(best, previous2.GetValue(col - 1U) + L2_MASK_5_KNIGHT);
                    }
                    if (col + 1U < width_) {
                        best = Min(best, previous2.GetValue(col + 1U) + L2_MASK_5_KNIGHT);
                    }
                }
                current.SetValue(col, best);
            }
            StoreDistanceRow(rowOffset, current);
        }

        uint32_t row = height_;
        while (row > 0U) {
            --row;
            const uint32_t rowOffset = imageOffset + row * width_;
            LocalTensor<float> current = GetBufferedRow(row % 3U);
            LoadInputRow(rowOffset, inputRow);
            LoadDistanceRow(rowOffset, current);
            PipeBarrier<PIPE_ALL>();
            uint32_t col = width_;
            while (col > 0U) {
                --col;
                if (inputRow.GetValue(col) == 0U) {
                    continue;
                }
                float best = current.GetValue(col);
                if (col + 1U < width_) {
                    best = Min(best, current.GetValue(col + 1U) + axialCost);
                }
                if (row + 1U < height_) {
                    LocalTensor<float> next = GetBufferedRow((row + 1U) % 3U);
                    best = Min(best, next.GetValue(col) + axialCost);
                    if (useDiagonal && col > 0U) {
                        best = Min(best, next.GetValue(col - 1U) + diagonalCost);
                    }
                    if (useDiagonal && col + 1U < width_) {
                        best = Min(best, next.GetValue(col + 1U) + diagonalCost);
                    }
                    if (useKnight && col > 1U) {
                        best = Min(best, next.GetValue(col - 2U) + L2_MASK_5_KNIGHT);
                    }
                    if (useKnight && col + 2U < width_) {
                        best = Min(best, next.GetValue(col + 2U) + L2_MASK_5_KNIGHT);
                    }
                }
                if (useKnight && row + 2U < height_) {
                    LocalTensor<float> next2 = GetBufferedRow((row + 2U) % 3U);
                    if (col > 0U) {
                        best = Min(best, next2.GetValue(col - 1U) + L2_MASK_5_KNIGHT);
                    }
                    if (col + 1U < width_) {
                        best = Min(best, next2.GetValue(col + 1U) + L2_MASK_5_KNIGHT);
                    }
                }
                current.SetValue(col, best);
            }
            StoreDistanceRow(rowOffset, current);
        }
    }

    __aicore__ inline void ProcessGmTwoPass(uint32_t imageOffset)
    {
        const bool useDiagonal = UsesDiagonal();
        const bool useKnight = UsesKnight();
        const float axialCost = AxialCost();
        const float diagonalCost = DiagonalCost();
        const float infinity = DistanceInfinity();

        // GM retains each tile's terminal rows as the next tile's dependency boundary.
        for (uint32_t tileStart = 0; tileStart < height_; tileStart += tileRows_) {
            const uint32_t tileEnd = MinU32(tileStart + tileRows_, height_);
            for (uint32_t row = tileStart; row < tileEnd; ++row) {
                for (uint32_t col = 0; col < width_; ++col) {
                    const uint32_t index = imageOffset + row * width_ + col;
                    if (xGm_.GetValue(index) == 0U) {
                        yGm_.SetValue(index, 0.0F);
                        continue;
                    }
                    float best = infinity;
                    if (col > 0U) {
                        best = Min(best, yGm_.GetValue(index - 1U) + axialCost);
                    }
                    if (row > 0U) {
                        best = Min(best, yGm_.GetValue(index - width_) + axialCost);
                        if (useDiagonal && col > 0U) {
                            best = Min(best, yGm_.GetValue(index - width_ - 1U) + diagonalCost);
                        }
                        if (useDiagonal && col + 1U < width_) {
                            best = Min(best, yGm_.GetValue(index - width_ + 1U) + diagonalCost);
                        }
                        if (useKnight && col > 1U) {
                            best = Min(best, yGm_.GetValue(index - width_ - 2U) + L2_MASK_5_KNIGHT);
                        }
                        if (useKnight && col + 2U < width_) {
                            best = Min(best, yGm_.GetValue(index - width_ + 2U) + L2_MASK_5_KNIGHT);
                        }
                    }
                    if (useKnight && row > 1U) {
                        if (col > 0U) {
                            best = Min(best, yGm_.GetValue(index - 2U * width_ - 1U) + L2_MASK_5_KNIGHT);
                        }
                        if (col + 1U < width_) {
                            best = Min(best, yGm_.GetValue(index - 2U * width_ + 1U) + L2_MASK_5_KNIGHT);
                        }
                    }
                    yGm_.SetValue(index, best);
                }
            }
        }

        uint32_t tileEnd = height_;
        while (tileEnd > 0U) {
            const uint32_t tileStart = tileEnd > tileRows_ ? tileEnd - tileRows_ : 0U;
            uint32_t row = tileEnd;
            while (row > tileStart) {
                --row;
                uint32_t col = width_;
                while (col > 0U) {
                    --col;
                    const uint32_t index = imageOffset + row * width_ + col;
                    if (xGm_.GetValue(index) == 0U) {
                        continue;
                    }
                    float best = yGm_.GetValue(index);
                    if (col + 1U < width_) {
                        best = Min(best, yGm_.GetValue(index + 1U) + axialCost);
                    }
                    if (row + 1U < height_) {
                        best = Min(best, yGm_.GetValue(index + width_) + axialCost);
                        if (useDiagonal && col > 0U) {
                            best = Min(best, yGm_.GetValue(index + width_ - 1U) + diagonalCost);
                        }
                        if (useDiagonal && col + 1U < width_) {
                            best = Min(best, yGm_.GetValue(index + width_ + 1U) + diagonalCost);
                        }
                        if (useKnight && col > 1U) {
                            best = Min(best, yGm_.GetValue(index + width_ - 2U) + L2_MASK_5_KNIGHT);
                        }
                        if (useKnight && col + 2U < width_) {
                            best = Min(best, yGm_.GetValue(index + width_ + 2U) + L2_MASK_5_KNIGHT);
                        }
                    }
                    if (useKnight && row + 2U < height_) {
                        if (col > 0U) {
                            best = Min(best, yGm_.GetValue(index + 2U * width_ - 1U) + L2_MASK_5_KNIGHT);
                        }
                        if (col + 1U < width_) {
                            best = Min(best, yGm_.GetValue(index + 2U * width_ + 1U) + L2_MASK_5_KNIGHT);
                        }
                    }
                    yGm_.SetValue(index, best);
                }
            }
            tileEnd = tileStart;
        }
    }

    __aicore__ inline void HorizontalL2Row(uint32_t rowOffset, float infSquared)
    {
        int32_t lastBackground = -1;
        for (uint32_t col = 0; col < width_; ++col) {
            const uint32_t index = rowOffset + col;
            if (xGm_.GetValue(index) == 0U) {
                lastBackground = static_cast<int32_t>(col);
                yGm_.SetValue(index, 0.0f);
            } else if (lastBackground >= 0) {
                const float delta = static_cast<float>(static_cast<int32_t>(col) - lastBackground);
                yGm_.SetValue(index, delta * delta);
            } else {
                yGm_.SetValue(index, infSquared);
            }
        }
        int32_t nextBackground = -1;
        uint32_t col = width_;
        while (col > 0U) {
            --col;
            const uint32_t index = rowOffset + col;
            if (xGm_.GetValue(index) == 0U) {
                nextBackground = static_cast<int32_t>(col);
            } else if (nextBackground >= 0) {
                const float delta = static_cast<float>(nextBackground - static_cast<int32_t>(col));
                yGm_.SetValue(index, Min(yGm_.GetValue(index), delta * delta));
            }
        }
    }

    __aicore__ inline void BuildEnvelopeUb(const LocalTensor<float>& tile, uint32_t lane, uint32_t tileStride,
                                           int32_t firstFinite, float infSquared, int32_t& envelopeSize)
    {
        LocalTensor<int32_t> envelopeIndex = envelopeIndexBuf_.Get<int32_t>();
        LocalTensor<float> envelopeBreak = envelopeBreakBuf_.Get<float>();
        LocalTensor<float> envelopeValue = envelopeValueBuf_.Get<float>();
        envelopeSize = 0;
        envelopeIndex.SetValue(0U, firstFinite);
        envelopeBreak.SetValue(0U, -infSquared);
        envelopeBreak.SetValue(1U, infSquared);
        envelopeValue.SetValue(0U, tile.GetValue(static_cast<uint32_t>(firstFinite) * tileStride + lane));
        for (uint32_t q = static_cast<uint32_t>(firstFinite) + 1U; q < height_; ++q) {
            const float fq = tile.GetValue(q * tileStride + lane);
            if (fq >= infSquared) {
                continue;
            }
            float intersection = 0.0F;
            while (true) {
                const int32_t vk = envelopeIndex.GetValue(envelopeSize);
                const float fvk = envelopeValue.GetValue(envelopeSize);
                const float qf = UintToFloat(q);
                const float vkf = static_cast<float>(vk);
                intersection = ((fq + qf * qf) - (fvk + vkf * vkf)) / (2.0F * (qf - vkf));
                if (envelopeSize == 0 || intersection > envelopeBreak.GetValue(envelopeSize)) {
                    break;
                }
                --envelopeSize;
            }
            ++envelopeSize;
            envelopeIndex.SetValue(envelopeSize, static_cast<int32_t>(q));
            envelopeBreak.SetValue(envelopeSize, intersection);
            envelopeBreak.SetValue(envelopeSize + 1, infSquared);
            envelopeValue.SetValue(envelopeSize, fq);
        }
    }

    __aicore__ inline void ProcessL2ColumnUb(const LocalTensor<float>& tile, uint32_t lane, uint32_t tileStride,
                                             float infSquared)
    {
        LocalTensor<int32_t> envelopeIndex = envelopeIndexBuf_.Get<int32_t>();
        LocalTensor<float> envelopeBreak = envelopeBreakBuf_.Get<float>();
        LocalTensor<float> envelopeValue = envelopeValueBuf_.Get<float>();
        int32_t firstFinite = -1;
        for (uint32_t row = 0U; row < height_; ++row) {
            if (tile.GetValue(row * tileStride + lane) < infSquared) {
                firstFinite = static_cast<int32_t>(row);
                break;
            }
        }
        if (firstFinite < 0) {
            for (uint32_t row = 0U; row < height_; ++row) {
                tile.SetValue(row * tileStride + lane, infSquared);
            }
            return;
        }

        int32_t envelopeSize = 0;
        BuildEnvelopeUb(tile, lane, tileStride, firstFinite, infSquared, envelopeSize);
        int32_t k = 0;
        for (uint32_t row = 0U; row < height_; ++row) {
            while (k < envelopeSize && envelopeBreak.GetValue(k + 1) < UintToFloat(row)) {
                ++k;
            }
            const int32_t nearestRow = envelopeIndex.GetValue(k);
            const float delta = static_cast<float>(static_cast<int32_t>(row) - nearestRow);
            tile.SetValue(row * tileStride + lane, envelopeValue.GetValue(k) + delta * delta);
        }
    }

    __aicore__ inline void BuildHorizontalL2Tile(uint32_t imageOffset, uint32_t tileStart, uint32_t tileWidth,
                                                 uint32_t tileStride, const LocalTensor<float>& tile)
    {
        const float infinity = UintToFloat(height_) + UintToFloat(width_);
        Duplicate(tile, 0.0F, height_ * tileStride);
        event_t mte2ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t vectorToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        for (uint32_t groupStart = 0U; groupStart < height_; groupStart += L1_HORIZONTAL_GROUP_ROWS) {
            const uint32_t groupRows = MinU32(L1_HORIZONTAL_GROUP_ROWS, height_ - groupStart);
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                LoadInputRow(imageOffset + (groupStart + lane) * width_, GetBufferedInputRow(lane));
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToVector);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
            const uint32_t sourceBank = ComputeVectorHorizontalRows(groupRows, infinity);
            for (uint32_t lane = 0U; lane < groupRows; ++lane) {
                LocalTensor<float> distance = GetL1VectorRow(sourceBank, lane);
                Mul(distance, distance, distance, width_);
                Adds(tile[(groupStart + lane) * tileStride], distance[tileStart], 0.0F, tileWidth);
            }
            SetFlag<HardEvent::V_MTE2>(vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        }
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
    }

    __aicore__ inline void ProcessL2Tile(uint32_t imageOffset, uint32_t tileStart, uint32_t tileWidth, float infSquared)
    {
        const uint32_t tileStride = (tileWidth + 7U) / 8U * 8U;
        LocalTensor<float> tile = exactTileBuf_.Get<float>();
        BuildHorizontalL2Tile(imageOffset, tileStart, tileWidth, tileStride, tile);
        for (uint32_t lane = 0U; lane < tileWidth; ++lane) {
            ProcessL2ColumnUb(tile, lane, tileStride, infSquared);
        }

        event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        Sqrt(tile, tile, static_cast<int32_t>(height_ * tileStride));
        event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);

        // UB source strides are measured in 32-byte blocks. DataCopyPad already
        // rounds each source block up to 32 bytes, matching tileStride exactly.
        DataCopyExtParams storeParams{static_cast<uint16_t>(height_), static_cast<uint32_t>(tileWidth * sizeof(float)),
                                      0U, static_cast<uint32_t>((width_ - tileWidth) * sizeof(float)), 0U};
        DataCopyPad(yGm_[imageOffset + tileStart], tile, storeParams);
        event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(mte3ToVector);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    }

    __aicore__ inline void ProcessExactL2Tasks()
    {
        const uint32_t tilesPerImage = (width_ + exactTileColumns_ - 1U) / exactTileColumns_;
        const uint32_t taskCount = imageCount_ * tilesPerImage;
        const float heightFloat = UintToFloat(height_);
        const float widthFloat = UintToFloat(width_);
        const float infSquared = heightFloat * heightFloat + widthFloat * widthFloat;
        for (uint32_t task = blockId_; task < taskCount; task += usedCoreNum_) {
            const uint32_t imageIndex = task / tilesPerImage;
            const uint32_t tileIndex = task - imageIndex * tilesPerImage;
            const uint32_t tileStart = tileIndex * exactTileColumns_;
            const uint32_t tileWidth = MinU32(exactTileColumns_, width_ - tileStart);
            ProcessL2Tile(imageIndex * imageStride_, tileStart, tileWidth, infSquared);
        }
    }

    __aicore__ inline void BuildEnvelope(uint32_t imageOffset, uint32_t col, int32_t firstFinite, float infSquared,
                                         int32_t& envelopeSize)
    {
        LocalTensor<int32_t> envelopeIndex = envelopeIndexBuf_.Get<int32_t>();
        LocalTensor<float> envelopeBreak = envelopeBreakBuf_.Get<float>();
        LocalTensor<float> envelopeValue = envelopeValueBuf_.Get<float>();
        envelopeSize = 0;
        envelopeIndex.SetValue(0, firstFinite);
        envelopeBreak.SetValue(0, -infSquared);
        envelopeBreak.SetValue(1, infSquared);
        envelopeValue.SetValue(0, yGm_.GetValue(imageOffset + static_cast<uint32_t>(firstFinite) * width_ + col));
        for (uint32_t q = static_cast<uint32_t>(firstFinite) + 1U; q < height_; ++q) {
            const float fq = yGm_.GetValue(imageOffset + q * width_ + col);
            if (fq >= infSquared) {
                continue;
            }
            float intersection = 0.0f;
            while (true) {
                const int32_t vk = envelopeIndex.GetValue(envelopeSize);
                const float fvk = envelopeValue.GetValue(envelopeSize);
                const float qf = UintToFloat(q);
                const float vkf = static_cast<float>(vk);
                intersection = ((fq + qf * qf) - (fvk + vkf * vkf)) / (2.0f * (qf - vkf));
                if (envelopeSize == 0 || intersection > envelopeBreak.GetValue(envelopeSize)) {
                    break;
                }
                --envelopeSize;
            }
            ++envelopeSize;
            envelopeIndex.SetValue(envelopeSize, static_cast<int32_t>(q));
            envelopeBreak.SetValue(envelopeSize, intersection);
            envelopeBreak.SetValue(envelopeSize + 1, infSquared);
            envelopeValue.SetValue(envelopeSize, fq);
        }
    }

    __aicore__ inline void ProcessL2Column(uint32_t imageOffset, uint32_t col, float infSquared)
    {
        LocalTensor<int32_t> envelopeIndex = envelopeIndexBuf_.Get<int32_t>();
        LocalTensor<float> envelopeBreak = envelopeBreakBuf_.Get<float>();
        LocalTensor<float> envelopeValue = envelopeValueBuf_.Get<float>();
        int32_t firstFinite = -1;
        for (uint32_t row = 0; row < height_; ++row) {
            if (yGm_.GetValue(imageOffset + row * width_ + col) < infSquared) {
                firstFinite = static_cast<int32_t>(row);
                break;
            }
        }
        if (firstFinite < 0) {
            for (uint32_t row = 0U; row < height_; ++row) {
                yGm_.SetValue(imageOffset + row * width_ + col, infSquared);
            }
            return;
        }

        int32_t envelopeSize = 0;
        BuildEnvelope(imageOffset, col, firstFinite, infSquared, envelopeSize);
        int32_t k = 0;
        for (uint32_t row = 0U; row < height_; ++row) {
            while (k < envelopeSize && envelopeBreak.GetValue(k + 1) < UintToFloat(row)) {
                ++k;
            }
            const int32_t nearestRow = envelopeIndex.GetValue(k);
            const float delta = static_cast<float>(static_cast<int32_t>(row) - nearestRow);
            yGm_.SetValue(imageOffset + row * width_ + col, envelopeValue.GetValue(k) + delta * delta);
        }
    }

    __aicore__ inline void FlushScalarOutput(uint32_t imageOffset)
    {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_[imageOffset]);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void VectorSqrt(uint32_t imageOffset)
    {
        FlushScalarOutput(imageOffset);
        for (uint32_t offset = 0U; offset < imageStride_; offset += vectorTileElems_) {
            const uint32_t count = MinU32(vectorTileElems_, imageStride_ - offset);
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};

            LocalTensor<float> dataLocal = sqrtQueue_.AllocTensor<float>();
            DataCopyPad(dataLocal, yGm_[imageOffset + offset], copyParams, padParams);
            sqrtQueue_.EnQue(dataLocal);

            dataLocal = sqrtQueue_.DeQue<float>();

            // Match ops-nn ForeachSqrt on Ascend 910B: compute in-place on the VECIN tensor.
            Sqrt(dataLocal, dataLocal, static_cast<int32_t>(count));

            event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            DataCopyPad(yGm_[imageOffset + offset], dataLocal, copyParams);

            event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            sqrtQueue_.FreeTensor(dataLocal);
        }
    }

    __aicore__ inline void ProcessL2(uint32_t imageOffset)
    {
        const float heightFloat = UintToFloat(height_);
        const float widthFloat = UintToFloat(width_);
        const float infSquared = heightFloat * heightFloat + widthFloat * widthFloat;

        for (uint32_t row = 0U; row < height_; ++row) {
            HorizontalL2Row(imageOffset + row * width_, infSquared);
        }
        for (uint32_t col = 0U; col < width_; ++col) {
            ProcessL2Column(imageOffset, col, infSquared);
        }
        VectorSqrt(imageOffset);
    }

private:
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<float> yGm_;
    TBuf<TPosition::VECCALC> envelopeIndexBuf_;
    TBuf<TPosition::VECCALC> envelopeBreakBuf_;
    TBuf<TPosition::VECCALC> envelopeValueBuf_;
    TQue<TPosition::VECIN, 2> sqrtQueue_;
    TBuf<TPosition::VECIN> inputRowBuf_;
    TBuf<TPosition::VECIN> inputRowBuf1_;
    TBuf<TPosition::VECIN> inputRowBuf2_;
    TBuf<TPosition::VECIN> rowBuf0_;
    TBuf<TPosition::VECIN> rowBuf1_;
    TBuf<TPosition::VECIN> rowBuf2_;
    TBuf<TPosition::VECIN> rowBuf3_;
    TBuf<TPosition::VECIN> rowBuf4_;
    TBuf<TPosition::VECIN> rowBuf5_;
    TBuf<TPosition::VECIN> rowBuf6_;
    TBuf<TPosition::VECIN> inputHalfBuf_;
    TBuf<TPosition::VECIN> vectorIndexBuf_;
    TBuf<TPosition::VECIN> chessIndexRightBuf_;
    TBuf<TPosition::VECIN> chamferPrefixIndexBuf_;
    TBuf<TPosition::VECIN> exactTileBuf_;

    uint32_t imageCount_ = 0;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t imageStride_ = 0;
    uint32_t distanceType_ = DISTANCE_L2;
    uint32_t maskSize_ = MASK_PRECISE;
    uint32_t usedCoreNum_ = 1;
    uint32_t tileRows_ = 1;
    uint32_t vectorTileElems_ = 8;
    uint32_t useRowBuffer_ = 0;
    uint32_t exactTileColumns_ = 0;
    uint32_t blockId_ = 0;
};
} // namespace NsDistanceTransform

#endif // OPS_CV_EXPERIMENTAL_IMAGE_DISTANCE_TRANSFORM_H_
