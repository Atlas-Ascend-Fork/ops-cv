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
 * \file scharr.h
 * \brief Scharr kernel implementation (multi-core vectorized, bit-exact with OpenCV fp32 Scharr).
 *
 * IS_X: true = X gradient, false = Y gradient.
 * BORDER: 0 = REFLECT_101, 1 = REPLICATE, 2 = CONSTANT(0).
 */
#ifndef SCHARR_KERNEL_H
#define SCHARR_KERNEL_H

// 第二轮优化：Y奇数末列跨行Gather快照、64行内批量补偿、集中写回与显式跨流水同步。
// 本版优化：非对齐全宽 Gather 合并、补偿系数常驻、批量计算后仅遍历边界行。
// 保留已验证的逐元素算术顺序；无 volatile；输出始终为 float32。
#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
#include <limits>
#include <type_traits>

using namespace AscendC;

// IS_X: true = X gradient, false = Y gradient.
// BORDER: 0 = REFLECT_101, 1 = REPLICATE, 2 = CONSTANT(0).
template <typename TYPE_X, bool IS_X, uint8_t BORDER>
class KernelScharrVector {
public:
    static_assert(std::is_same_v<TYPE_X, float> || std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, uint8_t>,
                  "Scharr only supports float, half and uint8_t input");
    static_assert(BORDER <= 2, "Invalid Scharr border mode");

    __aicore__ inline KernelScharrVector() {}

    // 初始化 GM、队列及暂存区；索引表与补偿系数每个核只生成一次。

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, GM_ADDR output, int32_t totalLength, int32_t batchCount,
                                int32_t H, int32_t W, int32_t usedCoreNum, int32_t splitMode, int32_t baseRowsPerCore,
                                int32_t extraRowCoreCount, int32_t basePixelsPerCore, int32_t extraPixelCoreCount,
                                int32_t basePlanesPerCore, int32_t extraPlaneCoreCount, int32_t blockW, int32_t tileH,
                                int32_t compRows, int32_t compPitch, int32_t bandW, int32_t rawPitch,
                                int32_t vectorPitch, int32_t inputSlabPitch, int32_t floatSlabPitch,
                                int32_t outputSlabPitch, uint32_t inputBufferBytes, uint32_t outputBufferBytes,
                                uint32_t floatSlabBytes, uint32_t scratchBytes, uint32_t slotsOffset,
                                uint32_t work0Offset, uint32_t work1Offset, uint32_t comp0Offset, uint32_t comp1Offset,
                                uint32_t comp2Offset, uint32_t fallbackOffset, uint32_t maskOffset,
                                uint32_t castHalfOffset, uint32_t offset0Offset, uint32_t offset1Offset,
                                uint32_t coefficientOffset, uint32_t tailBatchOffset, uint32_t bandTableOffset)
    {
        ASSERT(usedCoreNum > 0 && GetBlockNum() == usedCoreNum && "Scharr multi-core tiling mismatch");
        pipe = pipeIn;
        totalLength_ = static_cast<uint64_t>(totalLength);
        batchCount_ = batchCount;
        H_ = H;
        W_ = W;
        usedCoreNum_ = usedCoreNum;
        splitMode_ = splitMode;
        baseRowsPerCore_ = baseRowsPerCore;
        extraRowCoreCount_ = extraRowCoreCount;
        basePixelsPerCore_ = basePixelsPerCore;
        extraPixelCoreCount_ = extraPixelCoreCount;
        basePlanesPerCore_ = basePlanesPerCore;
        extraPlaneCoreCount_ = extraPlaneCoreCount;
        blockW_ = blockW;
        tileH_ = tileH;
        compRows_ = compRows;
        compPitch_ = compPitch;
        bandW_ = bandW;
        bandTableOffset_ = bandTableOffset;
        rawPitch_ = rawPitch;
        vectorPitch_ = vectorPitch;
        inputSlabPitch_ = inputSlabPitch;
        floatSlabPitch_ = floatSlabPitch;
        outputSlabPitch_ = outputSlabPitch;
        slotsOffset_ = slotsOffset;
        work0Offset_ = work0Offset;
        work1Offset_ = work1Offset;
        comp0Offset_ = comp0Offset;
        comp1Offset_ = comp1Offset;
        comp2Offset_ = comp2Offset;
        fallbackOffset_ = fallbackOffset;
        maskOffset_ = maskOffset;
        castHalfOffset_ = castHalfOffset;
        offset0Offset_ = offset0Offset;
        offset1Offset_ = offset1Offset;
        coefficientOffset_ = coefficientOffset;
        tailBatchOffset_ = tailBatchOffset;
        if (bandW_ > 0) {
            // 列分块行跨度复用通用字段，避免增大 TilingData 的拷贝开销。
            bandPitch_ = vectorPitch_;
            bandRawPitch_ = inputSlabPitch_;
            bandFloatPitch_ = floatSlabPitch_;
            bandOutPitch_ = outputSlabPitch_;
        }
        // OpenCV's aarch64 FP32 row filter first consumes four columns per
        // SIMD iteration, then two columns per unrolled scalar iteration,
        // and finally (for odd W) one scalar column.  The first two regions
        // use the center-term FMA ordering; only the final odd column uses
        // the side-term FMA ordering.
        openCvCenterFmaEnd_ = W_ - (W_ & 1);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(input), totalLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), totalLength_);

        boundaryRawPitch_ = 32 / sizeof(TYPE_X);
        boundaryVectorPitch_ = 64;
        pipe->InitBuffer(inputQueue_, 1, inputBufferBytes);
        pipe->InitBuffer(outputQueue_, 1, outputBufferBytes);
        if constexpr (!std::is_same_v<TYPE_X, float>) {
            if (floatSlabBytes > 0U) {
                pipe->InitBuffer(floatSlabBuf_, floatSlabBytes);
            }
        }
        pipe->InitBuffer(scratchArenaBuf_, scratchBytes);

        // CompensatedMulAdd compares an FP32-aligned tail when count is not a
        // multiple of 64. Initialize the fallback slot once; every invocation
        // overwrites its valid prefix and Select ignores the unused tail.
        if constexpr (NEED_SOFTWARE_FMA) {
            const int32_t batchScratch = compRows_ > 0 ? compRows_ * compPitch_ : 2 * vectorPitch_;
            Duplicate(ScratchFloat(fallbackOffset_), 0.0f, batchScratch);
            // 优化：X 常用系数 3、Y 常用系数 10 在每核 Init 时生成一次。
            // K 行扁平化批处理也能复用此向量；它独占布局区间，计算阶段仅只读。
            Duplicate(ScratchFloat(coefficientOffset_), IS_X ? 3.0f : 10.0f, batchScratch);
        }

        // Gather expects byte offsets in a uint32 tensor. Arange supports
        // int32 on Ascend 910B, so generate [0, 4, 8, ...] directly and then
        // reinterpret the same UB storage as uint32. All values are nonnegative.
        if (W_ > 2) {
            const int32_t offsetCount = tileH_ >= 1 ? W_ : ((W_ - 2 < blockW_) ? (W_ - 2) : blockW_);
            LocalTensor<int32_t> offset0 = ScratchInt(offset0Offset_);
            if (tileH_ >= 1) {
                if (bandW_ > 0) {
                    BuildBandTables();
                } else {
                    LocalTensor<int32_t> offset1 = ScratchInt(offset1Offset_);
                    // 优化：先将两个完整表填为合法的列0偏移，再覆盖真实列。
                    // W 后面的填充项也必须初始化；合并 Gather 会读取整个 vectorPitch。
                    Duplicate(offset0, static_cast<int32_t>(0), vectorPitch_);
                    Duplicate(offset1, static_cast<int32_t>(0), vectorPitch_);
                    Arange<int32_t>(offset0, -static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(sizeof(float)),
                                    W_);
                    Arange<int32_t>(offset1, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(sizeof(float)),
                                    W_);
                    PipeBarrier<PIPE_V>();
                    if constexpr (BORDER == 2) {
                        const int32_t zeroOffset = W_ * static_cast<int32_t>(sizeof(float));
                        offset0.SetValue(0, zeroOffset);
                        offset1.SetValue(W_ - 1, zeroOffset);
                    } else {
                        offset0.SetValue(0, Fold(-1, W_) * static_cast<int32_t>(sizeof(float)));
                        offset1.SetValue(W_ - 1, Fold(W_, W_) * static_cast<int32_t>(sizeof(float)));
                    }
                }
            } else {
                Arange<int32_t>(offset0, 0, static_cast<int32_t>(sizeof(float)), offsetCount);
            }
        }
        // 跨行表只依赖物理 pitch，在每核初始化一次；Gather 的偏移单位是字节。
        if (tailBatchOffset_ != 0U) {
            Arange<int32_t>(ScratchInt(tailBatchOffset_ + 3U * TAIL_BATCH_LANES * sizeof(float)), 0,
                            CachedRowPitch() * static_cast<int32_t>(sizeof(float)), TAIL_BATCH_LANES);
        }
        PipeBarrier<PIPE_V>();
    }

    // 列分块 Gather 表（每套=左表||右表 连续，供合并 Gather 读取 2*bandPitch 个元素）：
    //   A(offset0/offset1)：首带，行首=全局列0，输出局部列 [0,bw)；
    //   B(bandTable+0/1段)：内部带，行首=全局列 c0-1，输出局部列 [1,bw+1)；
    //   C(bandTable+2/3段)：末带，与 B 同构，但末列(全局 W-1)的右邻域按边界折叠/置零。
    __aicore__ inline void BuildBandTables()
    {
        const int32_t p = bandPitch_;
        const int32_t bw = bandW_;
        const int32_t s = BAND_SHIFT;
        const int32_t bandCount = (W_ + bw - 1) / bw;
        const int32_t bwLast = W_ - bw * (bandCount - 1);
        const int32_t f4 = static_cast<int32_t>(sizeof(float));
        LocalTensor<int32_t> aLeft = ScratchInt(offset0Offset_);
        LocalTensor<int32_t> aRight = ScratchInt(offset1Offset_);
        LocalTensor<int32_t> tables = ScratchInt(bandTableOffset_);
        LocalTensor<int32_t> bLeft = tables;
        LocalTensor<int32_t> bRight = tables[p];
        LocalTensor<int32_t> cLeft = tables[2 * p];
        LocalTensor<int32_t> cRight = tables[3 * p];
        Duplicate(aLeft, 0, p);
        Duplicate(aRight, 0, p);
        Duplicate(bLeft, 0, p);
        Duplicate(bRight, 0, p);
        Duplicate(cLeft, 0, p);
        Duplicate(cRight, 0, p);
        // A（首带，shift=0）：输出 lane ∈ [0,bw)，left[l]=(l-1)*4、right[l]=(l+1)*4。
        Arange<int32_t>(aLeft, -f4, f4, bw + 1);
        Arange<int32_t>(aRight, f4, f4, bw + 1);
        // B/C（内部/末带，shift=s）：输出 lane ∈ [s,s+bw)，表项覆盖到 s+bw-1。
        Arange<int32_t>(bLeft, -f4, f4, bw + s + 1);
        Arange<int32_t>(bRight, f4, f4, bw + s + 1);
        Arange<int32_t>(cLeft, -f4, f4, bw + s + 1);
        Arange<int32_t>(cRight, f4, f4, bw + s + 1);
        PipeBarrier<PIPE_V>();
        const int32_t aInCount = bw + 1;         // 首带 inCount
        const int32_t cInCount = bwLast + s;     // 末带 inCount（左 halo s 列）
        const int32_t lastLane = s + bwLast - 1; // 末带最后一个输出 lane
        if constexpr (BORDER == 2) {
            // 首带左边界=0：指向扩展拷贝后的 DMA 补零位（index=copyCount=aInCount+s）。
            aLeft.SetValue(0, (aInCount + s) * f4);
            cRight.SetValue(lastLane, cInCount * f4); // 末带右边界=0（零填充槽）
        } else {
            aLeft.SetValue(0, Fold(-1, W_) * f4);
            // 末带右边界折叠：R101→W-2、REP→W-1，映射到左 halo 坐标。
            const int32_t rightLane = (BORDER == 0) ? (lastLane - 1) : lastLane;
            cRight.SetValue(lastLane, rightLane * f4);
        }
        // 非输出 lane 的安全化（结果会被丢弃，仅避免越界读）。
        bLeft.SetValue(0, 0);
        cLeft.SetValue(0, 0);
        cRight.SetValue(lastLane + 1 < p ? lastLane + 1 : lastLane, 0);
    }

    // 根据 host 的行/像素/整切片切分分配任务；各核输出区间互不重叠。

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        if (blockIdx < 0 || blockIdx >= usedCoreNum_ || H_ <= 0 || W_ <= 0 || batchCount_ <= 0 || blockW_ <= 0 ||
            splitMode_ < 0 || splitMode_ > 2) {
            return;
        }
        const uint64_t plane = static_cast<uint64_t>(H_) * static_cast<uint64_t>(W_);
        if (static_cast<uint64_t>(batchCount_) * plane != totalLength_) {
            return;
        }

        if (splitMode_ == 1) {
            ProcessPixelRange(blockIdx, plane);
            return;
        }
        if (splitMode_ == 2) {
            ProcessPlaneRange(blockIdx, plane);
            return;
        }
        if (baseRowsPerCore_ <= 0 || extraRowCoreCount_ < 0 || extraRowCoreCount_ >= usedCoreNum_) {
            return;
        }

        // Quotient/remainder split: the first extraRowCoreCount cores are the
        // "big" cores and receive one more flattened output row. A core range
        // may cover multiple images, so split it again at every H-row plane
        // boundary before invoking the existing precision-stable row kernel.
        const uint64_t coreRows = static_cast<uint64_t>(baseRowsPerCore_) +
                                  static_cast<uint64_t>(blockIdx < extraRowCoreCount_);
        uint64_t flatRow = static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(baseRowsPerCore_) +
                           static_cast<uint64_t>(blockIdx < extraRowCoreCount_ ? blockIdx : extraRowCoreCount_);
        const uint64_t flatRowEnd = flatRow + coreRows;
        while (flatRow < flatRowEnd) {
            const int32_t batch = static_cast<int32_t>(flatRow / static_cast<uint64_t>(H_));
            const int32_t startRow = static_cast<int32_t>(flatRow % static_cast<uint64_t>(H_));
            const uint64_t rowsUntilPlaneEnd = static_cast<uint64_t>(H_ - startRow);
            const uint64_t rowsUntilCoreEnd = flatRowEnd - flatRow;
            const int32_t planeRows = static_cast<int32_t>(rowsUntilPlaneEnd < rowsUntilCoreEnd ? rowsUntilPlaneEnd :
                                                                                                  rowsUntilCoreEnd);
            ProcessPlaneRows(static_cast<uint64_t>(batch) * plane, startRow, planeRows);
            flatRow += static_cast<uint64_t>(planeRows);
        }
    }

private:
    static constexpr int32_t SLOT_COUNT = 6;
    static constexpr int32_t INTERIOR_ROW_COUNT = 3;
    static constexpr int32_t INTERIOR_SLOT_COUNT = 3;
    static constexpr bool NEED_SOFTWARE_FMA = std::is_same_v<TYPE_X, float> || std::is_same_v<TYPE_X, half>;

    // 窄图按完整切片划分，减少小图多核重复 halo 和调度开销。

    __aicore__ inline void ProcessPlaneRange(int32_t blockIdx, uint64_t plane)
    {
        if (basePlanesPerCore_ <= 0 || extraPlaneCoreCount_ < 0 || extraPlaneCoreCount_ >= usedCoreNum_) {
            return;
        }
        const int32_t corePlanes = basePlanesPerCore_ + (blockIdx < extraPlaneCoreCount_ ? 1 : 0);
        const int32_t firstPlane = blockIdx * basePlanesPerCore_ +
                                   (blockIdx < extraPlaneCoreCount_ ? blockIdx : extraPlaneCoreCount_);
        for (int32_t localPlane = 0; localPlane < corePlanes; ++localPlane) {
            const int32_t batch = firstPlane + localPlane;
            ProcessPlaneRows(static_cast<uint64_t>(batch) * plane, 0, H_);
        }
    }

    // 极宽图或行数不足时按像素均分；切分仍限制在各二维切片内。

    __aicore__ inline void ProcessPixelRange(int32_t blockIdx, uint64_t plane)
    {
        if (basePixelsPerCore_ <= 0 || extraPixelCoreCount_ < 0 || extraPixelCoreCount_ >= usedCoreNum_) {
            return;
        }
        const uint64_t corePixels = static_cast<uint64_t>(basePixelsPerCore_) +
                                    static_cast<uint64_t>(blockIdx < extraPixelCoreCount_);
        uint64_t flatPixel = static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(basePixelsPerCore_) +
                             static_cast<uint64_t>(blockIdx < extraPixelCoreCount_ ? blockIdx : extraPixelCoreCount_);
        const uint64_t flatPixelEnd = flatPixel + corePixels;
        while (flatPixel < flatPixelEnd) {
            const int32_t batch = static_cast<int32_t>(flatPixel / plane);
            const uint64_t inPlane = flatPixel % plane;
            const int32_t h = static_cast<int32_t>(inPlane / static_cast<uint64_t>(W_));
            const int32_t startCol = static_cast<int32_t>(inPlane % static_cast<uint64_t>(W_));
            const uint64_t pixelsUntilRowEnd = static_cast<uint64_t>(W_ - startCol);
            const uint64_t pixelsUntilCoreEnd = flatPixelEnd - flatPixel;
            const int32_t count = static_cast<int32_t>(pixelsUntilRowEnd < pixelsUntilCoreEnd ? pixelsUntilRowEnd :
                                                                                                pixelsUntilCoreEnd);
            ProcessRowColumns(static_cast<uint64_t>(batch) * plane, h, startCol, count);
            flatPixel += static_cast<uint64_t>(count);
        }
    }

    __aicore__ inline void ProcessRowColumns(uint64_t base, int32_t h, int32_t startCol, int32_t count)
    {
        const int32_t endCol = startCol + count;
        int32_t col = startCol;
        if (col == 0) {
            RunTile(base, h, 0, 1);
            col = 1;
        }
        const int32_t interiorEnd = endCol < W_ - 1 ? endCol : W_ - 1;
        while (col < interiorEnd) {
            const int32_t remaining = interiorEnd - col;
            int32_t tileCount = remaining < blockW_ ? remaining : blockW_;
            if (col < openCvCenterFmaEnd_ && col + tileCount > openCvCenterFmaEnd_) {
                tileCount = openCvCenterFmaEnd_ - col;
            }
            RunGatherTile(base, h, col, tileCount);
            col += tileCount;
        }
        if (endCol == W_ && W_ > 1) {
            RunTile(base, h, W_ - 1, 1);
        }
    }

    // 优先使用多行 slab，一次搬入多行及上下 halo；过宽或极小尺寸走既有回退路径。

    __aicore__ inline void ProcessPlaneRows(uint64_t base, int32_t startRow, int32_t rowCount)
    {
        if (tileH_ >= 1) {
            for (int32_t localRow = 0; localRow < rowCount; localRow += tileH_) {
                const int32_t remaining = rowCount - localRow;
                const int32_t rows = remaining < tileH_ ? remaining : tileH_;
                if (bandW_ > 0) {
                    RunRowStripBanded(base, startRow + localRow, rows);
                } else {
                    RunRowStrip(base, startRow + localRow, rows);
                }
            }
            return;
        }
        for (int32_t localRow = 0; localRow < rowCount; ++localRow) {
            const int32_t h = startRow + localRow;
            RunTile(base, h, 0, 1);
            int32_t col = 1;
            while (col < W_ - 1) {
                const int32_t remaining = W_ - 1 - col;
                int32_t count = remaining < blockW_ ? remaining : blockW_;
                if (col < openCvCenterFmaEnd_ && col + count > openCvCenterFmaEnd_) {
                    count = openCvCenterFmaEnd_ - col;
                }
                RunGatherTile(base, h, col, count);
                col += count;
            }
            if (W_ > 1) {
                RunTile(base, h, W_ - 1, 1);
            }
        }
    }

    static __aicore__ inline int32_t AlignUp(int32_t value, int32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline LocalTensor<float> ScratchFloat(uint32_t byteOffset)
    {
        return scratchArenaBuf_.Get<float>()[byteOffset / sizeof(float)];
    }

    __aicore__ inline LocalTensor<int32_t> ScratchInt(uint32_t byteOffset)
    {
        return scratchArenaBuf_.Get<int32_t>()[byteOffset / sizeof(int32_t)];
    }

    __aicore__ inline LocalTensor<uint32_t> ScratchUint(uint32_t byteOffset)
    {
        return scratchArenaBuf_.Get<uint32_t>()[byteOffset / sizeof(uint32_t)];
    }

    __aicore__ inline LocalTensor<uint8_t> ScratchByte(uint32_t byteOffset)
    {
        return scratchArenaBuf_.Get<uint8_t>()[byteOffset];
    }

    __aicore__ inline LocalTensor<half> ScratchHalf(uint32_t byteOffset)
    {
        return scratchArenaBuf_.Get<half>()[byteOffset / sizeof(half)];
    }

    static __aicore__ inline bool IsNonFinite(float value)
    {
        const float maxValue = std::numeric_limits<float>::max();
        return value != value || value > maxValue || value < -maxValue;
    }

    // 将越界坐标映射到原图；CONSTANT 返回 -1，后续生成零中间量。

    __aicore__ inline int32_t Fold(int32_t position, int32_t length) const
    {
        if (position >= 0 && position < length) {
            return position;
        }
        if constexpr (BORDER == 0) {
            if (length == 1) {
                return 0;
            }
            return (position < 0) ? 1 : (length - 2);
        } else if constexpr (BORDER == 1) {
            return (position < 0) ? 0 : (length - 1);
        } else {
            return -1;
        }
    }

    __aicore__ inline void MakeSlotCoordinates(int32_t h, int32_t outputCol, int32_t (&rows)[SLOT_COUNT],
                                               int32_t (&cols)[SLOT_COUNT]) const
    {
        const int32_t topRow = Fold(h - 1, H_);
        const int32_t middleRow = h;
        const int32_t bottomRow = Fold(h + 1, H_);
        const int32_t leftCol = Fold(outputCol - 1, W_);
        const int32_t middleCol = outputCol;
        const int32_t rightCol = Fold(outputCol + 1, W_);

        if constexpr (IS_X) {
            if constexpr (BORDER == 2) {
                // OpenCV's 1x1 CONSTANT X path evaluates the zero-weighted
                // center term. 0 * NaN/Inf is NaN, while finite inputs give 0.
                if (H_ == 1 && W_ == 1) {
                    rows[0] = middleRow;
                    cols[0] = middleCol;
                    rows[1] = topRow;
                    cols[1] = rightCol;
                    rows[2] = middleRow;
                    cols[2] = leftCol;
                    rows[3] = middleRow;
                    cols[3] = rightCol;
                    rows[4] = bottomRow;
                    cols[4] = leftCol;
                    rows[5] = bottomRow;
                    cols[5] = rightCol;
                    return;
                }
            }
            rows[0] = topRow;
            cols[0] = leftCol;
            rows[1] = topRow;
            cols[1] = rightCol;
            rows[2] = middleRow;
            cols[2] = leftCol;
            rows[3] = middleRow;
            cols[3] = rightCol;
            rows[4] = bottomRow;
            cols[4] = leftCol;
            rows[5] = bottomRow;
            cols[5] = rightCol;
        } else {
            rows[0] = topRow;
            cols[0] = leftCol;
            rows[1] = topRow;
            cols[1] = middleCol;
            rows[2] = topRow;
            cols[2] = rightCol;
            rows[3] = bottomRow;
            cols[3] = leftCol;
            rows[4] = bottomRow;
            cols[4] = middleCol;
            rows[5] = bottomRow;
            cols[5] = rightCol;
        }
    }

    __aicore__ inline void CopyOneSlot(LocalTensor<TYPE_X> raw, int32_t slot, uint64_t base, int32_t sourceRow,
                                       int32_t sourceCol, int32_t count, int32_t slotPitch)
    {
        const uint64_t gmOffset = base + static_cast<uint64_t>(sourceRow) * static_cast<uint64_t>(W_) +
                                  static_cast<uint64_t>(sourceCol);
        const uint32_t blockLenBytes = static_cast<uint32_t>(count * sizeof(TYPE_X));
        constexpr uint32_t alignElements = 32U / sizeof(TYPE_X);
        const uint32_t rightPadding = (alignElements - static_cast<uint32_t>(count) % alignElements) % alignElements;
        DataCopyExtParams copyParams{1, blockLenBytes, 0, 0, 0};
        DataCopyPadExtParams<TYPE_X> padParams{true, 0, static_cast<uint8_t>(rightPadding), static_cast<TYPE_X>(0)};
        DataCopyPad(raw[slot * slotPitch], xGm_[gmOffset], copyParams, padParams);
    }

    __aicore__ inline LocalTensor<TYPE_X> CopyInteriorRows(uint64_t base, int32_t h, int32_t outputCol, int32_t count,
                                                           int32_t (&rows)[3])
    {
        int32_t rowCount;
        if constexpr (IS_X) {
            rowCount = 3;
            rows[0] = Fold(h - 1, H_);
            rows[1] = h;
            rows[2] = Fold(h + 1, H_);
        } else {
            rowCount = 2;
            rows[0] = Fold(h - 1, H_);
            rows[1] = Fold(h + 1, H_);
            rows[2] = -1;
        }

        LocalTensor<TYPE_X> raw = inputQueue_.AllocTensor<TYPE_X>();
        bool hasDma = false;
        for (int32_t rowSlot = 0; rowSlot < rowCount; ++rowSlot) {
            if (rows[rowSlot] >= 0) {
                CopyOneSlot(raw, rowSlot, base, rows[rowSlot], outputCol - 1, count + 2, rawPitch_);
                hasDma = true;
            }
        }
        if (!hasDma) {
            CopyOneSlot(raw, 0, base, 0, 0, 1, rawPitch_);
        }
        inputQueue_.EnQue(raw);
        return inputQueue_.DeQue<TYPE_X>();
    }

    __aicore__ inline LocalTensor<TYPE_X> CopyIn(uint64_t base, const int32_t (&rows)[SLOT_COUNT],
                                                 const int32_t (&cols)[SLOT_COUNT], int32_t count)
    {
        LocalTensor<TYPE_X> raw = inputQueue_.AllocTensor<TYPE_X>();
        bool hasDma = false;
        for (int32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            if (rows[slot] >= 0 && cols[slot] >= 0) {
                CopyOneSlot(raw, slot, base, rows[slot], cols[slot], count, boundaryRawPitch_);
                hasDma = true;
            }
        }
        // All six logical sources can be CONSTANT padding for degenerate
        // H/W=1 cases. Issue a harmless DMA to create the queue event.
        if (!hasDma) {
            CopyOneSlot(raw, 0, base, 0, 0, 1, boundaryRawPitch_);
        }
        inputQueue_.EnQue(raw);
        return inputQueue_.DeQue<TYPE_X>();
    }

    // 将输入转成 FP32；uint8 经 half 中转，数值仍精确可表示。

    __aicore__ inline void ConvertOneSlot(LocalTensor<float> dst, LocalTensor<TYPE_X> src, bool valid, int32_t count)
    {
        if (!valid) {
            Duplicate(dst, 0.0f, count);
        } else if constexpr (std::is_same_v<TYPE_X, float>) {
            Adds(dst, src, 0.0f, count);
        } else if constexpr (std::is_same_v<TYPE_X, half>) {
            Cast(dst, src, RoundMode::CAST_NONE, count);
        } else {
            LocalTensor<half> castHalf = ScratchHalf(castHalfOffset_);
            Cast(castHalf, src, RoundMode::CAST_NONE, count);
            Cast(dst, castHalf, RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void PrepareFloatSlots(LocalTensor<TYPE_X> raw, const int32_t (&rows)[SLOT_COUNT],
                                             const int32_t (&cols)[SLOT_COUNT], int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        for (int32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            const bool valid = rows[slot] >= 0 && cols[slot] >= 0;
            ConvertOneSlot(slots[slot * boundaryVectorPitch_], raw[slot * boundaryRawPitch_], valid, count);
        }
    }

    // Software-emulated correctly-rounded mul-add for the two Scharr
    // coefficients. multiplier = highScale + lowScale, where both parts are
    // powers of two: 10 = 8 + 2 and 3 = 2 + 1. Therefore both partial
    // products are exact before TwoSum reconstructs the discarded low part.
    // 保留软件误差补偿的每一步运算及非有限值回退；addend 会被修改，不能当只读缓存。
    __aicore__ inline void CompensatedMulAdd(LocalTensor<float> dst, LocalTensor<float> multiplicand,
                                             LocalTensor<float> addend, float highScale, float lowScale, int32_t count)
    {
        LocalTensor<float> high = ScratchFloat(comp0Offset_);
        LocalTensor<float> low = ScratchFloat(comp1Offset_);
        LocalTensor<float> primary = ScratchFloat(comp2Offset_);
        LocalTensor<float> residual = ScratchFloat(work1Offset_);
        LocalTensor<float> fallback = ScratchFloat(fallbackOffset_);

        // The error-free transform below is valid only for finite results.
        // Keep the ordinary result before the transform mutates addend: an
        // infinite partial product would otherwise form Inf - Inf and turn
        // OpenCV's +/-Inf output into NaN.
        const int32_t compareCount = AlignUp(count, 64);
        Adds(fallback, multiplicand, 0.0f, count);
        // 优化：常见系数直接读取常驻向量，省去每次补偿的 Duplicate。
        // Y 奇数末列采用系数3的既有路径，继续使用临时系数，不改舍入顺序。
        if ((IS_X && highScale == 2.0f && lowScale == 1.0f) || (!IS_X && highScale == 8.0f && lowScale == 2.0f)) {
            FusedMulAdd(fallback, ScratchFloat(coefficientOffset_), addend, count);
        } else {
            Duplicate(high, highScale + lowScale, count);
            FusedMulAdd(fallback, high, addend, count);
        }

        Muls(high, multiplicand, highScale, count);
        Muls(low, multiplicand, lowScale, count);
        Add(primary, high, low, count);
        Sub(residual, primary, high, count);
        Sub(low, low, residual, count);

        Add(dst, primary, addend, count);
        Sub(high, dst, primary, count);
        Sub(residual, dst, high, count);
        Sub(primary, primary, residual, count);
        Sub(addend, addend, high, count);
        Add(primary, primary, addend, count);
        Add(primary, primary, low, count);
        Add(dst, dst, primary, count);

        // Compares requires a whole 256-byte FP32 block. The unused tail of
        // fallback was initialized to zero above, so it is safe to compare.
        Abs(high, fallback, compareCount);
        LocalTensor<uint8_t> finiteMask = ScratchByte(maskOffset_);
        Compares(finiteMask, high, std::numeric_limits<float>::max(), CMPMODE::LE, static_cast<uint32_t>(compareCount));
        Select(low, finiteMask, dst, fallback, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Adds(dst, low, 0.0f, count);
    }

    __aicore__ inline void ComputeVector(LocalTensor<float> output, int32_t count, bool useCompensation)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> s0 = slots[0 * boundaryVectorPitch_];
        LocalTensor<float> s1 = slots[1 * boundaryVectorPitch_];
        LocalTensor<float> s2 = slots[2 * boundaryVectorPitch_];
        LocalTensor<float> s3 = slots[3 * boundaryVectorPitch_];
        LocalTensor<float> s4 = slots[4 * boundaryVectorPitch_];
        LocalTensor<float> s5 = slots[5 * boundaryVectorPitch_];
        LocalTensor<float> work0 = ScratchFloat(work0Offset_);
        LocalTensor<float> work1 = ScratchFloat(work1Offset_);

        if constexpr (IS_X) {
            if constexpr (BORDER == 2) {
                if (H_ == 1 && W_ == 1) {
                    Muls(output, s0, 0.0f, count);
                    return;
                }
            }
            Sub(work0, s1, s0, count);
            Sub(output, s5, s4, count);
            Add(work0, work0, output, count);
            Sub(work1, s3, s2, count);
            Muls(s0, work1, 10.0f, count);
            if constexpr (NEED_SOFTWARE_FMA) {
                CompensatedMulAdd(output, work0, s0, 2.0f, 1.0f, count);
            } else {
                // uint8 values and the complete Scharr integer range are
                // exactly representable in FP32. Keep all computation in
                // FP32, but avoid coefficient materialization and FMA.
                Muls(work0, work0, 3.0f, count);
                Add(output, work0, s0, count);
            }
        } else {
            Add(work0, s0, s2, count);
            if constexpr (NEED_SOFTWARE_FMA) {
                if (useCompensation) {
                    Muls(work0, work0, 3.0f, count);
                    CompensatedMulAdd(output, s1, work0, 8.0f, 2.0f, count);
                } else {
                    Muls(s0, s1, 10.0f, count);
                    CompensatedMulAdd(output, work0, s0, 2.0f, 1.0f, count);
                }
            } else {
                Muls(work0, work0, 3.0f, count);
                Muls(output, s1, 10.0f, count);
                Add(output, output, work0, count);
            }
            Add(work0, s3, s5, count);
            if constexpr (NEED_SOFTWARE_FMA) {
                if (useCompensation) {
                    Muls(work0, work0, 3.0f, count);
                    CompensatedMulAdd(s0, s4, work0, 8.0f, 2.0f, count);
                } else {
                    Muls(s1, s4, 10.0f, count);
                    CompensatedMulAdd(s0, work0, s1, 2.0f, 1.0f, count);
                }
            } else {
                Muls(work0, work0, 3.0f, count);
                Muls(s0, s4, 10.0f, count);
                Add(s0, s0, work0, count);
            }
            Sub(output, s0, output, count);
        }
    }

    __aicore__ inline void PrepareGatherRow(LocalTensor<TYPE_X> raw, int32_t rawSlot, bool valid, int32_t countWithHalo)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        ConvertOneSlot(slots, raw[rawSlot * rawPitch_], valid, countWithHalo);
    }

    __aicore__ inline void GatherSides(int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<uint32_t> offsets = ScratchUint(offset0Offset_);
        Gather(slots[vectorPitch_], slots, offsets, 0U, static_cast<uint32_t>(count));
        Gather(slots[2 * vectorPitch_], slots, offsets, 2U * sizeof(float), static_cast<uint32_t>(count));
    }

    __aicore__ inline void GatherCenter(int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<uint32_t> offsets = ScratchUint(offset0Offset_);
        Gather(slots[vectorPitch_], slots, offsets, static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(count));
    }

    __aicore__ inline void ComputeGatherX(LocalTensor<TYPE_X> raw, const int32_t (&rows)[3], LocalTensor<float> output,
                                          int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> left = slots[vectorPitch_];
        LocalTensor<float> right = slots[2 * vectorPitch_];
        LocalTensor<float> work0 = ScratchFloat(work0Offset_);
        LocalTensor<float> work1 = ScratchFloat(work1Offset_);

        PrepareGatherRow(raw, 0, rows[0] >= 0, count + 2);
        GatherSides(count);
        Sub(work0, right, left, count);

        PrepareGatherRow(raw, 2, rows[2] >= 0, count + 2);
        GatherSides(count);
        Sub(output, right, left, count);
        Add(work0, work0, output, count);

        PrepareGatherRow(raw, 1, true, count + 2);
        GatherSides(count);
        Sub(work1, right, left, count);
        Muls(left, work1, 10.0f, count);
        if constexpr (NEED_SOFTWARE_FMA) {
            CompensatedMulAdd(output, work0, left, 2.0f, 1.0f, count);
        } else {
            Muls(work0, work0, 3.0f, count);
            Add(output, work0, left, count);
        }
    }

    __aicore__ inline void ComputeOneGatherYRowFma(LocalTensor<TYPE_X> raw, int32_t rawSlot, bool valid,
                                                   LocalTensor<float> smooth, LocalTensor<float> side, int32_t count,
                                                   bool useCompensation)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> left = slots[vectorPitch_];
        LocalTensor<float> right = slots[2 * vectorPitch_];
        PrepareGatherRow(raw, rawSlot, valid, count + 2);
        GatherSides(count);
        Add(side, left, right, count);
        GatherCenter(count);
        if constexpr (NEED_SOFTWARE_FMA) {
            if (useCompensation) {
                Muls(side, side, 3.0f, count);
                CompensatedMulAdd(smooth, left, side, 8.0f, 2.0f, count);
            } else {
                Muls(slots, left, 10.0f, count);
                CompensatedMulAdd(smooth, side, slots, 2.0f, 1.0f, count);
            }
        } else {
            Muls(side, side, 3.0f, count);
            Muls(smooth, left, 10.0f, count);
            Add(smooth, smooth, side, count);
        }
    }

    __aicore__ inline void ComputeGatherY(LocalTensor<TYPE_X> raw, const int32_t (&rows)[3], LocalTensor<float> output,
                                          int32_t count, bool useCompensation)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> bottomResult = slots[2 * vectorPitch_];
        LocalTensor<float> work0 = ScratchFloat(work0Offset_);
        LocalTensor<float> work1 = ScratchFloat(work1Offset_);
        if constexpr (NEED_SOFTWARE_FMA) {
            ComputeOneGatherYRowFma(raw, 0, rows[0] >= 0, output, work0, count, useCompensation);
            ComputeOneGatherYRowFma(raw, 1, rows[1] >= 0, bottomResult, work0, count, useCompensation);
            Sub(output, bottomResult, output, count);
        } else {
            ComputeOneGatherYRowFma(raw, 0, rows[0] >= 0, work0, work1, count, false);
            ComputeOneGatherYRowFma(raw, 1, rows[1] >= 0, output, work1, count, false);
            Sub(output, output, work0, count);
        }
    }

    // Full-width row strip: each physical input row is copied once, and all
    // output rows in the strip are staged before a single multi-block copy.
    __aicore__ inline LocalTensor<float> CachedRow(LocalTensor<TYPE_X> raw, int32_t localRow)
    {
        if constexpr (std::is_same_v<TYPE_X, float>) {
            return raw[localRow * inputSlabPitch_];
        } else {
            return floatSlabBuf_.Get<float>()[localRow * floatSlabPitch_];
        }
    }

    __aicore__ inline LocalTensor<float> CachedIntermediate(LocalTensor<TYPE_X> raw, int32_t logicalRow,
                                                            int32_t loadFirst)
    {
        return CachedRow(raw, logicalRow - loadFirst);
    }

    __aicore__ inline int32_t CachedRowPitch() const
    {
        if constexpr (std::is_same_v<TYPE_X, float>) {
            // FP32 overwrites the original input slab in place.
            return inputSlabPitch_;
        } else {
            // FP16/uint8 first convert into the dedicated FP32 slab.
            return floatSlabPitch_;
        }
    }

    __aicore__ inline void GatherFullWidthSides(LocalTensor<float> source)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<uint32_t> leftOffsets = ScratchUint(offset0Offset_);
        // 优化：左右两表各占 vectorPitch 个元素，且在 scratch 中连续。
        // 因填充表项也指向源行列0，可以一次收集 2*vectorPitch 个元素。
        // 有效结果仍分别位于 slots[pitch] 和 slots[2*pitch]；后续只计算 W 列。
        // 因而 W=255/pitch=256、W=65/pitch=128 等也无需两次 Gather。
        Gather(slots[vectorPitch_], source, leftOffsets, 0U, static_cast<uint32_t>(2 * vectorPitch_));
    }

    // The input/converted slab is overwritten in place with the horizontal
    // stage. Each physical source row is therefore evaluated exactly once.
    // 每个输入行仅计算一次左右差，并原地保存；奇数末列保留 OpenCV 的非有限值行为。
    __aicore__ inline void ComputeHorizontalX(LocalTensor<float> row)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> left = slots[vectorPitch_];
        LocalTensor<float> right = slots[2 * vectorPitch_];
        float tailCenter = 0.0f;
        if constexpr (NEED_SOFTWARE_FMA) {
            if ((W_ & 1) != 0) {
                // V->S: the scalar read consumes a Vector-produced lane.
                const auto vToS = pipe->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(vToS);
                WaitFlag<HardEvent::V_S>(vToS);
                tailCenter = row.GetValue(W_ - 1);
            }
        }
        GatherFullWidthSides(row);
        Sub(row, right, left, W_);
        if constexpr (NEED_SOFTWARE_FMA) {
            if ((W_ & 1) != 0 && IsNonFinite(tailCenter)) {
                // V->S then S->V around the scalar write of the tail lane.
                const auto vToS2 = pipe->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(vToS2);
                WaitFlag<HardEvent::V_S>(vToS2);
                row.SetValue(W_ - 1, std::numeric_limits<float>::quiet_NaN());
                const auto sToV = pipe->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(sToV);
                WaitFlag<HardEvent::S_V>(sToV);
            }
        }
    }

    // 每个输入行仅计算一次水平平滑；普通列和奇数末列沿用各自舍入顺序。

    __aicore__ inline void ComputeHorizontalY(LocalTensor<float> row, bool finishOddTail = true)
    {
        // 批量末列路径只计算前 W-1 列；末列按原有另一种 FMA 顺序统一补算。
        const int32_t count = finishOddTail ? W_ : W_ - 1;
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> center = slots;
        LocalTensor<float> left = slots[vectorPitch_];
        LocalTensor<float> right = slots[2 * vectorPitch_];
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        GatherFullWidthSides(row);
        Adds(center, row, 0.0f, count);
        Add(side, left, right, count);
        if constexpr (NEED_SOFTWARE_FMA) {
            Muls(side, side, 3.0f, count);
            CompensatedMulAdd(row, center, side, 8.0f, 2.0f, count);
            if (finishOddTail && (W_ & 1) != 0) {
                const int32_t tail = W_ - 1;
                // Vector API operands must start at a 32-byte-aligned UB
                // address. tail is not necessarily a multiple of 8 (for
                // W=255, tail=254), so stage the three scalar values in the
                // aligned lane 0, calculate there, then write the result back.
                // V->S: the three scalar reads consume Vector-produced lanes.
                const auto vToS = pipe->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(vToS);
                WaitFlag<HardEvent::V_S>(vToS);
                const float tailLeft = left.GetValue(tail);
                const float tailCenter = center.GetValue(tail);
                const float tailRight = right.GetValue(tail);
                left.SetValue(0, tailLeft);
                center.SetValue(0, tailCenter);
                right.SetValue(0, tailRight);
                // S->V: publish the staged lane-0 scalars to the Vector unit.
                const auto sToV = pipe->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(sToV);
                WaitFlag<HardEvent::S_V>(sToV);
                Add(side, left, right, 1);
                Muls(center, center, 10.0f, 1);
                CompensatedMulAdd(right, side, center, 2.0f, 1.0f, 1);
                // V->S: read back the Vector result, then S->V after the store.
                const auto vToS2 = pipe->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(vToS2);
                WaitFlag<HardEvent::V_S>(vToS2);
                row.SetValue(tail, right.GetValue(0));
                const auto sToV2 = pipe->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(sToV2);
                WaitFlag<HardEvent::S_V>(sToV2);
            }
        } else {
            Muls(side, side, 3.0f, count);
            Muls(row, center, 10.0f, count);
            Add(row, row, side, count);
        }
    }

    // uint8 has no NaN/Inf and every Scharr intermediate is a small integer
    // exactly representable by FP32. Process two independent rows together
    // to expose Vector instruction-level parallelism without changing any
    // float16/float32 operation or OpenCV rounding path.
    __aicore__ inline void ComputeHorizontalYPairUint8(LocalTensor<float> row0, LocalTensor<float> row1)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> left = slots[vectorPitch_];
        LocalTensor<float> right = slots[2 * vectorPitch_];
        LocalTensor<float> side0 = ScratchFloat(work0Offset_);
        LocalTensor<float> side1 = ScratchFloat(work1Offset_);

        GatherFullWidthSides(row0);
        Add(side0, left, right, W_);
        GatherFullWidthSides(row1);
        Add(side1, left, right, W_);

        Muls(side0, side0, 3.0f, W_);
        Muls(side1, side1, 3.0f, W_);
        Muls(row0, row0, 10.0f, W_);
        Muls(row1, row1, 10.0f, W_);
        Add(row0, row0, side0, W_);
        Add(row1, row1, side1, W_);
    }

    // K 行扁平化水平平滑：对 CachedRow(first .. first+rows) 按“主列顺序”
    // 整链计算（count = rows × 行跨度）。所有算子都是逐 lane 运算，与逐行
    // 版本相比只改变 Vector API 的调用粒度，不改变任何元素的运算顺序；
    // center 的“+0”有符号零归一化改为在源行上原地进行（multiplicand 与 dst
    // 同址在补偿链中是安全的，multiplicand 仅在链的前半段被读取）。
    __aicore__ inline void SmoothHorizontalYRows(LocalTensor<TYPE_X> raw, int32_t firstRow, int32_t rows)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        const int32_t pitch = CachedRowPitch();
        // 每个收集块按 max(pitch, vectorPitch) 预留，避免行尾收集结果越过块边界。
        const int32_t blockStride = pitch > vectorPitch_ ? pitch : vectorPitch_;
        LocalTensor<float> row0 = CachedRow(raw, firstRow);
        LocalTensor<float> left = slots;
        LocalTensor<float> right = slots[rows * blockStride];
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        LocalTensor<uint32_t> leftTable = ScratchUint(offset0Offset_);
        LocalTensor<uint32_t> rightTable = ScratchUint(offset1Offset_);
        // 先按行收集左右邻域（读取原值），再原地覆盖源行。
        for (int32_t r = 0; r < rows; ++r) {
            LocalTensor<float> row = row0[r * pitch];
            Gather(left[r * pitch], row, leftTable, 0U, static_cast<uint32_t>(vectorPitch_));
            Gather(right[r * pitch], row, rightTable, 0U, static_cast<uint32_t>(vectorPitch_));
        }
        const int32_t count = rows * pitch;
        Adds(row0, row0, 0.0f, count);
        Add(side, left, right, count);
        Muls(side, side, 3.0f, count);
        CompensatedMulAdd(row0, row0, side, 8.0f, 2.0f, count);
    }

    // Y 奇数末列：按 compRows 微批先快照原输入，再按 K 行扁平平滑，最后用
    // 末列专用舍入顺序集中补算并写回。逐 lane 运算与旧版逐行路径完全一致。
    __aicore__ inline void PrecomputeHorizontalYTailBatch(LocalTensor<TYPE_X> raw, int32_t physicalRows)
    {
        LocalTensor<float> tailLeft = ScratchFloat(tailBatchOffset_);
        LocalTensor<float> tailCenter = ScratchFloat(tailBatchOffset_ + TAIL_BATCH_LANES * sizeof(float));
        LocalTensor<float> tailRight = ScratchFloat(tailBatchOffset_ + 2U * TAIL_BATCH_LANES * sizeof(float));
        LocalTensor<uint32_t> rowOffsets = ScratchUint(tailBatchOffset_ + 3U * TAIL_BATCH_LANES * sizeof(float));
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        // 快照/写回保持 64 行一批（沿用旧版同步频率）；平滑按 K 行微批扁平化。
        const int32_t chunkRows = TAIL_BATCH_LANES;
        const int32_t smoothRows = compRows_ > 1 ? compRows_ : chunkRows;
        for (int32_t first = 0; first < physicalRows; first += chunkRows) {
            const int32_t count = physicalRows - first < chunkRows ? physicalRows - first : chunkRows;
            LocalTensor<float> firstRow = CachedRow(raw, first);
            Gather(tailLeft, firstRow, rowOffsets, static_cast<uint32_t>((W_ - 2) * sizeof(float)),
                   static_cast<uint32_t>(count));
            Gather(tailCenter, firstRow, rowOffsets, static_cast<uint32_t>((W_ - 1) * sizeof(float)),
                   static_cast<uint32_t>(count));
            if constexpr (BORDER == 2) {
                // CONSTANT右邻域为零，不读取下一行，避免边界跨行污染。
                Duplicate(tailRight, 0.0f, count);
            } else {
                const int32_t rightColumn = BORDER == 0 ? W_ - 2 : W_ - 1;
                Gather(tailRight, firstRow, rowOffsets, static_cast<uint32_t>(rightColumn * sizeof(float)),
                       static_cast<uint32_t>(count));
            }

            // Gather按离散索引读取多行，显式等待快照完成后再原地覆盖源行。
            PipeBarrier<PIPE_V>();
            for (int32_t local = 0; local < count; local += smoothRows) {
                const int32_t rows = count - local < smoothRows ? count - local : smoothRows;
                SmoothHorizontalYRows(raw, first + local, rows);
            }
            // 原逐行版本先 Adds(center, input, 0)，这里保留该步骤的有符号零行为。
            Adds(tailCenter, tailCenter, 0.0f, count);
            Add(side, tailLeft, tailRight, count);
            Muls(tailCenter, tailCenter, 10.0f, count);
            CompensatedMulAdd(tailRight, side, tailCenter, 2.0f, 1.0f, count);

            // 910B不支持通用Scatter；一次 V->S 同步后集中写回非对齐末列。
            // 写回每行仍是一对 UB 标量读写，但不再每行启动一次补偿向量链。
            const auto vToS = pipe->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(vToS);
            WaitFlag<HardEvent::V_S>(vToS);
            for (int32_t r = 0; r < count; ++r) {
                CachedRow(raw, first + r).SetValue(W_ - 1, tailRight.GetValue(r));
            }
            // 显式保证标量 UB 写入对后续垂直 Vector 组合可见。
            const auto sToV = pipe->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(sToV);
            WaitFlag<HardEvent::S_V>(sToV);
        }
    }

    // 输入行水平预计算一次，供相邻输出行复用；uint8 使用既有双行路径。

    __aicore__ inline void PrecomputeHorizontal(LocalTensor<TYPE_X> raw, int32_t physicalRows)
    {
        if constexpr (NEED_SOFTWARE_FMA && !IS_X) {
            if (tailBatchOffset_ != 0U && physicalRows >= 1) {
                PrecomputeHorizontalYTailBatch(raw, physicalRows);
                return;
            }
            if (compRows_ > 1 && (W_ & 1) == 0) {
                // 偶数宽 Y 没有末列特殊顺序，直接按 K 行扁平平滑。
                for (int32_t first = 0; first < physicalRows; first += compRows_) {
                    const int32_t count = physicalRows - first < compRows_ ? physicalRows - first : compRows_;
                    SmoothHorizontalYRows(raw, first, count);
                }
                return;
            }
        }
        if constexpr (std::is_same_v<TYPE_X, uint8_t> && !IS_X) {
            int32_t r = 0;
            for (; r + 1 < physicalRows; r += 2) {
                ComputeHorizontalYPairUint8(CachedRow(raw, r), CachedRow(raw, r + 1));
            }
            if (r < physicalRows) {
                ComputeHorizontalY(CachedRow(raw, r));
            }
            return;
        }
        for (int32_t r = 0; r < physicalRows; ++r) {
            LocalTensor<float> row = CachedRow(raw, r);
            if constexpr (IS_X) {
                ComputeHorizontalX(row);
            } else {
                ComputeHorizontalY(row);
            }
        }
    }

    __aicore__ inline void CombineSeparableX(LocalTensor<TYPE_X> raw, int32_t top, int32_t middle, int32_t bottom,
                                             int32_t loadFirst, LocalTensor<float> output)
    {
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        const bool topValid = top >= 0;
        const bool bottomValid = bottom >= 0;
        if (topValid && bottomValid) {
            Add(side, CachedIntermediate(raw, top, loadFirst), CachedIntermediate(raw, bottom, loadFirst), W_);
        } else if (topValid) {
            Adds(side, CachedIntermediate(raw, top, loadFirst), 0.0f, W_);
        } else if (bottomValid) {
            Adds(side, CachedIntermediate(raw, bottom, loadFirst), 0.0f, W_);
        } else {
            Duplicate(side, 0.0f, W_);
        }
        FinishSeparableX(CachedIntermediate(raw, middle, loadFirst), output, side, W_);
    }

    __aicore__ inline void FinishSeparableX(LocalTensor<float> middle, LocalTensor<float> output,
                                            LocalTensor<float> side, int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> middleWeighted = slots[vectorPitch_];
        Muls(middleWeighted, middle, 10.0f, count);
        if constexpr (NEED_SOFTWARE_FMA) {
            CompensatedMulAdd(output, side, middleWeighted, 2.0f, 1.0f, count);
        } else {
            Muls(side, side, 3.0f, count);
            Add(output, side, middleWeighted, count);
        }
    }

    // K 行扁平化版本的 X finish：与 FinishSeparableX 逐 lane 运算顺序一致，
    // middleWeighted 暂存放在 slots 内（host 按 (1+K) 行规划，K=2 时与旧布局等容）。
    __aicore__ inline void FinishSeparableXFlat(LocalTensor<float> middle, LocalTensor<float> output,
                                                LocalTensor<float> side, int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> middleWeighted = slots[vectorPitch_];
        Muls(middleWeighted, middle, 10.0f, count);
        CompensatedMulAdd(output, side, middleWeighted, 2.0f, 1.0f, count);
    }

    __aicore__ inline void FinishSeparableXPairUint8(LocalTensor<float> middle0, LocalTensor<float> middle1,
                                                     LocalTensor<float> side0, LocalTensor<float> side1, int32_t count)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        LocalTensor<float> middleWeighted0 = slots[vectorPitch_];
        LocalTensor<float> middleWeighted1 = slots[2 * vectorPitch_];
        Muls(middleWeighted0, middle0, 10.0f, count);
        Muls(middleWeighted1, middle1, 10.0f, count);
        Muls(side0, side0, 3.0f, count);
        Muls(side1, side1, 3.0f, count);
        Add(side0, side0, middleWeighted0, count);
        Add(side1, side1, middleWeighted1, count);
    }

    __aicore__ inline void CombineSeparableY(LocalTensor<TYPE_X> raw, int32_t top, int32_t bottom, int32_t loadFirst,
                                             LocalTensor<float> output)
    {
        const bool topValid = top >= 0;
        const bool bottomValid = bottom >= 0;
        if (topValid && bottomValid) {
            Sub(output, CachedIntermediate(raw, bottom, loadFirst), CachedIntermediate(raw, top, loadFirst), W_);
        } else if (bottomValid) {
            Adds(output, CachedIntermediate(raw, bottom, loadFirst), 0.0f, W_);
        } else if (topValid) {
            Muls(output, CachedIntermediate(raw, top, loadFirst), -1.0f, W_);
        } else {
            Duplicate(output, 0.0f, W_);
        }
    }

    __aicore__ inline bool CanBatchVerticalRows() const
    {
        // BinaryRepeatParams expresses repeat strides in 32-byte blocks with
        // an uint8 field. Rows and 64-column chunks are all 32-byte aligned.
        const int32_t cachedPitch = CachedRowPitch();
        return (cachedPitch & 7) == 0 && (outputSlabPitch_ & 7) == 0 && cachedPitch / 8 <= 255 &&
               outputSlabPitch_ / 8 <= 255;
    }

    template <bool DO_SUB>
    // 按 repeat 批量组合垂直方向；使用物理 pitch，分段限制 repeatTimes<=255。
    __aicore__ inline void ComputeVerticalRows(LocalTensor<float> dst, LocalTensor<float> src0, LocalTensor<float> src1,
                                               int32_t rows)
    {
        constexpr int32_t FP32_REPEAT_ELEMENTS = 64;
        const int32_t cachedPitch = CachedRowPitch();
        int32_t completedRows = 0;
        while (completedRows < rows) {
            const int32_t remainingRows = rows - completedRows;
            const uint8_t repeatTimes = static_cast<uint8_t>(remainingRows < 255 ? remainingRows : 255);
            const uint8_t dstRepStride = repeatTimes == 1 ? 0 : static_cast<uint8_t>(outputSlabPitch_ / 8);
            const uint8_t srcRepStride = repeatTimes == 1 ? 0 : static_cast<uint8_t>(cachedPitch / 8);
            const BinaryRepeatParams params{1, 1, 1, dstRepStride, srcRepStride, srcRepStride};
            const int32_t dstRowOffset = completedRows * outputSlabPitch_;
            const int32_t srcRowOffset = completedRows * cachedPitch;
            for (int32_t col = 0; col < W_; col += FP32_REPEAT_ELEMENTS) {
                const uint64_t mask = static_cast<uint64_t>(W_ - col < FP32_REPEAT_ELEMENTS ? W_ - col :
                                                                                              FP32_REPEAT_ELEMENTS);
                if constexpr (DO_SUB) {
                    Sub(dst[dstRowOffset + col], src0[srcRowOffset + col], src1[srcRowOffset + col], mask, repeatTimes,
                        params);
                } else {
                    Add(dst[dstRowOffset + col], src0[srcRowOffset + col], src1[srcRowOffset + col], mask, repeatTimes,
                        params);
                }
            }
            completedRows += static_cast<int32_t>(repeatTimes);
        }
    }

    // 统一行级回退：边界 Fold 和垂直组合沿用通过测试的实现。
    __aicore__ inline void ComputeStripBoundaryRow(LocalTensor<TYPE_X> raw, LocalTensor<float> staged, int32_t startRow,
                                                   int32_t localRow, int32_t loadFirst)
    {
        const int32_t h = startRow + localRow;
        const int32_t top = Fold(h - 1, H_);
        const int32_t bottom = Fold(h + 1, H_);
        LocalTensor<float> outRow = staged[localRow * outputSlabPitch_];
        if constexpr (IS_X) {
            CombineSeparableX(raw, top, h, bottom, loadFirst, outRow);
        } else {
            CombineSeparableY(raw, top, bottom, loadFirst, outRow);
        }
    }

    // 多行输入及输出批量搬运；保留真实 pitch 和 MTE3 行间 srcStride，避免非对齐行串位。

    __aicore__ inline void RunRowStrip(uint64_t base, int32_t startRow, int32_t rows)
    {
        const int32_t loadFirst = startRow > 0 ? startRow - 1 : 0;
        const int32_t loadLast = startRow + rows < H_ ? startRow + rows : H_ - 1;
        const int32_t physicalRows = loadLast - loadFirst + 1;
        LocalTensor<TYPE_X> raw = inputQueue_.AllocTensor<TYPE_X>();
        const uint32_t pad = static_cast<uint32_t>(inputSlabPitch_ - W_);
        DataCopyExtParams inParams{static_cast<uint16_t>(physicalRows), static_cast<uint32_t>(W_ * sizeof(TYPE_X)), 0,
                                   0, 0};
        DataCopyPadExtParams<TYPE_X> padParams{true, 0, static_cast<uint8_t>(pad), static_cast<TYPE_X>(0)};
        DataCopyPad(raw, xGm_[base + static_cast<uint64_t>(loadFirst) * W_], inParams, padParams);
        inputQueue_.EnQue(raw);
        raw = inputQueue_.DeQue<TYPE_X>();

        if constexpr (!std::is_same_v<TYPE_X, float>) {
            LocalTensor<float> converted = floatSlabBuf_.Get<float>();
            // Only CONSTANT addresses the zero sentinel at column W.  The
            // reflected/replicated paths fold the last Gather offset back
            // into [0, W), so converting W+1 elements would unnecessarily
            // cross the compact pitch when W is exactly vector aligned.
            constexpr int32_t SENTINEL_ELEMENTS = BORDER == 2 ? 1 : 0;
            for (int32_t r = 0; r < physicalRows; ++r) {
                ConvertOneSlot(converted[r * floatSlabPitch_], raw[r * inputSlabPitch_], true, W_ + SENTINEL_ELEMENTS);
            }
        }
        PrecomputeHorizontal(raw, physicalRows);

        LocalTensor<float> staged = outputQueue_.AllocTensor<float>();
        int32_t bulkFirst = startRow == 0 ? 1 : 0;
        int32_t bulkEnd = startRow + rows == H_ ? rows - 1 : rows;
        if (bulkEnd < bulkFirst) {
            bulkEnd = bulkFirst;
        }
        const int32_t bulkRows = bulkEnd - bulkFirst;
        const bool useBulkRows = bulkRows > 0 && CanBatchVerticalRows();
        if (useBulkRows) {
            const int32_t firstGlobalRow = startRow + bulkFirst;
            const int32_t topLocalRow = firstGlobalRow - 1 - loadFirst;
            LocalTensor<float> bulkOutput = staged[bulkFirst * outputSlabPitch_];
            LocalTensor<float> bulkTop = CachedRow(raw, topLocalRow);
            LocalTensor<float> bulkBottom = CachedRow(raw, topLocalRow + 2);
            if constexpr (IS_X) {
                // First calculate all (top + bottom) rows with one repeat-mode
                // Add per 64-column chunk. The compensated weighted sum still
                // keeps its original per-row arithmetic order.
                ComputeVerticalRows<false>(bulkOutput, bulkTop, bulkBottom, bulkRows);
                if constexpr (NEED_SOFTWARE_FMA) {
                    const int32_t cachedPitch = CachedRowPitch();
                    const int32_t batchRowsMax = compRows_ > 1 ? compRows_ : 1;
                    if (cachedPitch == outputSlabPitch_ && cachedPitch <= compPitch_) {
                        // The software transform is entirely lane-local, so
                        // flattening equally pitched rows does not change a
                        // single element's operation order. It only divides
                        // Scalar-side Vector API issue count by compRows.
                        for (int32_t r = bulkFirst; r < bulkEnd; r += batchRowsMax) {
                            const int32_t batchRows = bulkEnd - r < batchRowsMax ? bulkEnd - r : batchRowsMax;
                            const int32_t h = startRow + r;
                            FinishSeparableXFlat(CachedIntermediate(raw, h, loadFirst), staged[r * outputSlabPitch_],
                                                 staged[r * outputSlabPitch_], batchRows * cachedPitch);
                        }
                    } else {
                        for (int32_t r = bulkFirst; r < bulkEnd; ++r) {
                            const int32_t h = startRow + r;
                            FinishSeparableX(CachedIntermediate(raw, h, loadFirst), staged[r * outputSlabPitch_],
                                             staged[r * outputSlabPitch_], W_);
                        }
                    }
                } else {
                    int32_t r = bulkFirst;
                    for (; r + 1 < bulkEnd; r += 2) {
                        const int32_t h0 = startRow + r;
                        const int32_t h1 = h0 + 1;
                        FinishSeparableXPairUint8(CachedIntermediate(raw, h0, loadFirst),
                                                  CachedIntermediate(raw, h1, loadFirst), staged[r * outputSlabPitch_],
                                                  staged[(r + 1) * outputSlabPitch_], W_);
                    }
                    if (r < bulkEnd) {
                        const int32_t h = startRow + r;
                        FinishSeparableX(CachedIntermediate(raw, h, loadFirst), staged[r * outputSlabPitch_],
                                         staged[r * outputSlabPitch_], W_);
                    }
                }
            } else {
                // Y's vertical stage is only bottom - top, so all regular rows
                // can be completed directly in repeat mode.
                ComputeVerticalRows<true>(bulkOutput, bulkBottom, bulkTop, bulkRows);
            }
        }

        // 优化：内部行已经批量完成时，只处理最多两条图像边界行。
        // 不再逐行循环并 continue；核间切分处仍是普通内部行，不重复当作图像边界。
        if (useBulkRows) {
            if (bulkFirst > 0) {
                ComputeStripBoundaryRow(raw, staged, startRow, 0, loadFirst);
            }
            if (bulkEnd < rows) {
                ComputeStripBoundaryRow(raw, staged, startRow, rows - 1, loadFirst);
            }
        } else {
            for (int32_t r = 0; r < rows; ++r) {
                ComputeStripBoundaryRow(raw, staged, startRow, r, loadFirst);
            }
        }
        outputQueue_.EnQue(staged);
        staged = outputQueue_.DeQue<float>();
        // MTE3 advances the UB source by AlignUp(blockLen, 32) bytes. The
        // staged rows may use a wider physical pitch (for example 64 floats
        // for the compensated X path), so explicitly skip that row gap.
        const uint32_t outputRowBytes = static_cast<uint32_t>(W_ * sizeof(float));
        const uint32_t alignedOutputRowBytes = (outputRowBytes + 31U) & ~31U;
        const uint32_t outputPitchBytes = static_cast<uint32_t>(outputSlabPitch_ * sizeof(float));
        const uint32_t outputSrcStride = (outputPitchBytes - alignedOutputRowBytes) / 32U;
        DataCopyExtParams outParams{static_cast<uint16_t>(rows), outputRowBytes, outputSrcStride, 0, 0};
        DataCopyPad(yGm_[base + static_cast<uint64_t>(startRow) * W_], staged, outParams);
        outputQueue_.FreeTensor(staged);
        inputQueue_.FreeTensor(raw);
    }

    // ================= 列分块 slab（banded） =================
    // 几何：首带 shift=0（行首=全局列0）；内部/末带左 halo 取 8 列（shift=8），
    // 保证所有 Vector 操作数起点满足 32B 对齐；末带无右 halo，靠查表折叠/置零。

    __aicore__ inline void ComputeHorizontalXBand(LocalTensor<TYPE_X> raw, int32_t localRow,
                                                  LocalTensor<uint32_t> table, int32_t shift, int32_t bw, bool lastBand)
    {
        LocalTensor<float> row = CachedRow(raw, localRow);
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        float tailCenter = 0.0f;
        if constexpr (NEED_SOFTWARE_FMA) {
            if (lastBand && (W_ & 1) != 0) {
                // V->S: the scalar read consumes a Vector-produced lane.
                const auto vToS = pipe->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(vToS);
                WaitFlag<HardEvent::V_S>(vToS);
                tailCenter = row.GetValue(shift + bw - 1);
            }
        }
        Gather(slots[bandPitch_], row, table, 0U, static_cast<uint32_t>(2 * bandPitch_));
        // left/right 块按 lane 索引，输出位于 lane [shift, shift+bw)，因此读偏移同步 +shift。
        const int32_t s = shift;
        Sub(row[s], slots[2 * bandPitch_ + s], slots[bandPitch_ + s], bw);
        if constexpr (NEED_SOFTWARE_FMA) {
            if (lastBand && (W_ & 1) != 0 && IsNonFinite(tailCenter)) {
                // V->S then S->V around the scalar write of the band tail.
                const auto vToS2 = pipe->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(vToS2);
                WaitFlag<HardEvent::V_S>(vToS2);
                row.SetValue(shift + bw - 1, std::numeric_limits<float>::quiet_NaN());
                const auto sToV = pipe->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(sToV);
                WaitFlag<HardEvent::S_V>(sToV);
            }
        }
    }

    // 带内 K 行扁平水平平滑（Y）：输出位于各行的 [shift, shift+bw)，其余 lane 丢弃。
    __aicore__ inline void SmoothHorizontalYRowsBand(LocalTensor<TYPE_X> raw, int32_t firstRow, int32_t rows,
                                                     LocalTensor<uint32_t> leftTable, LocalTensor<uint32_t> rightTable)
    {
        LocalTensor<float> slots = ScratchFloat(slotsOffset_);
        const int32_t pitch = CachedRowPitch();
        const int32_t blockStride = pitch > bandPitch_ ? pitch : bandPitch_;
        LocalTensor<float> row0 = CachedRow(raw, firstRow);
        LocalTensor<float> left = slots;
        LocalTensor<float> right = slots[rows * blockStride];
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        for (int32_t r = 0; r < rows; ++r) {
            LocalTensor<float> row = row0[r * pitch];
            Gather(left[r * pitch], row, leftTable, 0U, static_cast<uint32_t>(bandPitch_));
            Gather(right[r * pitch], row, rightTable, 0U, static_cast<uint32_t>(bandPitch_));
        }
        const int32_t count = rows * pitch;
        if constexpr (NEED_SOFTWARE_FMA) {
            Adds(row0, row0, 0.0f, count);
            Add(side, left, right, count);
            Muls(side, side, 3.0f, count);
            CompensatedMulAdd(row0, row0, side, 8.0f, 2.0f, count);
        } else {
            Adds(row0, row0, 0.0f, count);
            Add(side, left, right, count);
            Muls(side, side, 3.0f, count);
            Muls(row0, row0, 10.0f, count);
            Add(row0, row0, side, count);
        }
    }

    // 末带奇数末列（全局 W-1，局部 lane=shift+bw-1）：跨行快照 + 专用顺序补算 + 标量写回。
    __aicore__ inline void PrecomputeHorizontalYTailBatchBand(LocalTensor<TYPE_X> raw, int32_t physicalRows, int32_t bw,
                                                              int32_t shift, LocalTensor<uint32_t> leftTable,
                                                              LocalTensor<uint32_t> rightTable)
    {
        LocalTensor<float> tailLeft = ScratchFloat(tailBatchOffset_);
        LocalTensor<float> tailCenter = ScratchFloat(tailBatchOffset_ + TAIL_BATCH_LANES * sizeof(float));
        LocalTensor<float> tailRight = ScratchFloat(tailBatchOffset_ + 2U * TAIL_BATCH_LANES * sizeof(float));
        LocalTensor<uint32_t> rowOffsets = ScratchUint(tailBatchOffset_ + 3U * TAIL_BATCH_LANES * sizeof(float));
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        const int32_t lane = shift + bw - 1;
        const int32_t f4 = static_cast<int32_t>(sizeof(float));
        const int32_t chunkRows = TAIL_BATCH_LANES;
        const int32_t smoothRows = compRows_ > 1 ? compRows_ : chunkRows;
        for (int32_t first = 0; first < physicalRows; first += chunkRows) {
            const int32_t count = physicalRows - first < chunkRows ? physicalRows - first : chunkRows;
            LocalTensor<float> firstRow = CachedRow(raw, first);
            Gather(tailLeft, firstRow, rowOffsets, static_cast<uint32_t>((lane - 1) * f4),
                   static_cast<uint32_t>(count));
            Gather(tailCenter, firstRow, rowOffsets, static_cast<uint32_t>(lane * f4), static_cast<uint32_t>(count));
            if constexpr (BORDER == 2) {
                Duplicate(tailRight, 0.0f, count);
            } else {
                const int32_t rightLane = (BORDER == 0) ? (lane - 1) : lane;
                Gather(tailRight, firstRow, rowOffsets, static_cast<uint32_t>(rightLane * f4),
                       static_cast<uint32_t>(count));
            }
            PipeBarrier<PIPE_V>();
            for (int32_t local = 0; local < count; local += smoothRows) {
                const int32_t rows = count - local < smoothRows ? count - local : smoothRows;
                SmoothHorizontalYRowsBand(raw, first + local, rows, leftTable, rightTable);
            }
            Adds(tailCenter, tailCenter, 0.0f, count);
            Add(side, tailLeft, tailRight, count);
            Muls(tailCenter, tailCenter, 10.0f, count);
            CompensatedMulAdd(tailRight, side, tailCenter, 2.0f, 1.0f, count);
            const auto vToS = pipe->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(vToS);
            WaitFlag<HardEvent::V_S>(vToS);
            for (int32_t r = 0; r < count; ++r) {
                CachedRow(raw, first + r).SetValue(lane, tailRight.GetValue(r));
            }
            const auto sToV = pipe->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(sToV);
            WaitFlag<HardEvent::S_V>(sToV);
        }
    }

    __aicore__ inline void PrecomputeHorizontalYBand(LocalTensor<TYPE_X> raw, int32_t physicalRows, bool lastBand,
                                                     int32_t bw, int32_t shift, LocalTensor<uint32_t> leftTable,
                                                     LocalTensor<uint32_t> rightTable)
    {
        if (lastBand && (W_ & 1) != 0 && tailBatchOffset_ != 0U) {
            PrecomputeHorizontalYTailBatchBand(raw, physicalRows, bw, shift, leftTable, rightTable);
            return;
        }
        const int32_t step = compRows_ > 1 ? compRows_ : 1;
        for (int32_t first = 0; first < physicalRows; first += step) {
            const int32_t count = physicalRows - first < step ? physicalRows - first : step;
            SmoothHorizontalYRowsBand(raw, first, count, leftTable, rightTable);
        }
    }

    template <bool DO_SUB>
    __aicore__ inline void ComputeVerticalRowsBand(LocalTensor<float> dst, LocalTensor<float> src0,
                                                   LocalTensor<float> src1, int32_t rows, int32_t bw, int32_t shift)
    {
        constexpr int32_t FP32_REPEAT_ELEMENTS = 64;
        const int32_t cachedPitch = CachedRowPitch();
        int32_t completedRows = 0;
        while (completedRows < rows) {
            const int32_t remainingRows = rows - completedRows;
            const uint8_t repeatTimes = static_cast<uint8_t>(remainingRows < 255 ? remainingRows : 255);
            const uint8_t dstRepStride = repeatTimes == 1 ? 0 : static_cast<uint8_t>(outputSlabPitch_ / 8);
            const uint8_t srcRepStride = repeatTimes == 1 ? 0 : static_cast<uint8_t>(cachedPitch / 8);
            const BinaryRepeatParams params{1, 1, 1, dstRepStride, srcRepStride, srcRepStride};
            const int32_t dstRowOffset = completedRows * outputSlabPitch_;
            const int32_t srcRowOffset = completedRows * cachedPitch;
            for (int32_t col = 0; col < bw; col += FP32_REPEAT_ELEMENTS) {
                const uint64_t mask = static_cast<uint64_t>(bw - col < FP32_REPEAT_ELEMENTS ? bw - col :
                                                                                              FP32_REPEAT_ELEMENTS);
                if constexpr (DO_SUB) {
                    Sub(dst[dstRowOffset + shift + col], src0[srcRowOffset + shift + col],
                        src1[srcRowOffset + shift + col], mask, repeatTimes, params);
                } else {
                    Add(dst[dstRowOffset + shift + col], src0[srcRowOffset + shift + col],
                        src1[srcRowOffset + shift + col], mask, repeatTimes, params);
                }
            }
            completedRows += static_cast<int32_t>(repeatTimes);
        }
    }

    __aicore__ inline bool CanBatchVerticalRowsBand() const
    {
        const int32_t cachedPitch = CachedRowPitch();
        return (cachedPitch & 7) == 0 && (bandOutPitch_ & 7) == 0 && cachedPitch / 8 <= 255 && bandOutPitch_ / 8 <= 255;
    }

    // 带内边界行（图像上下边界）：Fold 后按带宽计算，输出写入 staged 的 [shift, shift+bw)。
    __aicore__ inline void ComputeStripBoundaryRowBand(LocalTensor<TYPE_X> raw, LocalTensor<float> staged,
                                                       int32_t startRow, int32_t localRow, int32_t loadFirst,
                                                       int32_t bw, int32_t shift)
    {
        const int32_t h = startRow + localRow;
        const int32_t top = Fold(h - 1, H_);
        const int32_t bottom = Fold(h + 1, H_);
        LocalTensor<float> outRow = staged[localRow * outputSlabPitch_];
        LocalTensor<float> side = ScratchFloat(work0Offset_);
        const bool topValid = top >= 0;
        const bool bottomValid = bottom >= 0;
        if constexpr (IS_X) {
            if (topValid && bottomValid) {
                Add(side, CachedIntermediate(raw, top, loadFirst)[shift],
                    CachedIntermediate(raw, bottom, loadFirst)[shift], bw);
            } else if (topValid) {
                Adds(side, CachedIntermediate(raw, top, loadFirst)[shift], 0.0f, bw);
            } else if (bottomValid) {
                Adds(side, CachedIntermediate(raw, bottom, loadFirst)[shift], 0.0f, bw);
            } else {
                Duplicate(side, 0.0f, bw);
            }
            FinishSeparableX(CachedIntermediate(raw, h, loadFirst)[shift], outRow[shift], side, bw);
        } else {
            if (topValid && bottomValid) {
                Sub(outRow[shift], CachedIntermediate(raw, bottom, loadFirst)[shift],
                    CachedIntermediate(raw, top, loadFirst)[shift], bw);
            } else if (bottomValid) {
                Adds(outRow[shift], CachedIntermediate(raw, bottom, loadFirst)[shift], 0.0f, bw);
            } else if (topValid) {
                Muls(outRow[shift], CachedIntermediate(raw, top, loadFirst)[shift], -1.0f, bw);
            } else {
                Duplicate(outRow[shift], 0.0f, bw);
            }
        }
    }

    // 列分块 slab 主流程：逐带 搬入(±halo) → 转换 → 水平 → 垂直 → 写回。
    __aicore__ inline void RunRowStripBanded(uint64_t base, int32_t startRow, int32_t rows)
    {
        const int32_t loadFirst = startRow > 0 ? startRow - 1 : 0;
        const int32_t loadLast = startRow + rows < H_ ? startRow + rows : H_ - 1;
        const int32_t physicalRows = loadLast - loadFirst + 1;
        const int32_t bandCount = (W_ + bandW_ - 1) / bandW_;
        for (int32_t band = 0; band < bandCount; ++band) {
            const int32_t c0 = band * bandW_;
            const int32_t bw = W_ - c0 < bandW_ ? W_ - c0 : bandW_;
            const bool firstBand = (band == 0);
            const bool lastBand = (band == bandCount - 1);
            const int32_t shift = firstBand ? 0 : BAND_SHIFT;
            const int32_t startCol = firstBand ? 0 : c0 - BAND_SHIFT;
            const int32_t endCol = (c0 + bw < W_) ? (c0 + bw) : (W_ - 1);
            const int32_t inCount = endCol - startCol + 1;
            LocalTensor<uint32_t> leftTable;
            LocalTensor<uint32_t> rightTable;
            if (firstBand) {
                leftTable = ScratchUint(offset0Offset_);
                rightTable = ScratchUint(offset1Offset_);
            } else if (lastBand) {
                leftTable = ScratchUint(bandTableOffset_ + 2U * bandPitch_ * sizeof(uint32_t));
                rightTable = ScratchUint(bandTableOffset_ + 3U * bandPitch_ * sizeof(uint32_t));
            } else {
                leftTable = ScratchUint(bandTableOffset_);
                rightTable = ScratchUint(bandTableOffset_ + bandPitch_ * sizeof(uint32_t));
            }
            LocalTensor<TYPE_X> raw = inputQueue_.AllocTensor<TYPE_X>();
            // 首带无左 halo：拷到 bw+1 时单侧 padding = bRaw-(bw+1) 达 60B（fp32）会触发
            // MTE 异常（DataCopyPad 单侧 padding ≤32B）。向右多拷 8 列（仍在图像内，
            // 合法读取）把 padding 压回 ≤32B，且保证副本行 pitch 与其它带一致。
            const int32_t copyCount = firstBand ? (inCount + BAND_SHIFT) : inCount;
            const uint32_t pad = static_cast<uint32_t>(bandRawPitch_ - copyCount);
            DataCopyExtParams inParams{static_cast<uint16_t>(physicalRows),
                                       static_cast<uint32_t>(copyCount * sizeof(TYPE_X)),
                                       static_cast<uint32_t>((W_ - copyCount) * sizeof(TYPE_X)), 0, 0};
            DataCopyPadExtParams<TYPE_X> padParams{true, 0, static_cast<uint8_t>(pad), static_cast<TYPE_X>(0)};
            DataCopyPad(raw, xGm_[base + static_cast<uint64_t>(loadFirst) * static_cast<uint64_t>(W_) + startCol],
                        inParams, padParams);
            inputQueue_.EnQue(raw);
            raw = inputQueue_.DeQue<TYPE_X>();
            if constexpr (!std::is_same_v<TYPE_X, float>) {
                LocalTensor<float> converted = floatSlabBuf_.Get<float>();
                constexpr int32_t SENTINEL_ELEMENTS = BORDER == 2 ? 1 : 0;
                // CONSTANT 首带：左边界零槽位于扩展拷贝后的 DMA 补零位（index=copyCount），
                // 因此多转一个元素，使转换后 slab 的对应位置为 0。
                const int32_t convertCount = inCount + SENTINEL_ELEMENTS +
                                             ((firstBand && BORDER == 2) ? BAND_SHIFT : 0);
                for (int32_t r = 0; r < physicalRows; ++r) {
                    ConvertOneSlot(converted[r * bandFloatPitch_], raw[r * bandRawPitch_], true, convertCount);
                }
            }
            if constexpr (IS_X) {
                for (int32_t r = 0; r < physicalRows; ++r) {
                    ComputeHorizontalXBand(raw, r, leftTable, shift, bw, lastBand);
                }
            } else {
                PrecomputeHorizontalYBand(raw, physicalRows, lastBand, bw, shift, leftTable, rightTable);
            }
            LocalTensor<float> staged = outputQueue_.AllocTensor<float>();
            int32_t bulkFirst = startRow == 0 ? 1 : 0;
            int32_t bulkEnd = startRow + rows == H_ ? rows - 1 : rows;
            if (bulkEnd < bulkFirst) {
                bulkEnd = bulkFirst;
            }
            const int32_t bulkRows = bulkEnd - bulkFirst;
            const bool useBulkRows = bulkRows > 0 && CanBatchVerticalRowsBand();
            if (useBulkRows) {
                const int32_t firstGlobalRow = startRow + bulkFirst;
                const int32_t topLocalRow = firstGlobalRow - 1 - loadFirst;
                LocalTensor<float> bulkOutput = staged[bulkFirst * outputSlabPitch_];
                LocalTensor<float> bulkTop = CachedRow(raw, topLocalRow);
                LocalTensor<float> bulkBottom = CachedRow(raw, topLocalRow + 2);
                if constexpr (IS_X) {
                    ComputeVerticalRowsBand<false>(bulkOutput, bulkTop, bulkBottom, bulkRows, bw, shift);
                    for (int32_t r = bulkFirst; r < bulkEnd; ++r) {
                        const int32_t h = startRow + r;
                        FinishSeparableX(CachedIntermediate(raw, h, loadFirst)[shift],
                                         staged[r * outputSlabPitch_][shift], staged[r * outputSlabPitch_][shift], bw);
                    }
                } else {
                    ComputeVerticalRowsBand<true>(bulkOutput, bulkBottom, bulkTop, bulkRows, bw, shift);
                }
            }
            if (useBulkRows) {
                if (bulkFirst > 0) {
                    ComputeStripBoundaryRowBand(raw, staged, startRow, 0, loadFirst, bw, shift);
                }
                if (bulkEnd < rows) {
                    ComputeStripBoundaryRowBand(raw, staged, startRow, rows - 1, loadFirst, bw, shift);
                }
            } else {
                for (int32_t r = 0; r < rows; ++r) {
                    ComputeStripBoundaryRowBand(raw, staged, startRow, r, loadFirst, bw, shift);
                }
            }
            outputQueue_.EnQue(staged);
            staged = outputQueue_.DeQue<float>();
            const uint32_t outputRowBytes = static_cast<uint32_t>(bw * sizeof(float));
            const uint32_t alignedOutputRowBytes = (outputRowBytes + 31U) & ~31U;
            const uint32_t outputPitchBytes = static_cast<uint32_t>(bandOutPitch_ * sizeof(float));
            const uint32_t outputSrcStride = (outputPitchBytes - alignedOutputRowBytes) / 32U;
            DataCopyExtParams outParams{static_cast<uint16_t>(rows), outputRowBytes, outputSrcStride,
                                        static_cast<uint32_t>((W_ - bw) * sizeof(float)), 0};
            DataCopyPad(yGm_[base + static_cast<uint64_t>(startRow) * static_cast<uint64_t>(W_) + c0], staged[shift],
                        outParams);
            outputQueue_.FreeTensor(staged);
            inputQueue_.FreeTensor(raw);
        }
    }

    // 按有效元素字节数写回 GM，UB 对齐填充不写入输出。

    __aicore__ inline void CopyOut(LocalTensor<float> output, uint64_t outputOffset, int32_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm_[outputOffset], output, copyParams);
    }

    __aicore__ inline void RunTile(uint64_t base, int32_t h, int32_t outputCol, int32_t count)
    {
        int32_t rows[SLOT_COUNT];
        int32_t cols[SLOT_COUNT];
        MakeSlotCoordinates(h, outputCol, rows, cols);
        LocalTensor<TYPE_X> raw = CopyIn(base, rows, cols, count);
        PrepareFloatSlots(raw, rows, cols, count);

        LocalTensor<float> output = outputQueue_.AllocTensor<float>();
        const bool useCompensation = NEED_SOFTWARE_FMA && outputCol < openCvCenterFmaEnd_;
        ComputeVector(output, count, useCompensation);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        const uint64_t outputOffset = base + static_cast<uint64_t>(h) * static_cast<uint64_t>(W_) +
                                      static_cast<uint64_t>(outputCol);
        CopyOut(output, outputOffset, count);
        outputQueue_.FreeTensor(output);
        inputQueue_.FreeTensor(raw);
    }

    // 宽图内部块的既有 Gather 路径，继续沿用补偿和特殊值逻辑。

    __aicore__ inline void RunGatherTile(uint64_t base, int32_t h, int32_t outputCol, int32_t count)
    {
        int32_t rows[3];
        LocalTensor<TYPE_X> raw = CopyInteriorRows(base, h, outputCol, count, rows);
        LocalTensor<float> output = outputQueue_.AllocTensor<float>();
        const bool useCompensation = NEED_SOFTWARE_FMA && outputCol < openCvCenterFmaEnd_;
        if constexpr (IS_X) {
            ComputeGatherX(raw, rows, output, count);
        } else {
            ComputeGatherY(raw, rows, output, count, useCompensation);
        }
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        const uint64_t outputOffset = base + static_cast<uint64_t>(h) * static_cast<uint64_t>(W_) +
                                      static_cast<uint64_t>(outputCol);
        CopyOut(output, outputOffset, count);
        outputQueue_.FreeTensor(output);
        inputQueue_.FreeTensor(raw);
    }

private:
    static constexpr int32_t TAIL_BATCH_LANES = 64; // 末列快照通道数上限
    // 列分块左 halo 列数固定 8：Vector 操作数偏移需 32B 对齐（8×4B），
    // 同时首带通过多拷贝 8 列把 DataCopyPad 单侧 padding 控制在 ≤32B。
    static constexpr int32_t BAND_SHIFT = 8;
    // TPipe 由核函数入口创建、此处仅保存指针：避免 TPipe 对象与算子成员同址导致
    // 编译器对 scalar 成员放弃常量折叠/传播（官方头尾开销优化建议）。
    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> scratchArenaBuf_;
    TBuf<QuePosition::VECCALC> floatSlabBuf_;
    GlobalTensor<TYPE_X> xGm_;
    GlobalTensor<float> yGm_;
    uint64_t totalLength_ = 0;
    int32_t batchCount_ = 0;
    int32_t H_ = 0;
    int32_t W_ = 0;
    int32_t usedCoreNum_ = 0;
    int32_t splitMode_ = 0;
    int32_t baseRowsPerCore_ = 0;
    int32_t extraRowCoreCount_ = 0;
    int32_t basePixelsPerCore_ = 0;
    int32_t extraPixelCoreCount_ = 0;
    int32_t basePlanesPerCore_ = 0;
    int32_t extraPlaneCoreCount_ = 0;
    int32_t blockW_ = 0;
    int32_t tileH_ = 0;
    int32_t compRows_ = 0;
    int32_t compPitch_ = 0;
    int32_t bandW_ = 0;
    int32_t bandPitch_ = 0;
    int32_t bandRawPitch_ = 0;
    int32_t bandFloatPitch_ = 0;
    int32_t bandOutPitch_ = 0;
    uint32_t bandTableOffset_ = 0;
    int32_t inputSlabPitch_ = 0;
    int32_t floatSlabPitch_ = 0;
    int32_t outputSlabPitch_ = 0;
    int32_t rawPitch_ = 0;
    int32_t vectorPitch_ = 0;
    int32_t boundaryRawPitch_ = 0;
    int32_t boundaryVectorPitch_ = 0;
    int32_t openCvCenterFmaEnd_ = 0;
    uint32_t slotsOffset_ = 0;
    uint32_t work0Offset_ = 0;
    uint32_t work1Offset_ = 0;
    uint32_t comp0Offset_ = 0;
    uint32_t comp1Offset_ = 0;
    uint32_t comp2Offset_ = 0;
    uint32_t fallbackOffset_ = 0;
    uint32_t maskOffset_ = 0;
    uint32_t castHalfOffset_ = 0;
    uint32_t offset0Offset_ = 0;
    uint32_t offset1Offset_ = 0;
    uint32_t tailBatchOffset_ = 0;   // 只用于Y奇数末列，独立于普通行计算scratch
    uint32_t coefficientOffset_ = 0; // 只读常驻系数向量的 UB 字节偏移
};

#endif // SCHARR_KERNEL_H
