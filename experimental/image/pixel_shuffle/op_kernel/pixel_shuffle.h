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
 * \file pixel_shuffle.h
 * \brief
 */
#ifndef __PIXEL_SHUFFLE_H__
#define __PIXEL_SHUFFLE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pixel_shuffle_tiling_data.h"
#include "pixel_shuffle_tiling_key.h"

enum class PixelShuffleTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_FLOAT16 = 1,
};

namespace NsPixelShuffle {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr int64_t UB_BUF_BYTE_LIMIT = 32 * 1024;
constexpr int64_t BLOCK_BYTE_TARGET = 2 * 1024;
// 单核 UB 中块缓冲的队列数 × 双缓冲个数（inputQueueX + outputQueueY，各 BUFFER_NUM 份）
constexpr int64_t QUEUE_BUF_COUNT = 2 * BUFFER_NUM;
// 单块缓冲 + 索引表的 UB 总预算（字节）。910B 单核 UB 为 192KB，此处取 150KB 留余量。
// 必须与 host tiling 的 kUbTotalBudget（op_host/pixel_shuffle_tiling.cpp）保持一致，
// 否则两侧的块划分会不一致，导致多核划分错位。
constexpr int64_t UB_TOTAL_BUDGET_BYTES = 150 * 1024;

template <typename T>
class PixelShuffle {
public:
    __aicore__ inline PixelShuffle() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PixelShuffleTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BuildIndexMap();
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> indexBuf; // 组内置换索引表 (uint32)

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    // 形状信息
    int64_t N_;
    int64_t C_in_;
    int64_t H_;
    int64_t W_;
    int64_t r_;
    int64_t C_out_;
    int64_t outH_;
    int64_t outW_;

    int64_t D_;              // 5D 深度（4D 时为 1）
    int64_t rank_;           // 输入维度数（4 或 5）
    int64_t S_;              // 空间维度数（4D=2, 5D=3）
    int64_t chanPerBlock_;   // 每块通道数 = r^2（4D/5D 相同；5D 附加 di 平面偏移）
    int64_t chanStrideIn_;   // 输入通道间隔：H*W(4D) 或 D*H*W(5D)
    int64_t planeInStride_;  // 每个 d 平面元素数 = H*W
    int64_t planesPerGroup_; // 每组平面数：1(4D) 或 D*r(5D)
    int64_t bPerPlane_;      // 每平面块数 = H / blockH_
    int64_t outPlaneStride_; // 输出平面步长 = (H*r)*(W*r)
    int64_t groupElements_;  // 一组（一个 (n,c_out)）的输入/输出元素数（元素守恒）

    int64_t blockH_;     // 每块输入行数（H_ 的约数，保证整块切分无尾巴）
    int64_t wPerBlock_;  // 每块输入列数（W_ 方向分段宽度，≤ W_）
    int64_t bPerRow_;    // 每个平面内 W 方向段数 = ceil(W_ / wPerBlock_)
    int64_t wLast_;      // 末段有效列数 = W_ - (bPerRow_-1)*wPerBlock_
    int64_t blockElems_; // = blockH_ * r^2 * wPerBlock_
    int64_t chanStride_; // 局部布局中通道步长（元素数）：整组=blockH*wPerBlock，组内分块=32B对齐步长
    int64_t alignedBlockElems_; // 向上对齐到 32B 的块步长（元素数）
    int64_t blocksPerGroup_;    // = planesPerGroup_ * bPerPlane_ * bPerRow_
    int64_t totalBlocks_;       // = N * C_out * blocksPerGroup_
    int64_t startBlock_;        // 当前核心起始块
    int64_t coreBlocks_;        // 当前核心块数
    int64_t blockPerTile_;      // 每个 tile 的块数
    int64_t tileNum_;           // 当前核心 tile 数
    int64_t fullTileBlocks_;    // 满 tile 块数
    int64_t tailTileBlocks_;    // 末 tile 块数
};

template <typename T>
__aicore__ inline void PixelShuffle<T>::BuildIndexMap()
{
    LocalTensor<uint32_t> idx = indexBuf.Get<uint32_t>();
    // 块内置换：固定 (n,c,d_out平面) 后，5D 退化为 4D 像素重排（r^2 通道），
    // 块只覆盖 W 方向的一段（wPerBlock_ 列），故行宽用 wPerBlock_ 而非 W_
    const int64_t outWBlock = wPerBlock_ * r_;
    for (int64_t k = 0; k < blockElems_; ++k) {
        int64_t ow = k % outWBlock;
        int64_t rowInBlock = k / outWBlock; // [0, r*blockH_)
        int64_t hRel = rowInBlock / r_;     // 块内输入行 [0, blockH_)
        int64_t hi = rowInBlock % r_;
        int64_t w = ow / r_; // 段内输入列 [0, wPerBlock_)
        int64_t wi = ow % r_;
        int64_t cRel = hi * r_ + wi; // 块内通道 [0, r^2)
        uint32_t inOffset = static_cast<uint32_t>((cRel * chanStride_ + hRel * wPerBlock_ + w) *
                                                  static_cast<int64_t>(sizeof(T)));
        idx.SetValue(k, inOffset);
    }
}

template <typename T>
__aicore__ inline void PixelShuffle<T>::Init(GM_ADDR x, GM_ADDR y, const PixelShuffleTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();

    N_ = tilingData->N;
    C_in_ = tilingData->C_in;
    D_ = tilingData->D;
    H_ = tilingData->H;
    W_ = tilingData->W;
    r_ = tilingData->r;
    rank_ = tilingData->rank;
    S_ = (rank_ >= 5) ? 3 : 2;
    chanPerBlock_ = r_ * r_;
    planeInStride_ = H_ * W_;
    chanStrideIn_ = (S_ == 3) ? D_ * H_ * W_ : H_ * W_;
    planesPerGroup_ = (S_ == 3) ? D_ * r_ : 1;
    C_out_ = C_in_ / (chanPerBlock_ * ((S_ == 3) ? r_ : 1));
    outPlaneStride_ = (H_ * r_) * (W_ * r_);
    outW_ = W_ * r_;
    // 一组 = 一个 (n, c_out)：输入输出元素数守恒，用作组间基址步长
    groupElements_ = chanPerBlock_ * chanStrideIn_ * ((S_ == 3) ? r_ : 1);

    const int64_t unit = 32 / static_cast<int64_t>(sizeof(T));

    // ---- 块尺寸推导（必须与 host tiling 的推导完全一致，否则多核划分会错位）----
    // 单块 UB 占用（blockPerTile_ 取 1 时）：
    //   QUEUE_BUF_COUNT * A * sizeof(T) + A * sizeof(uint32_t)
    // 其中 A = alignedBlockElems_ = chanPerBlock_ * align(blockH_ * wPerBlock_, unit)。
    // 因此每通道可用的对齐列数上限由 UB 预算反推：
    const int64_t ubBudget = UB_TOTAL_BUDGET_BYTES;
    const int64_t perElemCost = QUEUE_BUF_COUNT * static_cast<int64_t>(sizeof(T)) + 4;
    int64_t alignedChanLimit = (ubBudget / perElemCost) / chanPerBlock_;
    alignedChanLimit = (alignedChanLimit / unit) * unit;
    if (alignedChanLimit < unit) {
        alignedChanLimit = unit;
    }

    // (1) 行块高度：既要满足 2KB 目标（块越小 cache 越友好），又受 UB 预算约束
    int64_t blockHCap = BLOCK_BYTE_TARGET / (chanPerBlock_ * W_ * static_cast<int64_t>(sizeof(T)));
    if (blockHCap < 1) {
        blockHCap = 1;
    }
    if (blockHCap > alignedChanLimit) {
        blockHCap = alignedChanLimit;
    }
    // blockH_ 必须是 H_ 的约数，否则按行切分会出现尾巴行
    blockH_ = 1;
    for (int64_t d = blockHCap; d >= 1; --d) {
        if (H_ % d == 0) {
            blockH_ = d;
            break;
        }
    }
    if (blockH_ < 1) {
        blockH_ = 1;
    }

    // (2) W 方向分段：避免单块字节数随 W 线性膨胀导致 InitBuffer 超出 UB。
    //     保证 blockH_ * wPerBlock_ <= alignedChanLimit。
    int64_t wMaxPerBlock = alignedChanLimit / blockH_;
    if (wMaxPerBlock < 1) {
        wMaxPerBlock = 1;
    }
    wPerBlock_ = (W_ < wMaxPerBlock) ? W_ : wMaxPerBlock;
    if (wPerBlock_ < 1) {
        wPerBlock_ = 1;
    }
    bPerRow_ = (W_ + wPerBlock_ - 1) / wPerBlock_;
    wLast_ = W_ - (bPerRow_ - 1) * wPerBlock_;
    if (wLast_ <= 0 || wLast_ > wPerBlock_) {
        wLast_ = wPerBlock_;
    }

    blockElems_ = blockH_ * chanPerBlock_ * wPerBlock_;
    if (blockElems_ <= 0) {
        blockElems_ = 1;
    }
    alignedBlockElems_ = ((blockElems_ + unit - 1) / unit) * unit;
    bPerPlane_ = H_ / blockH_;
    blocksPerGroup_ = planesPerGroup_ * bPerPlane_ * bPerRow_;
    totalBlocks_ = N_ * C_out_ * blocksPerGroup_;

    if (blocksPerGroup_ == 1) {
        // 单块覆盖整组：各通道在输入中连续，可整段搬入
        chanStride_ = blockH_ * wPerBlock_;
    } else {
        chanStride_ = ((blockH_ * wPerBlock_ + unit - 1) / unit) * unit;
        alignedBlockElems_ = chanPerBlock_ * chanStride_;
    }

    startBlock_ = (static_cast<int64_t>(coreIdx) * totalBlocks_) / coreNum;
    int64_t endBlock = (static_cast<int64_t>(coreIdx + 1) * totalBlocks_) / coreNum;
    coreBlocks_ = endBlock - startBlock_;

    int64_t tileDataNum = tilingData->tileDataNum;
    if (tileDataNum <= 0) {
        tileDataNum = 1024;
    }
    blockPerTile_ = (tileDataNum + blockElems_ - 1) / blockElems_;
    if (blockPerTile_ <= 0) {
        blockPerTile_ = 1;
    }

    int64_t maxBlockPerTile = UB_BUF_BYTE_LIMIT / (alignedBlockElems_ * static_cast<int64_t>(sizeof(T)));
    if (maxBlockPerTile < 1) {
        maxBlockPerTile = 1;
    }
    if (blockPerTile_ > maxBlockPerTile) {
        blockPerTile_ = maxBlockPerTile;
    }
    if (blockPerTile_ > coreBlocks_) {
        blockPerTile_ = coreBlocks_;
    }
    if (blockPerTile_ <= 0) {
        blockPerTile_ = 1;
    }

    tileNum_ = (coreBlocks_ + blockPerTile_ - 1) / blockPerTile_;
    fullTileBlocks_ = blockPerTile_;
    tailTileBlocks_ = coreBlocks_ - (tileNum_ - 1) * blockPerTile_;
    if (tailTileBlocks_ <= 0) {
        tailTileBlocks_ = blockPerTile_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    int64_t bufElems = blockPerTile_ * alignedBlockElems_;
    uint32_t bufSize = static_cast<uint32_t>(bufElems * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufSize);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufSize);
    pipe.InitBuffer(indexBuf, static_cast<uint32_t>(alignedBlockElems_ * sizeof(uint32_t)));

    BuildIndexMap();
}

template <typename T>
__aicore__ inline void PixelShuffle<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    int64_t blockBase = startBlock_ + static_cast<int64_t>(progress) * blockPerTile_;
    int64_t blocksNow = (static_cast<int64_t>(progress) == tileNum_ - 1) ? tailTileBlocks_ : fullTileBlocks_;
    int64_t rSq = chanPerBlock_;

    if (blocksPerGroup_ == 1) {
        for (int64_t bb = 0; bb < blocksNow; ++bb) {
            int64_t b = blockBase + bb;
            int64_t src = b * blockElems_;
            int64_t dst = bb * alignedBlockElems_;
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(blockElems_ * static_cast<int64_t>(sizeof(T)));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal[dst], inputGMX[src], copyParams, {false, 0, 0, 0});
        }
    } else {
        const int64_t perPlane = bPerPlane_ * bPerRow_;
        for (int64_t bb = 0; bb < blocksNow; ++bb) {
            int64_t b = blockBase + bb;
            int64_t g = b / blocksPerGroup_;
            int64_t remB = b % blocksPerGroup_;
            int64_t plane = remB / perPlane; // d_out（5D）或 0（4D）
            int64_t rest = remB % perPlane;
            int64_t wg = rest / bPerRow_;                                      // 平面内 h 块序号
            int64_t ws = rest % bPerRow_;                                      // W 方向段序号
            int64_t diBase = (S_ == 3) ? (plane % r_) * chanPerBlock_ : 0;     // 通道内 di 偏移
            int64_t dPlaneOff = (S_ == 3) ? (plane / r_) * planeInStride_ : 0; // 输入 d 平面偏移
            int64_t h0 = wg * blockH_;                                         // 块内起始输入行（平面内）
            int64_t w0 = ws * wPerBlock_;                                      // 段起始输入列
            // 末段只搬有效列；UB 中多出的列留作填充，其对应输出在 CopyOut 时被丢弃
            int64_t wNow = (ws == bPerRow_ - 1) ? wLast_ : wPerBlock_;
            int64_t groupInBase = g * groupElements_; // = 组输入基址
            int64_t dstBase = bb * alignedBlockElems_;
            for (int64_t c = 0; c < chanPerBlock_; ++c) {
                int64_t src = groupInBase + (diBase + c) * chanStrideIn_ + dPlaneOff + h0 * W_ + w0;
                int64_t dst = dstBase + c * chanStride_;
                if (bPerRow_ == 1) {
                    // 无 W 分段：blockH_ 行在输入中连续，整段单块搬入（与原实现一致）
                    DataCopyParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = static_cast<uint32_t>(blockH_ * W_ * static_cast<int64_t>(sizeof(T)));
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    DataCopyPad(xLocal[dst], inputGMX[src], copyParams, {false, 0, 0, 0});
                } else {
                    // W 分段：逐行单块搬入。blockCount 恒为 1，避免多块 DMA 对块长/起始地址的
                    // 对齐约束（wNow 可能只有几字节，多块模式下会读错数据）
                    for (int64_t h = 0; h < blockH_; ++h) {
                        DataCopyParams copyParams;
                        copyParams.blockCount = 1;
                        copyParams.blockLen = static_cast<uint32_t>(wNow * static_cast<int64_t>(sizeof(T)));
                        copyParams.srcStride = 0;
                        copyParams.dstStride = 0;
                        DataCopyPad(xLocal[dst + h * wPerBlock_], inputGMX[src + h * W_], copyParams, {false, 0, 0, 0});
                    }
                }
            }
        }
    }

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PixelShuffle<T>::Compute(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<uint32_t> idx = indexBuf.Get<uint32_t>();

    int64_t blocksNow = (static_cast<int64_t>(progress) == tileNum_ - 1) ? tailTileBlocks_ : fullTileBlocks_;

    // 向量化置换：每个块一次 Gather，把连续输入的块内元素搬到输出顺序
    for (int64_t b = 0; b < blocksNow; ++b) {
        uint32_t srcBase = static_cast<uint32_t>(b * alignedBlockElems_ * static_cast<int64_t>(sizeof(T)));
        AscendC::Gather(yLocal[b * alignedBlockElems_], xLocal, idx, srcBase, static_cast<uint32_t>(blockElems_));
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PixelShuffle<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    int64_t blockBase = startBlock_ + static_cast<int64_t>(progress) * blockPerTile_;
    int64_t blocksNow = (static_cast<int64_t>(progress) == tileNum_ - 1) ? tailTileBlocks_ : fullTileBlocks_;
    const int64_t perPlane = bPerPlane_ * bPerRow_;

    for (int64_t bb = 0; bb < blocksNow; ++bb) {
        int64_t b = blockBase + bb;
        int64_t g = b / blocksPerGroup_;
        int64_t remB = b % blocksPerGroup_;
        int64_t plane = remB / perPlane; // d_out（5D）或 0（4D）
        int64_t rest = remB % perPlane;
        int64_t wg = rest / bPerRow_;
        int64_t ws = rest % bPerRow_;
        int64_t h0 = wg * blockH_;
        int64_t w0 = ws * wPerBlock_;
        int64_t wNow = (ws == bPerRow_ - 1) ? wLast_ : wPerBlock_;
        int64_t outOffset = g * groupElements_ +      // 组输出基址（与输入同大小）
                            plane * outPlaneStride_ + // 输出 d_out 平面偏移
                            (h0 * r_) * outW_ +       // 平面内 h 块起始行偏移
                            w0 * r_;                  // 段起始输出列

        if (bPerRow_ == 1) {
            // 块覆盖整行宽（含 blocksPerGroup_==1 的整组情形）：输出在 GM 中连续，整段 DMA 搬出
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(blockElems_ * static_cast<int64_t>(sizeof(T)));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(outputGMY[outOffset], yLocal[bb * alignedBlockElems_], copyParams);
        } else {
            // W 分段：逐行单块搬出。blockCount 恒为 1，规避多块 DMA 对块长/起始地址的对齐约束
            for (int64_t row = 0; row < r_ * blockH_; ++row) {
                DataCopyParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(r_ * wNow * static_cast<int64_t>(sizeof(T)));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(outputGMY[outOffset + row * outW_], yLocal[bb * alignedBlockElems_ + row * r_ * wPerBlock_],
                            copyParams);
            }
        }
    }

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PixelShuffle<T>::Process()
{
    if (tileNum_ <= 0) {
        return;
    }

    CopyIn(0);
    for (int32_t i = 0; i < static_cast<int32_t>(tileNum_) - 1; ++i) {
        CopyIn(i + 1);
        Compute(i);
        CopyOut(i);
    }
    Compute(static_cast<int32_t>(tileNum_) - 1);
    CopyOut(static_cast<int32_t>(tileNum_) - 1);
}

} // namespace NsPixelShuffle
#endif // __PIXEL_SHUFFLE_H__
