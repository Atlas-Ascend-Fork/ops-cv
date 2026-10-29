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
 * \file ssim.h
 * \brief fused SSIM kernel
 */
#ifndef SSIM_H_
#define SSIM_H_
#include "kernel_operator.h"

using namespace AscendC;

namespace SsimKernel {
constexpr uint32_t SSIM_WIN = 11;
constexpr uint32_t SSIM_RING_SLOTS = 22;
constexpr uint32_t SSIM_QUANTITY_NUM = 5;
constexpr uint32_t SSIM_MAX_CORES = 128;
constexpr uint32_t SSIM_SYNC_ID = 0;
constexpr float SSIM_TWO = 2.0f;

// 11-tap 1D gaussian kernel, sigma = 1.5, normalized (float32)
constexpr float SSIM_GAUSS[SSIM_WIN] = {0.001028380f, 0.007598758f, 0.036000773f, 0.109360687f,
                                        0.213005543f, 0.266011715f, 0.213005543f, 0.109360687f,
                                        0.036000773f, 0.007598758f, 0.001028380f};

template <typename T>
class KernelSsim {
public:
    __aicore__ inline KernelSsim(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR yOut, GM_ADDR workspace, const SsimTilingData* tiling)
    {
        totalOutRows_ = tiling->totalOutRows;
        perImageOutRows_ = tiling->perImageOutRows;
        perImageInRows_ = tiling->perImageInRows;
        outW_ = tiling->outW;
        inW_ = tiling->inW;
        inH_ = tiling->inH;
        channels_ = tiling->channels;
        batch_ = tiling->batch;
        blockRows_ = tiling->blockRows;
        lastBlockRows_ = tiling->lastBlockRows;
        colTile_ = tiling->colTile;
        totalPixels_ = tiling->totalPixels;
        blockDim_ = tiling->blockDim;
        invTotalPixels_ = tiling->invTotalPixels;
        c1_ = tiling->c1;
        c2_ = tiling->c2;

        blockIdx_ = GetBlockIdx();
        rowStart_ = blockIdx_ * blockRows_;
        rowEnd_ = rowStart_ + blockRows_;
        if (blockIdx_ == blockDim_ - 1) {
            rowStart_ = totalOutRows_ - lastBlockRows_;
            rowEnd_ = totalOutRows_;
        }

        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ float*)yOut);
        wsGm_.SetGlobalBuffer((__gm__ float*)workspace);

        pipe_->InitBuffer(ringBuf_, SSIM_QUANTITY_NUM * SSIM_RING_SLOTS * colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(colAccBuf_, SSIM_QUANTITY_NUM * colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(inBuf_, colTile_ * channels_ * sizeof(T));
        pipe_->InitBuffer(halfTmpBuf_, colTile_ * channels_ * sizeof(half));
        pipe_->InitBuffer(xfBuf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(yfBuf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(qBuf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(t2Buf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(t3Buf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(ssimBuf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(totalBuf_, colTile_ * channels_ * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 8 * sizeof(float));
        pipe_->InitBuffer(partialsBuf_, SSIM_MAX_CORES * 8 * sizeof(float));

        ring_ = ringBuf_.Get<float>();
        colAcc_ = colAccBuf_.Get<float>();
        in_ = inBuf_.Get<T>();
        halfTmp_ = halfTmpBuf_.Get<half>();
        xf_ = xfBuf_.Get<float>();
        yf_ = yfBuf_.Get<float>();
        q_ = qBuf_.Get<float>();
        t1_ = tmpBuf_.Get<float>();
        t2_ = t2Buf_.Get<float>();
        t3_ = t3Buf_.Get<float>();
        ssim_ = ssimBuf_.Get<float>();
        total_ = totalBuf_.Get<float>();
        reduce_ = reduceBuf_.Get<float>();
        partials_ = partialsBuf_.Get<float>();

        Duplicate(total_, 0.0f, SlotLen());
        Duplicate(reduce_, 0.0f, 8);
        PipeBarrier<PIPE_V>();
    }

    // wide-layout helpers: with channels-last [..., H, W, C], every vector op
    // operates on outCols*C packed elements (column-major channels).
    __aicore__ inline uint32_t SlotLen() const { return colTile_ * channels_; }
    __aicore__ inline void Process()
    {
        uint32_t colTiles = (outW_ + colTile_ - 1) / colTile_;

        for (uint32_t tile = 0; tile < colTiles; tile++) {
            uint32_t colStart = tile * colTile_;
            uint32_t outCols = outW_ - colStart;
            if (outCols > colTile_) {
                outCols = colTile_;
            }
            uint32_t vLen = outCols * channels_;

            // Output rows assigned to this core are walked image by image: every image
            // owns perImageOutRows_ valid output rows and its own window neighbourhood,
            // so the sliding window restarts at each image boundary instead of spanning
            // the flattened batch.
            for (uint32_t row = rowStart_; row < rowEnd_;) {
                uint32_t imgIdx = row / perImageOutRows_;
                uint32_t outRowInImg = row % perImageOutRows_;
                uint32_t imgOutEnd = (imgIdx + 1) * perImageOutRows_;
                uint32_t rowsLeft = ((rowEnd_ < imgOutEnd) ? rowEnd_ : imgOutEnd) - row;
                for (uint32_t h = outRowInImg; h < outRowInImg + rowsLeft + (SSIM_WIN - 1); h++) {
                    uint32_t slot = h % SSIM_RING_SLOTS;
                    ComputeRowPass(imgIdx, h, colStart, outCols, slot);
                    if (h >= outRowInImg + SSIM_WIN - 1) {
                        EmitOutputRow(h, outCols);
                        Add(total_, total_, ssim_, vLen);
                        PipeBarrier<PIPE_V>();
                    }
                }
                row += rowsLeft;
            }

            // vector bisect reduction of this tile's column sums (mse_loss_v2 pattern)
            uint32_t len = vLen;
            while (len > 8) {
                uint32_t off = ((len + 15) >> 4) << 3;
                Add(total_, total_, total_[off], len - off);
                PipeBarrier<PIPE_V>();
                len = off;
            }
            Add(reduce_, reduce_, total_, len);
            PipeBarrier<PIPE_V>();
            Duplicate(total_, 0.0f, SlotLen());
            PipeBarrier<PIPE_V>();
        }

        // per-core partial: 8 floats at a distinct slot, [(blockDim + blockIdx) * 8]
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(wsGm_[(blockDim_ + blockIdx_) * 8], reduce_, kCopyEightFloats);
        PipeBarrier<PIPE_MTE3>();

        SyncAll<true>();

        if (blockIdx_ == 0) {
            DataCopyExtParams readParams{1, static_cast<uint32_t>(blockDim_ * 32), 0, 0, 0};
            DataCopyPadExtParams<float> readPadParams;
            readPadParams.isPad = false;
            readPadParams.leftPadding = 0;
            readPadParams.rightPadding = 0;
            readPadParams.paddingValue = 0.0f;
            DataCopyPad(partials_, wsGm_[blockDim_ * 8], readParams, readPadParams);
            PipeBarrier<PIPE_ALL>();
            ReduceSum(partials_, partials_, partials_, blockDim_ * 8);
            PipeBarrier<PIPE_V>();
            Muls(partials_, partials_, invTotalPixels_, 1);
            PipeBarrier<PIPE_V>();
            DataCopyPad(yGm_, partials_, kCopyOneFloat);
        }
    }

private:
    // compute row-pass for input row h of image imgIdx, columns [colStart, colStart+outCols),
    // store into ring slot
    __aicore__ inline void ComputeRowPass(uint32_t imgIdx, uint32_t h, uint32_t colStart, uint32_t outCols,
                                          uint32_t slot)
    {
        // WAR guard: before clearing/rewriting a (possibly reused) ring slot, make
        // sure every EmitOutputRow ring read issued in previous iterations has fully
        // drained, otherwise the Duplicate clear below may race with those reads.
        PipeBarrier<PIPE_ALL>();
        uint32_t vLen = outCols * channels_;
        // Wide layout: each row covers ALL channels, so only (image, height) matter.
        uint64_t base = (static_cast<uint64_t>(imgIdx) * inH_ + h) * static_cast<uint64_t>(inW_) * channels_;

        for (uint32_t q = 0; q < SSIM_QUANTITY_NUM; q++) {
            Duplicate(ring_[(q * SSIM_RING_SLOTS + slot) * SlotLen()], 0.0f, SlotLen());
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t tap = 0; tap < SSIM_WIN; tap++) {
            uint64_t srcOffset = base + static_cast<uint64_t>(colStart + tap) * channels_;
            LoadRowWide(xf_, x1Gm_, srcOffset, vLen);
            LoadRowWide(yf_, x2Gm_, srcOffset, vLen);
            PipeBarrier<PIPE_V>();

            Mul(q_, xf_, xf_, vLen);  // x^2
            Mul(t1_, yf_, yf_, vLen); // y^2
            Mul(t2_, xf_, yf_, vLen); // x*y
            PipeBarrier<PIPE_V>();

            Axpy(ring_[slot * SlotLen()], xf_, SSIM_GAUSS[tap], vLen);                         // muX
            Axpy(ring_[(SSIM_RING_SLOTS + slot) * SlotLen()], yf_, SSIM_GAUSS[tap], vLen);     // muY
            Axpy(ring_[(2 * SSIM_RING_SLOTS + slot) * SlotLen()], q_, SSIM_GAUSS[tap], vLen);  // uxx
            Axpy(ring_[(3 * SSIM_RING_SLOTS + slot) * SlotLen()], t1_, SSIM_GAUSS[tap], vLen); // uyy
            Axpy(ring_[(4 * SSIM_RING_SLOTS + slot) * SlotLen()], t2_, SSIM_GAUSS[tap], vLen); // uxy
            PipeBarrier<PIPE_V>();
        }
    }

    // column pass for output row outRow = s - (SSIM_WIN - 1); window rows are s-10..s
    __aicore__ inline void EmitOutputRow(uint32_t s, uint32_t outCols)
    {
        uint32_t vLen = outCols * channels_;
        for (uint32_t q = 0; q < SSIM_QUANTITY_NUM; q++) {
            Duplicate(colAcc_[q * SlotLen()], 0.0f, SlotLen());
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < SSIM_WIN; k++) {
            uint32_t slot = (s + k - (SSIM_WIN - 1)) % SSIM_RING_SLOTS;
            for (uint32_t q = 0; q < SSIM_QUANTITY_NUM; q++) {
                Axpy(colAcc_[q * SlotLen()], ring_[(q * SSIM_RING_SLOTS + slot) * SlotLen()], SSIM_GAUSS[k], vLen);
            }
        }
        PipeBarrier<PIPE_V>();
        // colAcc: 0=muX 1=muY 2=uxx 3=uyy 4=uxy
        Mul(xf_, colAcc_, colAcc_, vLen);                       // muX^2
        Mul(yf_, colAcc_[SlotLen()], colAcc_[SlotLen()], vLen); // muY^2
        Mul(t1_, colAcc_, colAcc_[SlotLen()], vLen);            // muX*muY
        PipeBarrier<PIPE_V>();
        Sub(t2_, colAcc_[2 * SlotLen()], xf_, vLen); // sigX2 = uxx - muX^2
        Sub(t3_, colAcc_[3 * SlotLen()], yf_, vLen); // sigY2 = uyy - muY^2
        Sub(q_, colAcc_[4 * SlotLen()], t1_, vLen);  // sigXY = uxy - muX*muY
        PipeBarrier<PIPE_V>();

        // l = (2*muX*muY + C1) / (muX^2 + muY^2 + C1)
        Muls(ssim_, t1_, SSIM_TWO, vLen);
        Adds(ssim_, ssim_, c1_, vLen);
        Add(xf_, xf_, yf_, vLen);
        Adds(xf_, xf_, c1_, vLen);
        PipeBarrier<PIPE_V>();
        Div(ssim_, ssim_, xf_, vLen);

        // cs = (2*sigXY + C2) / (sigX2 + sigY2 + C2)
        Muls(xf_, q_, SSIM_TWO, vLen);
        Adds(xf_, xf_, c2_, vLen);
        Add(yf_, t2_, t3_, vLen);
        Adds(yf_, yf_, c2_, vLen);
        PipeBarrier<PIPE_V>();
        Div(xf_, xf_, yf_, vLen);

        Mul(ssim_, ssim_, xf_, vLen);
        PipeBarrier<PIPE_V>();
    }

    // load vLen packed elements (outCols*C) contiguously from GM: the caller passes
    // the offset of the first needed (column, channel) element; channels are innermost,
    // so a contiguous span already covers all channels of the requested columns.
    __aicore__ inline void LoadRowWide(LocalTensor<float>& dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t vLen)
    {
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = vLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        if constexpr (sizeof(T) == 1) {
            DataCopyPad(in_, src[srcOffset], extParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(halfTmp_, in_, RoundMode::CAST_NONE, vLen);
            PipeBarrier<PIPE_V>();
            Cast(dst, halfTmp_, RoundMode::CAST_NONE, vLen);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(dst, src[srcOffset], extParams, padParams);
            PipeBarrier<PIPE_ALL>();
        }
    }
    TPipe* pipe_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> wsGm_;

    uint32_t totalOutRows_;
    uint32_t perImageOutRows_;
    uint32_t perImageInRows_;
    uint32_t outW_;
    uint32_t inW_;
    uint32_t inH_;
    uint32_t channels_;
    uint32_t batch_;
    uint32_t blockRows_;
    uint32_t lastBlockRows_;
    uint32_t colTile_;
    uint32_t totalPixels_;
    uint32_t blockDim_;
    float invTotalPixels_;
    float c1_;
    float c2_;
    uint32_t blockIdx_;
    uint32_t rowStart_;
    uint32_t rowEnd_;

    DataCopyExtParams kCopyOneFloat{1, sizeof(float), 0, 0, 0};
    DataCopyExtParams kCopyEightFloats{1, 8 * sizeof(float), 0, 0, 0};

    TBuf<TPosition::VECCALC> ringBuf_;
    TBuf<TPosition::VECCALC> colAccBuf_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> halfTmpBuf_;
    TBuf<TPosition::VECCALC> xfBuf_;
    TBuf<TPosition::VECCALC> yfBuf_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> t3Buf_;
    TBuf<TPosition::VECCALC> ssimBuf_;
    TBuf<TPosition::VECCALC> totalBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> partialsBuf_;

    LocalTensor<float> ring_;
    LocalTensor<float> colAcc_;
    LocalTensor<T> in_;
    LocalTensor<half> halfTmp_;
    LocalTensor<float> xf_;
    LocalTensor<float> yf_;
    LocalTensor<float> q_;
    LocalTensor<float> t1_;
    LocalTensor<float> t2_;
    LocalTensor<float> t3_;
    LocalTensor<float> ssim_;
    LocalTensor<float> total_;
    LocalTensor<float> reduce_;
    LocalTensor<float> partials_;
};
} // namespace SsimKernel

#endif // SSIM_H_
