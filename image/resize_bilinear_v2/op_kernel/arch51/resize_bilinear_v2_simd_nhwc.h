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
 * \file resize_bilinear_v2_simd_nhwc.h
 * \brief Ascend516（arch51）kernel：SIMD NHWC 双线性插值（TilingKey 50000）
 */
#ifndef RESIZE_BILINEAR_V2_SIMD_NHWC_H
#define RESIZE_BILINEAR_V2_SIMD_NHWC_H

#include "../arch35/resize_bilinear_v2_base.h"
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2 {
using namespace AscendC;
using AscendC::Reg::RegTensor;

constexpr int32_t POS_NW = 0;
constexpr int32_t POS_NE = 1;
constexpr int32_t POS_SW = 2;
constexpr int32_t POS_SE = 3;
constexpr int32_t POS_NUM = 4;
constexpr int64_t NEIGHBOR_MARGIN = 2; // 段边界两侧各留 1 个邻域列

template <typename T_X, typename T_Y>
class ResizeBilinearV2SimdNhwc : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2SimdNhwc(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe,
                                const ResizeBilinearV2TilingData* data);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyInSinglePoint(LocalTensor<uint8_t> xTensor, int64_t queIdx, int64_t hPos, int64_t wPos);
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute();
    __aicore__ inline float CalcInputPos(int64_t pos, float scale);
    __aicore__ inline void CalcW(int64_t wPos, int64_t& left, int64_t& right, float& wLerp);
    __aicore__ inline void ProcessWindow(int64_t nOffset4Block, int64_t nLength4Block, int64_t hwOffset4Block,
                                         int64_t hwLength4Block);
    __aicore__ inline void ProcessRowWindow(int64_t nOffset, int64_t rowH, int64_t wStart, int64_t wCount);
    __aicore__ inline void ComputeGatherSeg(__ubuf__ T_X* winTopX, __ubuf__ T_X* winBotX, int64_t srcColStart,
                                            int64_t segStart, int64_t segCount, float hLerp, __ubuf__ T_Y* ySegY);

    const ResizeBilinearV2TilingData* tilingData_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TQue<QuePosition::VECIN, 2> winTopQue_;
    TQue<QuePosition::VECIN, 2> winBotQue_;
    TBuf<QuePosition::VECCALC> stgNWBuf_;
    TBuf<QuePosition::VECCALC> stgNEBuf_;
    TBuf<QuePosition::VECCALC> stgSWBuf_;
    TBuf<QuePosition::VECCALC> stgSEBuf_;
    TBuf<QuePosition::VECCALC> idxNWBuf_;
    TBuf<QuePosition::VECCALC> idxNEBuf_;
    TBuf<QuePosition::VECCALC> wNWBuf_;
    TBuf<QuePosition::VECCALC> wNEBuf_;
    TBuf<QuePosition::VECCALC> wSWBuf_;
    TBuf<QuePosition::VECCALC> wSEBuf_;
    TQue<QuePosition::VECOUT, 2> ySegQue_;
    int64_t outSeg_ = 0;

    int64_t bufferLen_;

    DataCopyPadExtParams<uint8_t> padParams_ = {false, 0, 0, 0};

    constexpr static Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN}; // bf16 --float

    constexpr static Reg::CastTrait castTrait1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT}; // float---bf16

    constexpr static Reg::CastTrait castTraitFloor = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
    constexpr static Reg::CastTrait castInt32ToF = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};

    int64_t nStrideX_;
    int64_t hwStrideX_;

    int64_t nStrideY_;
    int64_t hwStrideY_;

    int64_t nOffset_;
    int64_t hwOffset_;
    int64_t cOffset_;

    int64_t nLength_;
    int64_t cLength_;

    float delta_[POS_NUM];
};

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe,
                                                                const ResizeBilinearV2TilingData* data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    // 窗口模式（ubWFactor>0）：整 C 非对齐小 C 时启用
    outSeg_ = tilingData_->ubWFactor;
    if (outSeg_ > 0) {
        const int64_t spanMax = Ops::Base::CeilDiv(outSeg_ * tilingData_->lenSrcW, tilingData_->lenDesW) +
                                NEIGHBOR_MARGIN;
        this->pipe_->InitBuffer(winTopQue_, 2, spanMax * tilingData_->lenC * sizeof(T_X));
        this->pipe_->InitBuffer(winBotQue_, 2, spanMax * tilingData_->lenC * sizeof(T_X));
        // 对齐粒度与 ComputeGatherSeg 运行时 lanePad 一致（均为 oneRepeat），保证分配覆盖补齐 lane
        const int64_t oneRepeat = Ops::Base::GetVRegSize() / sizeof(float);
        const int64_t lanePadMax = Ops::Base::CeilAlign<int64_t>(outSeg_ * tilingData_->lenC, oneRepeat);
        this->pipe_->InitBuffer(stgNWBuf_, lanePadMax * sizeof(T_X));
        this->pipe_->InitBuffer(stgNEBuf_, lanePadMax * sizeof(T_X));
        this->pipe_->InitBuffer(stgSWBuf_, lanePadMax * sizeof(T_X));
        this->pipe_->InitBuffer(stgSEBuf_, lanePadMax * sizeof(T_X));
        this->pipe_->InitBuffer(ySegQue_, 2, lanePadMax * sizeof(T_Y));
        constexpr int64_t idxSize = (sizeof(T_X) == 2) ? 2 : 4;
        this->pipe_->InitBuffer(idxNWBuf_, lanePadMax * idxSize);
        this->pipe_->InitBuffer(idxNEBuf_, lanePadMax * idxSize);
        this->pipe_->InitBuffer(wNWBuf_, lanePadMax * sizeof(float));
        this->pipe_->InitBuffer(wNEBuf_, lanePadMax * sizeof(float));
        this->pipe_->InitBuffer(wSWBuf_, lanePadMax * sizeof(float));
        this->pipe_->InitBuffer(wSEBuf_, lanePadMax * sizeof(float));
        return;
    }

    bufferLen_ = tilingData_->ubNFactor * tilingData_->ubCFactor;
    bufferLen_ = Ops::Base::CeilAlign<int64_t>(bufferLen_, ONE_BLOCK_BYTE / sizeof(T_X));
    this->pipe_->InitBuffer(xQue_, 2, POS_NUM * bufferLen_ * sizeof(T_X));
    this->pipe_->InitBuffer(yQue_, 2, bufferLen_ * sizeof(T_Y));
}

template <typename T_X, typename T_Y>
__aicore__ inline float ResizeBilinearV2SimdNhwc<T_X, T_Y>::CalcInputPos(int64_t pos, float scale)
{
    float posFp;

    if (tilingData_->halfPixelCenters > 0) {
        posFp = ((float)pos + 0.5f) * scale - 0.5f;
    } else {
        posFp = ((float)pos) * scale;
    }

    return posFp;
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::CopyInSinglePoint(LocalTensor<uint8_t> xTensor,
                                                                             int64_t queIdx, int64_t hPos, int64_t wPos)
{
    int64_t xOffsetInGM = nOffset_ * nStrideX_ + (hPos * tilingData_->lenSrcW + wPos) * hwStrideX_ + cOffset_;

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = nLength_;
    gm2ubParams.blockLen = cLength_ * sizeof(T_X);
    gm2ubParams.srcStride = (nStrideX_ - cLength_) * sizeof(T_X);
    gm2ubParams.dstStride = 0;

    DataCopyPad<uint8_t, PaddingMode::Compact>(xTensor[queIdx * bufferLen_ * sizeof(T_X)],
                                               xGM_[xOffsetInGM * sizeof(T_X)], gm2ubParams, padParams_);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::CopyIn()
{
    int64_t hPos = hwOffset_ / tilingData_->lenDesW;
    int64_t wPos = hwOffset_ - tilingData_->lenDesW * hPos;

    float hFp = CalcInputPos(hPos, tilingData_->scaleH);
    float hLerp = hFp - Floor(hFp);
    int64_t top = (hFp > 0.0f) ? Floor(hFp) : 0;
    top = min(tilingData_->lenSrcH - 1, top);
    int64_t bot = (hLerp > 0.0f) ? top + 1 : top;
    bot = min(tilingData_->lenSrcH - 1, bot);

    float wFp = CalcInputPos(wPos, tilingData_->scaleW);
    float wLerp = wFp - Floor(wFp);
    int64_t left = (wFp > 0.0f) ? Floor(wFp) : 0;
    left = min(tilingData_->lenSrcW - 1, left);
    int64_t right = (wLerp > 0.0f) ? left + 1 : left;
    right = min(tilingData_->lenSrcW - 1, right);

    delta_[POS_NW] = (1.0f - hLerp) * (1.0f - wLerp);
    delta_[POS_NE] = (1.0f - hLerp) * wLerp;
    delta_[POS_SW] = hLerp * (1.0f - wLerp);
    delta_[POS_SE] = hLerp * wLerp;

    LocalTensor<uint8_t> xTensor = xQue_.AllocTensor<uint8_t>();
    CopyInSinglePoint(xTensor, POS_NW, top, left);
    CopyInSinglePoint(xTensor, POS_NE, top, right);
    CopyInSinglePoint(xTensor, POS_SW, bot, left);
    CopyInSinglePoint(xTensor, POS_SE, bot, right);
    xQue_.EnQue(xTensor);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::CopyOut()
{
    int64_t yOffsetInGM = nOffset_ * nStrideY_ + hwOffset_ * hwStrideY_ + cOffset_;

    LocalTensor<uint8_t> yTensor = yQue_.DeQue<uint8_t>();

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = nLength_;
    ub2gmParams.blockLen = cLength_ * sizeof(T_Y);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = (nStrideY_ - cLength_) * sizeof(T_Y);
    DataCopyPad<uint8_t, PaddingMode::Compact>(yGM_[yOffsetInGM * sizeof(T_Y)], yTensor, ub2gmParams);

    yQue_.FreeTensor(yTensor);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::Compute()
{
    LocalTensor<T_X> xTensor = xQue_.DeQue<T_X>();
    LocalTensor<T_Y> yTensor = yQue_.AllocTensor<T_Y>();

    __ubuf__ T_X* xAddr = (__ubuf__ T_X*)xTensor.GetPhyAddr();
    __ubuf__ T_X* xAddrNW = xAddr + POS_NW * bufferLen_;
    __ubuf__ T_X* xAddrNE = xAddr + POS_NE * bufferLen_;
    __ubuf__ T_X* xAddrSW = xAddr + POS_SW * bufferLen_;
    __ubuf__ T_X* xAddrSE = xAddr + POS_SE * bufferLen_;
    __ubuf__ T_Y* yAddr = (__ubuf__ T_Y*)yTensor.GetPhyAddr();

    float weightNW = delta_[POS_NW];
    float weightNE = delta_[POS_NE];
    float weightSW = delta_[POS_SW];
    float weightSE = delta_[POS_SE];

    int64_t oneRepeat = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t totalLen = tilingData_->ubNFactor * tilingData_->ubCFactor;
    int64_t repeatTimes = Ops::Base::CeilDiv<int64_t>(totalLen, oneRepeat);

    __VEC_SCOPE__
    {
        Reg::MaskReg pregFp32;
        Reg::RegTensor<T_X> regNW;
        Reg::RegTensor<T_X> regNE;
        Reg::RegTensor<T_X> regSW;
        Reg::RegTensor<T_X> regSE;
        Reg::RegTensor<float> regNWFp32;
        Reg::RegTensor<float> regNEFp32;
        Reg::RegTensor<float> regSWFp32;
        Reg::RegTensor<float> regSEFp32;
        Reg::RegTensor<float> regSumTopFp32;
        Reg::RegTensor<float> regSumBotFp32;
        Reg::RegTensor<float> regSumFp32;
        Reg::RegTensor<T_Y> regRst;

        for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
            pregFp32 = Reg::UpdateMask<float>(totalLen);
            if constexpr (sizeof(T_X) == sizeof(int16_t)) {
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    regNW, xAddrNW, (int32_t)oneRepeat);
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    regNE, xAddrNE, (int32_t)oneRepeat);
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    regSW, xAddrSW, (int32_t)oneRepeat);
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    regSE, xAddrSE, (int32_t)oneRepeat);

                Reg::Cast<float, T_X, castTrait0>(regNWFp32, regNW, pregFp32);
                Reg::Cast<float, T_X, castTrait0>(regNEFp32, regNE, pregFp32);
                Reg::Cast<float, T_X, castTrait0>(regSWFp32, regSW, pregFp32);
                Reg::Cast<float, T_X, castTrait0>(regSEFp32, regSE, pregFp32);

                Reg::Muls(regNWFp32, regNWFp32, weightNW, pregFp32);
                Reg::Muls(regNEFp32, regNEFp32, weightNE, pregFp32);
                Reg::Muls(regSWFp32, regSWFp32, weightSW, pregFp32);
                Reg::Muls(regSEFp32, regSEFp32, weightSE, pregFp32);
            } else {
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE>(regNW, xAddrNW, (int32_t)oneRepeat);
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE>(regNE, xAddrNE, (int32_t)oneRepeat);
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE>(regSW, xAddrSW, (int32_t)oneRepeat);
                Reg::LoadAlign<T_X, Reg::PostLiteral::POST_MODE_UPDATE>(regSE, xAddrSE, (int32_t)oneRepeat);

                Reg::Muls(regNWFp32, regNW, weightNW, pregFp32);
                Reg::Muls(regNEFp32, regNE, weightNE, pregFp32);
                Reg::Muls(regSWFp32, regSW, weightSW, pregFp32);
                Reg::Muls(regSEFp32, regSE, weightSE, pregFp32);
            }

            Reg::Add(regSumTopFp32, regNWFp32, regNEFp32, pregFp32);
            Reg::Add(regSumBotFp32, regSWFp32, regSEFp32, pregFp32);
            Reg::Add(regSumFp32, regSumTopFp32, regSumBotFp32, pregFp32);

            if constexpr (sizeof(T_Y) == sizeof(int16_t)) {
                Reg::Cast<T_Y, float, castTrait1>(regRst, regSumFp32, pregFp32);
                Reg::StoreAlign<T_Y, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B32>(
                    yAddr, regRst, (int32_t)oneRepeat, pregFp32);
            } else {
                Reg::StoreAlign<T_Y, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, regSumFp32, (int32_t)oneRepeat,
                                                                         pregFp32);
            }
        }
    }

    yQue_.EnQue(yTensor);
    xQue_.FreeTensor(xTensor);
}

// W 方向源邻域定位（含钳位），供窗口路径段跨度推导复用；数值与 CopyIn 逐点路径逐位一致
template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::CalcW(int64_t wPos, int64_t& left, int64_t& right,
                                                                 float& wLerp)
{
    float wFp = CalcInputPos(wPos, tilingData_->scaleW);
    wLerp = wFp - Floor(wFp);
    left = (wFp > 0.0f) ? Floor(wFp) : 0;
    left = min(tilingData_->lenSrcW - 1, left);
    right = (wLerp > 0.0f) ? left + 1 : left;
    right = min(tilingData_->lenSrcW - 1, right);
}

// 窗口路径入口：将块内输出点序拆为行序列，逐行转 ProcessRowWindow 处理
template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::ProcessWindow(int64_t nOffset4Block, int64_t nLength4Block,
                                                                         int64_t hwOffset4Block, int64_t hwLength4Block)
{
    const int64_t lenDesW = tilingData_->lenDesW;

    hwStrideX_ = tilingData_->lenC;
    nStrideX_ = hwStrideX_ * tilingData_->lenSrcH * tilingData_->lenSrcW;
    hwStrideY_ = tilingData_->lenC;
    nStrideY_ = hwStrideY_ * tilingData_->lenDesH * tilingData_->lenDesW;

    int64_t hPos = hwOffset4Block / lenDesW;
    int64_t wPos = hwOffset4Block - lenDesW * hPos;

    for (int64_t nOff = 0; nOff < nLength4Block; nOff++) {
        const int64_t nOffset = nOffset4Block + nOff;
        int64_t rowH = hPos;
        int64_t rowW = wPos;
        int64_t remain = hwLength4Block;
        while (remain > 0) {
            const int64_t rowPts = this->Min(lenDesW - rowW, remain);
            ProcessRowWindow(nOffset, rowH, rowW, rowPts);
            remain -= rowPts;
            rowH++;
            rowW = 0;
        }
    }
}

// 行窗处理：行内按 outSeg_ 分段；每段 2-3 次连续 DMA（top/bot 各 1 次 + 回写 1 次），NHWC 行段单块连续拷贝
template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::ProcessRowWindow(int64_t nOffset, int64_t rowH,
                                                                            int64_t wStart, int64_t wCount)
{
    const int64_t lenC = tilingData_->lenC;
    const int64_t colBytes = lenC * sizeof(T_X);
    const int64_t colBytesY = lenC * sizeof(T_Y);

    // 行级共享量：top/bot 源行与 h 向权重
    float hFp = CalcInputPos(rowH, tilingData_->scaleH);
    float hLerp = hFp - Floor(hFp);
    int64_t top = (hFp > 0.0f) ? Floor(hFp) : 0;
    top = min(tilingData_->lenSrcH - 1, top);
    int64_t bot = (hLerp > 0.0f) ? top + 1 : top;
    bot = min(tilingData_->lenSrcH - 1, bot);

    const int64_t topRowBase = nOffset * nStrideX_ + top * tilingData_->lenSrcW * hwStrideX_;
    const int64_t botRowBase = nOffset * nStrideX_ + bot * tilingData_->lenSrcW * hwStrideX_;
    const int64_t yRowBase = nOffset * nStrideY_ + rowH * tilingData_->lenDesW * hwStrideY_;

    DataCopyExtParams segParams;
    segParams.blockCount = 1;
    segParams.srcStride = 0;
    segParams.dstStride = 0;

    for (int64_t segStart = wStart; segStart < wStart + wCount; segStart += outSeg_) {
        const int64_t segCount = this->Min(outSeg_, wStart + wCount - segStart);

        // 段内源列跨度：[left(首点), right(末点)]（钳位后必在窗内）
        int64_t l0 = 0;
        int64_t rTmp = 0;
        int64_t lTmp = 0;
        int64_t rightEnd = 0;
        float lerpTmp = 0.0f;
        CalcW(segStart, l0, rTmp, lerpTmp);
        CalcW(segStart + segCount - 1, lTmp, rightEnd, lerpTmp);
        const int64_t srcColStart = l0;
        const int64_t span = rightEnd - l0 + 1;

        // 窗口装载（MTE2→V 同步）：Alloc→DataCopyPad→EnQue→DeQue
        segParams.blockLen = span * colBytes;
        LocalTensor<uint8_t> winTopU8 = winTopQue_.AllocTensor<uint8_t>();
        DataCopyPad<uint8_t, PaddingMode::Compact>(winTopU8, xGM_[(topRowBase + srcColStart * lenC) * sizeof(T_X)],
                                                   segParams, padParams_);
        winTopQue_.EnQue(winTopU8);
        winTopU8 = winTopQue_.DeQue<uint8_t>();
        __ubuf__ T_X* winTopX = (__ubuf__ T_X*)winTopU8.GetPhyAddr();

        LocalTensor<uint8_t> winBotU8 = winTopU8; // bot==top 时复用 top Tensor
        __ubuf__ T_X* botRowX = winTopX;
        if (bot != top) {
            winBotU8 = winBotQue_.AllocTensor<uint8_t>();
            DataCopyPad<uint8_t, PaddingMode::Compact>(winBotU8, xGM_[(botRowBase + srcColStart * lenC) * sizeof(T_X)],
                                                       segParams, padParams_);
            winBotQue_.EnQue(winBotU8);
            winBotU8 = winBotQue_.DeQue<uint8_t>();
            botRowX = (__ubuf__ T_X*)winBotU8.GetPhyAddr();
        }

        // Gather 批计算（V→MTE3 同步）：Alloc→计算→EnQue→DeQue→Free
        LocalTensor<uint8_t> ySegU8 = ySegQue_.AllocTensor<uint8_t>();
        __ubuf__ T_Y* ySegY = (__ubuf__ T_Y*)ySegU8.GetPhyAddr();
        ComputeGatherSeg(winTopX, botRowX, srcColStart, segStart, segCount, hLerp, ySegY);
        ySegQue_.EnQue(ySegU8);
        ySegU8 = ySegQue_.DeQue<uint8_t>();

        segParams.blockLen = segCount * colBytesY;
        DataCopyPad<uint8_t, PaddingMode::Compact>(yGM_[(yRowBase + segStart * lenC) * sizeof(T_Y)], ySegU8, segParams);
        ySegQue_.FreeTensor(ySegU8);
        winTopQue_.FreeTensor(winTopU8);
        if (bot != top) {
            winBotQue_.FreeTensor(winBotU8);
        }
    }
}

// Gather 批计算（参照 nearest nchw_gather）：三阶段全向量化，消除逐点标量计算
//  A 索引/权重构造——lane 展平 [点×通道]，坐标链与 CopyIn 标量路径逐位一致（含 wFp<0 的截断语义）；
//  B Reg::Gather 按轮取 4 邻域入 stg 中转（16 位数据配 u16 索引，窗口元素数须 <65536）；
//  C 加权计算——fp32 乘加序列与逐点 Compute 逐位一致；补齐 lane 不落 GM。
template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::ComputeGatherSeg(__ubuf__ T_X* winTopX,
                                                                            __ubuf__ T_X* winBotX, int64_t srcColStart,
                                                                            int64_t segStart, int64_t segCount,
                                                                            float hLerp, __ubuf__ T_Y* ySegY)
{
    const int64_t lenC = tilingData_->lenC;
    const int64_t totalLanes = segCount * lenC;
    const int64_t oneRepeat = Ops::Base::GetVRegSize() / sizeof(float);
    const int64_t lanePad = Ops::Base::CeilAlign<int64_t>(totalLanes, oneRepeat);
    const float invC = 1.0f / static_cast<float>(lenC);
    const float oneMinusH = 1.0f - hLerp;

    __ubuf__ uint8_t* stgNW = (__ubuf__ uint8_t*)stgNWBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ uint8_t* stgNE = (__ubuf__ uint8_t*)stgNEBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ uint8_t* stgSW = (__ubuf__ uint8_t*)stgSWBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ uint8_t* stgSE = (__ubuf__ uint8_t*)stgSEBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ uint8_t* idxNWU8 = (__ubuf__ uint8_t*)idxNWBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ uint8_t* idxNEU8 = (__ubuf__ uint8_t*)idxNEBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ float* wNW = (__ubuf__ float*)wNWBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ float* wNE = (__ubuf__ float*)wNEBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ float* wSW = (__ubuf__ float*)wSWBuf_.Get<uint8_t>().GetPhyAddr();
    __ubuf__ float* wSE = (__ubuf__ float*)wSEBuf_.Get<uint8_t>().GetPhyAddr();

    // A：索引与权重构造（每 oneRepeat=8 lane 一轮，尾轮 mask 截断）
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> laneReg;
        Reg::RegTensor<int32_t> laneInt;
        Reg::RegTensor<int32_t> dInt;
        Reg::RegTensor<int32_t> cInt;
        Reg::RegTensor<int32_t> cScaled;
        Reg::RegTensor<int32_t> idxInt;
        Reg::RegTensor<int32_t> floorInt;
        Reg::RegTensor<int32_t> leftInt;
        Reg::RegTensor<int32_t> rightInt;
        Reg::RegTensor<int32_t> stepInt;
        Reg::RegTensor<int32_t> fomInt;
        Reg::RegTensor<int32_t> zeroReg;
        Reg::RegTensor<int32_t> oneIntReg;
        Reg::RegTensor<int32_t> srcWm1Reg;
        Reg::RegTensor<float> laneF;
        Reg::RegTensor<float> dF;
        Reg::RegTensor<float> cF;
        Reg::RegTensor<float> posF;
        Reg::RegTensor<float> wFp;
        Reg::RegTensor<float> leftF;
        Reg::RegTensor<float> wLerpF;
        Reg::RegTensor<float> halfLerp;
        Reg::RegTensor<float> oneMinusHalf;
        Reg::RegTensor<float> lRelF;
        Reg::RegTensor<float> lScaledF;
        Reg::RegTensor<float> idxNWF;
        Reg::RegTensor<float> idxNEF;
        Reg::RegTensor<float> oneRegF;
        Reg::RegTensor<float> oneMinusW;
        Reg::RegTensor<float> wNWv;
        Reg::RegTensor<float> wNEv;
        Reg::RegTensor<float> wSWv;
        Reg::RegTensor<float> wSEv;
        Reg::MaskReg preg;

        Reg::Duplicate<int32_t>(zeroReg, 0);
        Reg::Duplicate<int32_t>(oneIntReg, 1);
        Reg::Duplicate<int32_t>(srcWm1Reg, static_cast<int32_t>(tilingData_->lenSrcW - 1));
        Reg::Duplicate<float>(oneRegF, 1.0f);
        Reg::Arange(laneReg, 0);

        const int64_t roundsA = lanePad / oneRepeat;
        for (uint16_t r = 0; r < (uint16_t)roundsA; r++) {
            uint32_t remainA = static_cast<uint32_t>(lanePad - static_cast<int64_t>(r) * oneRepeat);
            uint32_t maskA = (remainA > static_cast<uint32_t>(oneRepeat)) ? static_cast<uint32_t>(oneRepeat) : remainA;
            preg = Reg::UpdateMask<int32_t>(maskA);

            Reg::Adds<int32_t>(laneInt, laneReg, static_cast<int32_t>(static_cast<int64_t>(r) * oneRepeat), preg);
            Reg::Cast<float, int32_t, castInt32ToF>(laneF, laneInt, preg);
            Reg::Muls<float>(dF, laneF, invC, preg);
            Reg::Cast<int32_t, float, castTraitFloor>(dInt, dF, preg);
            Reg::Muls<int32_t>(cScaled, dInt, static_cast<int32_t>(lenC), preg);
            Reg::Sub<int32_t>(cInt, laneInt, cScaled, preg);

            Reg::Cast<float, int32_t, castInt32ToF>(posF, dInt, preg);
            Reg::Adds<float>(posF, posF, static_cast<float>(segStart), preg);
            if (tilingData_->halfPixelCenters > 0) {
                Reg::Adds<float>(posF, posF, 0.5f, preg);
                Reg::Muls<float>(wFp, posF, tilingData_->scaleW, preg);
                Reg::Adds<float>(wFp, wFp, -0.5f, preg);
            } else {
                Reg::Muls<float>(wFp, posF, tilingData_->scaleW, preg);
            }
            Reg::Cast<int32_t, float, castTraitFloor>(floorInt, wFp, preg);
            Reg::Max<int32_t>(leftInt, floorInt, zeroReg, preg);
            Reg::Min<int32_t>(leftInt, leftInt, srcWm1Reg, preg);
            Reg::Cast<float, int32_t, castInt32ToF>(leftF, leftInt, preg);
            Reg::Sub<float>(wLerpF, wFp, leftF, preg);

            Reg::Muls<float>(halfLerp, wLerpF, 0.5f, preg);
            Reg::Sub<float>(oneMinusHalf, oneRegF, halfLerp, preg);
            Reg::Cast<int32_t, float, castTraitFloor>(fomInt, oneMinusHalf, preg);
            Reg::Sub<int32_t>(stepInt, oneIntReg, fomInt, preg);
            Reg::Add<int32_t>(rightInt, leftInt, stepInt, preg);
            Reg::Min<int32_t>(rightInt, rightInt, srcWm1Reg, preg);

            Reg::Cast<float, int32_t, castInt32ToF>(lRelF, leftInt, preg);
            Reg::Adds<float>(lRelF, lRelF, -static_cast<float>(srcColStart), preg);
            Reg::Muls<float>(lScaledF, lRelF, static_cast<float>(lenC), preg);
            Reg::Cast<float, int32_t, castInt32ToF>(cF, cInt, preg);
            Reg::Add<float>(idxNWF, lScaledF, cF, preg);
            Reg::Cast<int32_t, float, castTraitFloor>(idxInt, idxNWF, preg);
            Reg::Cast<float, int32_t, castInt32ToF>(lRelF, rightInt, preg);
            Reg::Adds<float>(lRelF, lRelF, -static_cast<float>(srcColStart), preg);
            Reg::Muls<float>(lScaledF, lRelF, static_cast<float>(lenC), preg);
            Reg::Add<float>(idxNEF, lScaledF, cF, preg);

            Reg::Sub<float>(oneMinusW, oneRegF, wLerpF, preg);
            Reg::Muls<float>(wNWv, oneMinusW, oneMinusH, preg);
            Reg::Muls<float>(wNEv, wLerpF, oneMinusH, preg);
            Reg::Muls<float>(wSWv, oneMinusW, hLerp, preg);
            Reg::Muls<float>(wSEv, wLerpF, hLerp, preg);

            if constexpr (sizeof(T_X) == sizeof(int16_t)) {
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B32>(
                    (__ubuf__ uint16_t*)idxNWU8 + static_cast<int64_t>(r) * oneRepeat,
                    (Reg::RegTensor<uint16_t>&)idxInt, preg);
                Reg::Cast<int32_t, float, castTraitFloor>(idxInt, idxNEF, preg);
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B32>(
                    (__ubuf__ uint16_t*)idxNEU8 + static_cast<int64_t>(r) * oneRepeat,
                    (Reg::RegTensor<uint16_t>&)idxInt, preg);
            } else {
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(
                    (__ubuf__ uint32_t*)idxNWU8 + static_cast<int64_t>(r) * oneRepeat,
                    (Reg::RegTensor<uint32_t>&)idxInt, preg);
                Reg::Cast<int32_t, float, castTraitFloor>(idxInt, idxNEF, preg);
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(
                    (__ubuf__ uint32_t*)idxNEU8 + static_cast<int64_t>(r) * oneRepeat,
                    (Reg::RegTensor<uint32_t>&)idxInt, preg);
            }
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(wNW + static_cast<int64_t>(r) * oneRepeat, wNWv, preg);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(wNE + static_cast<int64_t>(r) * oneRepeat, wNEv, preg);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(wSW + static_cast<int64_t>(r) * oneRepeat, wSWv, preg);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(wSE + static_cast<int64_t>(r) * oneRepeat, wSEv, preg);
        }
    }

    // B：Reg::Gather 按轮取 4 邻域入 stg 中转（T_X 16 位→u16 索引/16 lane；32 位→u32 索引/8 lane）
    const int64_t lanesPerGather = (sizeof(T_X) == 2) ? (Ops::Base::GetVRegSize() / sizeof(T_X)) : oneRepeat;
    const int64_t roundsB = Ops::Base::CeilDiv(lanePad, lanesPerGather);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T_X> gNW;
        Reg::RegTensor<T_X> gNE;
        Reg::RegTensor<T_X> gSW;
        Reg::RegTensor<T_X> gSE;
        Reg::MaskReg pregB;
        if constexpr (sizeof(T_X) == sizeof(int16_t)) {
            Reg::RegTensor<uint16_t> idxU16;
            for (uint16_t r = 0; r < (uint16_t)roundsB; r++) {
                const int64_t off = static_cast<int64_t>(r) * lanesPerGather;
                uint32_t remainB = static_cast<uint32_t>(lanePad - off);
                uint32_t maskB = (remainB > static_cast<uint32_t>(lanesPerGather)) ?
                                     static_cast<uint32_t>(lanesPerGather) :
                                     remainB;
                pregB = Reg::UpdateMask<T_X>(maskB);
                Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(idxU16, (__ubuf__ uint16_t*)idxNWU8 + off);
                Reg::Gather(gNW, winTopX, idxU16, pregB);
                Reg::Gather(gSW, winBotX, idxU16, pregB);
                Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(idxU16, (__ubuf__ uint16_t*)idxNEU8 + off);
                Reg::Gather(gNE, winTopX, idxU16, pregB);
                Reg::Gather(gSE, winBotX, idxU16, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgNW + off, gNW, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgNE + off, gNE, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgSW + off, gSW, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgSE + off, gSE, pregB);
            }
        } else {
            Reg::RegTensor<uint32_t> idxU32;
            for (uint16_t r = 0; r < (uint16_t)roundsB; r++) {
                const int64_t off = static_cast<int64_t>(r) * lanesPerGather;
                uint32_t remainB = static_cast<uint32_t>(lanePad - off);
                uint32_t maskB = (remainB > static_cast<uint32_t>(lanesPerGather)) ?
                                     static_cast<uint32_t>(lanesPerGather) :
                                     remainB;
                pregB = Reg::UpdateMask<T_X>(maskB);
                Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(idxU32, (__ubuf__ uint32_t*)idxNWU8 + off);
                Reg::Gather(gNW, winTopX, idxU32, pregB);
                Reg::Gather(gSW, winBotX, idxU32, pregB);
                Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(idxU32, (__ubuf__ uint32_t*)idxNEU8 + off);
                Reg::Gather(gNE, winTopX, idxU32, pregB);
                Reg::Gather(gSE, winBotX, idxU32, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgNW + off, gNW, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgNE + off, gNE, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgSW + off, gSW, pregB);
                Reg::StoreAlign<T_X, Reg::StoreDist::DIST_NORM>((__ubuf__ T_X*)stgSE + off, gSE, pregB);
            }
        }
    }

    // C：加权计算（fp32 域，与逐点 Compute 逐位一致）
    const int64_t roundsC = lanePad / oneRepeat;
    __VEC_SCOPE__
    {
        Reg::MaskReg pregFp32;
        Reg::MaskReg pregFp16;
        Reg::RegTensor<T_X> regNW;
        Reg::RegTensor<T_X> regNE;
        Reg::RegTensor<T_X> regSW;
        Reg::RegTensor<T_X> regSE;
        Reg::RegTensor<float> regWNW;
        Reg::RegTensor<float> regWNE;
        Reg::RegTensor<float> regWSW;
        Reg::RegTensor<float> regWSE;
        Reg::RegTensor<float> regNWFp32;
        Reg::RegTensor<float> regNEFp32;
        Reg::RegTensor<float> regSWFp32;
        Reg::RegTensor<float> regSEFp32;
        Reg::RegTensor<float> regNw;
        Reg::RegTensor<float> regEw;
        Reg::RegTensor<float> regSumTopFp32;
        Reg::RegTensor<float> regSumBotFp32;
        Reg::RegTensor<float> regSumFp32;
        Reg::RegTensor<T_Y> regRst;

        for (uint16_t r = 0; r < (uint16_t)roundsC; r++) {
            uint32_t remain = static_cast<uint32_t>(totalLanes - static_cast<int64_t>(r) * oneRepeat);
            uint32_t maskLen = (remain > static_cast<uint32_t>(oneRepeat)) ? static_cast<uint32_t>(oneRepeat) : remain;
            pregFp32 = Reg::UpdateMask<float>(maskLen);

            const int64_t off = static_cast<int64_t>(r) * oneRepeat;
            if constexpr (sizeof(T_X) == sizeof(int16_t)) {
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_UNPACK_B16>(regNW, (__ubuf__ T_X*)stgNW + off);
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_UNPACK_B16>(regNE, (__ubuf__ T_X*)stgNE + off);
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_UNPACK_B16>(regSW, (__ubuf__ T_X*)stgSW + off);
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_UNPACK_B16>(regSE, (__ubuf__ T_X*)stgSE + off);
                Reg::Cast<float, T_X, castTrait0>(regNWFp32, regNW, pregFp32);
                Reg::Cast<float, T_X, castTrait0>(regNEFp32, regNE, pregFp32);
                Reg::Cast<float, T_X, castTrait0>(regSWFp32, regSW, pregFp32);
                Reg::Cast<float, T_X, castTrait0>(regSEFp32, regSE, pregFp32);
            } else {
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_NORM>(regNW, (__ubuf__ T_X*)stgNW + off);
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_NORM>(regNE, (__ubuf__ T_X*)stgNE + off);
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_NORM>(regSW, (__ubuf__ T_X*)stgSW + off);
                Reg::LoadAlign<T_X, Reg::LoadDist::DIST_NORM>(regSE, (__ubuf__ T_X*)stgSE + off);
            }
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(regWNW, wNW + off);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(regWNE, wNE + off);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(regWSW, wSW + off);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(regWSE, wSE + off);

            if constexpr (sizeof(T_X) == sizeof(int16_t)) {
                Reg::Mul(regNw, regNWFp32, regWNW, pregFp32);
                Reg::Mul(regEw, regNEFp32, regWNE, pregFp32);
                Reg::Add(regSumTopFp32, regNw, regEw, pregFp32);
                Reg::Mul(regNw, regSWFp32, regWSW, pregFp32);
                Reg::Mul(regEw, regSEFp32, regWSE, pregFp32);
                Reg::Add(regSumBotFp32, regNw, regEw, pregFp32);
            } else {
                Reg::Mul(regNw, regNW, regWNW, pregFp32);
                Reg::Mul(regEw, regNE, regWNE, pregFp32);
                Reg::Add(regSumTopFp32, regNw, regEw, pregFp32);
                Reg::Mul(regNw, regSW, regWSW, pregFp32);
                Reg::Mul(regEw, regSE, regWSE, pregFp32);
                Reg::Add(regSumBotFp32, regNw, regEw, pregFp32);
            }
            Reg::Add(regSumFp32, regSumTopFp32, regSumBotFp32, pregFp32);

            if constexpr (sizeof(T_Y) == sizeof(int16_t)) {
                Reg::Cast<T_Y, float, castTrait1>(regRst, regSumFp32, pregFp32);
                Reg::StoreAlign<T_Y, Reg::StoreDist::DIST_PACK_B32>(ySegY + off, regRst, pregFp32);
            } else {
                Reg::StoreAlign<T_Y, Reg::StoreDist::DIST_NORM>(ySegY + off, regSumFp32, pregFp32);
            }
        }
    }
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2SimdNhwc<T_X, T_Y>::Process()
{
    int64_t nNum = Ops::Base::CeilDiv(tilingData_->lenN, tilingData_->nFactor);
    int64_t hwNum = Ops::Base::CeilDiv(tilingData_->lenDesH * tilingData_->lenDesW, tilingData_->hwFactor);
    int64_t cNum = Ops::Base::CeilDiv(tilingData_->lenC, tilingData_->cFactor);
    int64_t nTail = tilingData_->lenN - (nNum - 1) * tilingData_->nFactor;
    int64_t hwTail = tilingData_->lenDesH * tilingData_->lenDesW - (hwNum - 1) * tilingData_->hwFactor;
    int64_t cTail = tilingData_->lenC - (cNum - 1) * tilingData_->cFactor;

    hwStrideX_ = tilingData_->lenC;
    nStrideX_ = hwStrideX_ * tilingData_->lenSrcH * tilingData_->lenSrcW;

    hwStrideY_ = tilingData_->lenC;
    nStrideY_ = hwStrideY_ * tilingData_->lenDesH * tilingData_->lenDesW;

    int64_t block = GetBlockIdx();
    if (block >= tilingData_->realCoreNum) {
        return;
    }

    int64_t cIdx = block % cNum;
    block /= cNum;
    int64_t hwIdx = block % hwNum;
    block /= hwNum;
    int64_t nIdx = block % nNum;

    int64_t nLength4Block = (nIdx < nNum - 1) ? tilingData_->nFactor : nTail;
    int64_t hwLength4Block = (hwIdx < hwNum - 1) ? tilingData_->hwFactor : hwTail;
    int64_t cLength4Block = (cIdx < cNum - 1) ? tilingData_->cFactor : cTail;

    int64_t nOffset4Block = nIdx * tilingData_->nFactor;
    int64_t hwOffset4Block = hwIdx * tilingData_->hwFactor;
    int64_t cOffset4Block = cIdx * tilingData_->cFactor;

    // 窗口模式路由：整 C 块（多核 C 切分块走逐点）
    if (outSeg_ > 0 && cOffset4Block == 0 && cLength4Block == tilingData_->lenC) {
        ProcessWindow(nOffset4Block, nLength4Block, hwOffset4Block, hwLength4Block);
        return;
    }

    for (int64_t hwLoop = 0; hwLoop < hwLength4Block; hwLoop++) {
        hwOffset_ = hwOffset4Block + hwLoop;

        for (int64_t nLoop = 0; nLoop < nLength4Block; nLoop += tilingData_->ubNFactor) {
            nOffset_ = nOffset4Block + nLoop;
            nLength_ = this->Min(tilingData_->ubNFactor, nLength4Block - nLoop);
            for (int64_t cLoop = 0; cLoop < cLength4Block; cLoop += tilingData_->ubCFactor) {
                cOffset_ = cOffset4Block + cLoop;
                cLength_ = this->Min(tilingData_->ubCFactor, cLength4Block - cLoop);

                CopyIn();
                Compute();
                CopyOut();
            }
        }
    }
}

} // namespace ResizeBilinearV2

#endif // RESIZE_BILINEAR_V2_SIMD_NHWC_H
