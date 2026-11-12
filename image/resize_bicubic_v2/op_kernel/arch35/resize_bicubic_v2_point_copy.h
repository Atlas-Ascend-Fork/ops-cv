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
 * \file resize_bicubic_v2_point_copy.h
 * \brief resize_bicubic_v2_point_copy
 */
#ifndef RESIZE_BICUBIC_V2_POINT_COPY_H
#define RESIZE_BICUBIC_V2_POINT_COPY_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeBicubicV2 {
using namespace AscendC;

template <typename T_DATA, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
class ResizeBicubicV2PointCopy : public ResizeBicubicV2Base {
public:
    __aicore__ inline ResizeBicubicV2PointCopy(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe,
                                const ResizeBicubicV2TilingData* tilingData_);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void CalcTile();

    const ResizeBicubicV2TilingData* tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> dataQue_;
    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;

    int64_t nStrideX_;
    int64_t hStrideX_;
    int64_t wStrideX_;

    int64_t nStrideY_;
    int64_t hStrideY_;
    int64_t wStrideY_;

    int64_t nOffset4Block_;
    int64_t hOffset4Block_;
    int64_t wOffset4Block_;
    int64_t cOffset4Block_;

    int64_t nLength4Block_;
    int64_t hLength4Block_;
    int64_t wLength4Block_;
    int64_t cLength4Block_;

    int64_t nOffset_;
    int64_t hOffset_;
    int64_t wOffset_;
    int64_t cOffset_;

    int64_t nLength_;
    int64_t hLength_;
    int64_t wLength_;
    int64_t cLength_;
    int64_t wcLenAlign_;

    int64_t hPointStride_; // 代表y点在x阵中的网格距离，如yxxyxxy，则设为3
    int64_t wPointStride_;
};

template <typename T_DATA, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2PointCopy<T_DATA, halfPixel, mode, T_IDX, T_IDX2>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBicubicV2TilingData* data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    int64_t wcLenAlign = Ops::Base::CeilAlign<int64_t>(tilingData_->ubWFactor * tilingData_->ubCFactor,
                                                       ONE_BLOCK_BYTE / sizeof(T_DATA));
    int64_t bufferSize = tilingData_->ubNFactor * tilingData_->ubHFactor * wcLenAlign * sizeof(T_DATA);
    this->pipe_->InitBuffer(dataQue_, 2, bufferSize);
}

template <typename T_DATA, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2PointCopy<T_DATA, halfPixel, mode, T_IDX, T_IDX2>::CopyIn()
{
    LocalTensor<uint8_t> xTensor = dataQue_.AllocTensor<uint8_t>();

    int64_t xOffsetInGM = nOffset_ * nStrideX_ + cOffset_;
    if (halfPixel > 0) {
        xOffsetInGM += ((2 * hOffset_ + 1) * hPointStride_ - 1) / 2 * hStrideX_ +
                       ((2 * wOffset_ + 1) * wPointStride_ - 1) / 2 * wStrideX_;
    } else {
        xOffsetInGM += hOffset_ * hStrideX_ * hPointStride_ + wOffset_ * wStrideX_ * wPointStride_;
    }

    LoopModeParams loopParams;
    loopParams.loop2Size = nLength_;
    loopParams.loop2SrcStride = nStrideX_ * sizeof(T_DATA);
    loopParams.loop2DstStride = hLength_ * wcLenAlign_ * sizeof(T_DATA);
    loopParams.loop1Size = hLength_;
    loopParams.loop1SrcStride = hStrideX_ * hPointStride_ * sizeof(T_DATA);
    loopParams.loop1DstStride = wcLenAlign_ * sizeof(T_DATA);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = wLength_;
    gm2ubParams.blockLen = cLength_ * sizeof(T_DATA);
    gm2ubParams.srcStride = (wPointStride_ - 1) * cLength_ * sizeof(T_DATA);
    gm2ubParams.dstStride = 0;

    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad<uint8_t, PaddingMode::Compact>(xTensor, xGM_[xOffsetInGM * sizeof(T_DATA)], gm2ubParams, padParams);

    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    dataQue_.EnQue(xTensor);
}

template <typename T_DATA, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2PointCopy<T_DATA, halfPixel, mode, T_IDX, T_IDX2>::CopyOut()
{
    LocalTensor<uint8_t> yTensor = dataQue_.DeQue<uint8_t>();

    int64_t yOffsetInGM = nOffset_ * nStrideY_ + hOffset_ * hStrideY_ + wOffset_ * wStrideY_ + cOffset_;

    LoopModeParams loopParams;
    loopParams.loop2Size = nLength_;
    loopParams.loop2SrcStride = hLength_ * wcLenAlign_ * sizeof(T_DATA);
    loopParams.loop2DstStride = nStrideY_ * sizeof(T_DATA);
    loopParams.loop1Size = hLength_;
    loopParams.loop1SrcStride = wcLenAlign_ * sizeof(T_DATA);
    loopParams.loop1DstStride = hStrideY_ * sizeof(T_DATA);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = 1;
    ub2gmParams.blockLen = wLength_ * cLength_ * sizeof(T_DATA);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = 0;
    DataCopyPad<uint8_t, PaddingMode::Compact>(yGM_[yOffsetInGM * sizeof(T_DATA)], yTensor, ub2gmParams);

    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    dataQue_.FreeTensor(yTensor);
}

template <typename T_DATA, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2PointCopy<T_DATA, halfPixel, mode, T_IDX, T_IDX2>::CalcTile()
{
    int64_t nNum = Ops::Base::CeilDiv(tilingData_->lenN, tilingData_->nFactor);
    int64_t hNum = Ops::Base::CeilDiv(tilingData_->lenDesH, tilingData_->hFactor);
    int64_t wNum = Ops::Base::CeilDiv(tilingData_->lenDesW, tilingData_->wFactor);
    int64_t cNum = Ops::Base::CeilDiv(tilingData_->lenC, tilingData_->cFactor);
    int64_t nTail = tilingData_->lenN - (nNum - 1) * tilingData_->nFactor;
    int64_t hTail = tilingData_->lenDesH - (hNum - 1) * tilingData_->hFactor;
    int64_t wTail = tilingData_->lenDesW - (wNum - 1) * tilingData_->wFactor;
    int64_t cTail = tilingData_->lenC - (cNum - 1) * tilingData_->cFactor;

    wStrideY_ = tilingData_->lenC;
    hStrideY_ = wStrideY_ * tilingData_->lenDesW;
    nStrideY_ = hStrideY_ * tilingData_->lenDesH;

    wStrideX_ = tilingData_->lenC;
    hStrideX_ = wStrideX_ * tilingData_->lenSrcW;
    nStrideX_ = hStrideX_ * tilingData_->lenSrcH;

    if (halfPixel <= 0) {
        hPointStride_ = (tilingData_->lenSrcH - 1) / (tilingData_->lenDesH - 1);
        wPointStride_ = (tilingData_->lenSrcW - 1) / (tilingData_->lenDesW - 1);
    } else {
        hPointStride_ = tilingData_->lenSrcH / tilingData_->lenDesH;
        wPointStride_ = tilingData_->lenSrcW / tilingData_->lenDesW;
    }

    int64_t block = GetBlockIdx();
    int64_t cIdx = block % cNum;
    block /= cNum;
    int64_t wIdx = block % wNum;
    block /= wNum;
    int64_t hIdx = block % hNum;
    block /= hNum;
    int64_t nIdx = block % nNum;

    nOffset4Block_ = nIdx * tilingData_->nFactor;
    hOffset4Block_ = hIdx * tilingData_->hFactor;
    wOffset4Block_ = wIdx * tilingData_->wFactor;
    cOffset4Block_ = cIdx * tilingData_->cFactor;

    nLength4Block_ = (nIdx < nNum - 1) ? tilingData_->nFactor : nTail;
    hLength4Block_ = (hIdx < hNum - 1) ? tilingData_->hFactor : hTail;
    wLength4Block_ = (wIdx < wNum - 1) ? tilingData_->wFactor : wTail;
    cLength4Block_ = (cIdx < cNum - 1) ? tilingData_->cFactor : cTail;
}

template <typename T_DATA, uint64_t halfPixel, uint64_t mode, typename T_IDX, typename T_IDX2>
__aicore__ inline void ResizeBicubicV2PointCopy<T_DATA, halfPixel, mode, T_IDX, T_IDX2>::Process()
{
    CalcTile();

    for (int64_t nLoop = 0; nLoop < nLength4Block_; nLoop += tilingData_->ubNFactor) {
        nOffset_ = nOffset4Block_ + nLoop;
        nLength_ = this->Min(tilingData_->ubNFactor, nLength4Block_ - nLoop);
        for (int64_t hLoop = 0; hLoop < hLength4Block_; hLoop += tilingData_->ubHFactor) {
            hOffset_ = hOffset4Block_ + hLoop;
            hLength_ = this->Min(tilingData_->ubHFactor, hLength4Block_ - hLoop);
            for (int64_t wLoop = 0; wLoop < wLength4Block_; wLoop += tilingData_->ubWFactor) {
                wOffset_ = wOffset4Block_ + wLoop;
                wLength_ = this->Min(tilingData_->ubWFactor, wLength4Block_ - wLoop);
                for (int64_t cLoop = 0; cLoop < cLength4Block_; cLoop += tilingData_->ubCFactor) {
                    cOffset_ = cOffset4Block_ + cLoop;
                    cLength_ = this->Min(tilingData_->ubCFactor, cLength4Block_ - cLoop);
                    wcLenAlign_ = Ops::Base::CeilAlign<int64_t>(wLength_ * cLength_, ONE_BLOCK_BYTE / sizeof(T_DATA));

                    CopyIn();
                    CopyOut();
                }
            }
        }
    }
}

} // namespace ResizeBicubicV2

#endif // RESIZE_BICUBIC_V2_POINT_COPY_H
