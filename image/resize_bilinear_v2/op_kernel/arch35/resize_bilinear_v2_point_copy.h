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
 * \file resize_bilinear_v2_point_copy.h
 * \brief resize_bilinear_v2_point_copy
 */
#ifndef RESIZE_BILINEAR_V2_POINT_COPY_H
#define RESIZE_BILINEAR_V2_POINT_COPY_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2 {
using namespace AscendC;

template <typename T_DATA>
class ResizeBilinearV2PointCopy : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2PointCopy(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe,
                                const ResizeBilinearV2TilingData* data);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t& xOffsetInGM, int64_t& startHOffset, int64_t& startWOffset,
                                  int64_t& endHOffset, int64_t& endWOffset, int64_t& overStepH, int64_t& overStepW,
                                  int64_t& hMaxOffset, int64_t& wMaxOffset, int64_t& curBlockCount, bool& wOverStep,
                                  bool& hOverStep);
    __aicore__ inline void CopyOut();
    __aicore__ inline void CalcTile();
    __aicore__ inline void ComputeParams(int64_t& startHOffset, int64_t& startWOffset, int64_t& endHOffset,
                                         int64_t& endWOffset, bool& hOverStep, int64_t& overStepH, bool& wOverStep,
                                         int64_t& overStepW, int64_t& curBlockCount, int64_t& hMaxOffset,
                                         int64_t& wMaxOffset);
    __aicore__ inline void CopyInOverstepH(LocalTensor<uint8_t>& xTensor, bool& wOverStep, int64_t& xOffsetInGM,
                                           int64_t& hMaxOffset, int64_t& wMaxOffset, int64_t& startWOffset,
                                           int64_t& overStepH, int64_t& overStepW, DataCopyExtParams& gm2ubParams,
                                           DataCopyExtParams& gm2ubParamsW);

    const ResizeBilinearV2TilingData* tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue_;
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
    float EPSILON = 1e-6f;
};

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::Init(GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe,
                                                               const ResizeBilinearV2TilingData* data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    int64_t wcLenAlign = Ops::Base::CeilAlign<int64_t>(tilingData_->ubWFactor * tilingData_->ubCFactor,
                                                       ONE_BLOCK_BYTE / sizeof(T_DATA));
    int64_t bufferSize = tilingData_->ubNFactor * tilingData_->ubHFactor * wcLenAlign * sizeof(T_DATA);
    this->pipe_->InitBuffer(dataQue_, 2, bufferSize);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::ComputeParams(int64_t& startHOffset, int64_t& startWOffset,
                                                                        int64_t& endHOffset, int64_t& endWOffset,
                                                                        bool& hOverStep, int64_t& overStepH,
                                                                        bool& wOverStep, int64_t& overStepW,
                                                                        int64_t& curBlockCount, int64_t& hMaxOffset,
                                                                        int64_t& wMaxOffset)
{
    if (tilingData_->halfPixelCenters > 0) {
        startHOffset = min(hMaxOffset, ((2 * hOffset_ + 1) * hPointStride_ - 1) / 2 * hStrideX_);
        startWOffset = min(wMaxOffset, ((2 * wOffset_ + 1) * wPointStride_ - 1) / 2 * wStrideX_);
    } else {
        startHOffset = min(hMaxOffset, hOffset_ * hStrideX_ * hPointStride_);
        startWOffset = min(wMaxOffset, wOffset_ * wStrideX_ * wPointStride_);
    }

    endHOffset = startHOffset + (hLength_ - 1) * hStrideX_ * hPointStride_;
    endWOffset = startWOffset + (wLength_ - 1) * wStrideX_ * wPointStride_;

    if (endHOffset > hMaxOffset) {
        hOverStep = true;
        overStepH = Ops::Base::CeilDiv(endHOffset - hMaxOffset, hStrideX_ * hPointStride_);
    }

    if (endWOffset > wMaxOffset) {
        wOverStep = true;
        overStepW = Ops::Base::CeilDiv(endWOffset - wMaxOffset, wStrideX_ * wPointStride_);
        curBlockCount = 1;
    }
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::CopyInOverstepH(
    LocalTensor<uint8_t>& xTensor, bool& wOverStep, int64_t& xOffsetInGM, int64_t& hMaxOffset, int64_t& wMaxOffset,
    int64_t& startWOffset, int64_t& overStepH, int64_t& overStepW, DataCopyExtParams& gm2ubParams,
    DataCopyExtParams& gm2ubParamsW)
{
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    for (int64_t nIdx = 0; nIdx < nLength_; nIdx++) {
        for (int64_t heightIdx = hLength_ - overStepH; heightIdx < hLength_; heightIdx++) {
            int64_t xOffsetInTensor = nIdx * nStrideY_ + heightIdx * hStrideY_;
            if (!wOverStep) {
                int64_t newWOffset = (xOffsetInGM + hMaxOffset + startWOffset) * sizeof(T_DATA);
                DataCopyPad<uint8_t, PaddingMode::Compact>(xTensor[xOffsetInTensor * sizeof(T_DATA)], xGM_[newWOffset],
                                                           gm2ubParams, padParams);
            } else {
                int64_t newWOffset = (xOffsetInGM + hMaxOffset + wMaxOffset) * sizeof(T_DATA);
                DataCopyPad<uint8_t, PaddingMode::Compact>(
                    xTensor[xOffsetInTensor * sizeof(T_DATA)],
                    xGM_[(xOffsetInGM + hMaxOffset + startWOffset) * sizeof(T_DATA)], gm2ubParamsW, padParams);
                for (int64_t widthIdx = wLength_ - overStepW; widthIdx < wLength_; widthIdx++) {
                    DataCopyPad<uint8_t, PaddingMode::Compact>(
                        xTensor[xOffsetInTensor * sizeof(T_DATA) + widthIdx * cLength_ * sizeof(T_DATA)],
                        xGM_[newWOffset], gm2ubParams, padParams);
                }
            }
        }
    }
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::CopyIn(int64_t& xOffsetInGM, int64_t& startHOffset,
                                                                 int64_t& startWOffset, int64_t& endHOffset,
                                                                 int64_t& endWOffset, int64_t& overStepH,
                                                                 int64_t& overStepW, int64_t& hMaxOffset,
                                                                 int64_t& wMaxOffset, int64_t& curBlockCount,
                                                                 bool& wOverStep, bool& hOverStep)
{
    LocalTensor<uint8_t> xTensor = dataQue_.AllocTensor<uint8_t>();
    ComputeParams(startHOffset, startWOffset, endHOffset, endWOffset, hOverStep, overStepH, wOverStep, overStepW,
                  curBlockCount, hMaxOffset, wMaxOffset);

    LoopModeParams loopParams;
    loopParams.loop2Size = nLength_;
    loopParams.loop2SrcStride = nStrideX_ * sizeof(T_DATA);
    loopParams.loop2DstStride = hLength_ * wcLenAlign_ * sizeof(T_DATA);
    loopParams.loop1Size = hLength_ - overStepH;
    loopParams.loop1SrcStride = hStrideX_ * hPointStride_ * sizeof(T_DATA);
    loopParams.loop1DstStride = wcLenAlign_ * sizeof(T_DATA);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = curBlockCount;
    gm2ubParams.blockLen = cLength_ * sizeof(T_DATA);
    gm2ubParams.srcStride = (wPointStride_ - 1) * cLength_ * sizeof(T_DATA);
    gm2ubParams.dstStride = 0;

    // gm2ubParamsW 处理W轴越界情况下的未越界部分
    DataCopyExtParams gm2ubParamsW;
    gm2ubParamsW.blockCount = wLength_ - overStepW;
    gm2ubParamsW.blockLen = cLength_ * sizeof(T_DATA);
    gm2ubParamsW.srcStride = (wPointStride_ - 1) * cLength_ * sizeof(T_DATA);
    gm2ubParamsW.dstStride = 0;

    if (!wOverStep) {
        int64_t newWOffset = (xOffsetInGM + startHOffset + startWOffset) * sizeof(T_DATA);
        DataCopyPad<uint8_t, PaddingMode::Compact>(xTensor, xGM_[newWOffset], gm2ubParams, padParams);
    } else {
        int64_t newWOffset = (xOffsetInGM + startHOffset + wMaxOffset) * sizeof(T_DATA);
        DataCopyPad<uint8_t, PaddingMode::Compact>(
            xTensor, xGM_[(xOffsetInGM + startHOffset + startWOffset) * sizeof(T_DATA)], gm2ubParamsW, padParams);
        for (int64_t widthIdx = wLength_ - overStepW; widthIdx < wLength_; widthIdx++) {
            DataCopyPad<uint8_t, PaddingMode::Compact>(xTensor[widthIdx * cLength_ * sizeof(T_DATA)], xGM_[newWOffset],
                                                       gm2ubParams, padParams);
        }
    }

    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    // 如果有H轴方向越界，那么将越界的部分继续搬入
    if (hOverStep) {
        CopyInOverstepH(xTensor, wOverStep, xOffsetInGM, hMaxOffset, wMaxOffset, startWOffset, overStepH, overStepW,
                        gm2ubParams, gm2ubParamsW);
    }

    dataQue_.EnQue(xTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::CopyOut()
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

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::CalcTile()
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

    hPointStride_ = static_cast<int64_t>(tilingData_->scaleH + EPSILON);
    wPointStride_ = static_cast<int64_t>(tilingData_->scaleW + EPSILON);

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

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2PointCopy<T_DATA>::Process()
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

                    int64_t xOffsetInGM = nOffset_ * nStrideX_ + cOffset_;
                    int64_t startHOffset = 0;
                    int64_t startWOffset = 0;
                    int64_t endHOffset = 0;
                    int64_t endWOffset = 0;
                    int64_t overStepH = 0;
                    int64_t overStepW = 0;
                    int64_t hMaxOffset = (tilingData_->lenSrcH - 1) * hStrideX_;
                    int64_t wMaxOffset = (tilingData_->lenSrcW - 1) * wStrideX_;
                    // overStepH  映射计算后，H轴方向越界的块数
                    int64_t curBlockCount = wLength_;
                    bool wOverStep = false;
                    bool hOverStep = false;
                    CopyIn(xOffsetInGM, startHOffset, startWOffset, endHOffset, endWOffset, overStepH, overStepW,
                           hMaxOffset, wMaxOffset, curBlockCount, wOverStep, hOverStep);
                    CopyOut();
                }
            }
        }
    }
}

} // namespace ResizeBilinearV2

#endif // RESIZE_BILINEAR_V2_POINT_COPY_H
