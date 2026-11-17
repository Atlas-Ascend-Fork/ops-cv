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
 * \file resize_bilinear_v2_grad_point_copy.h
 * \brief resize_bilinear_v2_grad_point_copy
 */
#ifndef RESIZE_BILINEAR_V2_GRAD_POINT_COPY_H
#define RESIZE_BILINEAR_V2_GRAD_POINT_COPY_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "resize_bilinear_v2_grad_base.h"

namespace ResizeBilinearV2Grad {
using namespace AscendC;

struct PointCopyOffsetDefSt {
    int64_t nStrideSrc = 0;
    int64_t hStrideSrc = 0;
    int64_t wStrideSrc = 0;
    int64_t nStrideDes = 0;
    int64_t hStrideDes = 0;
    int64_t wStrideDes = 0;

    int64_t nOffset4Block = 0;
    int64_t hOffset4Block = 0;
    int64_t wOffset4Block = 0;
    int64_t cOffset4Block = 0;
    int64_t nLength4Block = 0;
    int64_t hLength4Block = 0;
    int64_t wLength4Block = 0;
    int64_t cLength4Block = 0;

    int64_t nOffset = 0;
    int64_t hOffset = 0;
    int64_t wOffset = 0;
    int64_t cOffset = 0;
    int64_t nLength = 0;
    int64_t hLength = 0;
    int64_t wLength = 0;
    int64_t cLength = 0;
    int64_t wcLenAlign = 0;
    int64_t hScales = 0;
    int64_t wScales = 0;
};

template <typename T_GRADS, typename T_OUT>
class ResizeBilinearV2GradPointCopy : public ResizeBilinearV2GradBase {
public:
    __aicore__ inline ResizeBilinearV2GradPointCopy(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, TPipe* pipe,
                                const ResizeBilinearV2GradTilingData* data);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyInGrad();
    __aicore__ inline void CopyOutY();
    __aicore__ inline void CalcTile();
    __aicore__ inline void ClearOutputGm();

    PointCopyOffsetDefSt offsetData_;
    const ResizeBilinearV2GradTilingData* tilingData_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQue_;
    int64_t blockIdx_ = 0;
    int64_t ubBlockSize_ = 0;
};

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradPointCopy<T_GRADS, T_OUT>::Init(GM_ADDR grads, GM_ADDR originalImage,
                                                                           GM_ADDR y, TPipe* pipe,
                                                                           const ResizeBilinearV2GradTilingData* data)
{
    this->BaseInit(grads, y, pipe);
    blockIdx_ = GetBlockIdx();
    ubBlockSize_ = Ops::Base::GetUbBlockSize();
    tilingData_ = data;

    int64_t cLengthAlign32 = Ops::Base::CeilAlign<int64_t>(tilingData_->ubCFactor, ubBlockSize_ / sizeof(T_GRADS));
    int64_t wcLenAlign = Ops::Base::CeilAlign<int64_t>(tilingData_->ubWFactor * cLengthAlign32,
                                                       ubBlockSize_ / sizeof(T_GRADS));
    int64_t bufferSize = tilingData_->ubNFactor * tilingData_->ubHFactor * wcLenAlign * sizeof(T_GRADS);
    this->pipe_->InitBuffer(dataQue_, 2, bufferSize);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradPointCopy<T_GRADS, T_OUT>::CopyOutY()
{
    LocalTensor<uint8_t> yTensor = dataQue_.DeQue<uint8_t>();

    int64_t yOffsetInGM = offsetData_.nOffset * offsetData_.nStrideSrc + offsetData_.cOffset;
    if (tilingData_->halfPixelCenters > 0) {
        yOffsetInGM += ((2 * offsetData_.hOffset + 1) * offsetData_.hScales - 1) / 2 * offsetData_.hStrideSrc +
                       ((2 * offsetData_.wOffset + 1) * offsetData_.wScales - 1) / 2 * offsetData_.wStrideSrc;
    } else {
        yOffsetInGM += offsetData_.hOffset * offsetData_.hStrideSrc * offsetData_.hScales +
                       offsetData_.wOffset * offsetData_.wStrideSrc * offsetData_.wScales;
    }

    LoopModeParams loopParams;
    loopParams.loop2Size = offsetData_.nLength;
    loopParams.loop2SrcStride = offsetData_.hLength * offsetData_.wcLenAlign * sizeof(T_OUT);
    loopParams.loop2DstStride = offsetData_.nStrideSrc * sizeof(T_OUT);
    loopParams.loop1Size = offsetData_.hLength;
    loopParams.loop1SrcStride = offsetData_.wcLenAlign * sizeof(T_OUT);
    loopParams.loop1DstStride = offsetData_.hStrideSrc * offsetData_.hScales * sizeof(T_OUT);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = offsetData_.wLength;
    ub2gmParams.blockLen = offsetData_.cLength * sizeof(T_OUT);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = (offsetData_.wScales - 1) * offsetData_.cLength * sizeof(T_OUT);

    DataCopyPad<uint8_t, PaddingMode::Compact>(yGM_[yOffsetInGM * sizeof(T_OUT)], yTensor, ub2gmParams);

    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    dataQue_.FreeTensor(yTensor);
    return;
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradPointCopy<T_GRADS, T_OUT>::CopyInGrad()
{
    LocalTensor<uint8_t> gradTensor = dataQue_.AllocTensor<uint8_t>();

    int64_t gradOffsetInGM = offsetData_.nOffset * offsetData_.nStrideDes +
                             offsetData_.hOffset * offsetData_.hStrideDes +
                             offsetData_.wOffset * offsetData_.wStrideDes + offsetData_.cOffset;

    LoopModeParams loopParams;
    loopParams.loop2Size = offsetData_.nLength;
    loopParams.loop2SrcStride = offsetData_.nStrideDes * sizeof(T_GRADS);
    loopParams.loop2DstStride = offsetData_.hLength * offsetData_.wcLenAlign * sizeof(T_GRADS);
    loopParams.loop1Size = offsetData_.hLength;
    loopParams.loop1SrcStride = offsetData_.hStrideDes * sizeof(T_GRADS);
    loopParams.loop1DstStride = offsetData_.wcLenAlign * sizeof(T_GRADS);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = 1;
    gm2ubParams.blockLen = offsetData_.wLength * offsetData_.cLength * sizeof(T_GRADS);
    gm2ubParams.srcStride = 0;
    gm2ubParams.dstStride = 0;
    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad<uint8_t, PaddingMode::Compact>(gradTensor, gradsGM_[gradOffsetInGM * sizeof(T_GRADS)], gm2ubParams,
                                               padParams);

    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    dataQue_.EnQue(gradTensor);
    return;
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradPointCopy<T_GRADS, T_OUT>::CalcTile()
{
    int64_t nNum = Ops::Base::CeilDiv(tilingData_->lenN, tilingData_->nFactor);
    int64_t hNum = Ops::Base::CeilDiv(tilingData_->lenDesH, tilingData_->hFactor);
    int64_t wNum = Ops::Base::CeilDiv(tilingData_->lenDesW, tilingData_->wFactor);
    int64_t cNum = Ops::Base::CeilDiv(tilingData_->lenC, tilingData_->cFactor);
    int64_t nTail = tilingData_->lenN - (nNum - 1) * tilingData_->nFactor;
    int64_t hTail = tilingData_->lenDesH - (hNum - 1) * tilingData_->hFactor;
    int64_t wTail = tilingData_->lenDesW - (wNum - 1) * tilingData_->wFactor;
    int64_t cTail = tilingData_->lenC - (cNum - 1) * tilingData_->cFactor;

    offsetData_.wStrideDes = tilingData_->lenC;
    offsetData_.hStrideDes = offsetData_.wStrideDes * tilingData_->lenDesW;
    offsetData_.nStrideDes = offsetData_.hStrideDes * tilingData_->lenDesH;

    offsetData_.wStrideSrc = tilingData_->lenC;
    offsetData_.hStrideSrc = offsetData_.wStrideSrc * tilingData_->lenSrcW;
    offsetData_.nStrideSrc = offsetData_.hStrideSrc * tilingData_->lenSrcH;

    if (tilingData_->alignCorners > 0) {
        // When lenDesH/lenDesW == 1 there is a single output point, so the scale (source pixels per
        // destination step) is unused for offset computation. Use 1 instead of computing
        // (lenSrc-1)/(lenDes-1), which would divide by zero; 1 also keeps the derived DataCopy
        // dstStride = (wScales-1)*... non-negative (dstStride is unsigned), unlike 0.
        offsetData_.hScales = (tilingData_->lenDesH > 1) ? (tilingData_->lenSrcH - 1) / (tilingData_->lenDesH - 1) : 1;
        offsetData_.wScales = (tilingData_->lenDesW > 1) ? (tilingData_->lenSrcW - 1) / (tilingData_->lenDesW - 1) : 1;
    } else {
        offsetData_.hScales = tilingData_->lenSrcH / tilingData_->lenDesH;
        offsetData_.wScales = tilingData_->lenSrcW / tilingData_->lenDesW;
    }

    int64_t block = blockIdx_;
    int64_t cIdx = block % cNum;
    block /= cNum;
    int64_t wIdx = block % wNum;
    block /= wNum;
    int64_t hIdx = block % hNum;
    block /= hNum;
    int64_t nIdx = block % nNum;

    offsetData_.nOffset4Block = nIdx * tilingData_->nFactor;
    offsetData_.hOffset4Block = hIdx * tilingData_->hFactor;
    offsetData_.wOffset4Block = wIdx * tilingData_->wFactor;
    offsetData_.cOffset4Block = cIdx * tilingData_->cFactor;

    offsetData_.nLength4Block = (nIdx < nNum - 1) ? tilingData_->nFactor : nTail;
    offsetData_.hLength4Block = (hIdx < hNum - 1) ? tilingData_->hFactor : hTail;
    offsetData_.wLength4Block = (wIdx < wNum - 1) ? tilingData_->wFactor : wTail;
    offsetData_.cLength4Block = (cIdx < cNum - 1) ? tilingData_->cFactor : cTail;
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradPointCopy<T_GRADS, T_OUT>::ClearOutputGm()
{
    int64_t yOffset = 0;
    int64_t yLength = tilingData_->initYSplitBlockFactor;
    int64_t yBaseOffset = tilingData_->initYSplitBlockFactor * blockIdx_;
    if (blockIdx_ >= tilingData_->initYRealCoreNum) {
        return;
    }
    if (blockIdx_ < tilingData_->initYSplitBlockTailFactor) {
        yLength = yLength + 1;
        yOffset = yBaseOffset + blockIdx_;
    } else {
        yOffset = yBaseOffset + tilingData_->initYSplitBlockTailFactor;
    }

    InitOutput<uint8_t>(yGM_[yOffset * sizeof(T_OUT)], yLength * sizeof(T_OUT), 0);
    SyncAll();
    return;
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradPointCopy<T_GRADS, T_OUT>::Process()
{
    int64_t cLengthAlign32 = 0;
    CalcTile();
    ClearOutputGm();

    for (int64_t nLoop = 0; nLoop < offsetData_.nLength4Block; nLoop += tilingData_->ubNFactor) {
        offsetData_.nOffset = offsetData_.nOffset4Block + nLoop;
        offsetData_.nLength = this->Min(tilingData_->ubNFactor, offsetData_.nLength4Block - nLoop);
        for (int64_t hLoop = 0; hLoop < offsetData_.hLength4Block; hLoop += tilingData_->ubHFactor) {
            offsetData_.hOffset = offsetData_.hOffset4Block + hLoop;
            offsetData_.hLength = this->Min(tilingData_->ubHFactor, offsetData_.hLength4Block - hLoop);
            for (int64_t wLoop = 0; wLoop < offsetData_.wLength4Block; wLoop += tilingData_->ubWFactor) {
                offsetData_.wOffset = offsetData_.wOffset4Block + wLoop;
                offsetData_.wLength = this->Min(tilingData_->ubWFactor, offsetData_.wLength4Block - wLoop);
                for (int64_t cLoop = 0; cLoop < offsetData_.cLength4Block; cLoop += tilingData_->ubCFactor) {
                    offsetData_.cOffset = offsetData_.cOffset4Block + cLoop;
                    offsetData_.cLength = this->Min(tilingData_->ubCFactor, offsetData_.cLength4Block - cLoop);
                    cLengthAlign32 = Ops::Base::CeilAlign<int64_t>(offsetData_.cLength, ubBlockSize_ / sizeof(T_GRADS));
                    offsetData_.wcLenAlign = Ops::Base::CeilAlign<int64_t>(offsetData_.wLength * cLengthAlign32,
                                                                           ubBlockSize_ / sizeof(T_GRADS));
                    CopyInGrad();
                    CopyOutY();
                }
            }
        }
    }
}

} // namespace ResizeBilinearV2Grad

#endif // RESIZE_BILINEAR_V2_GRAD_POINT_COPY_H
