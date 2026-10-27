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
 * \file aipp_yuv_rgb.h
 * \brief AIPP YUV420SP to RGB/BGR conversion kernel (with CSC matrix).
 *
 * Moved from aipp_yuv.h. Handles YUV420SP_U8 → RGB/BGR output via a
 * configurable 3x3 CSC matrix (ApplyCscMatrix).
 *
 * TilingKey = FORMAT_YUV_2_RGB_UINT32 (5)
 */

#ifndef AIPP_OP_KERNEL_ARCH35_YUV_RGB_H
#define AIPP_OP_KERNEL_ARCH35_YUV_RGB_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippYuvRgb : public AippBase<T, DataType> {
public:
    __aicore__ inline AippYuvRgb(){};
    __aicore__ inline void Init(const AippTilingData& tilingData, const tagAippDynamicParaHeader& tilingParamHeader,
                                const __gm__ uint8_t* gmParams, uint8_t dynamicTilingKey);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippYuvRgb<T, DataType>::Init(const AippTilingData& tilingData,
                                                     const tagAippDynamicParaHeader& tilingParamHeader,
                                                     const __gm__ uint8_t* gmParams, uint8_t dynamicTilingKey)
{
    this->BaseInit(tilingData, tilingParamHeader, gmParams, dynamicTilingKey);
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ComputeYuvRgbDstIdx(RgbPack<DataType>* dstRgbIdx,
                                                                               const CoordPack<DataType>& coord,
                                                                               const AippTilingData& tD)
{
    if (tD.outputFormat == NCHW_FORMAT_INDEX) {
        dstRgbIdx[YUV_DEAL_NUM_0].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) +
                                      DIGIT_2 * coord.hIdx * tD.outputSizeW + DIGIT_2 * coord.wIdx;
        dstRgbIdx[YUV_DEAL_NUM_0].g = dstRgbIdx[YUV_DEAL_NUM_0].r + tD.outputSizeH * tD.outputSizeW;
        dstRgbIdx[YUV_DEAL_NUM_0].b = dstRgbIdx[YUV_DEAL_NUM_0].g + tD.outputSizeH * tD.outputSizeW;

        dstRgbIdx[YUV_DEAL_NUM_1].r = dstRgbIdx[YUV_DEAL_NUM_0].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_1].g = dstRgbIdx[YUV_DEAL_NUM_0].g + 1;
        dstRgbIdx[YUV_DEAL_NUM_1].b = dstRgbIdx[YUV_DEAL_NUM_0].b + 1;

        dstRgbIdx[YUV_DEAL_NUM_2].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) +
                                      (DIGIT_2 * coord.hIdx + 1) * tD.outputSizeW + DIGIT_2 * coord.wIdx;
        dstRgbIdx[YUV_DEAL_NUM_2].g = dstRgbIdx[YUV_DEAL_NUM_2].r + tD.outputSizeH * tD.outputSizeW;
        dstRgbIdx[YUV_DEAL_NUM_2].b = dstRgbIdx[YUV_DEAL_NUM_2].g + tD.outputSizeH * tD.outputSizeW;

        dstRgbIdx[YUV_DEAL_NUM_3].r = dstRgbIdx[YUV_DEAL_NUM_2].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_3].g = dstRgbIdx[YUV_DEAL_NUM_2].g + 1;
        dstRgbIdx[YUV_DEAL_NUM_3].b = dstRgbIdx[YUV_DEAL_NUM_2].b + 1;
    } else {
        dstRgbIdx[YUV_DEAL_NUM_0].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) +
                                      DIGIT_2 * coord.hIdx * tD.outputSizeW * CHANNEL_THREE +
                                      DIGIT_2 * coord.wIdx * CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_0].g = dstRgbIdx[YUV_DEAL_NUM_0].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_0].b = dstRgbIdx[YUV_DEAL_NUM_0].g + 1;

        dstRgbIdx[YUV_DEAL_NUM_1].r = dstRgbIdx[YUV_DEAL_NUM_0].r + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_1].g = dstRgbIdx[YUV_DEAL_NUM_0].g + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_1].b = dstRgbIdx[YUV_DEAL_NUM_0].b + CHANNEL_THREE;

        dstRgbIdx[YUV_DEAL_NUM_2].r = coord.nIdx * (tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE) +
                                      (DIGIT_2 * coord.hIdx + 1) * tD.outputSizeW * CHANNEL_THREE +
                                      DIGIT_2 * coord.wIdx * CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_2].g = dstRgbIdx[YUV_DEAL_NUM_2].r + 1;
        dstRgbIdx[YUV_DEAL_NUM_2].b = dstRgbIdx[YUV_DEAL_NUM_2].g + 1;

        dstRgbIdx[YUV_DEAL_NUM_3].r = dstRgbIdx[YUV_DEAL_NUM_2].r + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_3].g = dstRgbIdx[YUV_DEAL_NUM_2].g + CHANNEL_THREE;
        dstRgbIdx[YUV_DEAL_NUM_3].b = dstRgbIdx[YUV_DEAL_NUM_2].b + CHANNEL_THREE;
    }
}

// IsPixelInPaddingForYuv is defined in aipp_base.h

template <typename T, typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ProcessYuvRgbPixel(__gm__ uint8_t* yuvGM, __gm__ T* outputGM,
                                                                              const RgbPack<DataType> dstIdx,
                                                                              DataType nIdx, uint32_t croodH,
                                                                              uint32_t croodW, const AippTilingData& tD)
{
    const uint64_t yuvPlaneSize = static_cast<uint64_t>(tD.inputSizeH) * tD.inputSizeW * DIGIT_3 / DIGIT_2;
    uint64_t srcYIdx = static_cast<uint64_t>(nIdx) * yuvPlaneSize +
                       (tD.cropParam.cropStartPosH + croodH) * tD.inputSizeW + (tD.cropParam.cropStartPosW + croodW);
    uint64_t srcUIdx = static_cast<uint64_t>(nIdx) * yuvPlaneSize + tD.inputSizeH * tD.inputSizeW +
                       ((tD.cropParam.cropStartPosH + (croodH & ~1)) >> 1) * tD.inputSizeW +
                       (tD.cropParam.cropStartPosW + (croodW & ~1));
    uint64_t srcVIdx = srcUIdx + 1;
    RgbPack<uint8_t> dstRgb;
    ApplyCscMatrix(dstRgb, yuvGM[srcYIdx], yuvGM[srcUIdx], yuvGM[srcVIdx], tD.cscParam);

    DataConversion(outputGM[dstIdx.r], dstRgb.r, tD.dtcParam, CHANNEL_NUM_0);
    DataConversion(outputGM[dstIdx.g], dstRgb.g, tD.dtcParam, CHANNEL_NUM_1);
    DataConversion(outputGM[dstIdx.b], dstRgb.b, tD.dtcParam, CHANNEL_NUM_2);
}

template <typename T, typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ProcessYuvRgbBlock(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM, const RgbPack<DataType>* dstRgbIdx, const CoordPack<DataType>& coord,
    const AippTilingData& tD, float padValue, bool allEvenPadding, bool blockAllInPadding)
{
    uint32_t actualH = coord.hIdx * DIGIT_2;
    uint32_t actualW = coord.wIdx * DIGIT_2;

    for (uint8_t i = 0; i < YUV_PER_DEAL_NUM; i++) {
        RgbPack<DataType> dstRgbIdxIndex = dstRgbIdx[i];
        uint32_t pixelH = (i >= YUV_DEAL_NUM_2) ? actualH + 1 : actualH;
        uint32_t pixelW = (i == YUV_DEAL_NUM_1 || i == YUV_DEAL_NUM_3) ? actualW + 1 : actualW;
        if (pixelH >= tD.outputSizeH || pixelW >= tD.outputSizeW) {
            continue;
        }
        if (IsPixelInPaddingForYuv(pixelH, pixelW, tD, allEvenPadding, blockAllInPadding)) {
            AssignPadValue(outputGM[dstRgbIdxIndex.r], padValue);
            AssignPadValue(outputGM[dstRgbIdxIndex.g], padValue);
            AssignPadValue(outputGM[dstRgbIdxIndex.b], padValue);
        } else {
            uint32_t cropCoordH = pixelH - tD.paddingParam.topPaddingSize;
            uint32_t cropCoordW = pixelW - tD.paddingParam.leftPaddingSize;
            ProcessYuvRgbPixel<T, DataType>(yuvGM, outputGM, dstRgbIdxIndex, coord.nIdx, cropCoordH, cropCoordW, tD);
        }
    }
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__
    void SimtComputeYuvRgb(__gm__ uint8_t* yuvGM, __gm__ T* outputGM, AippTilingData tD, const __gm__ uint8_t* gmParams,
                           uint32_t blockId, uint32_t blockNum, uint64_t batchSize, uint8_t dynamicTilingKey)
{
    uint32_t outputSizeH = tD.outputSizeH;
    uint32_t outputSizeW = tD.outputSizeW;
    uint32_t blockCountH = (outputSizeH + 1) >> 1;
    uint32_t blockCountW = (outputSizeW + 1) >> 1;

    for (DataType idx = threadIdx.x + blockId * blockDim.x; idx < batchSize; idx += blockNum * blockDim.x) {
        CoordPack<DataType> coord;
        coord.nIdx = idx / (blockCountH * blockCountW);
        DataType newIdx = idx - coord.nIdx * (blockCountH * blockCountW);
        coord.hIdx = newIdx / blockCountW;
        coord.wIdx = newIdx - coord.hIdx * blockCountW;
        if (dynamicTilingKey != 0) {
            UpdateDynamicBatchPara(coord, tD, gmParams);
        }

        int32_t paddingSwitch = tD.paddingParam.paddingSwitch;
        int32_t leftPaddingSize = tD.paddingParam.leftPaddingSize;
        int32_t rightPaddingSize = tD.paddingParam.rightPaddingSize;
        int32_t topPaddingSize = tD.paddingParam.topPaddingSize;
        int32_t bottomPaddingSize = tD.paddingParam.bottomPaddingSize;
        float padValue = tD.paddingParam.padValue;

        bool allEvenPadding = (topPaddingSize % DIGIT_2 == 0) && (bottomPaddingSize % DIGIT_2 == 0) &&
                              (leftPaddingSize % DIGIT_2 == 0) && (rightPaddingSize % DIGIT_2 == 0);
        // Block-level padding fast path is exact only when the crop region boundary is 2x2-block aligned
        bool blockAligned = allEvenPadding && (tD.cropParam.cropSizeH % DIGIT_2 == 0) &&
                            (tD.cropParam.cropSizeW % DIGIT_2 == 0);

        RgbPack<DataType> dstRgbIdx[YUV_PER_DEAL_NUM];
        ComputeYuvRgbDstIdx(dstRgbIdx, coord, tD);

        uint32_t actualH = coord.hIdx * DIGIT_2;
        uint32_t actualW = coord.wIdx * DIGIT_2;

        bool blockAllInPadding = (paddingSwitch != 0 && blockAligned) &&
                                 ((actualH < topPaddingSize) || (actualH >= topPaddingSize + tD.cropParam.cropSizeH) ||
                                  (actualW < leftPaddingSize) || (actualW >= leftPaddingSize + tD.cropParam.cropSizeW));

        ProcessYuvRgbBlock<T, DataType>(yuvGM, outputGM, dstRgbIdx, coord, tD, padValue, blockAligned,
                                        blockAllInPadding);
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippYuvRgb<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    uint64_t blockCountH = (this->tilingData_.outputSizeH + 1) >> 1;
    uint64_t blockCountW = (this->tilingData_.outputSizeW + 1) >> 1;
    uint64_t batchSize = static_cast<uint64_t>(this->tilingData_.batchNum) * blockCountH * blockCountW;

    asc_vf_call<Aipp_Kernel::SimtComputeYuvRgb<T, DataType>>(dim3(this->blockDimX_), (__gm__ uint8_t*)x, (__gm__ T*)y,
                                                             this->tilingData_, this->gmParams_, this->blockIdx_,
                                                             this->blockNum_, batchSize, this->dynamicTilingKey_);
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_YUV_RGB_H
