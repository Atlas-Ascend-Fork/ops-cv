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
 * \file aipp_yuv.h
 * \brief AIPP YUV420SP to YUV444/YVU444 upsampling kernel.
 *
 * TilingKey = FORMAT_YUV_INDICE_UINT32 (2), csc_switch = false.
 *
 * Algorithm (nearest-neighbour UV upsampling):
 *   Y444(x, y) = Y420(x, y)
 *   U444(x, y) = U420(x/2, y/2)
 *   V444(x, y) = V420(x/2, y/2)
 *
 * YVU444: rbuv_swap_switch=true → tiling side swaps CSC matrix columns 1&2,
 *         kernel is format-agnostic.
 * Thread model: one thread per 2x2 block (4 pixels share one UV pair).
 */

#ifndef AIPP_OP_KERNEL_ARCH35_YUV_H
#define AIPP_OP_KERNEL_ARCH35_YUV_H

#include "aipp_base.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippYuv : public AippBase<T, DataType> {
public:
    __aicore__ inline AippYuv(){};
    __aicore__ inline void Init(const AippTilingData& tilingData, const tagAippDynamicParaHeader& tilingParamHeader,
                                const __gm__ uint8_t* gmParams, uint8_t dynamicTilingKey);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippYuv<T, DataType>::Init(const AippTilingData& tilingData,
                                                  const tagAippDynamicParaHeader& tilingParamHeader,
                                                  const __gm__ uint8_t* gmParams, uint8_t dynamicTilingKey)
{
    this->BaseInit(tilingData, tilingParamHeader, gmParams, dynamicTilingKey);
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ComputeDstYuv444Idx(DataType dstYIdx[YUV_PER_DEAL_NUM],
                                                                               DataType& dstUBase, DataType& dstVBase,
                                                                               const CoordPack<DataType>& coord,
                                                                               const AippTilingData& tD)
{
    const DataType outputSizeH = tD.outputSizeH;
    const DataType outputSizeW = tD.outputSizeW;
    const DataType h0 = coord.hIdx * DIGIT_2;
    const DataType h1 = h0 + 1;
    const DataType w0 = coord.wIdx * DIGIT_2;
    const DataType w1 = w0 + 1;

    if (tD.outputFormat == NCHW_FORMAT_INDEX) {
        const DataType yPlaneBase = coord.nIdx * CHANNEL_THREE * outputSizeH * outputSizeW;
        dstYIdx[YUV_DEAL_NUM_0] = yPlaneBase + h0 * outputSizeW + w0;
        dstYIdx[YUV_DEAL_NUM_1] = yPlaneBase + h0 * outputSizeW + w1;
        dstYIdx[YUV_DEAL_NUM_2] = yPlaneBase + h1 * outputSizeW + w0;
        dstYIdx[YUV_DEAL_NUM_3] = yPlaneBase + h1 * outputSizeW + w1;
        dstUBase = yPlaneBase + outputSizeH * outputSizeW + h0 * outputSizeW + w0;
        dstVBase = yPlaneBase + DIGIT_2 * outputSizeH * outputSizeW + h0 * outputSizeW + w0;
    } else {
        // NHWC: each pixel triplet is [Y, ch1, ch2]
        const DataType pixelBase = coord.nIdx * outputSizeH * outputSizeW * CHANNEL_THREE;
        dstYIdx[YUV_DEAL_NUM_0] = pixelBase + (h0 * outputSizeW + w0) * CHANNEL_THREE;
        dstYIdx[YUV_DEAL_NUM_1] = pixelBase + (h0 * outputSizeW + w1) * CHANNEL_THREE;
        dstYIdx[YUV_DEAL_NUM_2] = pixelBase + (h1 * outputSizeW + w0) * CHANNEL_THREE;
        dstYIdx[YUV_DEAL_NUM_3] = pixelBase + (h1 * outputSizeW + w1) * CHANNEL_THREE;
        dstUBase = dstYIdx[YUV_DEAL_NUM_0] + 1;
        dstVBase = dstYIdx[YUV_DEAL_NUM_0] + DIGIT_2;
    }
}

template <typename T, typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ProcessYuv444Pixel(__gm__ uint8_t* inputGM,
                                                                              __gm__ T* outputGM, DataType dstYIdx,
                                                                              DataType dstUIdx, DataType dstVIdx,
                                                                              DataType nIdx, uint32_t croodH,
                                                                              uint32_t croodW, const AippTilingData& tD)
{
    const uint64_t yuvPlaneSize = static_cast<uint64_t>(tD.inputSizeH) * tD.inputSizeW * DIGIT_3 / DIGIT_2;

    uint64_t srcYIdx = static_cast<uint64_t>(nIdx) * yuvPlaneSize +
                       (tD.cropParam.cropStartPosH + croodH) * tD.inputSizeW + (tD.cropParam.cropStartPosW + croodW);
    uint64_t srcUIdx = static_cast<uint64_t>(nIdx) * yuvPlaneSize + tD.inputSizeH * tD.inputSizeW +
                       ((tD.cropParam.cropStartPosH + (croodH & ~1u)) >> 1) * tD.inputSizeW +
                       (tD.cropParam.cropStartPosW + (croodW & ~1u));
    uint64_t srcVIdx = srcUIdx + 1;

    RgbPack<uint8_t> dstYuv;
    ApplyCscMatrix(dstYuv, inputGM[srcYIdx], inputGM[srcUIdx], inputGM[srcVIdx], tD.cscParam);

    DataConversion(outputGM[dstYIdx], dstYuv.r, tD.dtcParam, CHANNEL_NUM_0);
    DataConversion(outputGM[dstUIdx], dstYuv.g, tD.dtcParam, CHANNEL_NUM_1);
    DataConversion(outputGM[dstVIdx], dstYuv.b, tD.dtcParam, CHANNEL_NUM_2);
}

template <typename T, typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ProcessYuv444Block(
    __gm__ uint8_t* inputGM, __gm__ T* outputGM, const DataType dstYIdx[YUV_PER_DEAL_NUM], DataType dstUBase,
    DataType dstVBase, const CoordPack<DataType>& coord, const AippTilingData& tD, float padValue, bool allEvenPadding,
    bool blockAllInPadding)
{
    const uint32_t actualH = coord.hIdx * DIGIT_2;
    const uint32_t actualW = coord.wIdx * DIGIT_2;

    DataType dstUIdx[YUV_PER_DEAL_NUM];
    DataType dstVIdx[YUV_PER_DEAL_NUM];
    if (tD.outputFormat == NCHW_FORMAT_INDEX) {
        const DataType outputW = tD.outputSizeW;
        dstUIdx[YUV_DEAL_NUM_0] = dstUBase;
        dstUIdx[YUV_DEAL_NUM_1] = dstUBase + 1;
        dstUIdx[YUV_DEAL_NUM_2] = dstUBase + outputW;
        dstUIdx[YUV_DEAL_NUM_3] = dstUBase + outputW + 1;
        dstVIdx[YUV_DEAL_NUM_0] = dstVBase;
        dstVIdx[YUV_DEAL_NUM_1] = dstVBase + 1;
        dstVIdx[YUV_DEAL_NUM_2] = dstVBase + outputW;
        dstVIdx[YUV_DEAL_NUM_3] = dstVBase + outputW + 1;
    } else {
        const DataType rowStride = tD.outputSizeW * CHANNEL_THREE;
        dstUIdx[YUV_DEAL_NUM_0] = dstUBase;
        dstUIdx[YUV_DEAL_NUM_1] = dstUBase + CHANNEL_THREE;
        dstUIdx[YUV_DEAL_NUM_2] = dstUBase + rowStride;
        dstUIdx[YUV_DEAL_NUM_3] = dstUBase + rowStride + CHANNEL_THREE;
        dstVIdx[YUV_DEAL_NUM_0] = dstVBase;
        dstVIdx[YUV_DEAL_NUM_1] = dstVBase + CHANNEL_THREE;
        dstVIdx[YUV_DEAL_NUM_2] = dstVBase + rowStride;
        dstVIdx[YUV_DEAL_NUM_3] = dstVBase + rowStride + CHANNEL_THREE;
    }

    for (uint8_t i = 0; i < YUV_PER_DEAL_NUM; i++) {
        uint32_t pixelH = (i >= YUV_DEAL_NUM_2) ? actualH + 1 : actualH;
        uint32_t pixelW = (i == YUV_DEAL_NUM_1 || i == YUV_DEAL_NUM_3) ? actualW + 1 : actualW;
        if (pixelH >= tD.outputSizeH || pixelW >= tD.outputSizeW) {
            continue;
        }
        if (IsPixelInPaddingForYuv(pixelH, pixelW, tD, allEvenPadding, blockAllInPadding)) {
            AssignPadValue(outputGM[dstYIdx[i]], padValue);
            AssignPadValue(outputGM[dstUIdx[i]], padValue);
            AssignPadValue(outputGM[dstVIdx[i]], padValue);
        } else {
            uint32_t cropCoordH = pixelH - tD.paddingParam.topPaddingSize;
            uint32_t cropCoordW = pixelW - tD.paddingParam.leftPaddingSize;
            ProcessYuv444Pixel<T, DataType>(inputGM, outputGM, dstYIdx[i], dstUIdx[i], dstVIdx[i], coord.nIdx,
                                            cropCoordH, cropCoordW, tD);
        }
    }
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__
    void SimtComputeYuv420ToYuv444(__gm__ uint8_t* inputGM, __gm__ T* outputGM, AippTilingData tD,
                                   const __gm__ uint8_t* gmParams, uint32_t blockId, uint32_t blockNum,
                                   uint64_t batchSize, uint8_t dynamicTilingKey)
{
    const uint32_t outputSizeH = tD.outputSizeH;
    const uint32_t outputSizeW = tD.outputSizeW;
    const uint32_t blockCountH = (outputSizeH + 1) >> 1;
    const uint32_t blockCountW = (outputSizeW + 1) >> 1;

    for (DataType idx = threadIdx.x + blockId * blockDim.x; idx < batchSize; idx += blockNum * blockDim.x) {
        // 1. Decode idx → 2x2-block coordinate (nIdx, hIdx, wIdx)
        CoordPack<DataType> coord;
        coord.nIdx = idx / (blockCountH * blockCountW);
        DataType rem = idx - coord.nIdx * (blockCountH * blockCountW);
        coord.hIdx = rem / blockCountW;
        coord.wIdx = rem - coord.hIdx * blockCountW;
        if (dynamicTilingKey != 0) {
            UpdateDynamicBatchPara(coord, tD, gmParams);
        }

        const int32_t topPaddingSize = tD.paddingParam.topPaddingSize;
        const int32_t bottomPaddingSize = tD.paddingParam.bottomPaddingSize;
        const int32_t leftPaddingSize = tD.paddingParam.leftPaddingSize;
        const int32_t rightPaddingSize = tD.paddingParam.rightPaddingSize;
        const float padValue = tD.paddingParam.padValue;
        const bool allEvenPadding = (topPaddingSize % DIGIT_2 == 0) && (bottomPaddingSize % DIGIT_2 == 0) &&
                                    (leftPaddingSize % DIGIT_2 == 0) && (rightPaddingSize % DIGIT_2 == 0);
        // Block-level padding fast path is exact only when the crop region boundary is 2x2-block aligned
        const bool blockAligned = allEvenPadding && (tD.cropParam.cropSizeH % DIGIT_2 == 0) &&
                                  (tD.cropParam.cropSizeW % DIGIT_2 == 0);

        // 2. Compute destination indices for this 2x2 block
        DataType dstYIdx[YUV_PER_DEAL_NUM];
        DataType dstUBase, dstVBase;
        ComputeDstYuv444Idx<DataType>(dstYIdx, dstUBase, dstVBase, coord, tD);

        // 3. Fast-path: check if entire 2x2 block is in padding
        const uint32_t actualH = coord.hIdx * DIGIT_2;
        const uint32_t actualW = coord.wIdx * DIGIT_2;
        const bool blockAllInPadding = (tD.paddingParam.paddingSwitch != 0 && blockAligned) &&
                                       ((actualH < topPaddingSize) ||
                                        (actualH >= topPaddingSize + tD.cropParam.cropSizeH) ||
                                        (actualW < leftPaddingSize) ||
                                        (actualW >= leftPaddingSize + tD.cropParam.cropSizeW));

        // 4. Process block (padding check + CSC + DataConversion per pixel)
        ProcessYuv444Block<T, DataType>(inputGM, outputGM, dstYIdx, dstUBase, dstVBase, coord, tD, padValue,
                                        blockAligned, blockAllInPadding);
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippYuv<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    uint64_t blockCountH = (static_cast<uint64_t>(this->tilingData_.outputSizeH) + 1) >> 1;
    uint64_t blockCountW = (static_cast<uint64_t>(this->tilingData_.outputSizeW) + 1) >> 1;
    uint64_t batchSize = (uint64_t)this->tilingData_.batchNum * blockCountH * blockCountW;

    asc_vf_call<Aipp_Kernel::SimtComputeYuv420ToYuv444<T, DataType>>(
        dim3(this->blockDimX_), (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->gmParams_, this->blockIdx_,
        this->blockNum_, batchSize, this->dynamicTilingKey_);
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_YUV_H
