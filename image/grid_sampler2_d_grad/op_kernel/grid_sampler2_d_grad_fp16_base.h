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
 * \file grid_sampler2_d_grad_fp16_base.h
 * \brief
 */

#ifndef _ASCENDC_GRID_SAMPLER2_D_GRAD_FP16_BASE_H_
#define _ASCENDC_GRID_SAMPLER2_D_GRAD_FP16_BASE_H_

#include "kernel_operator.h"

using namespace AscendC;

template <typename T, typename Dtype, typename GridSamplerGradTilingData>
class GridSampler2DGradFP16 {
public:
    __aicore__ inline GridSampler2DGradFP16(){};
    __aicore__ inline void Init(const GridSamplerGradTilingData& __restrict tilingData,
                                GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1]);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void InitBilinearLocalTensor();
    __aicore__ inline void InitNearestLocalTensor();
    __aicore__ inline void CopyOut(const int32_t offset, const int32_t copyElemCount);
    __aicore__ inline void CopyIn(const int64_t offset, const int32_t copyElemCount, const int32_t inputTensorIndex);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(const int32_t gridCoordElemCount, const int64_t curGridPointIndex);
    __aicore__ inline void ComputeWeight(LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2,
                                         LocalTensor<T> yCoorTensor1, LocalTensor<T> yCoorTensor2,
                                         const int32_t weightElemCount);
    __aicore__ inline void ComputeIndex(LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> dstIndex2,
                                        LocalTensor<int32_t> yCoor, LocalTensor<int32_t> xCoor,
                                        const int32_t indexElemCount);
    __aicore__ inline void ComputeSourceIndexSetGrad(LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size,
                                                     const int32_t coordElemCount);
    __aicore__ inline void ReflectCoordinatesCommon(LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size,
                                                    int32_t reflectElemCount);
    __aicore__ inline void ClipCoordinatesSetGrad(LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size,
                                                  int32_t clipElemCount);
    __aicore__ inline void ReflectCoordinatesSetGrad(LocalTensor<T> dataTensor, LocalTensor<T> dupTensor,
                                                     LocalTensor<T> tmpDataTensor, LocalTensor<T> tmpDupTensor,
                                                     LocalTensor<int32_t> tmpIntTensor, LocalTensor<T> extraTensor,
                                                     LocalTensor<T> flipTensor, int64_t twiceLow, int64_t twiceHigh,
                                                     int32_t reflectElemCount);
    __aicore__ inline void ComputeAfterTransposeGridGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> yCoor1,
                                                         LocalTensor<T> yCoor2, LocalTensor<T> xCoor1,
                                                         LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor,
                                                         LocalTensor<T> yIndex, LocalTensor<T> xIndex,
                                                         const int32_t coordElemIndex, const int32_t batchIdx);
    __aicore__ inline void ComputeAfterTransposeXGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> weight,
                                                      const int32_t coordElemIndex, const int64_t ncOffset,
                                                      LocalTensor<T> gOutLocalTensor, LocalTensor<T> yIndex,
                                                      LocalTensor<T> xIndex);
    __aicore__ inline void ComputeNearestXGrad(LocalTensor<int32_t> srcIndex, LocalTensor<T> weight,
                                               const int32_t coordElemIndex, const int32_t cycle,
                                               const int64_t ncOffset, LocalTensor<T> gOutLocalTensor,
                                               LocalTensor<T> yIndex, LocalTensor<T> xIndex);
    __aicore__ inline void WithinBounds2d(LocalTensor<T> dst, LocalTensor<T> iyT, LocalTensor<T> ixT,
                                          LocalTensor<T> weight, const int32_t calCount);
    __aicore__ inline void DupValue();
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    };

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> dataInQueue[INPUT_NUM];
    TQue<QuePosition::VECOUT, BUFFER_NUM> dataOutQueue[OUTPUT_NUM];
    TBuf<TPosition::VECCALC> xCoordinateBuf;
    TBuf<TPosition::VECCALC> inputCoordinateBuf;
    TBuf<TPosition::VECCALC> yCoordinateBuf;
    TBuf<TPosition::VECCALC> xGradInBuf;
    TBuf<TPosition::VECCALC> yGradInBuf;

    TBuf<TPosition::VECCALC> ixNwBuf;
    TBuf<TPosition::VECCALC> iyNwBuf;
    TBuf<TPosition::VECCALC> ixNeBuf;
    TBuf<TPosition::VECCALC> iyNeBuf;
    TBuf<TPosition::VECCALC> ixSwBuf;
    TBuf<TPosition::VECCALC> iySwBuf;
    TBuf<TPosition::VECCALC> ixSeBuf;
    TBuf<TPosition::VECCALC> iySeBuf;

    TBuf<TPosition::VECCALC> nwBuf;
    TBuf<TPosition::VECCALC> neBuf;
    TBuf<TPosition::VECCALC> swBuf;
    TBuf<TPosition::VECCALC> seBuf;

    TBuf<TPosition::VECCALC> ixNwIntBuf;
    TBuf<TPosition::VECCALC> iyNwIntBuf;

    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;

    TBuf<TPosition::VECCALC> ixNeIntBuf;
    TBuf<TPosition::VECCALC> iyNeIntBuf;
    TBuf<TPosition::VECCALC> ixSwIntBuf;
    TBuf<TPosition::VECCALC> iySwIntBuf;
    TBuf<TPosition::VECCALC> ixSeIntBuf;
    TBuf<TPosition::VECCALC> iySeIntBuf;

    TBuf<TPosition::VECCALC> mask1Buf;
    TBuf<TPosition::VECCALC> mask2Buf;
    TBuf<TPosition::VECCALC> mask3Buf;

    TBuf<TPosition::VECCALC> dupOneBuf;
    TBuf<TPosition::VECCALC> selBuf1;
    TBuf<TPosition::VECCALC> selBuf2;
    TBuf<TPosition::VECCALC> selBuf3;
    TBuf<TPosition::VECCALC> selBuf4;

    TBuf<TPosition::VECCALC> tmp5Buf;
    TBuf<TPosition::VECCALC> tmp6Buf;
    TBuf<TPosition::VECCALC> tmp7Buf;
    TBuf<TPosition::VECCALC> tmp8Buf;
    TBuf<TPosition::VECCALC> tmp9Buf;

    TBuf<TPosition::VECCALC> computeIndexBuf;
    TBuf<TPosition::VECCALC> computeIndexBuf1;
    TBuf<TPosition::VECCALC> computeIndexBuf2;

    TBuf<TPosition::VECCALC> gixBuf;
    TBuf<TPosition::VECCALC> giyBuf;
    TBuf<TPosition::VECCALC> sumXBuf;
    TBuf<TPosition::VECCALC> sumYBuf;
    TBuf<TPosition::VECCALC> ixNearIntBuf;
    TBuf<TPosition::VECCALC> iyNearIntBuf;
    TBuf<TPosition::VECCALC> ixFloatBuf;
    TBuf<TPosition::VECCALC> iyFloatBuf;
    TBuf<TPosition::VECCALC> clipLimitBuf;

    TBuf<TPosition::VECCALC> xTensorBuf;
    TBuf<TPosition::VECCALC> yTensorBuf;
    TBuf<TPosition::VECCALC> inputXLocalTensorBuf;
    TBuf<TPosition::VECCALC> localTensor1Buf;
    TBuf<TPosition::VECCALC> localTensor2Buf;
    TBuf<TPosition::VECCALC> gOutLocalTensorBuf;
    TBuf<TPosition::VECCALC> gOutLocalTensor2Buf;
    TBuf<TPosition::VECCALC> dstLocalBuf;
    GlobalTensor<Dtype> inputGm[INPUT_NUM + OUTPUT_NUM];
    GlobalTensor<T> inputGmT;

    uint32_t batch = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tailPNum = 0;
    int32_t channel = 0;
    int32_t alignChannel = 0;
    int32_t height = 0;
    int32_t width = 0;
    T fheight = 0;
    T fwidth = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t interpolation = 0; // 0:Bilinear, 1:Nearest
    uint32_t padding = 0;       // 0:Zeros, 1:Border
    uint32_t alignCorners = 0;  // 0:False, 1:True
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t outH = 0;
    uint32_t outW = 0;
    uint32_t perBlockCount = 0;
    uint32_t blockIdx = 0;
    uint32_t dataCount = 0;
    uint32_t batchOffset = 0;
    uint32_t baseOffset = 0;
    uint32_t alignBufferNum = 0;
    uint32_t dxStrideN = 0;
    uint32_t dxStrideC = 0;
    int32_t dxStrideH = 0;
    uint32_t dxStrideW = 0;
    uint32_t gradStrideH = 0;
    uint32_t gradStrideW = 0;
    uint32_t maskSize = 0;
    uint32_t maskNum = 0;
    int32_t inputStrideH = 0;
    uint32_t inputStrideW = 0;
    uint32_t inputStrideN = 0;
    int64_t pointIndex = 0;
    int64_t baseGradGmOffset = 0;
    int64_t gradGmOffset = 0;
    int64_t baseGmOffset = 0;
    int32_t pointOffset = 0;
    int64_t xGmOffset = 0;
    int32_t ncOffset = 0;
    int32_t group = 0;
    T gix = static_cast<T>(0);
    T giy = static_cast<T>(0);
    RoundMode mode;

    LocalTensor<uint8_t> mask1Tensor;
    LocalTensor<uint8_t> mask2Tensor;
    LocalTensor<uint8_t> mask3Tensor;
    LocalTensor<uint16_t> int8ToInt16Mask1;
    LocalTensor<uint16_t> int8ToInt16Mask2;
    LocalTensor<T> dupOneTensor;
    LocalTensor<T> selTensor1;
    LocalTensor<T> selTensor2;
    LocalTensor<T> selTensor3;
    LocalTensor<T> selTensor4;
    LocalTensor<T> tmp1Tensor;
    LocalTensor<T> tmp2Tensor;
    LocalTensor<int32_t> tmpIndex;

    LocalTensor<T> gixLocalTensor;
    LocalTensor<T> giyLocalTensor;
    LocalTensor<T> sumX;
    LocalTensor<T> sumY;
    LocalTensor<T> clipLimit;
};

#endif // GRID_SAMPLER_2D_GRAD_H_
