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
 * \file grid_sampler2_d_grad_fp16.h
 * \brief
 */

#ifndef GRID_SAMPLER_2D_GRAD_FP16_H_
#define GRID_SAMPLER_2D_GRAD_FP16_H_

#include "grid_sampler2_d_grad_fp16_base.h"

template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::Init(
    const GridSamplerGradTilingData& __restrict tilingData, GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM + 1])
{
    batch = tilingData.batch;
    pNumPerCore = tilingData.pNumPerCore;
    tailPNum = tilingData.tailPNum;
    channel = tilingData.channel;
    height = tilingData.height;
    width = tilingData.width;
    fheight = static_cast<T>(height);
    fwidth = static_cast<T>(width);
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
    interpolation = tilingData.interpolation;
    padding = tilingData.padding;
    alignCorners = tilingData.alignCorners;
    group = tilingData.group;
    gridH = tilingData.gridH;
    gridW = tilingData.gridW;
    outW = gridW;
    outH = gridH;
    dataCount = gridH * gridW;
    maskSize = CeilAlign(CeilDiv(ubFactorElement, 8), BLOCK_BYTES);
    maskNum = maskSize / sizeof(uint8_t);
    dxStrideN = channel * width * height;
    dxStrideC = width * height;
    dxStrideH = width;
    dxStrideW = 1;

    inputStrideH = width;
    inputStrideW = 1;
    inputStrideN = channel * width * height;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(Dtype);
    alignChannel = CeilAlign(channel, perBlockCount);
    mode = RoundMode::CAST_NONE;

    inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ Dtype*>(inputTensors[GRAD_INPUT_INDEX]));
    inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ Dtype*>(inputTensors[X_INPUT_INDEX]));
    inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ Dtype*>(inputTensors[GRID_INPUT_INDEX]));
    inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ Dtype*>(inputTensors[DX_INPUT_INDEX]));
    inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ Dtype*>(inputTensors[DGRID_INPUT_INDEX]));
    inputGmT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputTensors[WORKSPACE_INPUT_INDEX]));
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    // Bilinear branch
    if (interpolation == 0) {
        pipe->InitBuffer(dataInQueue[0], BUFFER_NUM, alignChannel * sizeof(Dtype));
        pipe->InitBuffer(dataInQueue[1], BUFFER_NUM, alignChannel * sizeof(Dtype));
        pipe->InitBuffer(dataInQueue[2], BUFFER_NUM, 2 * ubFactorElement * sizeof(Dtype));
        pipe->InitBuffer(dataOutQueue[0], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataOutQueue[1], BUFFER_NUM, 2 * ubFactorElement * sizeof(Dtype));

        pipe->InitBuffer(xCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(yCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(xGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(yGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixNwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixNeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySeBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixNwIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNwIntBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(nwBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(neBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(swBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(seBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(tmp1Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp2Buf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(ixNeIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyNeIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSwIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySwIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixSeIntBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iySeIntBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(mask1Buf, maskSize);
        pipe->InitBuffer(mask2Buf, maskSize);
        pipe->InitBuffer(mask3Buf, maskSize);

        pipe->InitBuffer(dupOneBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf1, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf2, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf3, ubFactorElement * sizeof(T));
        pipe->InitBuffer(selBuf4, ubFactorElement * sizeof(T));

        pipe->InitBuffer(tmp5Buf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(tmp6Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp7Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp8Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp9Buf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(computeIndexBuf1, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf2, ubFactorElement * sizeof(int32_t) * 8);

        pipe->InitBuffer(gixBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(giyBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumXBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(sumYBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(clipLimitBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(inputXLocalTensorBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(localTensor1Buf, alignChannel * sizeof(T));
        pipe->InitBuffer(gOutLocalTensorBuf, alignChannel * sizeof(T));
        pipe->InitBuffer(dstLocalBuf, (2 * ubFactorElement) * sizeof(T));
        pipe->InitBuffer(inputCoordinateBuf, (ubFactorElement * 2) * sizeof(T));
    }
    // nearest branch
    if (interpolation == 1) {
        pipe->InitBuffer(dataInQueue[0], BUFFER_NUM, alignChannel * sizeof(Dtype) * group);
        pipe->InitBuffer(dataInQueue[2], BUFFER_NUM, 2 * ubFactorElement * sizeof(Dtype));
        pipe->InitBuffer(dataOutQueue[0], BUFFER_NUM, alignChannel * sizeof(T));
        pipe->InitBuffer(dataOutQueue[1], BUFFER_NUM, 2 * ubFactorElement * sizeof(Dtype));
        pipe->InitBuffer(xCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(yCoordinateBuf, (ubFactorElement + ELE_NUM_PER_REPEAT) * sizeof(T));
        pipe->InitBuffer(xGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(yGradInBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(ixFloatBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(iyFloatBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(computeIndexBuf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf1, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(ixNearIntBuf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(iyNearIntBuf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(selBuf1, ubFactorElement * sizeof(T));
        pipe->InitBuffer(mask1Buf, maskSize);
        pipe->InitBuffer(mask2Buf, maskSize);
        pipe->InitBuffer(mask3Buf, maskSize);

        pipe->InitBuffer(dupOneBuf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(clipLimitBuf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(tmp5Buf, ubFactorElement * sizeof(int32_t));
        pipe->InitBuffer(tmp6Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp7Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp8Buf, ubFactorElement * sizeof(T));
        pipe->InitBuffer(tmp9Buf, ubFactorElement * sizeof(T));

        pipe->InitBuffer(gOutLocalTensor2Buf, alignChannel * sizeof(T) * group);
        pipe->InitBuffer(localTensor2Buf, alignChannel * sizeof(T));
        pipe->InitBuffer(dstLocalBuf, (2 * ubFactorElement) * sizeof(T));
        pipe->InitBuffer(inputCoordinateBuf, (ubFactorElement * 2) * sizeof(T));
    }
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::InitBilinearLocalTensor()
{
    mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
    mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);
    mask3Tensor = mask3Buf.Get<uint8_t>(maskNum);
    dupOneTensor = dupOneBuf.Get<T>(ubFactorElement);
    tmpIndex = computeIndexBuf1.Get<int32_t>(ubFactorElement);
    selTensor1 = selBuf1.Get<T>(ubFactorElement);
    tmp1Tensor = tmp1Buf.Get<T>(ubFactorElement);
    tmp2Tensor = tmp2Buf.Get<T>(ubFactorElement);
    selTensor2 = selBuf2.Get<T>(ubFactorElement);
    selTensor3 = selBuf3.Get<T>(ubFactorElement);
    selTensor4 = selBuf4.Get<T>(ubFactorElement);
    sumX = sumXBuf.Get<T>(alignChannel);
    sumY = sumYBuf.Get<T>(alignChannel);
    clipLimit = clipLimitBuf.Get<T>(ubFactorElement);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::InitNearestLocalTensor()
{
    mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
    mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);
    mask3Tensor = mask3Buf.Get<uint8_t>(maskNum);
    dupOneTensor = dupOneBuf.Get<T>(ubFactorElement);
    tmpIndex = computeIndexBuf1.Get<int32_t>(ubFactorElement);
    selTensor1 = selBuf1.Get<T>(ubFactorElement);
    clipLimit = clipLimitBuf.Get<T>(ubFactorElement);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ComputeSourceIndexSetGrad(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, const int32_t coordElemCount)
{
    if (alignCorners == 1) {
        T val = static_cast<T>(size - 1) / 2;
        Duplicate<T>(dupTensor, val, coordElemCount);
        Adds(dataTensor, dataTensor, static_cast<T>(1), coordElemCount);
        Muls(dataTensor, dataTensor, static_cast<T>(0.5), coordElemCount);
        Muls(dataTensor, dataTensor, static_cast<T>(size - 1), coordElemCount);
    } else {
        T val = static_cast<T>(size) / 2;
        Duplicate<T>(dupTensor, val, coordElemCount);
        Adds(dataTensor, dataTensor, static_cast<T>(1), coordElemCount);
        Muls(dataTensor, dataTensor, static_cast<T>(size), coordElemCount);
        Adds(dataTensor, dataTensor, static_cast<T>(-1), coordElemCount);
        Muls(dataTensor, dataTensor, static_cast<T>(0.5), coordElemCount);
    }
    int32_t alignedElemCount = ((coordElemCount * FLOAT_BYTES - 1 + ALGIN_256_BYTES) / ALGIN_256_BYTES *
                                ALGIN_256_BYTES) /
                               4;
    if (padding == 1) {
        // NAN NE NAN == TRUE, mask3是NAN的mask
        Compare(mask3Tensor, dataTensor, dataTensor, CMPMODE::EQ, alignedElemCount);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask1Tensor, dataTensor, static_cast<T>(0), CMPMODE::GT, alignedElemCount);
        Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               alignedElemCount);
        Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               alignedElemCount);
        CompareScalar(mask1Tensor, dataTensor, static_cast<T>(size - 1), CMPMODE::LT, alignedElemCount);
        Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(size - 1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               alignedElemCount);
        Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               alignedElemCount);
    } else if (padding == 2) {
        LocalTensor<T> tmpDataTensor = tmp6Buf.Get<T>();
        Muls(tmpDataTensor, dataTensor, (float)(0.0), alignedElemCount);
        PipeBarrier<PIPE_V>();
        Compare(mask3Tensor, tmpDataTensor, tmpDataTensor, CMPMODE::EQ, alignedElemCount);
        PipeBarrier<PIPE_V>();
        ReflectCoordinatesCommon(dataTensor, dupTensor, size, alignedElemCount);
    }
    // If the data is inf/-inf/nan, convert the data to -100.
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MAX - 1), CMPMODE::LE, alignedElemCount);
    PipeBarrier<PIPE_V>();
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           alignedElemCount);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(INT_MIN), CMPMODE::GE, alignedElemCount);
    PipeBarrier<PIPE_V>();
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           alignedElemCount);
    PipeBarrier<PIPE_V>();
    Compare(mask1Tensor, dataTensor, dataTensor, CMPMODE::EQ, alignedElemCount);
    PipeBarrier<PIPE_V>();
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           alignedElemCount);
    PipeBarrier<PIPE_V>();
    if (padding == 1 || padding == 2) {
        // nan输入重新赋值为-100
        Select(dataTensor, mask3Tensor, dataTensor, static_cast<T>(-100.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               alignedElemCount);
        PipeBarrier<PIPE_V>();
    }
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ReflectCoordinatesCommon(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, int32_t reflectElemCount)
{
    // init Reflect localTensor
    LocalTensor<int32_t> reflectIntScratch = tmp5Buf.Get<int32_t>();
    LocalTensor<T> tmpDataTensor = tmp6Buf.Get<T>();
    LocalTensor<T> extraTensor = tmp7Buf.Get<T>();
    LocalTensor<T> flipTensor = tmp8Buf.Get<T>();
    LocalTensor<T> reflectSignScratch = tmp9Buf.Get<T>();

    if (alignCorners == 1) {
        int64_t twiceLow = 0;
        int64_t twiceHigh = 2 * (size - 1);
        ReflectCoordinatesSetGrad(dataTensor, dupTensor, tmpDataTensor, reflectSignScratch, reflectIntScratch,
                                  extraTensor, flipTensor, twiceLow, twiceHigh, reflectElemCount);
    } else {
        int64_t twiceLow = -1;
        int64_t twiceHigh = 2 * size - 1;
        ReflectCoordinatesSetGrad(dataTensor, dupTensor, tmpDataTensor, reflectSignScratch, reflectIntScratch,
                                  extraTensor, flipTensor, twiceLow, twiceHigh, reflectElemCount);
    }
    ClipCoordinatesSetGrad(dataTensor, dupTensor, size, reflectElemCount);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ClipCoordinatesSetGrad(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, const T size, int32_t clipElemCount)
{
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(0), CMPMODE::GT, clipElemCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, clipElemCount);
    Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, clipElemCount);

    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(size - 1), CMPMODE::LT, clipElemCount);
    Select(dataTensor, mask1Tensor, dataTensor, static_cast<T>(size - 1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           clipElemCount);
    Select(dupTensor, mask1Tensor, dupTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, clipElemCount);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ReflectCoordinatesSetGrad(
    LocalTensor<T> dataTensor, LocalTensor<T> dupTensor, LocalTensor<T> tmpDataTensor, LocalTensor<T> tmpDupTensor,
    LocalTensor<int32_t> tmpIntTensor, LocalTensor<T> extraTensor, LocalTensor<T> flipTensor, int64_t twiceLow,
    int64_t twiceHigh, int32_t reflectElemCount)
{
    if (twiceLow == twiceHigh) {
        Duplicate(dataTensor, (float)0.0, reflectElemCount);
        Duplicate(dupTensor, (float)0.0, reflectElemCount);
        return;
    }

    T min = static_cast<T>(twiceLow) / 2;
    T span = static_cast<T>(twiceHigh - twiceLow) / 2;

    Duplicate(tmpDupTensor, (float)(1.0), reflectElemCount);
    Adds(tmpDataTensor, dataTensor, static_cast<T>(-1) * min, reflectElemCount);
    Abs(tmpDataTensor, tmpDataTensor, reflectElemCount);
    CompareScalar(mask1Tensor, dataTensor, static_cast<T>(min), CMPMODE::GE, reflectElemCount);
    Select(tmpDupTensor, mask1Tensor, tmpDupTensor, static_cast<T>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           reflectElemCount);

    // extra = x - Cast(x / span) * span
    Muls(extraTensor, tmpDataTensor, static_cast<T>(static_cast<T>(1.0) / span), reflectElemCount);
    Cast(tmpIntTensor, extraTensor, RoundMode::CAST_FLOOR, reflectElemCount);
    Cast(extraTensor, tmpIntTensor, RoundMode::CAST_NONE, reflectElemCount);
    Muls(extraTensor, extraTensor, span, reflectElemCount);
    Sub(extraTensor, tmpDataTensor, extraTensor, reflectElemCount);

    // flips = floor(x / span)
    Muls(tmpDataTensor, tmpDataTensor, (static_cast<T>(1) / span), reflectElemCount);
    Cast(tmpIntTensor, tmpDataTensor, RoundMode::CAST_FLOOR, reflectElemCount);
    Cast(tmpDataTensor, tmpIntTensor, RoundMode::CAST_NONE, reflectElemCount);

    // flips % 2
    Muls(flipTensor, tmpDataTensor, static_cast<T>(0.5), reflectElemCount);
    Cast(tmpIntTensor, flipTensor, RoundMode::CAST_FLOOR, reflectElemCount);
    Cast(flipTensor, tmpIntTensor, RoundMode::CAST_NONE, reflectElemCount);
    Muls(flipTensor, flipTensor, static_cast<T>(2), reflectElemCount);
    Sub(flipTensor, tmpDataTensor, flipTensor, reflectElemCount);
    CompareScalar(mask2Tensor, flipTensor, static_cast<float>(0.0), CMPMODE::EQ, reflectElemCount);

    // x = extra + min
    Adds(tmpDataTensor, extraTensor, min, reflectElemCount);

    // x = (span - extra) + min
    Muls(extraTensor, extraTensor, static_cast<T>(-1), reflectElemCount);
    Adds(extraTensor, extraTensor, span, reflectElemCount);
    Adds(extraTensor, extraTensor, min, reflectElemCount);

    // flips % 2 == 0 ?
    Select(dataTensor, mask2Tensor, tmpDataTensor, extraTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, reflectElemCount);
    Muls(flipTensor, tmpDupTensor, static_cast<T>(-1), reflectElemCount);
    Select(tmpDupTensor, mask2Tensor, tmpDupTensor, flipTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, reflectElemCount);
    Mul(dupTensor, dupTensor, tmpDupTensor, reflectElemCount);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::WithinBounds2d(
    LocalTensor<T> dst, LocalTensor<T> iyT, LocalTensor<T> ixT, LocalTensor<T> weight, const int32_t calCount)
{
    int32_t alignedElemCount = ((calCount * 4 - 1 + ALGIN_256_BYTES) / ALGIN_256_BYTES * ALGIN_256_BYTES) /
                               4; // align 256 bytes
    if (interpolation == 0) {
        CompareScalar(mask1Tensor, iyT, static_cast<T>(0), CMPMODE::GE, alignedElemCount);
        CompareScalar(mask2Tensor, iyT, static_cast<T>(fheight), CMPMODE::LT, alignedElemCount);
        int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
        int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();

        // Read data according to int16
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(0), CMPMODE::GE, alignedElemCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(fwidth), CMPMODE::LT, alignedElemCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        Select(dst, int8ToInt16Mask2, dupOneTensor, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               alignedElemCount);
    }
    if (interpolation == 1) {
        CompareScalar(mask1Tensor, iyT, static_cast<T>(0), CMPMODE::GE, alignedElemCount);
        CompareScalar(mask2Tensor, iyT, static_cast<T>(fheight), CMPMODE::LT, alignedElemCount);
        int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
        int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();

        // Read data according to int16
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(0), CMPMODE::GE, alignedElemCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<T>(fwidth), CMPMODE::LT, alignedElemCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
    }
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ComputeIndex(
    LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> dstIndex2, LocalTensor<int32_t> yCoor,
    LocalTensor<int32_t> xCoor, const int32_t indexElemCount)
{
    if (interpolation == 0) {
        Mins(yCoor, yCoor, height - 1, indexElemCount);
        Maxs(yCoor, yCoor, 0, indexElemCount);
        Mins(xCoor, xCoor, width - 1, indexElemCount);
        Maxs(xCoor, xCoor, 0, indexElemCount);
        Muls(tmpIndex, yCoor, inputStrideH, indexElemCount);
        Add(dstIndex, tmpIndex, xCoor, indexElemCount);
        Muls(dstIndex, dstIndex, channel, indexElemCount);

        Muls(tmpIndex, yCoor, dxStrideH, indexElemCount);
        Add(dstIndex2, tmpIndex, xCoor, indexElemCount);
        Muls(dstIndex2, dstIndex2, channel, indexElemCount);
    }
    if (interpolation == 1) {
        Mins(yCoor, yCoor, height - 1, indexElemCount);
        Maxs(yCoor, yCoor, 0, indexElemCount);
        Mins(xCoor, xCoor, width - 1, indexElemCount);
        Maxs(xCoor, xCoor, 0, indexElemCount);
        Muls(tmpIndex, yCoor, inputStrideH, indexElemCount);
        Add(dstIndex, tmpIndex, xCoor, indexElemCount);
        Muls(dstIndex, dstIndex, channel, indexElemCount);
    }
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ComputeAfterTransposeGridGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> yCoor1, LocalTensor<T> yCoor2, LocalTensor<T> xCoor1,
    LocalTensor<T> xCoor2, LocalTensor<T> gOutLocalTensor, LocalTensor<T> yIndex, LocalTensor<T> xIndex,
    const int32_t coordElemIndex, const int32_t batchIdx)
{
    int32_t y = yIndex.GetValue(coordElemIndex);
    int32_t x = xIndex.GetValue(coordElemIndex);

    bool inRangeY = y >= 0 && y < height;
    bool inRangeX = x >= 0 && x < width;

    if (!(inRangeY && inRangeX)) {
        return;
    }

    // Get the coordinates of the i-th point
    pointIndex = srcIndex.GetValue(coordElemIndex);
    xGmOffset = batchIdx * inputStrideN + pointIndex;
    T xVal = yCoor1.GetValue(coordElemIndex) - yCoor2.GetValue(coordElemIndex);
    T yVal = xCoor1.GetValue(coordElemIndex) - xCoor2.GetValue(coordElemIndex);

    LocalTensor<Dtype> inputXLocalTempTensor = dataInQueue[1].AllocTensor<Dtype>();
    DataCopyParams inputCopyParams = {1, 0, 0, 0};
    DataCopyPadParams inputPadParams = {true, 0, 0, 0};
    inputCopyParams.blockLen = channel * sizeof(Dtype);
    inputPadParams.rightPadding = alignChannel - channel;
    inputPadParams.paddingValue = GetScalarBitcodeValue((Dtype)0);
    DataCopyPad(inputXLocalTempTensor, inputGm[1][xGmOffset], inputCopyParams, inputPadParams);
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID1);
    WaitFlag<HardEvent::MTE2_V>(eventID1);

    LocalTensor<T> inputXLocalTensor = inputXLocalTensorBuf.Get<T>(alignChannel);
    Cast(inputXLocalTensor, inputXLocalTempTensor, mode, alignChannel);
    Muls(giyLocalTensor, inputXLocalTensor, yVal, channel);
    Mul(giyLocalTensor, gOutLocalTensor, giyLocalTensor, channel);
    Muls(gixLocalTensor, inputXLocalTensor, xVal, channel);
    Mul(gixLocalTensor, gOutLocalTensor, gixLocalTensor, channel);

    Add(sumY, giyLocalTensor, sumY, channel);
    Add(sumX, gixLocalTensor, sumX, channel);
    dataInQueue[1].FreeTensor(inputXLocalTempTensor);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ComputeAfterTransposeXGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coordElemIndex, const int64_t ncOffset,
    LocalTensor<T> gOutLocalTensor, LocalTensor<T> yIndex, LocalTensor<T> xIndex)
{
    int32_t y = yIndex.GetValue(coordElemIndex);
    int32_t x = xIndex.GetValue(coordElemIndex);

    bool inRangeY = y >= 0 && y < height;
    bool inRangeX = x >= 0 && x < width;

    if (!(inRangeX && inRangeY)) {
        return;
    }
    T weightVal = weight.GetValue(coordElemIndex);
    int64_t offset = ncOffset + srcIndex.GetValue(coordElemIndex);
    event_t eventIDMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3_V);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3_V);

    LocalTensor<T> dxLocalTensor = dataOutQueue[0].AllocTensor<T>();
    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID1);
    WaitFlag<HardEvent::S_V>(eventID1);
    Muls(dxLocalTensor, gOutLocalTensor, weightVal, channel);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    DataCopyParams copyParams{1, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);
    SetAtomicAdd<T>();
    DataCopyPad(inputGmT[offset], dxLocalTensor, copyParams);
    SetAtomicNone();
    PipeBarrier<PIPE_MTE3>();
    dataOutQueue[0].FreeTensor(dxLocalTensor);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ComputeNearestXGrad(
    LocalTensor<int32_t> srcIndex, LocalTensor<T> weight, const int32_t coordElemIndex, const int32_t cycle,
    const int64_t ncOffset, LocalTensor<T> gOutLocalTensor, LocalTensor<T> yIndex, LocalTensor<T> xIndex)
{
    int32_t x = xIndex.GetValue(coordElemIndex);
    int32_t y = yIndex.GetValue(coordElemIndex);

    bool inRangeX = x >= 0 && x < width;
    bool inRangeY = y >= 0 && y < height;

    if (!(inRangeX && inRangeY)) {
        return;
    }
    T weightVal = weight.GetValue(coordElemIndex);
    int64_t offset = ncOffset + srcIndex.GetValue(coordElemIndex);

    event_t eventMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV);

    LocalTensor<T> dxLocalTensor = dataOutQueue[0].AllocTensor<T>();
    Muls(dxLocalTensor, gOutLocalTensor[cycle * alignChannel], weightVal, channel);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    DataCopyParams copyParams{1, 0, 0, 0};
    copyParams.blockLen = channel * sizeof(T);
    SetAtomicAdd<T>();
    DataCopyPad(inputGmT[offset], dxLocalTensor, copyParams);
    SetAtomicNone();
    dataOutQueue[0].FreeTensor(dxLocalTensor);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::ComputeWeight(
    LocalTensor<T> dst, LocalTensor<T> xCoorTensor1, LocalTensor<T> xCoorTensor2, LocalTensor<T> yCoorTensor1,
    LocalTensor<T> yCoorTensor2, const int32_t weightElemCount)
{
    Muls(tmp1Tensor, xCoorTensor1, static_cast<T>(-1), weightElemCount);
    Add(tmp1Tensor, xCoorTensor2, tmp1Tensor, weightElemCount);
    Muls(tmp2Tensor, yCoorTensor1, static_cast<T>(-1), weightElemCount);
    Add(tmp2Tensor, yCoorTensor2, tmp2Tensor, weightElemCount);
    Mul(dst, tmp1Tensor, tmp2Tensor, weightElemCount);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::DupValue()
{
    Duplicate<T>(dupOneTensor, 1, ubFactorElement);
    if (interpolation == 0) {
        Duplicate<T>(sumX, 0, alignChannel);
        Duplicate<T>(sumY, 0, alignChannel);
    }
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::CopyIn(
    const int64_t offset, const int32_t copyElemCount, const int32_t inputTensorIndex)
{
    LocalTensor<Dtype> dataLocal = dataInQueue[inputTensorIndex].AllocTensor<Dtype>();
    DataCopyParams copyParams = {1, 0, 0, 0};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    int32_t alignCalCount = CeilAlign(copyElemCount, perBlockCount);
    copyParams.blockLen = copyElemCount * sizeof(Dtype);
    padParams.rightPadding = alignCalCount - copyElemCount;
    padParams.paddingValue = GetScalarBitcodeValue((Dtype)0);

    DataCopyPad(dataLocal, inputGm[inputTensorIndex][offset], copyParams, padParams);
    dataInQueue[inputTensorIndex].EnQue(dataLocal);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::CopyOut(const int32_t offset,
                                                                                           const int32_t copyElemCount)
{
    LocalTensor<Dtype> dstLocal = dataOutQueue[1].DeQue<Dtype>();
    DataCopyParams copyParams{1, 0, 0, 0};
    copyParams.blockLen = copyElemCount * sizeof(Dtype);
    DataCopyPad(inputGm[DGRID_INPUT_INDEX][offset], dstLocal, copyParams);
    dataOutQueue[1].FreeTensor(dstLocal);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::Compute(
    const int32_t gridCoordElemCount, const int64_t curGridPointIndex)
{
    int64_t flatGridPointIndex = 0;
    int64_t gradStrideN = channel * outH * outW;
    int32_t gradStrideH = outW;
    int32_t gradStrideW = 1;
    int64_t w = 0;
    int64_t h = 0;
    int64_t n = 0;
    int64_t ncBaseOffset = 0;
    uint32_t mask = 0;
    uint64_t gatherReservedCount = 0;
    uint8_t gatherXPattern = 1;
    uint8_t gatherYPattern = 2;
    bool reduceMode = false;
    uint8_t src0BlockStride = 1;
    uint16_t gatherRepeatCount = CeilDiv(gridCoordElemCount, ELE_NUM_PER_REPEAT);
    uint8_t src0RepeatStride = REPEAT_STRIDE;
    uint8_t src1RepeatStride = REPEAT_STRIDE;

    LocalTensor<T> gridXTensor = xCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);
    LocalTensor<T> gridYTensor = yCoordinateBuf.Get<T>(ubFactorElement + ELE_NUM_PER_REPEAT);
    LocalTensor<T> xGradIn = xGradInBuf.Get<T>(ubFactorElement);
    LocalTensor<T> yGradIn = yGradInBuf.Get<T>(ubFactorElement);
    LocalTensor<T> inputCoordinate = inputCoordinateBuf.Get<T>(ubFactorElement * 2);
    LocalTensor<Dtype> inputCoordinateTemp = dataInQueue[2].DeQue<Dtype>();
    LocalTensor<Dtype> dstTempLocal = dataOutQueue[1].AllocTensor<Dtype>();
    LocalTensor<T> dstLocal = dstLocalBuf.Get<T>(2 * ubFactorElement);

    Cast(inputCoordinate, inputCoordinateTemp, mode, (ubFactorElement * 2));
    DupValue();
    GatherMask(gridXTensor, inputCoordinate, gatherXPattern, reduceMode, mask,
               {1, gatherRepeatCount, src0RepeatStride, src1RepeatStride}, gatherReservedCount);
    GatherMask(gridYTensor, inputCoordinate, gatherYPattern, reduceMode, mask,
               {1, gatherRepeatCount, src0RepeatStride, src1RepeatStride}, gatherReservedCount);

    // gather ix
    ComputeSourceIndexSetGrad(gridXTensor, xGradIn, fwidth, gridCoordElemCount / 2);

    // gather iy
    ComputeSourceIndexSetGrad(gridYTensor, yGradIn, fheight, gridCoordElemCount / 2);
    if (interpolation == 0) {
        LocalTensor<T> ixNw = ixNwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iyNw = iyNwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ixNe = ixNeBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iyNe = iyNeBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ixSw = ixSwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iySw = iySwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ixSe = ixSeBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iySe = iySeBuf.Get<T>(ubFactorElement);

        LocalTensor<int32_t> ixNwInt = ixNwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iyNwInt = iyNwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> ixNeInt = ixNeIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iyNeInt = iyNeIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> ixSwInt = ixSwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iySwInt = iySwIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> ixSeInt = ixSeIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iySeInt = iySeIntBuf.Get<int32_t>(ubFactorElement);

        LocalTensor<T> nw = nwBuf.Get<T>(ubFactorElement);
        LocalTensor<T> ne = neBuf.Get<T>(ubFactorElement);
        LocalTensor<T> sw = swBuf.Get<T>(ubFactorElement);
        LocalTensor<T> se = seBuf.Get<T>(ubFactorElement);

        LocalTensor<int32_t> nwIndex = computeIndexBuf2.Get<int32_t>();
        LocalTensor<int32_t> neIndex = nwIndex[ubFactorElement * 1];
        LocalTensor<int32_t> swIndex = nwIndex[ubFactorElement * 2];
        LocalTensor<int32_t> seIndex = nwIndex[ubFactorElement * 3];

        LocalTensor<int32_t> nwIndex2 = nwIndex[ubFactorElement * 4];
        LocalTensor<int32_t> neIndex2 = nwIndex[ubFactorElement * 5];
        LocalTensor<int32_t> swIndex2 = nwIndex[ubFactorElement * 6];
        LocalTensor<int32_t> seIndex2 = nwIndex[ubFactorElement * 7];

        gixLocalTensor = gixBuf.Get<T>(alignChannel);
        giyLocalTensor = giyBuf.Get<T>(alignChannel);

        Cast(ixNwInt, gridXTensor, RoundMode::CAST_FLOOR, gridCoordElemCount / 2);
        Cast(iyNwInt, gridYTensor, RoundMode::CAST_FLOOR, gridCoordElemCount / 2);
        Cast(iyNeInt, gridYTensor, RoundMode::CAST_FLOOR, gridCoordElemCount / 2);
        Cast(ixSwInt, gridXTensor, RoundMode::CAST_FLOOR, gridCoordElemCount / 2);
        Adds(ixNeInt, ixNwInt, static_cast<int32_t>(1), gridCoordElemCount / 2);
        Adds(iySwInt, iyNwInt, static_cast<int32_t>(1), gridCoordElemCount / 2);
        Adds(ixSeInt, ixNwInt, static_cast<int32_t>(1), gridCoordElemCount / 2);
        Adds(iySeInt, iyNwInt, static_cast<int32_t>(1), gridCoordElemCount / 2);
        // convert to Float32
        Cast(ixNw, ixNwInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(iyNw, iyNwInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(ixNe, ixNeInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(iyNe, iyNeInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(ixSw, ixSwInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(iySw, iySwInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(ixSe, ixSeInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(iySe, iySeInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        // compute nw
        ComputeWeight(nw, gridXTensor, ixSe, gridYTensor, iySe, gridCoordElemCount / 2);
        // compute ne
        ComputeWeight(ne, ixSw, gridXTensor, gridYTensor, iySw, gridCoordElemCount / 2);
        // compute sw
        ComputeWeight(sw, gridXTensor, ixNe, iyNe, gridYTensor, gridCoordElemCount / 2);
        // compute se
        ComputeWeight(se, ixNw, gridXTensor, iyNw, gridYTensor, gridCoordElemCount / 2);
        WithinBounds2d(selTensor1, iyNw, ixNw, nw, gridCoordElemCount / 2);
        WithinBounds2d(selTensor2, iyNe, ixNe, ne, gridCoordElemCount / 2);
        WithinBounds2d(selTensor3, iySw, ixSw, sw, gridCoordElemCount / 2);
        WithinBounds2d(selTensor4, iySe, ixSe, se, gridCoordElemCount / 2);
        ComputeIndex(nwIndex, nwIndex2, iyNwInt, ixNwInt, gridCoordElemCount / 2);
        ComputeIndex(neIndex, neIndex2, iyNeInt, ixNeInt, gridCoordElemCount / 2);
        ComputeIndex(swIndex, swIndex2, iySwInt, ixSwInt, gridCoordElemCount / 2);
        ComputeIndex(seIndex, seIndex2, iySeInt, ixSeInt, gridCoordElemCount / 2);

        for (int32_t i = 0; i < gridCoordElemCount / 2; i++) {
            flatGridPointIndex = curGridPointIndex + i;
            w = flatGridPointIndex % outW;
            h = (flatGridPointIndex / outW) % outH;
            n = flatGridPointIndex / (outH * outW);

            ncBaseOffset = n * dxStrideN;
            gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
            LocalTensor<Dtype> gOutLocalTempTensor = dataInQueue[0].AllocTensor<Dtype>();
            LocalTensor<T> gOutLocalTensor = gOutLocalTensorBuf.Get<T>(alignChannel);
            DataCopyParams copyParams = {1, 0, 0, 0};
            DataCopyPadParams padParams = {true, 0, 0, 0};
            int32_t alignCalCount = CeilAlign(channel, perBlockCount);
            copyParams.blockLen = channel * sizeof(Dtype);
            padParams.rightPadding = alignCalCount - channel;
            padParams.paddingValue = GetScalarBitcodeValue((Dtype)0);
            DataCopyPad(gOutLocalTempTensor, inputGm[0][gradGmOffset], copyParams, padParams);
            event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventID1);
            WaitFlag<HardEvent::MTE2_V>(eventID1);
            Cast(gOutLocalTensor, gOutLocalTempTensor, mode, alignChannel);

            ComputeAfterTransposeGridGrad(nwIndex, iySe, gridYTensor, ixSe, gridXTensor, gOutLocalTensor, iyNw, ixNw, i,
                                          n);
            ComputeAfterTransposeXGrad(nwIndex2, nw, i, ncBaseOffset, gOutLocalTensor, iyNw, ixNw);
            ComputeAfterTransposeGridGrad(neIndex, gridYTensor, iySw, gridXTensor, ixSw, gOutLocalTensor, iyNe, ixNe, i,
                                          n);
            ComputeAfterTransposeXGrad(neIndex2, ne, i, ncBaseOffset, gOutLocalTensor, iyNe, ixNe);
            ComputeAfterTransposeGridGrad(swIndex, gridYTensor, iyNe, gridXTensor, ixNe, gOutLocalTensor, iySw, ixSw, i,
                                          n);
            ComputeAfterTransposeXGrad(swIndex2, sw, i, ncBaseOffset, gOutLocalTensor, iySw, ixSw);
            ComputeAfterTransposeGridGrad(seIndex, iyNw, gridYTensor, ixNw, gridXTensor, gOutLocalTensor, iySe, ixSe, i,
                                          n);
            ComputeAfterTransposeXGrad(seIndex2, se, i, ncBaseOffset, gOutLocalTensor, iySe, ixSe);

            ReduceSum<T>(sumY, sumY, sumY, channel); // resue sumY as worklocal
            ReduceSum<T>(sumX, sumX, sumX, channel);
            gix -= sumX.GetValue(0);
            giy -= sumY.GetValue(0);
            dstLocal.SetValue(2 * i, gix * xGradIn.GetValue(i));
            dstLocal.SetValue(2 * i + 1, giy * yGradIn.GetValue(i));
            Duplicate<T>(sumX, 0, alignChannel);
            Duplicate<T>(sumY, 0, alignChannel);
            gix = static_cast<T>(0);
            giy = static_cast<T>(0);
            dataInQueue[0].FreeTensor(gOutLocalTempTensor);
        }
    }
    if (interpolation == 1) {
        LocalTensor<int32_t> ixNearInt = ixNearIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<int32_t> iyNearInt = iyNearIntBuf.Get<int32_t>(ubFactorElement);
        LocalTensor<T> ixFloat = ixFloatBuf.Get<T>(ubFactorElement);
        LocalTensor<T> iyFloat = iyFloatBuf.Get<T>(ubFactorElement);
        LocalTensor<int32_t> xIndex = computeIndexBuf.Get<int32_t>(ubFactorElement);
        Cast(ixNearInt, gridXTensor, RoundMode::CAST_RINT, gridCoordElemCount / 2);
        Cast(iyNearInt, gridYTensor, RoundMode::CAST_RINT, gridCoordElemCount / 2);
        Cast(ixFloat, ixNearInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        Cast(iyFloat, iyNearInt, RoundMode::CAST_NONE, gridCoordElemCount / 2);
        WithinBounds2d(selTensor1, iyFloat, ixFloat, dupOneTensor, gridCoordElemCount / 2);
        ComputeIndex(xIndex, xIndex, iyNearInt, ixNearInt, gridCoordElemCount / 2);
        DataCopyParams copyParams = {1, 0, 0, 0};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        copyParams.blockLen = channel * sizeof(Dtype);
        padParams.rightPadding = alignChannel - channel;
        padParams.paddingValue = GetScalarBitcodeValue((Dtype)0);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        if (channel <= 1024) {
            int32_t times = (gridCoordElemCount / 2) / group;
            int32_t tailNums = (gridCoordElemCount / 2) % group;

            for (int32_t i = 0; i < times; i++) {
                LocalTensor<Dtype> gOutLocalTempTensor = dataInQueue[0].AllocTensor<Dtype>();
                LocalTensor<T> gOutLocalTensor = gOutLocalTensor2Buf.Get<T>(alignChannel * group);
                for (int32_t j = 0; j < group; j++) {
                    flatGridPointIndex = curGridPointIndex + i * group + j;
                    w = flatGridPointIndex % outW;
                    h = (flatGridPointIndex / outW) % outH;
                    n = flatGridPointIndex / (outH * outW);
                    ncBaseOffset = n * dxStrideN;
                    gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                    DataCopyPad(gOutLocalTempTensor[j * alignChannel], inputGm[0][gradGmOffset], copyParams, padParams);
                }
                SetFlag<HardEvent::MTE2_S>(eventID);
                WaitFlag<HardEvent::MTE2_S>(eventID);
                Cast(gOutLocalTensor, gOutLocalTempTensor, mode, alignChannel * group);
                for (int32_t k = 0; k < group; k++) {
                    flatGridPointIndex = curGridPointIndex + i * group + k;
                    n = flatGridPointIndex / (outH * outW);
                    ncBaseOffset = n * dxStrideN;
                    ComputeNearestXGrad(xIndex, dupOneTensor, i * group + k, k, ncBaseOffset, gOutLocalTensor, iyFloat,
                                        ixFloat);
                }
                dataInQueue[0].FreeTensor(gOutLocalTempTensor);
            }
            for (int32_t i = 0; i < tailNums; i++) {
                flatGridPointIndex = curGridPointIndex + times * group + i;
                h = (flatGridPointIndex / outW) % outH;
                w = flatGridPointIndex % outW;
                n = flatGridPointIndex / (outH * outW);
                ncBaseOffset = n * dxStrideN;
                gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                LocalTensor<Dtype> gOutLocalTempTensor = dataInQueue[0].AllocTensor<Dtype>();
                LocalTensor<T> gOutLocalTensor = gOutLocalTensor2Buf.Get<T>(alignChannel * group);
                DataCopyPad(gOutLocalTempTensor, inputGm[0][gradGmOffset], copyParams, padParams);
                SetFlag<HardEvent::MTE2_S>(eventID);
                WaitFlag<HardEvent::MTE2_S>(eventID);
                Cast(gOutLocalTensor, gOutLocalTempTensor, mode, alignChannel * group);
                ComputeNearestXGrad(xIndex, dupOneTensor, times * group + i, 0, ncBaseOffset, gOutLocalTensor, iyFloat,
                                    ixFloat);
                dataInQueue[0].FreeTensor(gOutLocalTempTensor);
            }
        } else {
            for (int32_t i = 0; i < gridCoordElemCount / 2; i++) {
                flatGridPointIndex = curGridPointIndex + i;
                w = flatGridPointIndex % outW;
                h = (flatGridPointIndex / outW) % outH;
                n = flatGridPointIndex / (outH * outW);
                ncBaseOffset = n * dxStrideN;
                gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                LocalTensor<Dtype> gOutLocalTempTensor = dataInQueue[0].AllocTensor<Dtype>();
                LocalTensor<T> gOutLocalTensor = gOutLocalTensor2Buf.Get<T>(alignChannel * group);
                DataCopyPad(gOutLocalTempTensor, inputGm[0][gradGmOffset], copyParams, padParams);
                SetFlag<HardEvent::MTE2_S>(eventID);
                WaitFlag<HardEvent::MTE2_S>(eventID);
                Cast(gOutLocalTensor, gOutLocalTempTensor, mode, alignChannel * group);
                ComputeNearestXGrad(xIndex, dupOneTensor, i, 0, ncBaseOffset, gOutLocalTensor, iyFloat, ixFloat);
                dataInQueue[0].FreeTensor(gOutLocalTempTensor);
            }
        }
        Duplicate<T>(dstLocal, 0, 2 * ubFactorElement);
    }
    Cast(dstTempLocal, dstLocal, RoundMode::CAST_RINT, 2 * ubFactorElement);
    dataOutQueue[1].EnQue(dstTempLocal);
    dataInQueue[2].FreeTensor(inputCoordinateTemp);
}
template <typename T, typename Dtype, typename GridSamplerGradTilingData>
__aicore__ inline void GridSampler2DGradFP16<T, Dtype, GridSamplerGradTilingData>::Process()
{
    uint32_t corePointCount = 0;
    int64_t gridGmOffset = 0;
    int32_t gridOffset = 0;
    int32_t cycleOffset = 0;
    int64_t curGridPointIndex = 0;
    if (blockIdx < tailPNum) {
        corePointCount = pNumPerCore + 1;
        gridOffset = blockIdx * corePointCount;
    } else {
        corePointCount = pNumPerCore;
        gridOffset = blockIdx * pNumPerCore + tailPNum;
    }

    int32_t coordElemsPerCopy = 2 * ubFactorElement;
    int32_t tileCoordElemCount = coordElemsPerCopy;
    int32_t copyIterationCount = CeilDiv(corePointCount * 2, coordElemsPerCopy);
    for (int copyIter = 0; copyIter < copyIterationCount; copyIter++) {
        if (copyIter == copyIterationCount - 1) {
            tileCoordElemCount = corePointCount * 2 - (copyIterationCount - 1) * coordElemsPerCopy;
        }
        cycleOffset = copyIter * coordElemsPerCopy;
        gridGmOffset = cycleOffset + static_cast<int64_t>(gridOffset) * 2;
        curGridPointIndex = gridOffset + static_cast<int64_t>(copyIter) * coordElemsPerCopy / 2;
        CopyIn(gridGmOffset, tileCoordElemCount, 2);
        Compute(tileCoordElemCount, curGridPointIndex);
        CopyOut(gridGmOffset, tileCoordElemCount);
    }
}

#endif // GRID_SAMPLER_2D_GRAD_H_
