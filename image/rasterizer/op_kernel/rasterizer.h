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
 * \file rasterizer.h
 * \brief
 */
#ifndef __RASTERIZER_H__
#define __RASTERIZER_H__
#include <limits>
#include "kernel_operator.h"

namespace NsRasterizer {

using namespace AscendC;

constexpr int32_t NUM_PIXEL = 2 * 1024;
constexpr uint32_t NUM_F_PER_TURN = 8;
constexpr uint32_t NUM_V_PER_F = 3;
constexpr uint32_t NUM_VAL_PER_V = 4;
constexpr uint32_t NUM_VAL_PER_BARY = 3;
constexpr uint32_t NUM_FACES = 1024;
constexpr uint32_t PIXEL_COORDINATE_DIM = 2;
constexpr uint32_t BARYCENTRIC_COORDINATE_DIM = 3;
constexpr int32_t NUM_BIT_UINT8 = 8;
constexpr int32_t LOGIC_COUNT = NUM_PIXEL / NUM_BIT_UINT8 * sizeof(uint8_t) / sizeof(uint16_t);
constexpr size_t RESERVE_SIZE = 64;

template <typename T>
class Rasterizer {
public:
    __aicore__ inline Rasterizer(){};

    __aicore__ inline void Init(GM_ADDR v, GM_ADDR f, GM_ADDR d, GM_ADDR findices, GM_ADDR barycentric,
                                GM_ADDR workSpace, RasterizerTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuf();
    __aicore__ inline void InitGM();
    __aicore__ inline void CopyInFaces(uint32_t progress, uint32_t curFNum);
    __aicore__ inline void ComputeZBuffer(uint32_t progress, uint32_t curFNum);
    __aicore__ inline void ProcessZBuffer();
    __aicore__ inline void CopyInVertices(const uint32_t num, const uint32_t count);
    __aicore__ inline void CalcSignedArea(T a0, T a1, T t0, T t1, const uint32_t isBeta, const int32_t count);
    __aicore__ inline void ToScreenCoordinate();
    __aicore__ inline void CalcTriangle(const uint32_t idx, int32_t preFaces);
    __aicore__ inline void CalcBaryCoordinate(T area, const int32_t count);
    __aicore__ inline void CalcMaskInTriangle();
    __aicore__ inline void CalcZBufferMask(int32_t currentFIdx);
    __aicore__ inline void UpdateZBuffer(int32_t currFIdx);
    __aicore__ inline void CalcPixelDepth(T z0, T z1, T z2, int32_t count);
    __aicore__ inline void CopyInZBuffer(int32_t x, int32_t y, int32_t count);
    __aicore__ inline void CopyOutZBuffer(int32_t x, int32_t y, int32_t count);
    __aicore__ inline uint64_t CalcMask(uint32_t idx, LocalTensor<uint16_t>& tensor);

private:
    TPipe pipe;

    // input
    GlobalTensor<T> vGM;
    GlobalTensor<int32_t> fGM;

    // zbuffer
    GlobalTensor<int32_t> fIdxGM;
    GlobalTensor<T> depthGM;
    // output
    GlobalTensor<int32_t> findicesGM;
    GlobalTensor<T> barycentricGM;

    TBuf<TPosition::VECCALC> facesBuf;
    TBuf<TPosition::VECCALC> verticesBuf;
    TBuf<TPosition::VECCALC> coordinatesBuf;
    TBuf<TPosition::VECCALC> dBuf;
    TBuf<TPosition::VECCALC> signedAreaBuf;
    TBuf<TPosition::VECCALC> baryCoBuf;
    TBuf<TPosition::VECCALC> compareBuf;
    TBuf<TPosition::VECCALC> pixelDepthBuf;
    TBuf<TPosition::VECCALC> fIdxBuf;
    TBuf<TPosition::VECCALC> depthBuf;

    // faces to be processed in one turn
    LocalTensor<int32_t> facesLocal;
    // vertices coordinates of faces in faces
    LocalTensor<T> verticesLocal;
    // coordinates of part of pixels in rectangle
    LocalTensor<T> coordinatesLocal;
    LocalTensor<T> dLocal;
    LocalTensor<T> depthThresLocal;
    // barycentric coordinates of part of pixels in rectangle
    LocalTensor<T> baryCoLocal;
    LocalTensor<T> signedAreaTempLocal;
    LocalTensor<T> signedAreaBetaLocal;
    LocalTensor<T> signedAreaGammaLocal;
    // compare result
    LocalTensor<uint8_t> compareFstLocal;
    LocalTensor<uint8_t> compareSecLocal;
    LocalTensor<uint8_t> compareTrdLocal;

    LocalTensor<uint16_t> maskFstLocal;
    LocalTensor<uint16_t> maskSecLocal;
    LocalTensor<uint16_t> maskTrdLocal;
    // depth of part of pixels in rectangle
    LocalTensor<T> pixelDepthLocal;
    // findices in zbuffer
    LocalTensor<int32_t> fIdxLocal;
    // depth in zbuffer
    LocalTensor<T> depthLocal;

    int32_t height;
    int32_t width;
    float occlusionTruncation;
    uint32_t useDepthPrior;
    // faces to be processed by this vec core
    uint32_t numFaces;
    // faces to be processd by vec cores before this
    int32_t numPreFaces;
    int32_t numFindices;
};

template <typename T>
__aicore__ inline void Rasterizer<T>::Init(GM_ADDR v, GM_ADDR f, GM_ADDR d, GM_ADDR findices, GM_ADDR barycentric,
                                           GM_ADDR workSpace, RasterizerTilingData* tilingData)
{
    this->height = tilingData->height;
    this->width = tilingData->width;
    this->occlusionTruncation = tilingData->occlusionTruncation;
    this->useDepthPrior = tilingData->useDepthPrior;

    vGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v));

    const int64_t blockNum = GetBlockNum();
    const int64_t blockIdx = GetBlockIdx();

    uint32_t offsetF = 0;
    uint32_t facesPerBlock = static_cast<uint32_t>(tilingData->numFaces / blockNum);
    this->numFaces = facesPerBlock;
    uint32_t numFacesReminder = tilingData->numFaces % blockNum;
    this->numPreFaces = facesPerBlock * blockIdx;
    if (blockIdx < numFacesReminder) {
        this->numFaces += 1;
        this->numPreFaces += blockIdx;
        offsetF = blockIdx * (facesPerBlock + 1) * NUM_V_PER_F;
    } else {
        offsetF = blockIdx * facesPerBlock * NUM_V_PER_F + numFacesReminder * NUM_V_PER_F;
        this->numPreFaces += numFacesReminder;
    }

    fGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(f) + offsetF);

    uint32_t offsetFid = 0;
    uint32_t numFindices = static_cast<uint32_t>(height * width / blockNum);
    uint32_t numFindicesReminder = height * width % blockNum;
    this->numFindices = numFindices;
    if (blockIdx < numFindicesReminder) {
        offsetFid = blockIdx * (numFindices + 1);
        this->numFindices += 1;
    } else {
        offsetFid = blockIdx * numFindices + numFindicesReminder;
    }

    this->findicesGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(findices) + offsetFid);
    this->barycentricGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(barycentric) + offsetFid * NUM_VAL_PER_BARY);

    const int64_t findicesSize = height * width;

    fIdxGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workSpace) + findicesSize * blockIdx);
    depthGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workSpace) + findicesSize * blockNum + RESERVE_SIZE +
                            findicesSize * blockIdx);

    InitBuf();
    InitGM();
}

template <typename T>
__aicore__ inline void Rasterizer<T>::InitGM()
{
    Duplicate(fIdxLocal, std::numeric_limits<int32_t>::max(), NUM_PIXEL);
    Duplicate(depthLocal, std::numeric_limits<T>::max(), NUM_PIXEL);
    int32_t turns = height * width / NUM_PIXEL;
    int32_t remainder = height * width % NUM_PIXEL;
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    for (int32_t i = 0; i < turns; ++i) {
        DataCopy(fIdxGM[i * NUM_PIXEL], fIdxLocal, NUM_PIXEL);
        DataCopy(depthGM[i * NUM_PIXEL], depthLocal, NUM_PIXEL);
    }

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainder * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(fIdxGM[turns * NUM_PIXEL], fIdxLocal, copyParams);

    copyParams.blockLen = remainder * sizeof(T);
    DataCopyPad(depthGM[turns * NUM_PIXEL], depthLocal, copyParams);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    PipeBarrier<PIPE_V>();
    Duplicate(fIdxLocal, 0, NUM_PIXEL);
    turns = this->numFindices / NUM_PIXEL;
    remainder = this->numFindices % NUM_PIXEL;
    SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (int32_t i = 0; i < turns; ++i) {
        DataCopy(findicesGM[i * NUM_PIXEL], fIdxLocal, NUM_PIXEL);
    }
    copyParams.blockLen = static_cast<uint32_t>(remainder * sizeof(int32_t));
    DataCopyPad(findicesGM[turns * NUM_PIXEL], fIdxLocal, copyParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Rasterizer<T>::InitBuf()
{
    this->pipe.InitBuffer(compareBuf, NUM_PIXEL * sizeof(uint8_t) / 8 * BARYCENTRIC_COORDINATE_DIM);
    this->pipe.InitBuffer(verticesBuf, NUM_F_PER_TURN * NUM_V_PER_F * NUM_VAL_PER_V * sizeof(T));
    this->pipe.InitBuffer(facesBuf, NUM_FACES * NUM_V_PER_F * sizeof(int32_t));
    this->pipe.InitBuffer(coordinatesBuf, NUM_PIXEL * sizeof(T) * PIXEL_COORDINATE_DIM);
    this->pipe.InitBuffer(baryCoBuf, NUM_PIXEL * sizeof(T) * BARYCENTRIC_COORDINATE_DIM);
    this->pipe.InitBuffer(signedAreaBuf, NUM_PIXEL * sizeof(T) * BARYCENTRIC_COORDINATE_DIM);
    this->pipe.InitBuffer(pixelDepthBuf, NUM_PIXEL * sizeof(T));
    this->pipe.InitBuffer(fIdxBuf, NUM_PIXEL * sizeof(int32_t));
    this->pipe.InitBuffer(depthBuf, NUM_PIXEL * sizeof(T));

    facesLocal = facesBuf.Get<int32_t>();
    verticesLocal = verticesBuf.Get<T>();
    coordinatesLocal = coordinatesBuf.Get<T>();
    baryCoLocal = baryCoBuf.Get<T>();
    signedAreaTempLocal = signedAreaBuf.Get<T>();
    signedAreaBetaLocal = signedAreaBuf.GetWithOffset<T>(NUM_PIXEL, NUM_PIXEL * sizeof(T));
    signedAreaGammaLocal = signedAreaBuf.GetWithOffset<T>(NUM_PIXEL, NUM_PIXEL * sizeof(T) * 2);
    compareFstLocal = compareBuf.Get<uint8_t>(NUM_PIXEL / 8);
    compareSecLocal = compareBuf.GetWithOffset<uint8_t>(NUM_PIXEL / 8, NUM_PIXEL / 8);
    compareTrdLocal = compareBuf.GetWithOffset<uint8_t>(NUM_PIXEL / 8, NUM_PIXEL / 8 * 2);
    pixelDepthLocal = pixelDepthBuf.Get<T>();
    fIdxLocal = fIdxBuf.Get<int32_t>();
    depthLocal = depthBuf.Get<T>();
    maskFstLocal = compareFstLocal.ReinterpretCast<uint16_t>();
    maskSecLocal = compareSecLocal.ReinterpretCast<uint16_t>();
    maskTrdLocal = compareTrdLocal.ReinterpretCast<uint16_t>();
}

template <typename T>
__aicore__ inline void Rasterizer<T>::ToScreenCoordinate()
{
    const uint32_t count = NUM_V_PER_F * NUM_F_PER_TURN;
    const uint32_t offsetY = count;
    const uint32_t offsetZ = count * 2;
    const uint32_t offsetW = count * 3;
    const T diff = static_cast<T>(0.5f);
    Div(verticesLocal, verticesLocal, verticesLocal[offsetW], count);
    PipeBarrier<PIPE_V>();
    Muls(verticesLocal, verticesLocal, diff, count);
    PipeBarrier<PIPE_V>();
    Adds(verticesLocal, verticesLocal, diff, count);
    PipeBarrier<PIPE_V>();
    Muls(verticesLocal, verticesLocal, static_cast<T>(this->width - 1), count);
    PipeBarrier<PIPE_V>();
    Adds(verticesLocal, verticesLocal, diff, count);
    Muls(verticesLocal[offsetY], verticesLocal[offsetY], diff, count);
    PipeBarrier<PIPE_V>();
    Div(verticesLocal[offsetY], verticesLocal[offsetY], verticesLocal[offsetW], count);
    PipeBarrier<PIPE_V>();
    Adds(verticesLocal[offsetY], verticesLocal[offsetY], diff, count);
    PipeBarrier<PIPE_V>();
    Muls(verticesLocal[offsetY], verticesLocal[offsetY], static_cast<T>(this->height - 1), count);
    PipeBarrier<PIPE_V>();
    Adds(verticesLocal[offsetY], verticesLocal[offsetY], diff, count);

    Div(verticesLocal[offsetZ], verticesLocal[offsetZ], verticesLocal[offsetW], count);
    PipeBarrier<PIPE_V>();
    Muls(verticesLocal[offsetZ], verticesLocal[offsetZ], static_cast<T>(0.49999f), count);
    PipeBarrier<PIPE_V>();
    Adds(verticesLocal[offsetZ], verticesLocal[offsetZ], diff, count);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CalcSignedArea(T a0, T a1, T t0, T t1, const uint32_t isBeta, const int32_t count)
{
    if (isBeta) {
        Duplicate(pixelDepthLocal, a1, count);
        PipeBarrier<PIPE_V>();
        Sub(signedAreaBetaLocal, coordinatesLocal[NUM_PIXEL], pixelDepthLocal, count);
        PipeBarrier<PIPE_V>();
        Muls(signedAreaBetaLocal, signedAreaBetaLocal, t0 - a0, count);
        Duplicate(pixelDepthLocal, a0, count);
        PipeBarrier<PIPE_V>();
        Sub(signedAreaTempLocal, coordinatesLocal, pixelDepthLocal, count);
        PipeBarrier<PIPE_V>();
        Muls(signedAreaTempLocal, signedAreaTempLocal, t1 - a1, count);
        PipeBarrier<PIPE_V>();
        Sub(signedAreaBetaLocal, signedAreaBetaLocal, signedAreaTempLocal, count);
        return;
    }

    Duplicate(pixelDepthLocal, a0, count);
    PipeBarrier<PIPE_V>();
    Sub(signedAreaGammaLocal, coordinatesLocal, pixelDepthLocal, count);
    PipeBarrier<PIPE_V>();
    Muls(signedAreaGammaLocal, signedAreaGammaLocal, t1 - a1, count);
    Duplicate(pixelDepthLocal, a1, count);
    PipeBarrier<PIPE_V>();
    Sub(signedAreaTempLocal, coordinatesLocal[NUM_PIXEL], pixelDepthLocal, count);
    PipeBarrier<PIPE_V>();
    Muls(signedAreaTempLocal, signedAreaTempLocal, t0 - a0, count);
    PipeBarrier<PIPE_V>();
    Sub(signedAreaGammaLocal, signedAreaGammaLocal, signedAreaTempLocal, count);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CalcBaryCoordinate(T area, const int32_t count)
{
    Duplicate(signedAreaTempLocal, area, count);
    Duplicate(baryCoLocal, -1.0f, NUM_PIXEL * BARYCENTRIC_COORDINATE_DIM);
    PipeBarrier<PIPE_V>();
    Div(baryCoLocal[NUM_PIXEL], signedAreaBetaLocal, signedAreaTempLocal, count);
    Div(baryCoLocal[NUM_PIXEL * 2], signedAreaGammaLocal, signedAreaTempLocal, count);
    Duplicate(baryCoLocal, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Sub(baryCoLocal, baryCoLocal, baryCoLocal[NUM_PIXEL], count);
    PipeBarrier<PIPE_V>();
    Sub(baryCoLocal, baryCoLocal, baryCoLocal[NUM_PIXEL * 2], count);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CalcMaskInTriangle()
{
    CompareScalar(compareFstLocal, baryCoLocal, 0.0f, CMPMODE::GE, NUM_PIXEL);
    CompareScalar(compareSecLocal, baryCoLocal, 1.0f, CMPMODE::LE, NUM_PIXEL);
    CompareScalar(compareTrdLocal, baryCoLocal[NUM_PIXEL], 0.0f, CMPMODE::GE, NUM_PIXEL);
    PipeBarrier<PIPE_V>();
    // result of alpha
    And(maskFstLocal, maskFstLocal, maskSecLocal, LOGIC_COUNT);
    PipeBarrier<PIPE_V>();
    CompareScalar(compareSecLocal, baryCoLocal[NUM_PIXEL], 1.0f, CMPMODE::LE, NUM_PIXEL);
    PipeBarrier<PIPE_V>();
    // result of beta
    And(maskSecLocal, maskSecLocal, maskTrdLocal, LOGIC_COUNT);
    PipeBarrier<PIPE_V>();
    CompareScalar(compareTrdLocal, baryCoLocal[NUM_PIXEL * 2], 0.0f, CMPMODE::GE, NUM_PIXEL);
    // alpha & beta
    And(maskFstLocal, maskFstLocal, maskSecLocal, LOGIC_COUNT);
    PipeBarrier<PIPE_V>();
    CompareScalar(compareSecLocal, baryCoLocal[NUM_PIXEL * 2], 1.0f, CMPMODE::LE, NUM_PIXEL);
    PipeBarrier<PIPE_V>();
    And(maskSecLocal, maskSecLocal, maskTrdLocal, LOGIC_COUNT);
    PipeBarrier<PIPE_V>();
    And(maskFstLocal, maskFstLocal, maskSecLocal, LOGIC_COUNT);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CalcZBufferMask(int32_t currentFIdx)
{
    CalcMaskInTriangle();
    PipeBarrier<PIPE_V>();
    Compare(compareSecLocal, pixelDepthLocal, depthLocal, CMPMODE::LT, NUM_PIXEL);
    PipeBarrier<PIPE_V>();
    // mask depth less
    And(maskSecLocal, maskFstLocal, maskSecLocal, LOGIC_COUNT);
}

template <typename T>
__aicore__ inline uint64_t Rasterizer<T>::CalcMask(uint32_t idx, LocalTensor<uint16_t>& tensor)
{
    return static_cast<uint64_t>(tensor.GetValue(idx)) | (static_cast<uint64_t>(tensor.GetValue(idx + 1)) << 16) |
           (static_cast<uint64_t>(tensor.GetValue(idx + 2)) << 32) |
           (static_cast<uint64_t>(tensor.GetValue(idx + 3)) << 48);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::UpdateZBuffer(int32_t currFIdx)
{
    constexpr uint32_t COUNT = sizeof(uint64_t) / sizeof(uint16_t);
    uint64_t mask[1] = {0};
    constexpr uint8_t repeatTime = 1;
    const UnaryRepeatParams repeatParams{1, 1, 8, 8};
    for (uint32_t idx = 0; idx < NUM_PIXEL / 16; idx += COUNT) {
        mask[0] = CalcMask(idx, maskSecLocal);
        PipeBarrier<PIPE_ALL>();
        if (mask[0] != 0) {
            Adds(depthLocal[idx / COUNT * 64], pixelDepthLocal[idx / COUNT * 64], 0.0f, mask, repeatTime, repeatParams);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        if (mask[0] != 0) {
            Duplicate(fIdxLocal[idx / COUNT * 64], currFIdx + 1, mask, repeatTime, 1, 8);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
    }
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CalcPixelDepth(T z0, T z1, T z2, int32_t count)
{
    Muls(pixelDepthLocal, baryCoLocal, z0, count);
    Muls(signedAreaTempLocal, baryCoLocal[NUM_PIXEL], z1, count);
    PipeBarrier<PIPE_V>();
    Add(pixelDepthLocal, pixelDepthLocal, signedAreaTempLocal, count);
    PipeBarrier<PIPE_V>();
    Muls(signedAreaTempLocal, baryCoLocal[NUM_PIXEL * 2], z2, count);
    PipeBarrier<PIPE_V>();
    Add(pixelDepthLocal, pixelDepthLocal, signedAreaTempLocal, count);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CalcTriangle(const uint32_t idx, int32_t preFaces)
{
    T x0 = verticesLocal.GetValue(idx * NUM_V_PER_F);
    T x1 = verticesLocal.GetValue(idx * NUM_V_PER_F + 1);
    T x2 = verticesLocal.GetValue(idx * NUM_V_PER_F + 2);
    T xMin = Std::max(Std::min(Std::min(x0, x1), x2), static_cast<T>(0));
    T xMax = Std::min(Std::max(Std::max(x0, x1), x2), static_cast<T>(width - 1));
    T y0 = verticesLocal.GetValue(idx * NUM_V_PER_F + NUM_V_PER_F * NUM_F_PER_TURN);
    T y1 = verticesLocal.GetValue(idx * NUM_V_PER_F + 1 + NUM_V_PER_F * NUM_F_PER_TURN);
    T y2 = verticesLocal.GetValue(idx * NUM_V_PER_F + 2 + NUM_V_PER_F * NUM_F_PER_TURN);
    T yMin = Std::max(Std::min(Std::min(y0, y1), y2), static_cast<T>(0));
    T yMax = Std::min(Std::max(Std::max(y0, y1), y2), static_cast<T>(height - 1));
    T z0 = verticesLocal.GetValue(idx * NUM_V_PER_F + NUM_V_PER_F * NUM_F_PER_TURN * 2);
    T z1 = verticesLocal.GetValue(idx * NUM_V_PER_F + 1 + NUM_V_PER_F * NUM_F_PER_TURN * 2);
    T z2 = verticesLocal.GetValue(idx * NUM_V_PER_F + 2 + NUM_V_PER_F * NUM_F_PER_TURN * 2);

    float area = (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0);

    if (area == 0.0f) {
        return;
    }

    int32_t xEnd = static_cast<int32_t>(xMax) + 1;
    int32_t yEnd = static_cast<int32_t>(yMax) + 1;

    for (int32_t i = yMin; i < yEnd; ++i) {
        int32_t count = xMin;
        while (count < xEnd) {
            int32_t num = Std::min(NUM_PIXEL, xEnd - count);
            // generate centric point coordinates
            ArithProgression(coordinatesLocal, static_cast<T>(count) + static_cast<T>(0.5f), static_cast<T>(1.0f), num);
            PipeBarrier<PIPE_ALL>();
            Duplicate(coordinatesLocal[NUM_PIXEL], i + 0.5f, num);
            CopyInZBuffer(count, i, num);
            // first gamma then beta, otherwise may need PIPE_V sync
            CalcSignedArea(x0, y0, x2, y2, 1, num);
            CalcSignedArea(x0, y0, x1, y1, 0, num);
            PipeBarrier<PIPE_V>();
            CalcBaryCoordinate(area, num);
            PipeBarrier<PIPE_V>();
            CalcPixelDepth(z0, z1, z2, num);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            CalcZBufferMask(preFaces + idx);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            UpdateZBuffer(preFaces + idx);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyOutZBuffer(count, i, num);
            count += num;
        }
    }
}

// count: 已计算的面的数量，num：当前需要计算的面的数量
template <typename T>
__aicore__ inline void Rasterizer<T>::CopyInVertices(const uint32_t num, const uint32_t count)
{
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < NUM_V_PER_F; ++j) {
            uint32_t offsetFLocal = (count + i) * NUM_V_PER_F + j;
            for (int k = 0; k < NUM_VAL_PER_V; ++k) {
                uint32_t offsetVLocal = i * NUM_V_PER_F + j + NUM_V_PER_F * NUM_F_PER_TURN * k;
                int32_t vIdx = facesLocal.GetValue(offsetFLocal) * NUM_VAL_PER_V + k;
                T vVal = vGM.GetValue(vIdx);
                verticesLocal.SetValue(offsetVLocal, vVal);
            }
        }
    }
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CopyInFaces(uint32_t progress, uint32_t num)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * NUM_V_PER_F * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(facesLocal, this->fGM[progress * NUM_FACES * NUM_V_PER_F], copyParams, padParams);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CopyInZBuffer(int32_t x, int32_t y, int32_t count)
{
    int32_t offset = y * this->width + x;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyExtParams depthCopyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<T> depthPadParams{false, 0, 0, 0.0f};
    DataCopyPad(fIdxLocal, fIdxGM[offset], copyParams, padParams);
    DataCopyPad(depthLocal, depthGM[offset], depthCopyParams, depthPadParams);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::CopyOutZBuffer(int32_t x, int32_t y, int32_t count)
{
    int32_t offset = y * this->width + x;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyExtParams depthCopyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(fIdxGM[offset], fIdxLocal, copyParams);
    DataCopyPad(depthGM[offset], depthLocal, depthCopyParams);
}

template <typename T>
__aicore__ inline void Rasterizer<T>::ComputeZBuffer(uint32_t progress, uint32_t num)
{
    uint32_t count = 0;
    while (count < num) {
        uint32_t curNum = Std::min(NUM_F_PER_TURN, num - count);
        CopyInVertices(curNum, count);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        ToScreenCoordinate();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t i = 0; i < curNum; ++i) {
            CalcTriangle(i, this->numPreFaces + count);
        }
        count += curNum;
    }
}

template <typename T>
__aicore__ inline void Rasterizer<T>::ProcessZBuffer()
{
    const int32_t turns = numFaces / NUM_FACES;
    const int32_t remainder = numFaces % NUM_FACES;
    Duplicate(verticesLocal, 1.0f, NUM_F_PER_TURN * NUM_V_PER_F * NUM_VAL_PER_V);
    for (int32_t i = 0; i < turns; ++i) {
        CopyInFaces(i, NUM_FACES);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        ComputeZBuffer(i, NUM_FACES);
        this->numPreFaces += NUM_FACES;
    }
    if (remainder != 0) {
        CopyInFaces(turns, remainder);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        ComputeZBuffer(turns, remainder);
    }
}

template <typename T>
__aicore__ inline void Rasterizer<T>::Process()
{
    ProcessZBuffer();
    PipeBarrier<PIPE_ALL>();
}

} // namespace NsRasterizer
#endif // RASTERIZER_H
