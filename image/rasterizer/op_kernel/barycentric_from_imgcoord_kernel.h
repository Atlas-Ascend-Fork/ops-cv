/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BARYCENTRIC_FROM_IMGCOORD_AIV_H
#define BARYCENTRIC_FROM_IMGCOORD_AIV_H

#include "barycentric_from_imgcoord_kernel_base.h"

using namespace AscendC;

namespace BarycentricFromImgcoord {

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::Init(GM_ADDR v, GM_ADDR f, GM_ADDR findices, GM_ADDR barycentric,
                                                           GM_ADDR workspace, RasterizerTilingData* tilingData)
{
    ParseTilingData(tilingData);
    InitParam();

    this->invGM.SetGlobalBuffer((__gm__ T*)v);
    this->infGM.SetGlobalBuffer((__gm__ int32_t*)f);
    this->findicesGM.SetGlobalBuffer((__gm__ int32_t*)findices);
    this->barycentricGM.SetGlobalBuffer((__gm__ T*)barycentric);

    uint64_t wkspOffset = 0;
    this->zBufIdxGM.SetGlobalBuffer((__gm__ int32_t*)workspace);
    wkspOffset += static_cast<uint64_t>(this->vecCoreNum * this->totalPixNum + rsv) * sizeof(int32_t);
    this->zBufDepthGM.SetGlobalBuffer((__gm__ float*)(workspace + wkspOffset));
    wkspOffset += static_cast<uint64_t>(this->vecCoreNum * this->totalPixNum + rsv) * sizeof(float);
    this->maskGM.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkspOffset)); // 1920 * 3 * 4 byte

    InitUbuf();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::InitParam()
{
    this->vecIdx = GetBlockIdx();
    this->vecCoreNum = GetBlockNum() * GetSubBlockNum(); // 20 * 2
    this->totalPixNum = this->height * this->width;

    uint32_t basePixNum = this->totalPixNum / this->vecCoreNum;
    uint32_t basePixTail = this->totalPixNum % this->vecCoreNum;

    if (this->vecIdx < basePixTail) {
        this->calPixNum = basePixNum + 1;
        this->startPixId = this->vecIdx * (basePixNum + 1);
    } else {
        this->calPixNum = basePixNum;
        this->startPixId = this->vecIdx * basePixNum + basePixTail;
    }
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::InitUbuf()
{
    this->pipe.InitBuffer(this->uBuf, 184 * 1024); // revserve 8k for select
    this->ubufLocal = uBuf.Get<float>();

    uint32_t offset = 0;
    this->inDepthLocal = this->ubufLocal[offset]; // 48.5kb
    offset += BUFFER_NUM * (ELENUM_REPEAT_FP32 * TRANS_COL_ELENUM + rsv);
    this->transDepthLocal = this->ubufLocal[offset]; // 48.5kb
    offset += BUFFER_NUM * (ELENUM_REPEAT_FP32 * TRANS_COL_ELENUM + rsv);
    this->reduceMinIdxLocal = this->ubufLocal[offset]; // 15.5kb
    offset += BUFFER_NUM * (MAX_PROC_ELENUM + rsv);
    this->barycentricFlagLocal = this->ubufLocal[offset].template ReinterpretCast<int32_t>(); // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;

    // save screen coordinate (x, y)
    this->screenVertxLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->screenVertyLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;

    this->screenVert0xLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->screenVert1xLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->screenVert2xLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->screenVert0yLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->screenVert1yLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->screenVert2yLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;

    ReuseUbuf();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::ReuseUbuf()
{
    uint32_t offset = 0;
    // save origin coordinate (x, y, z, w)
    this->vert0xLocal = this->ubufLocal[offset]; // 7.75kb
    this->areaZeroMaskLocal = this->ubufLocal[offset].template ReinterpretCast<uint8_t>();
    offset += MAX_PROC_ELENUM + rsv;
    this->vert1xLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert2xLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert0yLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert1yLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert2yLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert0wLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert1wLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->vert2wLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->pixIdTmpLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->baryCentricxLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->baryCentricyLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
    this->baryCentriczLocal = this->ubufLocal[offset]; // 7.75kb
    offset += MAX_PROC_ELENUM + rsv;
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::ParseTilingData(RasterizerTilingData* tilingData)
{
    this->numFaces = tilingData->numFaces;
    this->numVertices = tilingData->numVertices;
    this->height = tilingData->height;
    this->width = tilingData->width;
    this->heightF32 = static_cast<float>(this->height);
    this->widthF32 = static_cast<float>(this->width);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::GenMaskDataInGM()
{
    LocalTensor<uint32_t> maskLocal = this->inDepthLocal.template ReinterpretCast<uint32_t>();
    uint32_t baseMaskColNum = MAX_PROC_ELENUM / this->vecCoreNum;
    uint32_t baseMaskColTail = MAX_PROC_ELENUM % this->vecCoreNum;

    uint32_t calMaskColNum = 0;
    uint32_t startMaskColId = 0;

    if (this->vecIdx < baseMaskColTail) {
        calMaskColNum = baseMaskColNum + 1;
        startMaskColId = this->vecIdx * (baseMaskColNum + 1);
    } else {
        calMaskColNum = baseMaskColNum;
        startMaskColId = this->vecIdx * baseMaskColNum + baseMaskColTail;
    }

    for (uint32_t colId = 0; colId < calMaskColNum; colId++) {
        for (uint32_t rowId = 0; rowId < BARY_COORD_NUM; rowId++) {
            uint32_t maskValue = (rowId * (MAX_PROC_ELENUM + rsv) + colId + startMaskColId) * sizeof(float);
            uint32_t offset = colId * BARY_COORD_NUM + rowId;
            maskLocal.SetValue(offset, maskValue);
        }
    }
    PipeBarrier<PIPE_ALL>();

    uint16_t blockCnt = 1;
    uint32_t blockLen = calMaskColNum * BARY_COORD_NUM * sizeof(uint32_t);
    uint64_t startMaskId = static_cast<uint64_t>(startMaskColId * BARY_COORD_NUM);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPad(this->maskGM[startMaskId], maskLocal, copyParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::PreProcess(uint32_t loopId, uint32_t curCalPixNum)
{
    uint32_t preLoopNum = curCalPixNum / TRANS_COL_ELENUM;
    uint32_t preTailNum = curCalPixNum % TRANS_COL_ELENUM;

    if (preTailNum > 0) {
        preLoopNum++;
    }

    uint32_t pingFlag = 1;

    for (uint32_t preLoopId = 0; preLoopId < preLoopNum; preLoopId++) {
        uint32_t curPrePixNum = TRANS_COL_ELENUM;
        if ((preLoopId == preLoopNum - 1) && (preTailNum > 0)) {
            curPrePixNum = preTailNum;
        }

        auto eventId = pingFlag ? EVENT_ID0 : EVENT_ID1;

        // copy in depth and transpose, reducemin
        CopyInDepth(loopId, preLoopId, curPrePixNum, pingFlag);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        TransposeDepth(curPrePixNum, pingFlag);

        PipeBarrier<PIPE_V>();

        ReduceMinDepth(preLoopId, curPrePixNum, pingFlag);

        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);

        pingFlag = 1 - pingFlag;
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::CopyInDepth(uint32_t loopId, uint32_t preLoopId,
                                                                  uint32_t curPrePixNum, uint32_t pingFlag)
{
    LocalTensor<float>
        inLocal = this->inDepthLocal[pingFlag * (ELENUM_REPEAT_FP32 * TRANS_COL_ELENUM + rsv)]; // 40 * 96 + 64

    uint64_t inDepthOffset = this->startPixId + loopId * MAX_PROC_ELENUM + preLoopId * TRANS_COL_ELENUM;

    uint16_t blockCnt = this->vecCoreNum;
    uint32_t blockLen = curPrePixNum * sizeof(float);
    uint32_t srcStride = (this->height * this->width - curPrePixNum) * sizeof(float);
    uint32_t dstStride = (TRANS_COL_ELENUM - curPrePixNum) / ELENUM_BLOCK_FP32;

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    DataCopyPad(inLocal, this->zBufDepthGM[inDepthOffset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::TransposeDepth(uint32_t curPrePixNum, uint32_t pingFlag)
{
    LocalTensor<float>
        srcLocal = this->inDepthLocal[pingFlag * (ELENUM_REPEAT_FP32 * TRANS_COL_ELENUM + rsv)]; // 64 * 96 + 64
    LocalTensor<float>
        dstLocal = this->transDepthLocal[pingFlag * (ELENUM_REPEAT_FP32 * TRANS_COL_ELENUM + rsv)]; // 64 * 96 + 64

    uint64_t repeatsInRow = (this->vecCoreNum - 1) / NCHW_CONV_ADDR_LIST_SIZE + 1;
    uint64_t repeatsInCol = (curPrePixNum - 1) / ELENUM_BLOCK_FP32 + 1;

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = static_cast<uint8_t>(repeatsInRow); // less than 256
    transDataParams.dstRepStride = NCHW_CONV_ADDR_LIST_SIZE / ELENUM_BLOCK_FP32;
    transDataParams.srcRepStride = NCHW_CONV_ADDR_LIST_SIZE * (TRANS_COL_ELENUM / ELENUM_BLOCK_FP32);

    for (uint64_t j = 0; j < repeatsInCol; j++) {
        uint64_t srcLocalList[16];
        uint64_t dstLocalList[16];

        for (uint64_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
            srcLocalList[i] = (uint64_t)(srcLocal[j * ELENUM_BLOCK_FP32 + TRANS_COL_ELENUM * i].GetPhyAddr());

            dstLocalList[i] = (uint64_t)(dstLocal[j * ELENUM_REPEAT_FP32 * ELENUM_BLOCK_FP32 +
                                                  (i / 2) * ELENUM_REPEAT_FP32 + (i % 2) * ELENUM_BLOCK_FP32]
                                             .GetPhyAddr());
        }
        TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParams);
    }
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::ReduceMinDepth(uint32_t preLoopId, uint32_t curPrePixNum,
                                                                     uint32_t pingFlag)
{
    LocalTensor<float>
        srcLocal = this->transDepthLocal[pingFlag * (ELENUM_REPEAT_FP32 * TRANS_COL_ELENUM + rsv)]; // 64 * 96 + 64

    uint32_t dstLocalOffset = preLoopId * TRANS_COL_ELENUM;
    LocalTensor<float> dstLocal = this->reduceMinIdxLocal[dstLocalOffset]; // 2 * (1920 + 64)

    int32_t mask = this->vecCoreNum;
    int32_t repeatNum = curPrePixNum;
    int32_t srcRepStride = ELENUM_REPEAT_FP32 / ELENUM_BLOCK_FP32;
    WholeReduceMin(dstLocal, srcLocal, mask, repeatNum, 1, 1, srcRepStride, ReduceOrder::ORDER_ONLY_INDEX);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::GenFindicesAndPreVertData(uint32_t loopId, uint32_t curCalPixNum)
{
    LocalTensor<uint32_t> srcLocal = this->reduceMinIdxLocal.template ReinterpretCast<uint32_t>(); // 2 * (1920 + 64)
    LocalTensor<int32_t> flagLocal = this->barycentricFlagLocal; // save barycentric calculation message
    LocalTensor<int32_t> findicesLocal = this->baryCentricxLocal
                                             .template ReinterpretCast<int32_t>(); // save findices result

    uint64_t findicesStartOffset = static_cast<uint64_t>(this->startPixId) + loopId * MAX_PROC_ELENUM;

    for (uint32_t pixId = 0; pixId < curCalPixNum; pixId++) {
        uint64_t minDepthVecIdx = static_cast<uint64_t>(srcLocal.GetValue(pixId));
        uint64_t findicesOffset = static_cast<uint64_t>(findicesStartOffset + pixId);
        uint64_t zbufIdxOffset = minDepthVecIdx * this->height * this->width + findicesOffset; // up precision
        int32_t findicesValue = this->zBufIdxGM.GetValue(zbufIdxOffset);
        PipeBarrier<PIPE_ALL>();

        // default value of findices space is MAXINT
        if (findicesValue != std::numeric_limits<int32_t>::max()) {
            // zbuffer saves (faceIdx + 1); output the zero-based face index
            int32_t faceIdx = findicesValue - 1;
            if (faceIdx < 0) {
                findicesLocal.SetValue(pixId, static_cast<int32_t>(0));
                flagLocal.SetValue(pixId, static_cast<int32_t>(0));
            } else {
                findicesLocal.SetValue(pixId, faceIdx);
                flagLocal.SetValue(pixId, static_cast<int32_t>(1));
                // prepare vertex data in ub when face idx is valid
                GetVertData(faceIdx, pixId);
            }
        } else {
            // invalid index: default use 0
            findicesLocal.SetValue(pixId, static_cast<int32_t>(0));
            flagLocal.SetValue(pixId, static_cast<int32_t>(0));
        }
    }
    PipeBarrier<PIPE_ALL>();

    // copy out findices, and no need pipe here because of pipe all after ArithProgression
    uint16_t blockCnt = 1;
    uint32_t blockLen = curCalPixNum * sizeof(int32_t);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPad(this->findicesGM[findicesStartOffset], findicesLocal, copyParams);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::GetVertData(int32_t faceIdx, uint32_t pixId)
{
    int32_t vert0 = this->infGM.GetValue(FACE_VERT_NUM * faceIdx);
    int32_t vert1 = this->infGM.GetValue(FACE_VERT_NUM * faceIdx + 1);
    int32_t vert2 = this->infGM.GetValue(FACE_VERT_NUM * faceIdx + 2);
    PipeBarrier<PIPE_ALL>();

    float vert0x = this->invGM.GetValue(VERT_COORD_NUM * vert0);
    float vert0y = this->invGM.GetValue(VERT_COORD_NUM * vert0 + 1);
    float vert0w = this->invGM.GetValue(VERT_COORD_NUM * vert0 + 3);

    float vert1x = this->invGM.GetValue(VERT_COORD_NUM * vert1);
    float vert1y = this->invGM.GetValue(VERT_COORD_NUM * vert1 + 1);
    float vert1w = this->invGM.GetValue(VERT_COORD_NUM * vert1 + 3);

    float vert2x = this->invGM.GetValue(VERT_COORD_NUM * vert2);
    float vert2y = this->invGM.GetValue(VERT_COORD_NUM * vert2 + 1);
    float vert2w = this->invGM.GetValue(VERT_COORD_NUM * vert2 + 3);
    PipeBarrier<PIPE_ALL>();

    this->vert0xLocal.SetValue(pixId, vert0x);
    this->vert0yLocal.SetValue(pixId, vert0y);
    this->vert0wLocal.SetValue(pixId, vert0w);

    this->vert1xLocal.SetValue(pixId, vert1x);
    this->vert1yLocal.SetValue(pixId, vert1y);
    this->vert1wLocal.SetValue(pixId, vert1w);

    this->vert2xLocal.SetValue(pixId, vert2x);
    this->vert2yLocal.SetValue(pixId, vert2y);
    this->vert2wLocal.SetValue(pixId, vert2w);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::CalculateBarycentric(uint32_t loopId, uint32_t curCalPixNum)
{
    GenScreenVertData(loopId, curCalPixNum);

    CalcSignedArea(curCalPixNum);

    GenBarycentricCoord(curCalPixNum);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::GenScreenVertData(uint32_t loopId, uint32_t curCalPixNum)
{
    LocalTensor<int32_t> screenVertxIntLocal = this->screenVertxLocal.template ReinterpretCast<int32_t>();
    LocalTensor<float> tmpLocal = this->pixIdTmpLocal;

    float xmulsCoef = SCREEN_PIX_COEF * (this->widthF32 - 1.0f);
    float xaddsCoef = SCREEN_PIX_COEF * this->widthF32;
    float ymulsCoef = SCREEN_PIX_COEF * (this->heightF32 - 1.0f);
    float yaddsCoef = SCREEN_PIX_COEF * this->heightF32;
    float pixMulsCoef = 1.0f / this->widthF32; // greater than zero

    int32_t count = ((curCalPixNum - 1) / ELENUM_REPEAT_FP32 + 1) * ELENUM_REPEAT_FP32;
    int32_t firstValue = static_cast<int32_t>(this->startPixId + loopId * MAX_PROC_ELENUM);

    ArithProgression(screenVertxIntLocal, firstValue, static_cast<int32_t>(1), count); // gen int32 pix id
    Div(this->screenVert0xLocal, this->vert0xLocal, this->vert0wLocal, count);         // vt0_ptr[0] / vt0_ptr[3]
    Div(this->screenVert1xLocal, this->vert1xLocal, this->vert1wLocal, count);         // vt1_ptr[0] / vt1_ptr[3]
    Div(this->screenVert2xLocal, this->vert2xLocal, this->vert2wLocal, count);         // vt2_ptr[0] / vt2_ptr[3]
    Div(this->screenVert0yLocal, this->vert0yLocal, this->vert0wLocal, count);         // vt0_ptr[1] / vt0_ptr[3]
    Div(this->screenVert1yLocal, this->vert1yLocal, this->vert1wLocal, count);         // vt1_ptr[1] / vt1_ptr[3]
    Div(this->screenVert2yLocal, this->vert2yLocal, this->vert2wLocal, count);         // vt2_ptr[1] / vt2_ptr[3]
    PipeBarrier<PIPE_ALL>();

    Cast(tmpLocal, screenVertxIntLocal, RoundMode::CAST_NONE, count);         // cast int pix to float
    Muls(this->screenVert0xLocal, this->screenVert0xLocal, xmulsCoef, count); // 0.5f * (width - 1)
    Muls(this->screenVert1xLocal, this->screenVert1xLocal, xmulsCoef, count); // 0.5f * (width - 1)
    Muls(this->screenVert2xLocal, this->screenVert2xLocal, xmulsCoef, count); // 0.5f * (width - 1)
    Muls(this->screenVert0yLocal, this->screenVert0yLocal, ymulsCoef, count); // 0.5f * (height - 1)
    Muls(this->screenVert1yLocal, this->screenVert1yLocal, ymulsCoef, count); // 0.5f * (height - 1)
    Muls(this->screenVert2yLocal, this->screenVert2yLocal, ymulsCoef, count); // 0.5f * (height - 1)
    PipeBarrier<PIPE_V>();

    Muls(this->screenVertxLocal, tmpLocal, pixMulsCoef, count);               // pix / width in float
    Adds(this->screenVert0xLocal, this->screenVert0xLocal, xaddsCoef, count); // 0.5f * (width - 1) + 0.5f
    Adds(this->screenVert1xLocal, this->screenVert1xLocal, xaddsCoef, count); // 0.5f * (width - 1) + 0.5f
    Adds(this->screenVert2xLocal, this->screenVert2xLocal, xaddsCoef, count); // 0.5f * (width - 1) + 0.5f
    Adds(this->screenVert0yLocal, this->screenVert0yLocal, yaddsCoef, count); // 0.5f * (height - 1) + 0.5f
    Adds(this->screenVert1yLocal, this->screenVert1yLocal, yaddsCoef, count); // 0.5f * (height - 1) + 0.5f
    Adds(this->screenVert2yLocal, this->screenVert2yLocal, yaddsCoef, count); // 0.5f * (height - 1) + 0.5f
    PipeBarrier<PIPE_V>();

    Cast(this->screenVertyLocal, this->screenVertxLocal, RoundMode::CAST_FLOOR, count); // float(pix / width)
    PipeBarrier<PIPE_V>();
    Muls(this->screenVertxLocal, this->screenVertyLocal, this->widthF32, count); // float(pix / width) * width
    PipeBarrier<PIPE_V>();
    Sub(this->screenVertxLocal, tmpLocal, this->screenVertxLocal, count); // float(pix % width)
    PipeBarrier<PIPE_V>();
    Adds(this->screenVertxLocal, this->screenVertxLocal, SCREEN_PIX_COEF, count); // float(pix % width) + 0.5f
    Adds(this->screenVertyLocal, this->screenVertyLocal, SCREEN_PIX_COEF, count); // float(pix / width) + 0.5f
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::CalcSignedArea(uint32_t curCalPixNum)
{
    LocalTensor<float> betaTriLocal = this->vert0yLocal;
    LocalTensor<float> gammaTriLocal = this->vert1yLocal;
    LocalTensor<float> areaLocal = this->vert2yLocal;

    LocalTensor<float> alphaLocal = this->baryCentricxLocal;
    LocalTensor<float> betaLocal = this->baryCentricyLocal;
    LocalTensor<float> gammaLocal = this->baryCentriczLocal;

    int32_t count = ((curCalPixNum - 1) / ELENUM_REPEAT_FP32 + 1) * ELENUM_REPEAT_FP32;

    Sub(betaTriLocal, this->screenVertxLocal, this->screenVert0xLocal, count); // p - a
    Sub(gammaTriLocal, this->screenVertyLocal, this->screenVert0yLocal, count);

    Sub(areaLocal, this->screenVert1xLocal, this->screenVert0xLocal, count); // b - a
    Sub(alphaLocal, this->screenVert1yLocal, this->screenVert0yLocal, count);

    Sub(betaLocal, this->screenVert2xLocal, this->screenVert0xLocal, count); // c - a
    Sub(gammaLocal, this->screenVert2yLocal, this->screenVert0yLocal, count);
    PipeBarrier<PIPE_V>();

    Mul(this->screenVertxLocal, betaLocal, gammaTriLocal, count); // (c0 - a0) * (p1 - a1)
    Mul(this->screenVertyLocal, gammaLocal, betaTriLocal, count); // (c1 - a1) * (p0 - a0)

    Mul(this->screenVert1xLocal, betaLocal, alphaLocal, count); // (c0 - a0) * (b1 - a1)
    Mul(this->screenVert1yLocal, gammaLocal, areaLocal, count); // (c1 - a1) * (b0 - a0)

    Mul(this->screenVert2xLocal, betaTriLocal, alphaLocal, count); // (p0 - a0) * (b1 - a1)
    Mul(this->screenVert2yLocal, gammaTriLocal, areaLocal, count); // (p1 - a1) * (b0 - a0)
    PipeBarrier<PIPE_V>();

    Sub(betaTriLocal, this->screenVertxLocal, this->screenVertyLocal, count);    // apc
    Sub(gammaTriLocal, this->screenVert2xLocal, this->screenVert2yLocal, count); // abp
    Sub(areaLocal, this->screenVert1xLocal, this->screenVert1yLocal, count);     // abc
    PipeBarrier<PIPE_V>();

    // save where is zero area
    CompareScalar(this->areaZeroMaskLocal, areaLocal, 0.0f, CMPMODE::NE, count);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::GenBarycentricCoord(uint32_t curCalPixNum)
{
    LocalTensor<float> betaTriLocal = this->vert0yLocal;
    LocalTensor<float> gammaTriLocal = this->vert1yLocal;
    LocalTensor<float> areaLocal = this->vert2yLocal;
    LocalTensor<float> alphaLocal = this->baryCentricxLocal;
    LocalTensor<float> betaLocal = this->baryCentricyLocal;
    LocalTensor<float> gammaLocal = this->baryCentriczLocal;

    LocalTensor<int32_t> alphaIntLocal = alphaLocal.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> betaIntLocal = betaLocal.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> gammaIntLocal = gammaLocal.template ReinterpretCast<int32_t>();
    LocalTensor<float> wCoefLocal = this->pixIdTmpLocal;
    int32_t count = ((curCalPixNum - 1) / ELENUM_REPEAT_FP32 + 1) * ELENUM_REPEAT_FP32;

    Div(betaLocal, betaTriLocal, areaLocal, count);   // beta = beta_tri / area
    Div(gammaLocal, gammaTriLocal, areaLocal, count); // gamma = gamma_tri / area
    PipeBarrier<PIPE_V>();
    Add(alphaLocal, betaLocal, gammaLocal, count); // beta + gamma
    PipeBarrier<PIPE_V>();
    Muls(alphaLocal, alphaLocal, -1.0f, count); // - beta - gamma
    PipeBarrier<PIPE_V>();
    Adds(alphaLocal, alphaLocal, 1.0f, count); // alpha = 1.0 - beta - gamma
    PipeBarrier<PIPE_V>();

    Select(alphaLocal, this->areaZeroMaskLocal, alphaLocal, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    Select(betaLocal, this->areaZeroMaskLocal, betaLocal, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    Select(gammaLocal, this->areaZeroMaskLocal, gammaLocal, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    // PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_ALL>(); // avoid mask buffer in use

    // barycentric / vertw
    Div(alphaLocal, alphaLocal, this->vert0wLocal, count);
    Div(betaLocal, betaLocal, this->vert1wLocal, count);
    Div(gammaLocal, gammaLocal, this->vert2wLocal, count);
    PipeBarrier<PIPE_V>();

    Add(wCoefLocal, alphaLocal, betaLocal, count);
    PipeBarrier<PIPE_V>();
    Add(wCoefLocal, wCoefLocal, gammaLocal, count);
    PipeBarrier<PIPE_V>();

    // barycentric / w
    Div(alphaLocal, alphaLocal, wCoefLocal, count);
    Div(betaLocal, betaLocal, wCoefLocal, count);
    Div(gammaLocal, gammaLocal, wCoefLocal, count);
    PipeBarrier<PIPE_V>();

    // set zero for invalid pix
    Mul(alphaIntLocal, alphaIntLocal, this->barycentricFlagLocal, count);
    Mul(betaIntLocal, betaIntLocal, this->barycentricFlagLocal, count);
    Mul(gammaIntLocal, gammaIntLocal, this->barycentricFlagLocal, count);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::CopyOutRslt(uint32_t loopId, uint32_t curCalPixNum)
{
    LocalTensor<uint32_t> maskLocal = this->inDepthLocal.template ReinterpretCast<uint32_t>();
    LocalTensor<float> srcLocal = this->baryCentricxLocal;
    LocalTensor<float> dstLocal = this->vert0wLocal;

    uint16_t maskBlockCnt = 1;
    uint32_t maskBlockLen = MAX_PROC_ELENUM * BARY_COORD_NUM * sizeof(uint32_t);

    DataCopyExtParams maskCopyParams{maskBlockCnt, maskBlockLen, 0, 0, 0};
    DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};

    DataCopyPad(maskLocal, this->maskGM, maskCopyParams, padParams);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    int32_t calCount = static_cast<int32_t>(((curCalPixNum * BARY_COORD_NUM - 1) / ELENUM_REPEAT_FP32 + 1) *
                                            ELENUM_REPEAT_FP32);

    Gather(dstLocal, srcLocal, maskLocal, (uint32_t)0, calCount);
    PipeBarrier<PIPE_ALL>();

    uint16_t rsltBlockCnt = 1;
    uint32_t rsltBlockLen = curCalPixNum * BARY_COORD_NUM * sizeof(float);

    uint64_t outOffset = static_cast<uint64_t>((this->startPixId + loopId * MAX_PROC_ELENUM) * BARY_COORD_NUM);

    DataCopyExtParams rsltCopyParams{rsltBlockCnt, rsltBlockLen, 0, 0, 0};
    DataCopyPad(this->barycentricGM[outOffset], dstLocal, rsltCopyParams);

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void BarycentricFromImgcoordAIV<T>::Process()
{
    GenMaskDataInGM();
    // maskGM is shared by all vec cores and gathered in full by CopyOutRslt.
    // Sync here so every core's mask write is globally visible before any
    // core issues the Gather, otherwise stale workspace data may be read.
    SyncAll();

    if (this->calPixNum == 0) {
        return;
    }
    uint32_t loopNum = this->calPixNum / MAX_PROC_ELENUM;
    uint32_t tailNum = this->calPixNum % MAX_PROC_ELENUM;

    if (tailNum > 0) {
        loopNum++;
    }
    for (uint32_t loopId = 0; loopId < loopNum; loopId++) {
        uint32_t curCalPixNum = MAX_PROC_ELENUM;
        if ((loopId == loopNum - 1) && (tailNum > 0)) {
            curCalPixNum = tailNum;
        }
        // copy in depth and transpose, reducemin
        PreProcess(loopId, curCalPixNum);

        GenFindicesAndPreVertData(loopId, curCalPixNum);

        CalculateBarycentric(loopId, curCalPixNum);

        CopyOutRslt(loopId, curCalPixNum);
    }
    PipeBarrier<PIPE_ALL>();
}
} // namespace BarycentricFromImgcoord

#endif // BARYCENTRIC_FROM_IMGCOORD_AIV_H
