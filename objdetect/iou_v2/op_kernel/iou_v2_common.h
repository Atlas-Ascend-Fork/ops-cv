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
 * \file iou_v2_common.h
 * \brief 公共常量、Align 系列 CopyIn/CopyOut、NotAlign 系列公共基类
 */
#ifndef _IOU_V2_COMMON_H
#define _IOU_V2_COMMON_H

#include "kernel_operator.h"

namespace IouV2 {
using namespace AscendC;

constexpr uint8_t POS_NUM = 4;
constexpr uint8_t src1Pattern3 = 3;
constexpr uint8_t src1Pattern4 = 4;
constexpr uint8_t src1Pattern5 = 5;
constexpr uint8_t src1Pattern6 = 6;
constexpr uint32_t BLOCK_NUM = 8;

// -------- Align 系列：CopyIn / CopyOut 复用 --------
template <typename inType>
__aicore__ inline void AlignCopyIn(TQue<QuePosition::VECIN, 1>& box1Que, TQue<QuePosition::VECIN, 1>& box2Que,
                                   const GlobalTensor<inType>& box1Gm, const GlobalTensor<inType>& box2Gm,
                                   uint64_t totalLength, uint64_t tileLen, uint64_t quadTileLen,
                                   TBuf<TPosition::VECCALC>& fp16Tensor, uint32_t bBoxLoop)
{
    LocalTensor<float> box1Local = box1Que.AllocTensor<float>();
    LocalTensor<float> box2Local = box2Que.AllocTensor<float>();
    bool isLastCore = GetBlockIdx() == GetBlockNum() - 1;
    uint32_t lastCoreLen = totalLength % tileLen == 0 ? tileLen : totalLength % tileLen;
    uint32_t blockLen = isLastCore ? lastCoreLen : tileLen;
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLen * sizeof(inType)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<inType> padParams{false, 0, 0, 0};
    if constexpr (!std::is_same<inType, float>::value) {
        LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
        event_t eventVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVToMTE2);
        event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
        for (uint8_t posId = 0; posId < POS_NUM; ++posId) { // 第一维输入为4
#if __CCE_AICORE__ == 200                                   // 310p
            DataCopy(fp16Buf[tileLen * posId], box1Gm[bBoxLoop * tileLen + posId * totalLength], tileLen);
            DataCopy(fp16Buf[tileLen * (posId + 4)], box2Gm[bBoxLoop * tileLen + posId * totalLength],
                     tileLen); // 第一维输入为4
#else
            AscendC::DataCopyPad(fp16Buf[tileLen * posId], box1Gm[bBoxLoop * tileLen + posId * totalLength], copyParams,
                                 padParams);
            AscendC::DataCopyPad(fp16Buf[tileLen * (posId + 4)], box2Gm[bBoxLoop * tileLen + posId * totalLength],
                                 copyParams, padParams);
#endif
        }
        event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        Cast(box1Local, fp16Buf, RoundMode::CAST_NONE, quadTileLen);
        Cast(box2Local, fp16Buf[quadTileLen], RoundMode::CAST_NONE, quadTileLen);
    } else {
        for (uint8_t posId = 0; posId < POS_NUM; ++posId) { // 第一维输入为4
#if __CCE_AICORE__ == 200                                   // 310p
            DataCopy(box1Local[tileLen * posId].ReinterpretCast<inType>(),
                     box1Gm[bBoxLoop * tileLen + posId * totalLength], tileLen);
            DataCopy(box2Local[tileLen * posId].ReinterpretCast<inType>(),
                     box2Gm[bBoxLoop * tileLen + posId * totalLength], tileLen); // 第一维输入为4
#else
            AscendC::DataCopyPad(box1Local[tileLen * posId].ReinterpretCast<inType>(),
                                 box1Gm[bBoxLoop * tileLen + posId * totalLength], copyParams, padParams);
            AscendC::DataCopyPad(box2Local[tileLen * posId].ReinterpretCast<inType>(),
                                 box2Gm[bBoxLoop * tileLen + posId * totalLength], copyParams, padParams);
#endif
        }
    }
    box1Que.EnQue(box1Local);
    box2Que.EnQue(box2Local);
}

template <typename inType>
__aicore__ inline void AlignCopyOut(TQue<QuePosition::VECOUT, 1>& outQue, const GlobalTensor<inType>& outGm,
                                    TBuf<TPosition::VECCALC>& fp16Tensor, uint64_t totalLength, uint64_t tileLen,
                                    uint32_t bBoxLoop)
{
    LocalTensor<float> outLocal = outQue.DeQue<float>();
    bool isLastCore = GetBlockIdx() == GetBlockNum() - 1;
    uint32_t lastCoreLen = totalLength % tileLen == 0 ? tileLen : totalLength % tileLen;
    uint32_t blockLen = isLastCore ? lastCoreLen : tileLen;
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLen * sizeof(inType)), 0, 0, 0};
    if constexpr (!std::is_same<inType, float>::value) {
        LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
#if __CCE_AICORE__ == 200 // 310p
        Cast(fp16Buf, outLocal, RoundMode::CAST_NONE, tileLen);
#else
        Cast(fp16Buf, outLocal, RoundMode::CAST_RINT, tileLen);
#endif
        event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
#if __CCE_AICORE__ == 200 // 310p
        DataCopy(outGm[bBoxLoop * tileLen], fp16Buf, tileLen);
#else
        AscendC::DataCopyPad(outGm[bBoxLoop * tileLen], fp16Buf, copyParams);
#endif
    } else {
#if __CCE_AICORE__ == 200 // 310p
        DataCopy(outGm[bBoxLoop * tileLen], outLocal.ReinterpretCast<inType>(), tileLen);
#else
        AscendC::DataCopyPad(outGm[bBoxLoop * tileLen], outLocal.ReinterpretCast<inType>(), copyParams);
#endif
    }
    outQue.FreeTensor(outLocal);
}

// -------- Align 系列：公共基类 Init / Process / 成员 --------
template <typename inType, template <typename> class AlignDerived>
class KernelIouV2AlignBase {
public:
    __aicore__ inline KernelIouV2AlignBase() {}

    __aicore__ inline void Init(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap, const IouV2TilingData* tilingData,
                                uint64_t tmpTensorBytes)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be 0!");
        this->eps = tilingData->eps;
        this->totalLength = tilingData->gtBoxLength;
        this->tileLen = tilingData->tileLength;
        this->doubleTileLen = 2 * tileLen;         // 2倍tileLen
        this->tripTileLen = 3 * tileLen;           // 3倍tileLen
        this->quadTileLen = 4 * tileLen;           // 4倍tileLen
        this->octaTileLen = 8 * tileLen;           // 8倍tileLen
        this->doubleTotalLength = 2 * totalLength; // 2倍totalLength
        this->tripTotalLength = 3 * totalLength;   // 3倍totalLength

        uint32_t blockId = GetBlockIdx();
        // blockOffsetLen是核间偏移量
        uint32_t blockOffsetLen = blockId > tilingData->frontCoreNum ?
                                      tileLen * (tilingData->frontCoreNum + tilingData->loopNum * blockId) :
                                      tileLen * (tilingData->loopNum + 1) * blockId;
        this->loopNum = blockId < tilingData->frontCoreNum ? (tilingData->loopNum + 1) : tilingData->loopNum;

        uint64_t bufferSize = tileLen * 4; // 输入shape为(4, n)
        box1Gm.SetGlobalBuffer((__gm__ inType*)bboxes + blockOffsetLen, bufferSize);
        box2Gm.SetGlobalBuffer((__gm__ inType*)gtboxes + blockOffsetLen, bufferSize);
        outGm.SetGlobalBuffer((__gm__ inType*)overlap + blockOffsetLen, tileLen);

        // 核内每次循环的大小，数据类型经过cast之后都是float
        pipe.InitBuffer(box1Que, 1, quadTileLen * 4); // float大小为4
        pipe.InitBuffer(box2Que, 1, quadTileLen * 4); // float大小为4
        pipe.InitBuffer(outQue, 1, tileLen * 4);      // float大小为4
        pipe.InitBuffer(tmpTensor, tmpTensorBytes);

        if constexpr (!std::is_same<inType, float>::value) {
            pipe.InitBuffer(fp16Tensor, octaTileLen * 2); // sizeof(inType) = 2
        }
    }

    __aicore__ inline void Process()
    {
        for (int32_t bBoxLoop = 0; bBoxLoop < this->loopNum; ++bBoxLoop) {
            AlignCopyIn<inType>(box1Que, box2Que, box1Gm, box2Gm, totalLength, tileLen, quadTileLen, fp16Tensor,
                                bBoxLoop);
            static_cast<AlignDerived<inType>*>(this)->Compute();
            AlignCopyOut<inType>(outQue, outGm, fp16Tensor, totalLength, tileLen, bBoxLoop);
        }
    }

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> box1Que;
    TQue<QuePosition::VECIN, 1> box2Que;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> tmpTensor;
    TBuf<TPosition::VECCALC> fp16Tensor;
    GlobalTensor<inType> box1Gm;
    GlobalTensor<inType> box2Gm;
    GlobalTensor<inType> outGm;
    uint64_t loopNum;
    uint64_t tileLen;
    uint64_t doubleTileLen;
    uint64_t tripTileLen;
    uint64_t quadTileLen;
    uint64_t octaTileLen;
    uint64_t totalLength;
    uint64_t doubleTotalLength;
    uint64_t tripTotalLength;
    float eps;
};

// -------- NotAlign 系列：公共基类 Init / CopyIn / CopyOut / CommonCopyOut --------
template <typename inType, template <typename> class Derived>
class KernelIouV2NotAlignBase {
public:
    __aicore__ inline KernelIouV2NotAlignBase() {}

    __aicore__ inline void Init(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap, const IouV2TilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be 0!");
        this->eps = tilingData->eps;
        this->bBoxLength = tilingData->bBoxLength;
        this->gtBoxLength = tilingData->gtBoxLength;
        this->tileLen = tilingData->tileLength;
        this->quadTileLen = 4 * tileLen; // 4倍tileLen
        this->loopTileLen = tilingData->subTileLen;
        this->doubleLoopTileLen = 2 * loopTileLen; // 2倍loopTileLen
        this->tripLoopTileLen = 3 * loopTileLen;   // 3倍loopTileLen
        this->quadLoopTileLen = 4 * loopTileLen;   // 4倍loopTileLen
        this->mulLen = tileLen * loopTileLen;
        this->doubleMulLen = 2 * mulLen; // 2倍mulLen
        this->tripMulLen = 3 * mulLen;   // 3倍mulLen
        this->quadMulLen = 4 * mulLen;   // 4倍mulLen
        this->pentaMulLen = 5 * mulLen;  // 5倍mulLen
        this->hexaMulLen = 6 * mulLen;   // 6倍mulLen
        this->septaMulLen = 7 * mulLen;  // 7倍mulLen
        this->octaMulLen = 8 * mulLen;   // 8倍mulLen
        this->totalLen = tileLen + loopTileLen;
        this->doubleTotalLen = 2 * totalLen;             // 2倍totalLen
        this->tripTotalLen = 3 * totalLen;               // 3倍totalLen
        this->quadTotalLen = 4 * totalLen;               // 4倍totalLen
        this->totalLen1 = 3 * loopTileLen + 2 * tileLen; // 3倍loopTileLen+2倍tileLen的偏移量
        this->totalLen2 = 2 * loopTileLen + tileLen;     // 2倍loopTileLen+1倍tileLen的偏移量
        this->loopTileNum = (bBoxLength + tileLen - 1) / tileLen;

        uint32_t blockId = GetBlockIdx();
        // blockOffsetLen是核间偏移量
        this->blockOffsetLen = blockId > tilingData->frontCoreNum ?
                                   tileLen * (tilingData->frontCoreNum + tilingData->loopNum * blockId) :
                                   tileLen * (tilingData->loopNum + 1) * blockId;
        this->loopNum = blockId < tilingData->frontCoreNum ? tilingData->loopNum + 1 : tilingData->loopNum;

        box1Gm.SetGlobalBuffer((__gm__ inType*)bboxes, loopTileNum * loopTileLen * 4);
        box2Gm.SetGlobalBuffer((__gm__ inType*)gtboxes + blockOffsetLen * 4, tileLen * 4);
        outGm.SetGlobalBuffer((__gm__ inType*)overlap + blockOffsetLen * bBoxLength, bBoxLength * tileLen);

        // 核内每次循环的大小，数据类型经过cast之后都是float
        pipe.InitBuffer(inQue, 1, octaMulLen * 4);     // float大小为4
        pipe.InitBuffer(outQue, 1, mulLen * 4);        // float大小为4
        pipe.InitBuffer(tmpTensor1, quadTotalLen * 4); // float大小为4
        pipe.InitBuffer(tmpTensor2, quadTotalLen * 4); // float大小为4

        if constexpr (!std::is_same<inType, float>::value) {
            pipe.InitBuffer(fp16Tensor, mulLen * 2); // sizeof(inType) = 2
        }
    }

    __aicore__ inline void Process()
    {
        for (int32_t gtBoxLoop = 0; gtBoxLoop < this->loopNum; ++gtBoxLoop) {
            gmOffset = gtBoxLoop * quadTileLen;
            for (int32_t bBoxLoop = loopTileNum - 1; bBoxLoop >= 0; --bBoxLoop) {
                CopyIn(bBoxLoop);
                static_cast<Derived<inType>*>(this)->Compute(bBoxLoop);
                CopyOut(gtBoxLoop, bBoxLoop);
            }
        }
    }

protected:
    __aicore__ inline void CopyIn(uint32_t bBoxLoop)
    {
        LocalTensor<float> boxLocal = inQue.AllocTensor<float>();
        bool isLastCore = GetBlockIdx() == GetBlockNum() - 1;
        bool isLastLoop = bBoxLoop == loopTileNum - 1;
        uint32_t lastCoreLen = gtBoxLength % tileLen == 0 ? tileLen : gtBoxLength % tileLen;
        uint32_t lastLoopLen = bBoxLength % loopTileLen == 0 ? loopTileLen : bBoxLength % loopTileLen;
        uint32_t blockLen0 = isLastLoop ? lastLoopLen * 4 : quadLoopTileLen;
        uint32_t blockLen1 = isLastCore ? lastCoreLen * 4 : quadTileLen;
        AscendC::DataCopyExtParams copyParams0{1, static_cast<uint32_t>(blockLen0 * sizeof(inType)), 0, 0, 0};
        AscendC::DataCopyExtParams copyParams1{1, static_cast<uint32_t>(blockLen1 * sizeof(inType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<inType> padParams{false, 0, 0, 0};
        if constexpr (!std::is_same<inType, float>::value) {
            LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
            event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
#if __CCE_AICORE__ == 200 // 310p
            DataCopy(fp16Buf, box1Gm[bBoxLoop * quadLoopTileLen], quadLoopTileLen);
            DataCopy(fp16Buf[quadLoopTileLen], box2Gm[gmOffset], quadTileLen);
#else
            AscendC::DataCopyPad(fp16Buf, box1Gm[bBoxLoop * quadLoopTileLen], copyParams0, padParams);
            AscendC::DataCopyPad(fp16Buf[quadLoopTileLen], box2Gm[gmOffset], copyParams1, padParams);
#endif
            event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            Cast(boxLocal, fp16Buf, RoundMode::CAST_NONE, quadTotalLen);
        } else {
            uint32_t blockLen = quadLoopTileLen;
#if __CCE_AICORE__ == 200 // 310p
            DataCopy(boxLocal.ReinterpretCast<inType>(), box1Gm[bBoxLoop * quadLoopTileLen], quadLoopTileLen);
            DataCopy(boxLocal[quadLoopTileLen].ReinterpretCast<inType>(), box2Gm[gmOffset], quadTileLen);
#else
            AscendC::DataCopyPad(boxLocal.ReinterpretCast<inType>(), box1Gm[bBoxLoop * quadLoopTileLen], copyParams0,
                                 padParams);
            AscendC::DataCopyPad(boxLocal[quadLoopTileLen].ReinterpretCast<inType>(), box2Gm[gmOffset], copyParams1,
                                 padParams);
#endif
        }
        event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        uint64_t rsvdCnt = 0;
        LocalTensor<float> tmpBuffer1 = tmpTensor1.Get<float>();
        GatherMask(tmpBuffer1, boxLocal, src1Pattern3, true, quadTotalLen, {1, 1, 8, 8},
                   rsvdCnt); // src1Pattern = 3，每四个元素取第一个
        GatherMask(tmpBuffer1[totalLen], boxLocal, src1Pattern4, true, quadTotalLen, {1, 1, 8, 8},
                   rsvdCnt); // src1Pattern = 4，每四个元素取第二个
        GatherMask(tmpBuffer1[doubleTotalLen], boxLocal, src1Pattern5, true, quadTotalLen, {1, 1, 8, 8},
                   rsvdCnt); // src1Pattern = 5，每四个元素取第三个
        GatherMask(tmpBuffer1[tripTotalLen], boxLocal, src1Pattern6, true, quadTotalLen, {1, 1, 8, 8},
                   rsvdCnt); // src1Pattern = 6，每四个元素取第四个
        inQue.FreeTensor(boxLocal);
        LocalTensor<float> tmpBuffer2 = tmpTensor2.Get<float>();
        uint16_t blockLen = loopTileLen / 8; // 8个数组成一个block
        uint16_t srcStride = tileLen / 8;    // 8个数组成一个block
        DataCopy(tmpBuffer2, tmpBuffer1, {4, blockLen, srcStride, 0});
        blockLen = tileLen / 8;      // 8个数组成一个block
        srcStride = loopTileLen / 8; // 8个数组成一个block
        DataCopy(tmpBuffer2[quadLoopTileLen], tmpBuffer1[loopTileLen], {4, blockLen, srcStride, 0});
    }

    __aicore__ inline void NotAlignComputePrepareBox(LocalTensor<float>& boxLocal, LocalTensor<float>& tmpBuffer2) const
    {
        event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        Duplicate(boxLocal, (float)0.0, octaMulLen);
        uint8_t bBoxRepStride = loopTileLen / 8; // 8个数组成一个block
        BinaryRepeatParams bBoxAddRepParams{1, 1, 1, bBoxRepStride, bBoxRepStride, 0};
        Add(boxLocal, boxLocal, tmpBuffer2, loopTileLen, tileLen, bBoxAddRepParams);
        Add(boxLocal[mulLen], boxLocal[mulLen], tmpBuffer2[loopTileLen], loopTileLen, tileLen, bBoxAddRepParams);
        Add(boxLocal[doubleMulLen], boxLocal[doubleMulLen], tmpBuffer2[doubleLoopTileLen], loopTileLen, tileLen,
            bBoxAddRepParams);
        Add(boxLocal[tripMulLen], boxLocal[tripMulLen], tmpBuffer2[tripLoopTileLen], loopTileLen, tileLen,
            bBoxAddRepParams);
        uint16_t gtBoxRepTimes = loopTileLen / 8; // 8个数组成一个block
        uint8_t brcbRepTimes = tileLen / 2;
        BrcbRepeatParams gtBoxBrcbRepParams{gtBoxRepTimes, static_cast<uint16_t>(loopTileLen)};
        for (uint16_t repTime = 0; repTime < gtBoxRepTimes; ++repTime) {
            Brcb(boxLocal[quadMulLen + repTime * BLOCK_NUM], tmpBuffer2[quadLoopTileLen], brcbRepTimes,
                 gtBoxBrcbRepParams); // 8个数组成一个block
        }
    }

    template <typename T>
    __aicore__ inline void CommonCopyOutBoundaryStep(LocalTensor<T>& tmpLocal, uint64_t i, uint64_t num1,
                                                     uint64_t bBoxNotAlign, uint64_t tmpGmOffsetVal,
                                                     uint64_t& tmpGmOffset, uint64_t& tmpUbOffset, uint64_t& addNum,
                                                     uint64_t& headNum) const
    {
        event_t eventMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
        if (addNum == 0) {
            headNum = loopTileLen - (tileLen - i) * bBoxNotAlign;
            if (headNum != bBoxNotAlign) {
                for (uint64_t index = 0; index < headNum; ++index) {
                    T tensorValue = tmpLocal.GetValue(tmpUbOffset + bBoxNotAlign - headNum + index);
                    tmpLocal.SetValue(tmpUbOffset + index, tensorValue);
                }
            }
            addNum += headNum;
        }
        for (uint64_t index = 0; index < bBoxNotAlign; ++index) {
            T tensorValue = tmpLocal.GetValue(loopTileLen * i + index);
            tmpLocal.SetValue(tmpUbOffset + addNum + index, tensorValue);
        }
        addNum += bBoxNotAlign;
        if (i == num1 - 1) {
            uint32_t blockLen = static_cast<uint32_t>(addNum);
            blockLen = blockLen > loopTileLen ? static_cast<uint32_t>(loopTileLen) : blockLen;
            if (blockLen == 0) {
                return;
            }

            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLen * sizeof(inType)), 0, 0, 0};
#if __CCE_AICORE__ == 200 // 310p
            DataCopy(outGm[tmpGmOffsetVal + bBoxLength - headNum], tmpLocal[tmpUbOffset], loopTileLen);
#else
            AscendC::DataCopyPad(outGm[tmpGmOffsetVal + bBoxLength - headNum], tmpLocal[tmpUbOffset], copyParams);
#endif
        }
    }

    template <typename T>
    __aicore__ inline void CommonCopyOut(LocalTensor<T>& tmpLocal, uint32_t gtBoxLoop, uint32_t bBoxLoop)
    {
        uint64_t num1 = tileLen;
        if (gtBoxLoop == (loopNum - 1) && (gtBoxLength - blockOffsetLen - gtBoxLoop * tileLen) < tileLen) {
            num1 = gtBoxLength - blockOffsetLen - gtBoxLoop * tileLen;
        }
        uint64_t gmLenEachLoop = bBoxLength * tileLen;
        uint64_t bBoxNotAlign = bBoxLength - bBoxLoop * loopTileLen;
        uint64_t tmpGmOffset{0};
        uint64_t tmpUbOffset{0};
        uint64_t addNum{0};
        uint64_t headNum{0};
        for (uint64_t i = 0; i < num1; ++i) {
            uint64_t offset = gtBoxLoop * gmLenEachLoop + bBoxLoop * loopTileLen + i * bBoxLength;
            PipeBarrier<PIPE_MTE3>();
            bool isBoundary = (bBoxLoop == (loopTileNum - 1) && bBoxNotAlign > 0 &&
                               (offset + loopTileLen) > gmLenEachLoop * (gtBoxLoop + 1));
            if (isBoundary) {
                CommonCopyOutBoundaryStep(tmpLocal, i, num1, bBoxNotAlign, tmpGmOffset, tmpGmOffset, tmpUbOffset,
                                          addNum, headNum);
            } else {
                // blockLen 是当前 bBox 切片沿列方向的有效行数，不能再用 gmLen - offset：
                // gtBoxLoop>=1 时 offset 会 >= 单次 loop 的 gmLen，无符号减法得到 0 或回绕值，导致漏写。
                uint32_t blockLen = static_cast<uint32_t>(loopTileLen);
                if (bBoxLoop * loopTileLen + blockLen > bBoxLength) {
                    blockLen = static_cast<uint32_t>(bBoxLength - bBoxLoop * loopTileLen);
                }
                if (blockLen == 0) {
                    continue;
                }
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockLen * sizeof(inType)), 0, 0, 0};
#if __CCE_AICORE__ == 200 // 310p
                DataCopy(outGm[offset], tmpLocal[loopTileLen * i], loopTileLen);
#else
                AscendC::DataCopyPad(outGm[offset], tmpLocal[loopTileLen * i], copyParams);
#endif
                tmpGmOffset = offset;
                tmpUbOffset = loopTileLen * i;
            }
        }
    }

    __aicore__ inline void CopyOut(uint32_t gtBoxLoop, uint32_t bBoxLoop)
    {
        LocalTensor<float> outLocal = outQue.DeQue<float>();

        if constexpr (!std::is_same<inType, float>::value) {
            LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
#if __CCE_AICORE__ == 200 // 310p
            Cast(fp16Buf, outLocal, RoundMode::CAST_NONE, mulLen);
#else
            Cast(fp16Buf, outLocal, RoundMode::CAST_RINT, mulLen);
#endif
            event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
            CommonCopyOut<inType>(fp16Buf, gtBoxLoop, bBoxLoop);
        } else {
            CommonCopyOut<float>(outLocal, gtBoxLoop, bBoxLoop);
        }
        outQue.FreeTensor(outLocal);
    }

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> tmpTensor1;
    TBuf<TPosition::VECCALC> tmpTensor2;
    TBuf<TPosition::VECCALC> fp16Tensor;
    GlobalTensor<inType> box1Gm;
    GlobalTensor<inType> box2Gm;
    GlobalTensor<inType> outGm;
    uint64_t loopNum;
    uint64_t tileLen;
    uint64_t quadTileLen;
    uint64_t mulLen;
    uint64_t doubleMulLen;
    uint64_t tripMulLen;
    uint64_t quadMulLen;
    uint64_t pentaMulLen;
    uint64_t hexaMulLen;
    uint64_t septaMulLen;
    uint64_t octaMulLen;
    uint64_t totalLen1;
    uint64_t totalLen2;
    uint64_t gmOffset;
    uint64_t loopTileNum;
    uint64_t bBoxLength;
    uint64_t gtBoxLength;
    uint64_t loopTileLen;
    uint64_t doubleLoopTileLen;
    uint64_t tripLoopTileLen;
    uint64_t quadLoopTileLen;
    uint64_t totalLen;
    uint64_t doubleTotalLen;
    uint64_t tripTotalLen;
    uint64_t quadTotalLen;
    uint32_t blockOffsetLen;
    float eps;
};

} // namespace IouV2
#endif // _IOU_V2_COMMON_H
