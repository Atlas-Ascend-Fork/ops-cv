/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_nearest_neighbor_v2_nchw_gather_hw.h
 * \brief resize_nearest_neighbor_v2_nchw_gather_hw.h
 */
#ifndef RESIZE_NEAREAST_NEIGHBOR_V2_NCHW_GATHER_H
#define RESIZE_NEAREAST_NEIGHBOR_V2_NCHW_GATHER_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "resize_nearest_neighbor_v2_tiling_key.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;
using AscendC::Reg::AddrReg;
using AscendC::Reg::CreateAddrReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

template <typename T, typename T1, int schId, bool alignCorners>
class ResizeGather {
public:
    __aicore__ inline ResizeGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR y,
                                const ResizeNearestNeighborV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t ComputeOriIds(int64_t dstIds, int64_t maxIds, float scale, float bais);
    __aicore__ inline void GatherOutput(LocalTensor<T>& outUb, LocalTensor<T>& srcUb, int64_t ubFactor, int64_t srcLen,
                                        int64_t dstLen, int64_t srcHwStart);
    __aicore__ inline void DataCopyIn(LocalTensor<T>& xLocal, int64_t blockCount, int64_t blockLen, int64_t srcStride,
                                      int64_t offset);
    __aicore__ inline void DataCopyOut(LocalTensor<T>& yLocal, int64_t blockCount, int64_t blockLen, int64_t srcStride,
                                       int64_t offset);
    __aicore__ inline void ComputeDataCopyGather(LocalTensor<T>& srcUb, LocalTensor<T>& outUb, LocalTensor<T1>& idxHwUb,
                                                 int64_t num);
    __aicore__ inline void ComputeAllHw();
    __aicore__ inline void ComputeIdsSpecial(int64_t dstWSizeAlgin, int64_t srcWSizeAlgin);
    __aicore__ inline void ComputeCutH();
    __aicore__ inline void ComputeOriHIdx(LocalTensor<T1>& idxHUb, LocalTensor<T1>& idxH1Ub, int64_t onceHsize,
                                          int64_t hoStart, int64_t hiStart);
    __aicore__ inline void ComputeOriHWidx(LocalTensor<T1>& idxWUb, LocalTensor<T1>& idxH1Ub, LocalTensor<T1>& idxHwUb,
                                           int64_t onceHsize);
    __aicore__ inline void ComputeHWids(LocalTensor<T1>& idxUb, LocalTensor<T1>& idxHUb, LocalTensor<T1>& idxWUb,
                                        int64_t dstWSizeAlgin);
    __aicore__ inline void ComputeHids(LocalTensor<T1>& idxHUb, int64_t hFactor);
    constexpr static int32_t bufferNum = 2;

    constexpr static AscendC::Reg::CastTrait castTraitRound = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::Reg::CastTrait castTraitFloor = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};
    constexpr static AscendC::Reg::CastTrait castInt32ToF = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> idxBuf_;
    TBuf<QuePosition::VECCALC> idxDstBuf_;
    TBuf<QuePosition::VECCALC> idxHBuf_;
    TBuf<QuePosition::VECCALC> idxH1Buf_;
    TBuf<QuePosition::VECCALC> idxWBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int32_t blockIdx_ = 0;
    int32_t vlLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    int32_t vlLenB32_ = Ops::Base::GetVRegSize() / sizeof(int32_t);
    float hScale_ = 0.0f;
    float wScale_ = 0.0f;
    int64_t srcHSize_ = 0;
    int64_t srcWSize_ = 0;
    int64_t srcWAlignSize_ = 0;
    int64_t dstHSize_ = 0;
    int64_t dstWSize_ = 0;
    int64_t dstWAlignSize_ = 0;
    int64_t ncFactor_ = 0;
    int64_t dstHwNum_ = 0;
    int64_t srcHwNum_ = 0;
    int64_t xUb_ = 0;
    int64_t yUb_ = 0;
    int64_t idsUb_ = 0;
    int64_t tailTimes_ = 0;
    int64_t hTimes_ = 0; // 切h时表示hLoopTimes
    int64_t beforeTimes_ = 0;
    int64_t tailTNum_ = 0;
    int64_t beforeTNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t blockFactor_ = 0;
    int64_t realCoreNum_ = 0;
    float bias_ = 0.0f;
    int32_t blockNumSize_ = Ops::Base::GetUbBlockSize() / sizeof(T);
    // tiling params
    const ResizeNearestNeighborV2TilingData* tiling_;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyParams_{1, 1, 0, 0, 0};
};

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, const ResizeNearestNeighborV2TilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    tiling_ = tilingData;

    // 接收tilingdata信息

    hScale_ = tiling_->scaleH;
    wScale_ = tiling_->scaleW;
    srcHSize_ = tiling_->lenSrcH;
    srcWSize_ = tiling_->lenSrcW;
    dstHSize_ = tiling_->lenDesH;
    dstWSize_ = tiling_->lenDesW;

    dstWAlignSize_ = (dstWSize_ + blockNumSize_ - 1) / blockNumSize_ * blockNumSize_;
    srcWAlignSize_ = (srcWSize_ + blockNumSize_ - 1) / blockNumSize_ * blockNumSize_;

    dstHwNum_ = dstHSize_ * dstWSize_;
    srcHwNum_ = srcHSize_ * srcWSize_;
    xUb_ = tilingData->splitFactorTailDesW; // 输入x的ub大小
    yUb_ = tilingData->splitFactorDesW;     // 输出y的ub大小
    idsUb_ = tilingData->ubSize;
    realCoreNum_ = tilingData->realCoreNum;
    tailTimes_ = tilingData->nLoopTimesLast;
    beforeTimes_ = tilingData->nLoopTimesBefore;
    tailTNum_ = tilingData->nLoopTailLast;

    beforeTNum_ = tilingData->nLoop;
    ubFactor_ = tilingData->splitFactorDesH;
    blockFactor_ = tilingData->splitBlockFactor;
    if (tilingData->halfPixelCenters == 1) {
        bias_ = 0.5f;
    }
    hTimes_ = tilingData->splitCountDesH; // 切h时,h反向循环次数

    pipe.InitBuffer(inQue_, bufferNum, xUb_);
    pipe.InitBuffer(outQue_, bufferNum, yUb_);
    pipe.InitBuffer(idxBuf_, idsUb_);                       // 存放h*w坐标
    pipe.InitBuffer(idxWBuf_, dstWAlignSize_ * sizeof(T1)); // 放w的坐标
    if (schId == TPL_SCH_MODE_GATHER_ALL_HW) {
        int64_t dstHAlignSize = (dstHSize_ + blockNumSize_ - 1) / blockNumSize_ * blockNumSize_;
        pipe.InitBuffer(idxHBuf_, dstHAlignSize * sizeof(T1)); // 放h的坐标
    }
    if constexpr (schId == TPL_SCH_MODE_GATHER_CUT_H) {
        int64_t dstHAlignSize = (ubFactor_ + blockNumSize_ - 1) / blockNumSize_ * blockNumSize_;
        pipe.InitBuffer(idxH1Buf_, dstHAlignSize * sizeof(T1)); // 对应的输入h坐标
        pipe.InitBuffer(idxHBuf_, dstHAlignSize * sizeof(T1));  // 输出h坐标
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline int64_t ResizeGather<T, T1, schId, alignCorners>::ComputeOriIds(int64_t dstIds, int64_t maxIds,
                                                                                  float scale, float bais)
{
    int64_t srcIds = 0;
    if constexpr (alignCorners) {
        // round
        srcIds = static_cast<int64_t>(static_cast<int32_t>(((static_cast<float>(dstIds) + bais) * scale) + 0.5f));
    } else {
        // floor
        srcIds = static_cast<int64_t>(static_cast<int32_t>((static_cast<float>(dstIds) + bais) * scale));
    }
    if (srcIds > maxIds) {
        srcIds = maxIds;
    }
    return srcIds;
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::GatherOutput(LocalTensor<T>& outUb,
                                                                              LocalTensor<T>& srcUb, int64_t ubFactor,
                                                                              int64_t srcLen, int64_t dstLen,
                                                                              int64_t srcHwStart)
{
    uint16_t times = CeilDivision(dstLen, vlLen_);
    LocalTensor<T1> idxUb = idxBuf_.AllocTensor<T1>();
    auto idxUbAddr = (__ubuf__ T1*)idxUb.GetPhyAddr();
    auto srcUbAddr = (__ubuf__ T*)srcUb.GetPhyAddr();
    auto dstUbAddr = (__ubuf__ T*)outUb.GetPhyAddr();
    T1 srcHwNum = srcHwStart;
    auto dstUbAddr1 = (__ubuf__ T*)outUb[dstLen].GetPhyAddr();

    uint32_t vfLen = vlLen_;
    uint32_t dstHwAlign = dstLen;
    T1 srcLenNum = srcLen;
    uint16_t ubFactorTimes = ubFactor > 1 ? static_cast<uint16_t>(ubFactor) - 1 : 0;
    uint16_t timesNc = ubFactorTimes == 0 ? 0 : 1;
    uint32_t hwNum = dstLen;
    uint32_t hwNum1 = dstLen;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> startReg;
        AscendC::Reg::RegTensor<T1> idxRegT;
        AscendC::Reg::RegTensor<T> dstReg;
        AscendC::Reg::MaskReg preg;
        // 先处理第一行的hw
        Duplicate<T1>(startReg, srcHwNum);
        for (uint16_t j = 0; j < times; j++) {
            preg = AscendC::Reg::UpdateMask<T>(hwNum);
            AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T1>(j, vfLen);
            AscendC::Reg::LoadAlign(idxRegT, idxUbAddr, srcIdxOffset);
            AscendC::Reg::Sub(idxRegT, idxRegT, startReg, preg);
            Gather(dstReg, srcUbAddr, idxRegT, preg);
            AscendC::Reg::StoreAlign(dstUbAddr, dstReg, srcIdxOffset, preg);
        }
        // 从第二行开始处理
        for (uint16_t nc = 0; nc < timesNc; nc++) {
            for (uint16_t jj = 0; jj < times; jj++) {
                preg = AscendC::Reg::UpdateMask<T>(hwNum1);
                AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T1>(jj, vfLen);
                AscendC::Reg::LoadAlign(idxRegT, idxUbAddr, srcIdxOffset);
                AscendC::Reg::Sub(idxRegT, idxRegT, startReg, preg);
                for (uint16_t i = 0; i < ubFactorTimes; i++) {
                    AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<T>(jj, vfLen, i, dstHwAlign);
                    Adds(idxRegT, idxRegT, srcLenNum, preg);
                    Gather(dstReg, srcUbAddr, idxRegT, preg);
                    AscendC::Reg::StoreAlign(dstUbAddr1, dstReg, outOffset, preg);
                }
            }
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::DataCopyIn(LocalTensor<T>& xLocal, int64_t blockCount,
                                                                            int64_t blockLen, int64_t srcStride,
                                                                            int64_t offset)
{
    copyParams_.blockCount = blockCount;
    copyParams_.blockLen = blockLen * sizeof(T);
    copyParams_.srcStride = srcStride * sizeof(T);
    copyParams_.dstStride = 0;
    AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams_, padParams_);
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::DataCopyOut(LocalTensor<T>& yLocal, int64_t blockCount,
                                                                             int64_t blockLen, int64_t srcStride,
                                                                             int64_t offset)
{
    copyParams_.blockCount = blockCount;
    copyParams_.blockLen = blockLen * sizeof(T);
    copyParams_.srcStride = srcStride * sizeof(T);
    copyParams_.dstStride = 0;
    AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams_);
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeDataCopyGather(LocalTensor<T>& srcUb,
                                                                                       LocalTensor<T>& outUb,
                                                                                       LocalTensor<T1>& idxHwUb,
                                                                                       int64_t num)
{
    uint16_t times = CeilDivision(num, vlLen_);
    auto idxUbAddr = (__ubuf__ T1*)idxHwUb.GetPhyAddr();
    auto srcUbAddr = (__ubuf__ T*)srcUb.GetPhyAddr();
    auto dstUbAddr = (__ubuf__ T*)outUb.GetPhyAddr();
    uint32_t vfLen = vlLen_;
    uint32_t onceSize = num;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> idxRegT;
        AscendC::Reg::RegTensor<T> dstReg;
        AscendC::Reg::MaskReg preg;
        for (uint16_t j = 0; j < times; j++) {
            preg = AscendC::Reg::UpdateMask<T>(onceSize);
            AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T1>(j, vfLen);
            AscendC::Reg::LoadAlign(idxRegT, idxUbAddr, srcIdxOffset);
            Gather(dstReg, srcUbAddr, idxRegT, preg);
            AscendC::Reg::StoreAlign(dstUbAddr, dstReg, srcIdxOffset, preg);
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeHids(LocalTensor<T1>& idxHUb, int64_t hFactor)
{
    auto idxUbAddr = (__ubuf__ T1*)idxHUb.GetPhyAddr();
    auto idxUbRemainAddr = (__ubuf__ T1*)idxHUb[vlLenB32_].GetPhyAddr();
    uint16_t times = 0;
    uint32_t remainNum = 0;
    uint32_t numH = vlLenB32_;
    if (hFactor > static_cast<int64_t>(vlLenB32_)) {
        remainNum = hFactor - static_cast<int64_t>(vlLenB32_);
        times = CeilDivision(remainNum, static_cast<uint32_t>(vlLenB32_));
    } else {
        numH = hFactor;
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> idxInt32Reg;
        AscendC::Reg::MaskReg pregB32 = AscendC::Reg::UpdateMask<uint32_t>(numH);
        AscendC::Reg::MaskReg pregRemainB32;
        Arange(idxInt32Reg, 0);
        if constexpr (sizeof(T1) == sizeof(int32_t)) {
            StoreAlign(idxUbAddr, (Reg::RegTensor<T1>&)idxInt32Reg, pregB32);
        } else {
            StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B32>(idxUbAddr, (Reg::RegTensor<T1>&)idxInt32Reg,
                                                                   pregB32);
        }
        for (uint16_t i = 0; i < times; i++) {
            pregRemainB32 = AscendC::Reg::UpdateMask<int32_t>(remainNum);
            Adds(idxInt32Reg, idxInt32Reg, vlLenB32_, pregRemainB32);
            AscendC::Reg::AddrReg dstOffset = AscendC::Reg::CreateAddrReg<T1>(i, vlLenB32_);
            if constexpr (sizeof(T1) == sizeof(int32_t)) {
                StoreAlign(idxUbRemainAddr, (Reg::RegTensor<T1>&)idxInt32Reg, dstOffset, pregRemainB32);
            } else {
                StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    idxUbRemainAddr, (Reg::RegTensor<T1>&)idxInt32Reg, dstOffset, pregRemainB32);
            }
        }
    }
}

template <typename T1, bool isH, bool alignCorners>
__aicore__ inline void ComputeHOrWids(LocalTensor<T1>& idxUb, float bias, float scale, int64_t srcSize, int64_t dstSize,
                                      int64_t srcWsize)
{
    uint32_t vfLenb32 = Ops::Base::GetVRegSize() / sizeof(float);
    auto idxUbAddr = (__ubuf__ T1*)idxUb.GetPhyAddr();
    auto idxUbRemainAddr = (__ubuf__ T1*)idxUb[vfLenb32].GetPhyAddr();
    uint32_t oneTimeNum = vfLenb32;
    uint32_t remainNum = 0;
    int32_t srcW = srcWsize;
    int32_t maxData = srcSize - 1;
    uint16_t times = 0;

    if (dstSize <= static_cast<int64_t>(vfLenb32)) {
        oneTimeNum = dstSize;
    } else {
        remainNum = dstSize - vfLenb32;
        times = CeilDivision(remainNum, vfLenb32);
    }
    constexpr static AscendC::Reg::CastTrait castTraitRound = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::Reg::CastTrait castTraitFloor = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};
    constexpr static AscendC::Reg::CastTrait castInt32ToF = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> idxInt32Reg;
        AscendC::Reg::RegTensor<float> hIdxF;

        AscendC::Reg::RegTensor<int32_t> hIdxInt32;
        AscendC::Reg::RegTensor<int32_t> hIdxInt32C;
        AscendC::Reg::MaskReg pregB32 = AscendC::Reg::UpdateMask<uint32_t>(oneTimeNum);
        AscendC::Reg::MaskReg pregRemainB32;
        Arange(idxInt32Reg, 0.0f);

        Adds(hIdxF, idxInt32Reg, bias, pregB32);

        Muls(hIdxF, hIdxF, scale, pregB32);

        if constexpr (alignCorners == 1) {
            Cast<int32_t, float, castTraitRound>(hIdxInt32, hIdxF, pregB32);
        } else {
            Cast<int32_t, float, castTraitFloor>(hIdxInt32, hIdxF, pregB32);
        }
        Mins(hIdxInt32C, hIdxInt32, maxData, pregB32);
        if constexpr (isH) {
            Muls(hIdxInt32C, hIdxInt32C, srcW, pregB32);
        }
        if constexpr (sizeof(T1) == sizeof(int32_t)) {
            StoreAlign(idxUbAddr, (Reg::RegTensor<T1>&)hIdxInt32C, pregB32);
        } else {
            StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B32>(idxUbAddr, (Reg::RegTensor<T1>&)hIdxInt32C, pregB32);
        }
        for (uint16_t i = 0; i < times; i++) {
            pregRemainB32 = AscendC::Reg::UpdateMask<int32_t>(remainNum);
            Adds(idxInt32Reg, idxInt32Reg, static_cast<float>(vfLenb32), pregRemainB32);
            Adds(hIdxF, idxInt32Reg, bias, pregRemainB32);
            Muls(hIdxF, hIdxF, scale, pregRemainB32);
            if constexpr (alignCorners == 1) {
                Cast<int32_t, float, castTraitRound>(hIdxInt32, hIdxF, pregRemainB32);
            } else {
                Cast<int32_t, float, castTraitFloor>(hIdxInt32, hIdxF, pregRemainB32);
            }
            Mins(hIdxInt32C, hIdxInt32, maxData, pregRemainB32);
            if constexpr (isH) {
                Muls(hIdxInt32C, hIdxInt32C, srcW, pregRemainB32);
            }
            AscendC::Reg::AddrReg dstOffset = AscendC::Reg::CreateAddrReg<T1>(i, vfLenb32);
            if constexpr (sizeof(T1) == sizeof(int32_t)) {
                StoreAlign(idxUbRemainAddr, (Reg::RegTensor<T1>&)hIdxInt32C, dstOffset, pregRemainB32);
            } else {
                StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B32>(idxUbRemainAddr, (Reg::RegTensor<T1>&)hIdxInt32C,
                                                                       dstOffset, pregRemainB32);
            }
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeOriHIdx(LocalTensor<T1>& idxHUb,
                                                                                LocalTensor<T1>& idxH1Ub,
                                                                                int64_t onceHsize, int64_t hoStart,
                                                                                int64_t hiStart)
{
    // 先计算h的原始坐标
    auto idxHubAddr = (__ubuf__ T1*)idxHUb.GetPhyAddr();
    auto idxH1UbAddr = (__ubuf__ T1*)idxH1Ub.GetPhyAddr();
    uint32_t size = onceHsize;
    uint32_t vfLenB32 = vlLenB32_;
    uint16_t hTimes = CeilDivision(size, vfLenB32);
    float bias = bias_;
    float hScale = hScale_;
    int32_t maxData = srcHSize_ - 1;
    int32_t wSize = srcWAlignSize_;
    int32_t hoStartData = hoStart;
    int64_t hiStartData = hiStart;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> hiStartReg;
        AscendC::Reg::RegTensor<int32_t> idxInt32Reg;
        AscendC::Reg::RegTensor<float> hIdxF;
        AscendC::Reg::RegTensor<int32_t> idxInt32OriReg;
        AscendC::Reg::RegTensor<int32_t> idxInt32OriWReg;
        AscendC::Reg::MaskReg preg;
        Duplicate<int32_t>(hiStartReg, hiStartData);

        for (uint16_t i = 0; i < hTimes; i++) {
            preg = AscendC::Reg::UpdateMask<int32_t>(size);
            AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T1>(i, vfLenB32);
            if constexpr (sizeof(T1) == sizeof(int32_t)) {
                LoadAlign((Reg::RegTensor<T1>&)idxInt32Reg, idxHubAddr, srcIdxOffset);
            } else {
                LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>((Reg::RegTensor<T1>&)idxInt32Reg, idxHubAddr,
                                                              srcIdxOffset);
            }
            //
            Adds(idxInt32Reg, idxInt32Reg, hoStartData, preg); // 输出位置
            Cast<float, int32_t, castInt32ToF>(hIdxF, idxInt32Reg, preg);
            Adds(hIdxF, hIdxF, bias, preg);
            Muls(hIdxF, hIdxF, hScale, preg);

            if constexpr (alignCorners == 1) {
                Cast<int32_t, float, castTraitRound>(idxInt32OriReg, hIdxF, preg);
            } else {
                Cast<int32_t, float, castTraitFloor>(idxInt32OriReg, hIdxF, preg);
            }
            Mins(idxInt32OriReg, idxInt32OriReg, maxData, preg);
            Sub(idxInt32OriReg, idxInt32OriReg, hiStartReg, preg);
            Muls(idxInt32OriWReg, idxInt32OriReg, wSize, preg);
            if constexpr (sizeof(T1) == sizeof(int32_t)) {
                StoreAlign(idxH1UbAddr, (Reg::RegTensor<T1>&)idxInt32OriWReg, srcIdxOffset, preg);
            } else {
                StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    idxH1UbAddr, (Reg::RegTensor<T1>&)idxInt32OriWReg, srcIdxOffset, preg);
            }
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeOriHWidx(LocalTensor<T1>& idxWUb,
                                                                                 LocalTensor<T1>& idxH1Ub,
                                                                                 LocalTensor<T1>& idxHwUb,
                                                                                 int64_t onceHsize)
{
    // 计算h*w的合轴的坐标
    auto idxHwUbAddr = (__ubuf__ T1*)idxHwUb.GetPhyAddr();
    auto idxWubAddr = (__ubuf__ T1*)idxWUb.GetPhyAddr();

    uint16_t onceHTimes = onceHsize;
    uint32_t dstWSize = dstWAlignSize_;
    uint16_t wTimes = dstWAlignSize_ / vlLen_;
    uint16_t wTailTimes = 0;
    uint32_t tail = dstWAlignSize_ % vlLen_;
    if (tail != 0) {
        wTailTimes = 1;
    }
    uint32_t vfLen = vlLen_;
    auto idxWubAddr1 = (__ubuf__ T1*)idxWUb[wTimes * vfLen].GetPhyAddr();
    auto idxHwUbAddr1 = (__ubuf__ T1*)idxHwUb[wTimes * vfLen].GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> idxWReg;
        AscendC::Reg::RegTensor<T1> addsReg;
        AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<T1>(tail);
        AscendC::Reg::MaskReg pregB32 = AscendC::Reg::CreateMask<T1, AscendC::Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < onceHTimes; i++) {
            T1 hIdx = idxH1Ub.GetValue(i);
            AscendC::Reg::AddrReg outIdxOffset1 = AscendC::Reg::CreateAddrReg<T1>(i, dstWSize);
            for (uint16_t j = 0; j < static_cast<uint16_t>(wTimes); j++) {
                AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T1>(j, vfLen);
                AscendC::Reg::AddrReg outIdxOffset = AscendC::Reg::CreateAddrReg<T1>(i, dstWSize, j, vfLen);
                LoadAlign(idxWReg, idxWubAddr, srcIdxOffset);
                Adds(addsReg, idxWReg, hIdx, pregB32);
                StoreAlign(idxHwUbAddr, addsReg, outIdxOffset, pregB32);
            }
            for (uint16_t jj = 0; jj < wTailTimes; jj++) {
                LoadAlign(idxWReg, idxWubAddr1);
                Adds(addsReg, idxWReg, hIdx, preg);
                StoreAlign(idxHwUbAddr1, addsReg, outIdxOffset1, preg);
            }
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeCutH()
{
    LocalTensor<T1> idxWUb = idxWBuf_.Get<T1>();
    ComputeHOrWids<T1, false, alignCorners>(idxWUb, bias_, wScale_, srcWSize_, dstWAlignSize_, srcWAlignSize_);
    LocalTensor<T1> idxHUb = idxHBuf_.Get<T1>();
    ComputeHids(idxHUb, ubFactor_);
    LocalTensor<T1> idxH1Ub = idxH1Buf_.Get<T1>();
    LocalTensor<T1> idxHwUb = idxBuf_.Get<T1>();
    int64_t ncTimes = blockIdx_ == realCoreNum_ - 1 ? beforeTNum_ : blockFactor_; // 每个核处理的nc个数
    for (int64_t h = 0; h < hTimes_; h++) {
        int64_t onceHsize = h == hTimes_ - 1 ? tailTNum_ : ubFactor_; // 每次处理的h数量
        int64_t hoStart = h * ubFactor_;
        int64_t hoEnd = hoStart + onceHsize - 1;
        int64_t allSize = onceHsize * dstWAlignSize_;
        int64_t hiStart = static_cast<int32_t>(ComputeOriIds(hoStart, srcHSize_ - 1, hScale_, bias_));
        int64_t hiEnd = static_cast<int32_t>(ComputeOriIds(hoEnd, srcHSize_ - 1, hScale_, bias_));
        int64_t hiSize = hiEnd - hiStart + 1;
        ComputeOriHIdx(idxHUb, idxH1Ub, onceHsize, hoStart, hiStart);
        ComputeOriHWidx(idxWUb, idxH1Ub, idxHwUb, onceHsize);
        for (int64_t nc = 0; nc < ncTimes; nc++) {
            LocalTensor<T> xLocal = inQue_.AllocTensor<T>();
            int64_t inOffset = (blockIdx_ * blockFactor_ + nc) * srcHwNum_ + hiStart * srcWSize_;
            if (srcWAlignSize_ == srcWSize_) {
                DataCopyIn(xLocal, 1, hiSize * srcWSize_, 0, inOffset);
            } else {
                DataCopyIn(xLocal, hiSize, srcWSize_, 0, inOffset);
            }
            inQue_.EnQue<T>(xLocal);
            xLocal = inQue_.DeQue<T>();
            LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
            ComputeDataCopyGather(xLocal, yLocal, idxHwUb, allSize);
            inQue_.FreeTensor(xLocal);
            int64_t outOffset = (blockIdx_ * blockFactor_ + nc) * dstHwNum_ + hoStart * dstWSize_;
            outQue_.EnQue<T>(yLocal);
            yLocal = outQue_.DeQue<T>();
            if (dstWAlignSize_ == dstWSize_) {
                DataCopyOut(yLocal, 1, allSize, 0, outOffset);
            } else {
                DataCopyOut(yLocal, onceHsize, dstWSize_, 0, outOffset);
            }
            outQue_.FreeTensor(yLocal);
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeHWids(LocalTensor<T1>& idxUb,
                                                                              LocalTensor<T1>& idxHUb,
                                                                              LocalTensor<T1>& idxWUb,
                                                                              int64_t dstWSizeAlgin)
{
    T1 hSize = dstHSize_;
    uint32_t vfLen = vlLen_;
    T1 wTimes = dstWSizeAlgin / vlLen_;
    uint32_t tail = dstWSizeAlgin % vlLen_;
    uint16_t tailTimes = tail > 0 ? 1 : 0;
    auto idxUbAddr = (__ubuf__ T1*)idxUb.GetPhyAddr();
    auto idxUbAddr1 = (__ubuf__ T1*)idxUb[wTimes * vfLen].GetPhyAddr();

    auto idxWUbAddr = (__ubuf__ T1*)idxWUb.GetPhyAddr();
    auto idxWUbAddr1 = (__ubuf__ T1*)idxWUb[wTimes * vfLen].GetPhyAddr();
    uint32_t wAlign = dstWSizeAlgin;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> wIdxReg;
        AscendC::Reg::RegTensor<T1> idxReg;

        AscendC::Reg::RegTensor<int32_t> hIdxInt32;
        AscendC::Reg::RegTensor<int32_t> hIdxInt32C;
        AscendC::Reg::MaskReg pregB32 = AscendC::Reg::CreateMask<T1, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg pregTail = AscendC::Reg::UpdateMask<T1>(tail);
        for (uint16_t i = 0; i < static_cast<uint16_t>(hSize); i++) {
            T1 hIdx = idxHUb.GetValue(i);
            AscendC::Reg::AddrReg outIdxOffset = AscendC::Reg::CreateAddrReg<T1>(i, wAlign);
            for (uint16_t j = 0; j < static_cast<uint16_t>(wTimes); j++) {
                AscendC::Reg::AddrReg srcIdxOffset = AscendC::Reg::CreateAddrReg<T1>(j, vfLen);
                LoadAlign(wIdxReg, idxWUbAddr, srcIdxOffset);
                Adds(idxReg, wIdxReg, hIdx, pregB32);
                AscendC::Reg::AddrReg srcOutOffset = AscendC::Reg::CreateAddrReg<T1>(i, wAlign, j, vfLen);
                StoreAlign(idxUbAddr, idxReg, srcOutOffset, pregB32);
            }
            for (uint16_t jj = 0; jj < tailTimes; jj++) {
                LoadAlign(wIdxReg, idxWUbAddr1);
                Adds(idxReg, wIdxReg, hIdx, pregTail);
                StoreAlign(idxUbAddr1, idxReg, outIdxOffset, pregTail);
            }
        }
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeIdsSpecial(int64_t dstWSizeAlgin,
                                                                                   int64_t srcWSizeAlgin)
{
    LocalTensor<T1> idxUb = idxBuf_.Get<T1>();
    LocalTensor<T1> idxHUb = idxHBuf_.Get<T1>();
    LocalTensor<T1> idxWUb = idxWBuf_.Get<T1>();
    ComputeHOrWids<T1, true, alignCorners>(idxHUb, bias_, hScale_, srcHSize_, dstHSize_, srcWSizeAlgin);
    ComputeHOrWids<T1, false, alignCorners>(idxWUb, bias_, wScale_, srcWSize_, dstWSizeAlgin, srcWSizeAlgin);
    ComputeHWids(idxUb, idxHUb, idxWUb, dstWSizeAlgin); // h和w坐标合成1维坐标
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::ComputeAllHw()
{
    ComputeIdsSpecial(dstWAlignSize_, srcWAlignSize_); // h*w一次可以放下，对应的h*w坐标
    int64_t ncLoopTimes = blockIdx_ == realCoreNum_ - 1 ? tailTimes_ : beforeTimes_;
    int64_t ncLoopTailNum = blockIdx_ == realCoreNum_ - 1 ? tailTNum_ : beforeTNum_;
    int64_t hiStart = ComputeOriIds(0, srcHSize_ - 1, hScale_, bias_);
    int64_t hiEnd = ComputeOriIds(dstHSize_ - 1, srcHSize_ - 1, hScale_, bias_);
    int64_t srcHwOffsetStart = hiStart * srcWAlignSize_;
    int64_t hiSize = hiEnd - hiStart + 1;
    int64_t blockLen = hiSize * srcWAlignSize_;
    int64_t dstLen = dstHSize_ * dstWAlignSize_;
    for (int64_t i = 0; i < ncLoopTimes; i++) {
        int64_t onceNum = i == ncLoopTimes - 1 ? ncLoopTailNum : ubFactor_;
        int64_t srcOffset = (blockIdx_ * blockFactor_ + i * ubFactor_) * srcHwNum_ + hiStart * srcWSize_;
        LocalTensor<T> xLocal = inQue_.AllocTensor<T>();
        if (srcWAlignSize_ == srcWSize_) { // srcW对齐
            DataCopyIn(xLocal, onceNum, blockLen, srcHwNum_ - blockLen, srcOffset);
        } else {
            // srcW不对齐，需要用硬件4层for循环能力
            LoopModeParams loopParams;
            copyParams_.blockCount = hiSize;
            copyParams_.blockLen = srcWSize_ * sizeof(T);
            copyParams_.srcStride = 0;
            copyParams_.dstStride = 0;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = onceNum;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            loopParams.loop1SrcStride = srcHwNum_ * sizeof(T);
            loopParams.loop1DstStride = hiSize * srcWAlignSize_ * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(xLocal, xGm_[srcOffset], copyParams_, padParams_);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }

        inQue_.EnQue<T>(xLocal);
        xLocal = inQue_.DeQue<T>();
        LocalTensor<T> yLocal = outQue_.AllocTensor<T>();
        // 搬入后根据ids做gather选出输出数据
        GatherOutput(yLocal, xLocal, onceNum, blockLen, dstLen, srcHwOffsetStart);
        inQue_.FreeTensor(xLocal);
        outQue_.EnQue<T>(yLocal);
        yLocal = outQue_.DeQue<T>();
        int64_t outOffset = (blockIdx_ * blockFactor_ + i * ubFactor_) * dstHwNum_;
        if (dstWAlignSize_ == dstWSize_) { // dstW对齐
            DataCopyOut(yLocal, onceNum, dstHwNum_, 0, outOffset);
        } else {
            LoopModeParams loopParams;
            copyParams_.blockCount = dstHSize_;
            copyParams_.blockLen = dstWSize_ * sizeof(T);
            copyParams_.srcStride = 0;
            copyParams_.dstStride = 0;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = onceNum;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            loopParams.loop1SrcStride = dstHSize_ * dstWAlignSize_ * sizeof(T);
            loopParams.loop1DstStride = dstHwNum_ * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
            DataCopyPad(yGm_[outOffset], yLocal, copyParams_);
            ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        }
        outQue_.FreeTensor(yLocal);
    }
}

template <typename T, typename T1, int schId, bool alignCorners>
__aicore__ inline void ResizeGather<T, T1, schId, alignCorners>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }

    if constexpr (schId == TPL_SCH_MODE_GATHER_ALL_HW) {
        ComputeAllHw();
    }
    if constexpr (schId == TPL_SCH_MODE_GATHER_CUT_H) {
        ComputeCutH();
    }
}
} // namespace ResizeNearestNeighborV2
#endif // RESIZE_NEAREAST_NEIGHBOR_V2_NCHW_GATHER_H
