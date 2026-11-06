/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_linear_grad_simt_no_determine.h
 * \brief resize_linear_grad_simt_no_determine
 */

#ifndef CANN_RESIZE_LINEAR_GRAD_NO_DETERMINE_H
#define CANN_RESIZE_LINEAR_GRAD_NO_DETERMINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ResizeLinearGrad {
using namespace AscendC;

template <typename T1, typename T2, uint64_t isCenter, uint64_t mode>
class ResizeLinearGradNoDetermine {
public:
    __aicore__ inline ResizeLinearGradNoDetermine(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y,
                                const ResizeLinearGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    uint32_t blockIdx_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T1> outputGm_;
    const ResizeLinearGradTilingData* tilingData_;
};

template <typename T1, typename T2, uint64_t isCenter, uint64_t mode>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtComputeNoDeter(
    T2 blkStartOffset, T2 blkProcessNum, T2 mL, T2 shiftL, T2 lenDesL, T2 lenSrcLOrUb, float scaleL, __gm__ T1* inputGm,
    __gm__ T1* outputGm)
{
    T2 srcL1 = lenSrcLOrUb - 1;
    for (T2 idx = static_cast<T2>(threadIdx.x); idx < blkProcessNum; idx += static_cast<T2>(blockDim.x)) {
        T2 yGmIdx = blkStartOffset + idx;
        T2 NC = Simt::UintDiv(yGmIdx, mL, shiftL);

        T2 L = yGmIdx - NC * lenDesL;
        T2 origBaseIdx = NC * lenSrcLOrUb;
        float origWidth = ComputeOriL<T2, isCenter>(L, scaleL);

        if constexpr (mode == 5) {
            // POINT COPY
            T2 leftX = floorf(origWidth);
            leftX = min(leftX, srcL1);
            T1 gradVal = inputGm[yGmIdx];
            asc_atomic_add(outputGm + origBaseIdx + leftX, gradVal);
            continue;
        }
        if constexpr (mode == 0) {
            // LINEAR_GRAD 操作
            T2 leftX = floorf(origWidth);
            leftX = min(leftX, srcL1);
            T2 RightX = static_cast<T2>(leftX < srcL1);
            float deltaX = origWidth - static_cast<float>(leftX);
            float val = static_cast<float>(inputGm[yGmIdx]);
            float val2 = val * deltaX;
            float val1 = val * (1.0f - deltaX);
            asc_atomic_add(outputGm + origBaseIdx + leftX, static_cast<T1>(val1));
            asc_atomic_add(outputGm + origBaseIdx + leftX + RightX, static_cast<T1>(val2));
        }
    }
}

template <typename T1, typename T2, uint64_t isCenter, uint64_t mode>
__simt_vf__ LAUNCH_BOUND(512) __aicore__
    void calleeInt64NoDeter(T2 blkStartOffset, T2 blkProcessNum, T2 mL, T2 shiftL, T2 lenDesL, T2 lenSrcLOrUb,
                            float scaleL, __gm__ T1* inputGm, __gm__ T1* outputGm)
{
    SimtComputeNoDeter<T1, T2, isCenter, mode>(blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcLOrUb, scaleL,
                                               inputGm, outputGm);
}

template <typename T1, typename T2, uint64_t isCenter, uint64_t mode>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__
    void calleeInt32NoDeter(T2 blkStartOffset, T2 blkProcessNum, T2 mL, T2 shiftL, T2 lenDesL, T2 lenSrcLOrUb,
                            float scaleL, __gm__ T1* inputGm, __gm__ T1* outputGm)
{
    SimtComputeNoDeter<T1, T2, isCenter, mode>(blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcLOrUb, scaleL,
                                               inputGm, outputGm);
}

template <typename T1, typename T2, uint64_t isCenter, uint64_t mode>
__aicore__ inline void ResizeLinearGradNoDetermine<T1, T2, isCenter, mode>::Init(
    GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, const ResizeLinearGradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    inputGm_.SetGlobalBuffer((__gm__ T1*)grads);
    outputGm_.SetGlobalBuffer((__gm__ T1*)y);
}

template <typename T1, typename T2, uint64_t isCenter, uint64_t mode>
__aicore__ inline void ResizeLinearGradNoDetermine<T1, T2, isCenter, mode>::Process()
{
    if (blockIdx_ < tilingData_->initCoreNum) {
        T2 blkProcessNumC = tilingData_->ubFactor;
        T2 blkStartOffsetC = blockIdx_ * blkProcessNumC;
        if (blockIdx_ >= tilingData_->ubFactorTailB) {
            blkProcessNumC = tilingData_->ubFactorTailT;
            blkStartOffsetC = blockIdx_ * blkProcessNumC + tilingData_->ubFactorTailB;
        }
        InitOutput<T1>(outputGm_[blkStartOffsetC], blkProcessNumC, T1(0));
    }
    SyncAll();

    if (blockIdx_ < tilingData_->realCoreNum) {
        T2 blkProcessNum = tilingData_->blkProcessNum;
        T2 blkStartOffset = blockIdx_ * blkProcessNum;
        if (blockIdx_ >= tilingData_->ubLoopSizeB) {
            blkProcessNum = tilingData_->ubLoopSizeT;
            blkStartOffset = blockIdx_ * blkProcessNum + tilingData_->ubLoopSizeB;
        }
        T2 mL = 0;
        T2 shiftL = 0;
        T2 lenDesL = (T2)(tilingData_->lenDesL);
        GetUintDivMagicAndShift(mL, shiftL, lenDesL);
        T2 lenSrcLOrUb = (T2)(tilingData_->lenSrcLOrUb);
        float scaleL = tilingData_->scaleL;
        if constexpr (sizeof(T2) == sizeof(uint64_t)) {
            asc_vf_call<calleeInt64NoDeter<T1, T2, isCenter, mode>>(
                dim3(512), blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcLOrUb, scaleL,
                (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()));
        } else {
            asc_vf_call<calleeInt32NoDeter<T1, T2, isCenter, mode>>(
                dim3(1024), blkStartOffset, blkProcessNum, mL, shiftL, lenDesL, lenSrcLOrUb, scaleL,
                (__gm__ T1*)(inputGm_.GetPhyAddr()), (__gm__ T1*)(outputGm_.GetPhyAddr()));
        }
    }
}
} // namespace ResizeLinearGrad
#endif // CANN_RESIZE_LINEAR_GRAD_NO_DETERMINE_H
