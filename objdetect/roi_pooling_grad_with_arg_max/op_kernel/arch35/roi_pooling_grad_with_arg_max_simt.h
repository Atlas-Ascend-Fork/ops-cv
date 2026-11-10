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
 * \file roi_pooling_grad_with_arg_max_simt.h
 * \brief roi_pooling_grad_with_arg_max_simt
 */

#ifndef ROI_POOLING_GRAD_WITH_ARG_MAX_H
#define ROI_POOLING_GRAD_WITH_ARG_MAX_H

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "roi_pooling_grad_with_arg_max_tiling_data.h"

namespace RoiPoolingGradWithArgMaxOps {

using namespace AscendC;

const uint32_t VF_MAX_THREAD_NUM = 1024;

template <typename ACC_T, typename D_T>
class RoiPoolingGradWithArgMaxSimt {
public:
    __aicore__ inline RoiPoolingGradWithArgMaxSimt(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR rois, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace,
                                const RoiPoolingGradWithArgMaxRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<D_T> gradGm_;
    GlobalTensor<D_T> xGm_;
    GlobalTensor<D_T> roisGm_;
    GlobalTensor<int32_t> argMaxGm_;
    GlobalTensor<D_T> yGm_;
    GlobalTensor<ACC_T> userWSGm_;
    uint32_t coreIdx_;
    const RoiPoolingGradWithArgMaxRegBaseTilingData* tiling_;
};

template <typename ACC_T, typename D_T>
__aicore__ inline void RoiPoolingGradWithArgMaxSimt<ACC_T, D_T>::Init(
    GM_ADDR grad, GM_ADDR x, GM_ADDR rois, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace,
    const RoiPoolingGradWithArgMaxRegBaseTilingData* __restrict tilingData)
{
    gradGm_.SetGlobalBuffer((__gm__ D_T*)(grad));
    xGm_.SetGlobalBuffer((__gm__ D_T*)(x));
    roisGm_.SetGlobalBuffer((__gm__ D_T*)(rois));
    argMaxGm_.SetGlobalBuffer((__gm__ int32_t*)(argmax));
    yGm_.SetGlobalBuffer((__gm__ D_T*)(y));
    userWSGm_.SetGlobalBuffer((__gm__ ACC_T*)(workspace));

    tiling_ = tilingData;
    coreIdx_ = GetBlockIdx();
}

template <typename ACC_T, typename D_T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) inline void RoiPoolingGradWithArgMaxComputeSIMT(
    __gm__ D_T* gradGmAddr, __gm__ D_T* xGmAddr, __gm__ D_T* roisGmAddr, __gm__ int32_t* argMaxGmAddr,
    __gm__ D_T* yGmAddr, __gm__ ACC_T* userWSGmAddr, const int32_t pooled_h, const int32_t pooled_w,
    const int32_t pool_channel, const int32_t height, const int32_t width, const int32_t batch, const int32_t count)
{
    asc_syncthreads();
    const int64_t pixelNum = static_cast<int64_t>(height) * static_cast<int64_t>(width);
    for (int32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x * blockDim.x) {
        // (n, c) is an element in the pooled output
        int32_t c = static_cast<int32_t>(
            static_cast<int32_t>(static_cast<int32_t>(static_cast<int32_t>(idx / pooled_w) / pooled_h)) % pool_channel);
        int32_t n = static_cast<int32_t>(static_cast<int32_t>(static_cast<int32_t>(idx / pooled_w) / pooled_h) /
                                         pool_channel);

        const int32_t roi_batch_ind = static_cast<int32_t>(roisGmAddr[n * 5]);
        if (roi_batch_ind < 0 || roi_batch_ind >= batch) {
            continue;
        }
        const int64_t y_offset = (static_cast<int64_t>(roi_batch_ind) * pool_channel + c) *
                                 static_cast<int64_t>(height) * width;
        const int32_t argmax_index = argMaxGmAddr[idx];

        if (argmax_index >= 0 && static_cast<int64_t>(argmax_index) < pixelNum) {
            __asc_simt_vf::asc_atomic_add(userWSGmAddr + y_offset + argmax_index, static_cast<ACC_T>(gradGmAddr[idx]));
        }
    }
}

template <typename ACC_T, typename D_T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) inline void RoiPoolingGradWithArgMaxCopyGmSIMT(
    __gm__ D_T* gradGmAddr, __gm__ D_T* xGmAddr, __gm__ D_T* roisGmAddr, __gm__ int32_t* argMaxGmAddr,
    __gm__ D_T* yGmAddr, __gm__ ACC_T* userWSGmAddr, const int32_t pooled_h, const int32_t pooled_w,
    const int32_t pool_channel, const int32_t height, const int32_t width, const int32_t batch, const int32_t count)
{
    asc_syncthreads();
    const int64_t pixelNum = static_cast<int64_t>(height) * static_cast<int64_t>(width);
    for (int32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x * blockDim.x) {
        // (n, c) is an element in the pooled output
        int32_t c = static_cast<int32_t>(
            static_cast<int32_t>(static_cast<int32_t>(static_cast<int32_t>(idx / pooled_w) / pooled_h)) % pool_channel);
        int32_t n = static_cast<int32_t>(static_cast<int32_t>(static_cast<int32_t>(idx / pooled_w) / pooled_h) /
                                         pool_channel);

        const int32_t roi_batch_ind = static_cast<int32_t>(roisGmAddr[n * 5]);
        if (roi_batch_ind < 0 || roi_batch_ind >= batch) {
            continue;
        }
        const int64_t y_offset = (static_cast<int64_t>(roi_batch_ind) * pool_channel + c) *
                                 static_cast<int64_t>(height) * width;
        const int32_t argmax_index = argMaxGmAddr[idx];

        if (argmax_index >= 0 && static_cast<int64_t>(argmax_index) < pixelNum) {
            yGmAddr[y_offset + argmax_index] = static_cast<D_T>(userWSGmAddr[y_offset + argmax_index]);
        }
    }
}

template <typename ACC_T, typename D_T>
__aicore__ inline void RoiPoolingGradWithArgMaxSimt<ACC_T, D_T>::Process()
{
    const uint32_t yTotalCoreNum = static_cast<uint32_t>(tiling_->yTotalCoreNum);
    const uint32_t yDataPerCore = static_cast<uint32_t>(tiling_->yDataPerCore);
    const uint32_t yDataTailCore = static_cast<uint32_t>(tiling_->yDataTailCore);
    AscendC::SyncAll();

    // 输出、userWorkSpace清零
    if (coreIdx_ < yTotalCoreNum - 1) {
        InitOutput<D_T>(yGm_[static_cast<uint32_t>(coreIdx_ * yDataPerCore)], yDataPerCore, static_cast<D_T>(0.0f));
        InitOutput<ACC_T>(userWSGm_[static_cast<uint32_t>(coreIdx_ * yDataPerCore)], yDataPerCore,
                          static_cast<ACC_T>(0.0f));
    } else if (coreIdx_ == yTotalCoreNum - 1) {
        InitOutput<D_T>(yGm_[static_cast<uint32_t>(coreIdx_ * yDataPerCore)], yDataTailCore, static_cast<D_T>(0.0f));
        InitOutput<ACC_T>(userWSGm_[static_cast<uint32_t>(coreIdx_ * yDataPerCore)], yDataTailCore,
                          static_cast<ACC_T>(0.0f));
    }
    SyncAll();

    // simt执行计算
    asc_vf_call<RoiPoolingGradWithArgMaxComputeSIMT<ACC_T, D_T>>(
        dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ D_T*)(gradGm_.GetPhyAddr()), (__gm__ D_T*)(xGm_.GetPhyAddr()),
        (__gm__ D_T*)(roisGm_.GetPhyAddr()), (__gm__ int32_t*)(argMaxGm_.GetPhyAddr()),
        (__gm__ D_T*)(yGm_.GetPhyAddr()), (__gm__ ACC_T*)(userWSGm_.GetPhyAddr()),
        static_cast<int32_t>(tiling_->pooledH), static_cast<int32_t>(tiling_->pooledW),
        static_cast<int32_t>(tiling_->poolChannel), static_cast<int32_t>(tiling_->height),
        static_cast<int32_t>(tiling_->width), static_cast<int32_t>(tiling_->batch),
        static_cast<int32_t>(tiling_->totalLength));
    SyncAll();

    // 数据搬到输出
    asc_vf_call<RoiPoolingGradWithArgMaxCopyGmSIMT<ACC_T, D_T>>(
        dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ D_T*)(gradGm_.GetPhyAddr()), (__gm__ D_T*)(xGm_.GetPhyAddr()),
        (__gm__ D_T*)(roisGm_.GetPhyAddr()), (__gm__ int32_t*)(argMaxGm_.GetPhyAddr()),
        (__gm__ D_T*)(yGm_.GetPhyAddr()), (__gm__ ACC_T*)(userWSGm_.GetPhyAddr()),
        static_cast<int32_t>(tiling_->pooledH), static_cast<int32_t>(tiling_->pooledW),
        static_cast<int32_t>(tiling_->poolChannel), static_cast<int32_t>(tiling_->height),
        static_cast<int32_t>(tiling_->width), static_cast<int32_t>(tiling_->batch),
        static_cast<int32_t>(tiling_->totalLength));
}
} // namespace RoiPoolingGradWithArgMaxOps
#endif
