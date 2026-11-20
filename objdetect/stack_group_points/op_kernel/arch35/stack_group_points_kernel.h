/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STACK_GROUP_POINTS_H_
#define STACK_GROUP_POINTS_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "stack_group_points_tiling_data.h"
#include "stack_group_points_tiling_key.h"

namespace NsStackGroupPoints {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpStackGroupPointsSimt(
    int64_t m, int64_t c, int64_t nsample, int64_t b, int64_t n, int64_t totalElements, __gm__ T* features,
    __gm__ int32_t* featuresBatchCnt, __gm__ int32_t* indices, __gm__ int32_t* indicesBatchCnt, __gm__ T* y)
{
    int64_t nsampleC = nsample * c;
    int64_t mNsample = m * nsample;
    uint64_t stride = static_cast<uint64_t>(blockDim.x * gridDim.x);
    int64_t featLen = n * c;

    if (b <= 0) {
        for (uint64_t i = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
             i < static_cast<uint64_t>(totalElements); i += stride) {
            y[i] = static_cast<T>(0);
        }
        return;
    }

    // B==1 域机制路由（sgp-r2-hybrid-route）：单批次下原 k 线性扫描零迭代、游标化收益解析为零，
    // 跨迭代游标状态只剩纯开销；本分支体逐字节复原基线逐元素 grid-stride 循环（baseline-tree
    // 同文件 L44-81，状态全部在迭代内定义），b>=2 仍走下方游标路径（逐字节不变）
    if (b == 1) {
        for (uint64_t index = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
             index < static_cast<uint64_t>(totalElements); index += stride) {
            int64_t sampleIdx = index % nsample;
            int64_t cIdx = (index / nsample) % c;
            int64_t ptIdx = index / nsampleC;

            // 批次归属：indices_batch_cnt 前缀和上的线性扫描（非负计数域上与
            // searchsorted right 结果一致）；计数/计数和全 int64 域
            int64_t bsIdx = 0;
            int64_t ptCnt = indicesBatchCnt[0];
            for (int64_t k = 1; k < b; k++) {
                if (ptIdx >= ptCnt) {
                    ptCnt += indicesBatchCnt[k];
                    bsIdx = k;
                }
            }

            int64_t featuresBatchStartIdx = 0;
            for (int64_t k = 0; k < bsIdx; k++) {
                featuresBatchStartIdx += featuresBatchCnt[k];
            }
            const int32_t featuresBatchLocalCnt = featuresBatchCnt[bsIdx];

            int64_t tmpCin = ptIdx * nsample + sampleIdx;
            int32_t cin = 0;
            if (tmpCin < mNsample) {
                cin = indices[tmpCin];
            }

            T result = static_cast<T>(0);
            if (cin >= 0 && cin < featuresBatchLocalCnt) {
                int64_t fsIdx = (featuresBatchStartIdx + static_cast<int64_t>(cin)) * c + cIdx;
                if (fsIdx >= 0 && fsIdx < featLen) {
                    result = features[fsIdx];
                }
            }
            y[index] = result;
        }
        return;
    }

    // 批次归属游标：声明于 grid-stride 循环外，仅随 ptIdx 单调前进（grid-stride 线程序保证
    // ptIdx 单调不降，bsIdx/fbsIdx 随之单调不退）；与原逐元素 O(B) 线性扫描在 (ptIdx, ibc)
    // 组合域上逐点等价（241131 试验对拍，analysis/round1/theory/cursor-equivalence.txt）
    int64_t bsIdx = 0;
    int64_t ptCnt = indicesBatchCnt[0];
    int64_t fbsIdx = 0;
    int64_t featuresBatchStartIdx = 0;

    for (uint64_t index = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < static_cast<uint64_t>(totalElements); index += stride) {
        int64_t sampleIdx = index % nsample;
        int64_t cIdx = (index / nsample) % c;
        int64_t ptIdx = index / nsampleC;

        // indices_batch_cnt 前缀和上的单调早退游标：ptIdx >= ptCnt 与 bsIdx < b - 1 双早退
        // （非负计数域上与原线性扫描、searchsorted right 结果一致）；计数/计数和全 int64 域
        while (ptIdx >= ptCnt && bsIdx < b - 1) {
            bsIdx++;
            ptCnt += indicesBatchCnt[bsIdx];
        }

        // features_batch_cnt 前缀和游标：仅累加 (fbsIdx, bsIdx] 新增区间，与原 0..bsIdx-1
        // 全量求和逐点等价
        while (fbsIdx < bsIdx) {
            featuresBatchStartIdx += featuresBatchCnt[fbsIdx];
            fbsIdx++;
        }
        const int32_t featuresBatchLocalCnt = featuresBatchCnt[bsIdx];

        int64_t tmpCin = ptIdx * nsample + sampleIdx;
        int32_t cin = 0;
        if (tmpCin < mNsample) {
            cin = indices[tmpCin];
        }

        T result = static_cast<T>(0);
        if (cin >= 0 && cin < featuresBatchLocalCnt) {
            int64_t fsIdx = (featuresBatchStartIdx + static_cast<int64_t>(cin)) * c + cIdx;
            if (fsIdx >= 0 && fsIdx < featLen) {
                result = features[fsIdx];
            }
        }
        y[index] = result;
    }
}

template <typename T>
__aicore__ inline void ZeroFill(__gm__ T* yGm, int64_t totalElements);

template <typename T>
__aicore__ inline void Process(GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices, GM_ADDR indices_batch_cnt,
                               GM_ADDR y, int64_t m, int64_t c, int64_t nsample, int64_t b, int64_t n,
                               int64_t totalElements, int64_t zeroFill)
{
    __gm__ T* featuresGm = (__gm__ T*)features;
    __gm__ int32_t* featuresBatchCntGm = (__gm__ int32_t*)features_batch_cnt;
    __gm__ int32_t* indicesGm = (__gm__ int32_t*)indices;
    __gm__ int32_t* indicesBatchCntGm = (__gm__ int32_t*)indices_batch_cnt;
    __gm__ T* yGm = (__gm__ T*)y;

    if (zeroFill == 1) {
        ZeroFill(yGm, totalElements);
        return;
    }
    asc_vf_call<OpStackGroupPointsSimt<T>>(dim3(THREAD_NUM), m, c, nsample, b, n, totalElements, featuresGm,
                                           featuresBatchCntGm, indicesGm, indicesBatchCntGm, yGm);
}
// N=0（zeroFill=1）输出全零的向量化填充：TBuf(VECCALC) 每核仅一次 Duplicate(0) 于 copy
// 循环之外，每核按 GetBlockIdx 跨步 MTE3 全宽搬运，尾块 DataCopyPad 按 copyElements*sizeof(T)
// 字节参（与 arch22 同族 kernel 写法一致）；FILL_UB_BYTES=65536B 预算 ≤ SIMT 有效 UB
// （GetCoreMemSize(UB) 运行值 253952B − 顶部 40KB SIMT 预留 = 212992B，KB PB-32）
template <typename T>
__aicore__ inline void ZeroFill(__gm__ T* yGm, int64_t totalElements)
{
    constexpr uint32_t FILL_UB_BYTES = 65536;
    constexpr int32_t FILL_ELEMS = static_cast<int32_t>(FILL_UB_BYTES / sizeof(T));
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fillBuf;
    pipe.InitBuffer(fillBuf, FILL_UB_BYTES);
    LocalTensor<T> fillLocal = fillBuf.Get<T>();

    Duplicate<T>(fillLocal, 0, FILL_ELEMS);
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<T> yGlobal;
    yGlobal.SetGlobalBuffer(yGm);
    const int64_t coreNum = GetBlockNum();
    const int64_t coreIdx = GetBlockIdx();
    const int64_t chunkCnt = totalElements / FILL_ELEMS;
    for (int64_t ci = coreIdx; ci < chunkCnt; ci += coreNum) {
        DataCopy(yGlobal[static_cast<uint64_t>(ci) * static_cast<uint64_t>(FILL_ELEMS)], fillLocal, FILL_ELEMS);
    }
    const int64_t tailElems = totalElements - chunkCnt * FILL_ELEMS;
    if (tailElems > 0 && coreIdx == chunkCnt % coreNum) {
        DataCopyExtParams tailParams{1, static_cast<uint32_t>(tailElems * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(yGlobal[static_cast<uint64_t>(chunkCnt) * static_cast<uint64_t>(FILL_ELEMS)], fillLocal,
                    tailParams);
    }
}

} // namespace NsStackGroupPoints
#endif // STACK_GROUP_POINTS_H_
