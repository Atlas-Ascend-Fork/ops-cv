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
 * \file dilation2_d_backprop_filter_simt_scatter.h
 * \brief Deterministic dense-scatter route for dilation2_d_backprop_filter
 *        (scheme r1_dense_keyarb_v1, Ascend950PR arch35, SIMT)
 *
 * Replaces, ONLY for the capacity-gated dense route (host predicate
 * routeDenseScatter = routeDense && totalElements <= 2^21 && filterSize >= 1),
 * the frozen-baseline "per-thread filterSize accumulator + 1024->1 tree
 * reduction + core-0 serial merge" structure by three phases:
 *
 *   D1 ScanMaterializeSimt (dim3(1024), one thread per element grid-stride
 *      inside the core's contiguous element slice)
 *        idx -> (b, h_out, w_out, d) (UintDiv magic/shift from UB, the same
 *        decomposition and format branch as the baseline ComputeSimt)
 *        window argmax: h ascending, then w ascending; the x/filter loads of a
 *        candidate group of up to 4 candidates are issued before the first
 *        compare of that group; sentinel -FLT_MAX, strict greater-than update
 *        and the (0,0) fallback are the baseline ComputeSimt semantics
 *        posBuf[idx] = yOffset(hMax, wMax, d)       (int32, 4B/element)
 *      explicit barrier (PipeBarrier<PIPE_ALL>) between D1 and D2
 *   D2 TargetReduceSimt (dim3(1024), one warp per filter target)
 *        target t in [0, filterSize) is owned by exactly one warp; the 32
 *        lanes split the core's element slice into 32 contiguous idx
 *        intervals (lane ascending); each lane accumulates out_backprop in
 *        ascending idx where posBuf[idx] == t; the lane partials are folded in
 *        ascending lane order by asc_shfl, and lane 0 writes
 *        perCorePartials[core * filterSize + t]
 *   D3 CrossCoreMergeSimt (dim3(1024), one warp per filter target)
 *        lanes cover the core range in ascending core id (multi-round when
 *        needCoreNum > 32); the rounds are folded core-ascending and
 *        y[target] is written with '=' for every filter target
 *
 * Determinism: no float atomics. The accumulation grouping is a fixed
 * function of (element index, core order) only — never of needCoreNum or of
 * the thread count — so changing the launch width cannot change the
 * accumulation order (this is the structural difference from the baseline
 * tree reduction, whose merge grouping depends on needCoreNum).
 *
 * Workspace layout (user workspace, 128B aligned by the host):
 *   [0, 4*totalElements)                       posBuf   (int32 per element)
 *   [.., +4*needCoreNum*filterSize)            perCorePartials (float)
 *
 * Kernel-side members are shape-gated at tiling time; the host never selects
 * this route by case id / CSV row / fixed shape list (capacity predicate only)
 * and the legacy Dense path stays the byte-identical fallback.
 */

#ifndef DILATION2D_BACKPROP_FILTER_SIMT_SCATTER_H_
#define DILATION2D_BACKPROP_FILTER_SIMT_SCATTER_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/math_constants.h"
#include "simt_api/math_functions.h"
#include "dilation2_d_backprop_filter_tiling_data.h"

namespace NsDilation2DBackpropFilter {
using namespace AscendC;

namespace DenseScatter {

// This header is included before the routing header opens its own copy of the
// shared constants, so the launch geometry is mirrored here (compile-time
// constants only, identical values).
constexpr uint32_t kThreadNum = 1024; // = THREAD_NUM of the routing header
constexpr uint32_t kLaneNum = 32;     // = WARP_LANE_NUM (950PR SIMT warp size)

// Sentinel of the argmax scan: the lowest finite value of T (baseline
// GetLowestVal semantics, -FLT_MAX for T = float).
template <typename T>
__simt_callee__ __aicore__ inline T ScatterLowestVal()
{
    return -ASCRT_MAX_NORMAL_F;
}

// Candidate batch of the window scan: the loads of a group of up to
// kScanBatch consecutive w candidates are issued before the first compare of
// the group (the compare order and the update predicate are unchanged).
constexpr int64_t kScanBatch = 4;

// ============================================================================
// Phase D1: element-wise argmax materialization
//
// One thread per element, grid-stride inside the core's contiguous element
// slice [sliceBeg, sliceEnd). Every element is covered exactly once across the
// launch, and every core writes only its own posBuf slice, so D2 (which reads
// the same slice) depends on D1 within the core only.
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kThreadNum) inline void ScanMaterializeSimt(
    int64_t totalElements, int64_t inputH, int64_t inputW, int64_t depth, int64_t filterH, int64_t filterW,
    int64_t outH, int64_t outW, int64_t strideH, int64_t strideW, int64_t rateH, int64_t rateW, int64_t padTop,
    int64_t padLeft, int64_t padInputH, int64_t padInputW, int32_t isNCHW, int32_t needCoreNum, int32_t coreId,
    __ubuf__ uint64_t* uintdivUb, __gm__ T* xGm, __gm__ T* filterGm, __gm__ int32_t* posBuf)
{
    // 1. Read UintDiv magic/shift from UB (same slot order as the baseline
    //    ComputeSimt prologue: NHWC -> (depth, outW, outH),
    //    NCHW -> (outW, outH, depth))
    const uint64_t magic0 = uintdivUb[0];
    const uint64_t shift0 = uintdivUb[1];
    const uint64_t magic1 = uintdivUb[2];
    const uint64_t shift1 = uintdivUb[3];
    const uint64_t magic2 = uintdivUb[4];
    const uint64_t shift2 = uintdivUb[5];

    // 2. Format-dependent strides (baseline ComputeSimt expressions, verbatim)
    const int64_t xDepthStride = isNCHW ? (static_cast<int64_t>(inputH) * inputW) : 1;
    const int64_t xRowStride = isNCHW ? static_cast<int64_t>(inputW) : (static_cast<int64_t>(inputW) * depth);
    const int64_t xWStride = isNCHW ? 1 : static_cast<int64_t>(depth);
    const int64_t inputBatchStride = static_cast<int64_t>(inputH) * inputW * depth;

    const int64_t filterDepthStride = isNCHW ? (static_cast<int64_t>(filterH) * filterW) : 1;
    const int64_t filterRowStride = isNCHW ? static_cast<int64_t>(filterW) : (static_cast<int64_t>(filterW) * depth);
    const int64_t filterWStride = isNCHW ? 1 : static_cast<int64_t>(depth);

    const uint64_t div0 = isNCHW ? static_cast<uint64_t>(outW) : static_cast<uint64_t>(depth);
    const uint64_t div1 = isNCHW ? static_cast<uint64_t>(outH) : static_cast<uint64_t>(outW);
    const uint64_t div2 = isNCHW ? static_cast<uint64_t>(depth) : static_cast<uint64_t>(outH);

    const T lowestVal = ScatterLowestVal<T>();

    // 3. The core's contiguous element slice (identical formula in D2). coreId
    //    is the scalar-side GetBlockIdx(): the SIMT VF launch geometry does not
    //    expose the AI-core index, so it is passed in explicitly (the same
    //    contract the frozen-baseline ComputeSimt uses).
    const int64_t uNeedCoreNum = static_cast<int64_t>(needCoreNum > 0 ? needCoreNum : 1);
    const int64_t perCoreElems = (totalElements + uNeedCoreNum - 1) / uNeedCoreNum;
    const int64_t sliceBeg = static_cast<int64_t>(coreId) * perCoreElems;
    int64_t sliceEnd = sliceBeg + perCoreElems;
    if (sliceEnd > totalElements) {
        sliceEnd = totalElements;
    }

    // 4. Grid-stride over the core's slice: one thread per element
    const int64_t stride = static_cast<int64_t>(kThreadNum);
    for (int64_t idx = sliceBeg + static_cast<int64_t>(threadIdx.x); idx < sliceEnd; idx += stride) {
        // 4.1 Index decomposition (baseline order)
        uint64_t rem = static_cast<uint64_t>(idx);
        uint64_t q = Simt::UintDiv<uint64_t>(rem, magic0, shift0);
        uint64_t c0 = rem - q * div0;
        rem = q;
        q = Simt::UintDiv<uint64_t>(rem, magic1, shift1);
        uint64_t c1 = rem - q * div1;
        rem = q;
        q = Simt::UintDiv<uint64_t>(rem, magic2, shift2);
        uint64_t c2 = rem - q * div2;
        uint64_t b = q;

        uint64_t d, hOut, wOut;
        if (isNCHW) {
            wOut = c0;
            hOut = c1;
            d = c2;
        } else {
            d = c0;
            wOut = c1;
            hOut = c2;
        }

        // 4.2 Window start (baseline)
        const int64_t hBeg = static_cast<int64_t>(hOut) * strideH - padTop;
        const int64_t wBeg = static_cast<int64_t>(wOut) * strideW - padLeft;

        T curVal = lowestVal;
        int64_t hMax = 0;
        int64_t wMax = 0;

        const int64_t xBase = static_cast<int64_t>(b) * inputBatchStride + static_cast<int64_t>(d) * xDepthStride;
        const int64_t filterBase = static_cast<int64_t>(d) * filterDepthStride;

        // 4.3 Window argmax (baseline semantics; candidate group batch of up to
        //     kScanBatch loads issued before the first compare of the group)
        for (int64_t h = 0; h < filterH; ++h) {
            const int64_t hIn = hBeg + h * rateH;
            if (hIn >= 0 && hIn < padInputH) {
                const int64_t xRowOffset = xBase + hIn * xRowStride;
                const int64_t filterRowOffset = filterBase + static_cast<int64_t>(h) * filterRowStride;
                const bool hInInput = (hIn < inputH);
                for (int64_t w = 0; w < filterW; w += kScanBatch) {
                    const int64_t wIn0 = wBeg + w * rateW;
                    const int64_t wIn1 = wIn0 + rateW;
                    const int64_t wIn2 = wIn1 + rateW;
                    const int64_t wIn3 = wIn2 + rateW;
                    const bool ok0 = (w < filterW) && (wIn0 >= 0) && (wIn0 < padInputW);
                    const bool ok1 = (w + 1 < filterW) && (wIn1 >= 0) && (wIn1 < padInputW);
                    const bool ok2 = (w + 2 < filterW) && (wIn2 >= 0) && (wIn2 < padInputW);
                    const bool ok3 = (w + 3 < filterW) && (wIn3 >= 0) && (wIn3 < padInputW);

                    // Group loads (issued before the first compare of the group)
                    T g0 = lowestVal;
                    T g1 = lowestVal;
                    T g2 = lowestVal;
                    T g3 = lowestVal;
                    if (ok0) {
                        const T x0 = (hInInput && wIn0 < inputW) ? xGm[xRowOffset + wIn0 * xWStride] :
                                                                   static_cast<T>(0);
                        g0 = x0 + filterGm[filterRowOffset + w * filterWStride];
                    }
                    if (ok1) {
                        const T x1 = (hInInput && wIn1 < inputW) ? xGm[xRowOffset + wIn1 * xWStride] :
                                                                   static_cast<T>(0);
                        g1 = x1 + filterGm[filterRowOffset + (w + 1) * filterWStride];
                    }
                    if (ok2) {
                        const T x2 = (hInInput && wIn2 < inputW) ? xGm[xRowOffset + wIn2 * xWStride] :
                                                                   static_cast<T>(0);
                        g2 = x2 + filterGm[filterRowOffset + (w + 2) * filterWStride];
                    }
                    if (ok3) {
                        const T x3 = (hInInput && wIn3 < inputW) ? xGm[xRowOffset + wIn3 * xWStride] :
                                                                   static_cast<T>(0);
                        g3 = x3 + filterGm[filterRowOffset + (w + 3) * filterWStride];
                    }

                    // Group compares (ascending w, strict greater-than)
                    if (ok0 && g0 > curVal) {
                        curVal = g0;
                        hMax = h;
                        wMax = w;
                    }
                    if (ok1 && g1 > curVal) {
                        curVal = g1;
                        hMax = h;
                        wMax = w + 1;
                    }
                    if (ok2 && g2 > curVal) {
                        curVal = g2;
                        hMax = h;
                        wMax = w + 2;
                    }
                    if (ok3 && g3 > curVal) {
                        curVal = g3;
                        hMax = h;
                        wMax = w + 3;
                    }
                }
            }
        }

        // 4.4 Materialize the argmax target: the y offset of (hMax, wMax, d) in
        //     the native filter layout, i.e. baseline ComputeSimt's yOffset.
        //     For depth == 1 (the four covered tuning cases) this is exactly
        //     hMax * filterW + wMax; for depth > 1 the depth term keeps the
        //     target inside the [0, filterSize) slot space D2/D3 own.
        const int64_t yOffset = hMax * filterRowStride + wMax * filterWStride +
                                static_cast<int64_t>(d) * filterDepthStride;
        posBuf[idx] = static_cast<int32_t>(yOffset);
    }
}

// ============================================================================
// Phase D2: target-ownership reduction (one warp per filter target)
//
// Target t in [0, filterSize) is owned by exactly one warp. The warp's 32
// lanes split the core's element slice into 32 contiguous idx intervals (lane
// ascending, the interval math is lane-invariant); each lane scans its own
// interval in ascending idx and accumulates out_backprop where posBuf == t.
// The lane partials are folded in ascending lane order, so the accumulation
// order is (interval, ascending idx) x ascending lane — independent of the
// launch width.
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kThreadNum) inline void TargetReduceSimt(
    int64_t totalElements, int64_t filterSize, int32_t needCoreNum, int32_t coreId, __gm__ T* outBackpropGm,
    __gm__ int32_t* posBuf, __gm__ T* perCorePartials)
{
    const uint32_t lane = threadIdx.x & (kLaneNum - 1U);
    // Every core owns EVERY filter target (one warp per target, strided by the
    // core's warp count), so a partial row exists for all (core, target) pairs.
    const int64_t warpsPerCore = static_cast<int64_t>(kThreadNum) >> 5;
    const int64_t warpInCore = static_cast<int64_t>(threadIdx.x) >> static_cast<int64_t>(5);

    // The core's contiguous element slice (same formula as D1)
    const int64_t uNeedCoreNum = static_cast<int64_t>(needCoreNum > 0 ? needCoreNum : 1);
    const int64_t perCoreElems = (totalElements + uNeedCoreNum - 1) / uNeedCoreNum;
    const int64_t sliceBeg = static_cast<int64_t>(coreId) * perCoreElems;
    int64_t sliceEnd = sliceBeg + perCoreElems;
    if (sliceEnd > totalElements) {
        sliceEnd = totalElements;
    }
    int64_t sliceLen = sliceEnd - sliceBeg;
    if (sliceLen < 0) {
        sliceLen = 0;
    }

    // 32 contiguous lane intervals (lane ascending)
    const int64_t interval = (sliceLen + static_cast<int64_t>(kLaneNum) - 1) / static_cast<int64_t>(kLaneNum);
    const int64_t laneBeg = sliceBeg + static_cast<int64_t>(lane) * interval;
    int64_t laneEnd = laneBeg + interval;
    if (laneEnd > sliceEnd) {
        laneEnd = sliceEnd;
    }

    const int64_t laneBegSafe = (laneBeg < sliceEnd) ? laneBeg : sliceEnd;
    const int64_t laneEndSafe = (laneEnd > laneBegSafe) ? laneEnd : laneBegSafe;

    for (int64_t t = warpInCore; t < filterSize; t += warpsPerCore) {
        T sum = static_cast<T>(0);
        for (int64_t idx = laneBegSafe; idx < laneEndSafe; ++idx) {
            if (posBuf[idx] == static_cast<int32_t>(t)) {
                sum += outBackpropGm[idx];
            }
        }
        // Lane-ascending fold: every lane converges to the same bitwise sum.
        // The folded value must be the stable lane partial (folding the running
        // accumulator would feedback into itself).
        const T laneSum = sum;
        sum = static_cast<T>(0);
        for (int32_t j = 0; j < static_cast<int32_t>(kLaneNum); ++j) {
            sum += asc_shfl(laneSum, j, static_cast<int32_t>(kLaneNum));
        }
        if (lane == 0) {
            perCorePartials[static_cast<int64_t>(coreId) * filterSize + t] = sum;
        }
    }
}

// ============================================================================
// Phase D3: cross-core ordered merge (one warp per filter target)
//
// Cores are folded in ascending core id; when needCoreNum > 32 the fold runs
// in rounds of 32 cores (round r covers cores r*32 .. r*32+31) and each round
// is folded lane-ascending before the next round is added. y[target] is
// written with '=' for every target in [0, filterSize), i.e. the full filter
// tensor is overwritten.
// ============================================================================
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(kThreadNum) inline void CrossCoreMergeSimt(int64_t filterSize,
                                                                                    int32_t needCoreNum, int32_t coreId,
                                                                                    __gm__ T* perCorePartials,
                                                                                    __gm__ T* yGm)
{
    const uint32_t lane = threadIdx.x & (kLaneNum - 1U);
    // Every core owns EVERY filter target (one warp per target, strided by the
    // core's warp count), so a partial row exists for all (core, target) pairs.
    const int64_t warpsPerCore = static_cast<int64_t>(kThreadNum) >> 5;
    const int64_t warpInCore = static_cast<int64_t>(threadIdx.x) >> static_cast<int64_t>(5);
    const int64_t uNeedCoreNum = static_cast<int64_t>(needCoreNum > 0 ? needCoreNum : 1);

    for (int64_t t = warpInCore; t < filterSize; t += warpsPerCore) {
        T sum = static_cast<T>(0);
        for (int64_t base = 0; base < uNeedCoreNum; base += static_cast<int64_t>(kLaneNum)) {
            const int64_t core = base + static_cast<int64_t>(lane);
            const T v = (core < uNeedCoreNum) ? perCorePartials[core * filterSize + t] : static_cast<T>(0);
            for (int32_t j = 0; j < static_cast<int32_t>(kLaneNum); ++j) {
                sum += asc_shfl(v, j, static_cast<int32_t>(kLaneNum));
            }
        }
        if (lane == 0) {
            yGm[t] = sum;
        }
    }
}

// ============================================================================
// Scalar prologue: UintDiv magic/shift for D1's index decomposition
// (baseline Dense::LaunchComputeVf prologue, verbatim slot order)
// ============================================================================
template <typename T>
__aicore__ inline void LaunchScanVf(const Dilation2DBackpropFilterTilingData* tilingData, __gm__ T* xGm,
                                    __gm__ T* filterGm, __gm__ int32_t* posBuf, int32_t coreId)
{
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<uint64_t> ub = ubAlloc.Alloc<uint64_t>(8);

    uint64_t magic = 0;
    uint64_t shift = 0;

    if (tilingData->isNCHW) {
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outW));
        ub.SetValue(0, magic);
        ub.SetValue(1, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outH));
        ub.SetValue(2, magic);
        ub.SetValue(3, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub.SetValue(4, magic);
        ub.SetValue(5, shift);
    } else {
        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->depth));
        ub.SetValue(0, magic);
        ub.SetValue(1, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outW));
        ub.SetValue(2, magic);
        ub.SetValue(3, shift);

        GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(tilingData->outH));
        ub.SetValue(4, magic);
        ub.SetValue(5, shift);
    }

    DataSyncBarrier<MemDsbT::UB>();

    __ubuf__ uint64_t* uintdivPtr = (__ubuf__ uint64_t*)ub.GetPhyAddr();

    asc_vf_call<ScanMaterializeSimt<T>>(dim3(kThreadNum), tilingData->totalElements, tilingData->inputH,
                                        tilingData->inputW, tilingData->depth, tilingData->filterH, tilingData->filterW,
                                        tilingData->outH, tilingData->outW, tilingData->strideH, tilingData->strideW,
                                        tilingData->rateH, tilingData->rateW, tilingData->padTop, tilingData->padLeft,
                                        tilingData->padInputH, tilingData->padInputW, tilingData->isNCHW,
                                        tilingData->needCoreNum, coreId, uintdivPtr, xGm, filterGm, posBuf);
}

// ============================================================================
// Process: D1 -> (explicit barrier) -> D2 -> (SyncAll) -> D3 -> kernel tail
//
// Exactly one SyncAll (the D2 -> D3 cross-core publish). D1 -> D2 is a
// same-core data dependency and is guarded by an explicit PIPE_ALL barrier —
// the phase order never relies on implicit VF ordering.
// ============================================================================
template <typename T>
__aicore__ inline void Process(const Dilation2DBackpropFilterTilingData* tilingData, __gm__ T* xGm, __gm__ T* filterGm,
                               __gm__ T* outBackpropGm, __gm__ T* yGm, __gm__ T* wsBuf)
{
    // Workspace: posBuf (int32) first, per-core target partials after it
    __gm__ int32_t* posBuf = reinterpret_cast<__gm__ int32_t*>(wsBuf);
    __gm__ T* perCorePartials = wsBuf + tilingData->totalElements;

    // ===== D1: element-wise argmax materialization =====
    const int32_t coreId = static_cast<int32_t>(GetBlockIdx());
    LaunchScanVf<T>(tilingData, xGm, filterGm, posBuf, coreId);

    // ===== explicit D1 -> D2 barrier (no reliance on implicit VF ordering) =====
    PipeBarrier<PIPE_ALL>();

    // ===== D2: target-ownership reduction (warp per filter target) =====
    asc_vf_call<TargetReduceSimt<T>>(dim3(kThreadNum), tilingData->totalElements, tilingData->filterSize,
                                     tilingData->needCoreNum, coreId, outBackpropGm, posBuf, perCorePartials);

    // ===== D2 -> D3: cross-core publish (single SyncAll) =====
    SyncAll();

    // The cross-core merge is a single-writer phase (core 0 owns y): the same
    // scalar-gated launch shape the frozen-baseline dense path uses for its
    // core-0 final reduce.
    if (coreId == 0) {
        asc_vf_call<CrossCoreMergeSimt<T>>(dim3(kThreadNum), tilingData->filterSize, tilingData->needCoreNum, coreId,
                                           perCorePartials, yGm);
    }

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace DenseScatter
} // namespace NsDilation2DBackpropFilter
#endif // DILATION2D_BACKPROP_FILTER_SIMT_SCATTER_H_
