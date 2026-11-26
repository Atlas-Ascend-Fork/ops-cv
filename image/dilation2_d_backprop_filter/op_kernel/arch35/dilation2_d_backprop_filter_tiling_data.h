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
 * \file dilation2_d_backprop_filter_tiling_data.h
 * \brief Tiling data struct for dilation2_d_backprop_filter operator
 *
 * MDE v2.1 §3.1: 18 fields, no threadNum (compile-time constant),
 * no magic/shift (computed in kernel Process and passed via UB).
 *
 * v2.1 change: deterministic accumulation (per-core buffer + final reduce)
 *   - Added perCoreBufElems: per-core buffer size in elements (128B aligned)
 *   - ComputeSimt atomic_add targets per-core workspace, not yGm directly
 *   - New Phase 3: sequential reduce of per-core buffers to yGm
 */

#ifndef DILATION2D_BACKPROP_FILTER_TILING_DATA_H_
#define DILATION2D_BACKPROP_FILTER_TILING_DATA_H_

struct Dilation2DBackpropFilterTilingData {
    int64_t totalElements = 0;   // batch × outH × outW × depth
    int64_t filterSize = 0;      // filterH × filterW × depth (ZeroOut element count)
    int32_t needCoreNum = 0;     // actual launched core count
    int64_t perCoreBufElems = 0; // per-core buffer size in elements (128B aligned), v2.1
    int64_t batch = 0;           // N
    int64_t inputH = 0;          // H_in
    int64_t inputW = 0;          // W_in
    int64_t depth = 0;           // C
    int64_t filterH = 0;         // filter height
    int64_t filterW = 0;         // filter width
    int64_t outH = 0;            // H_out
    int64_t outW = 0;            // W_out
    int64_t strideH = 0;         // strides[1]
    int64_t strideW = 0;         // strides[2]
    int64_t rateH = 0;           // rates[1]
    int64_t rateW = 0;           // rates[2]
    int64_t padTop = 0;          // top padding
    int64_t padLeft = 0;         // left padding
    int64_t padInputH = 0;       // padded input H (inputH + padTop + padBottom), v2.3
    int64_t padInputW = 0;       // padded input W (inputW + padLeft + padRight), v2.3
    int32_t isNCHW = 0;          // 0=NHWC, 1=NCHW (v2.5: NCHW support)
    // v3.0 (chanexcl-fhw-merge-v1): det scheme fields, append-only; nondet branch never reads them
    int32_t groupNum = 0;      // det: number of spatial groups G (scan/merge split, F4/F6)
    int64_t scratchStride = 0; // det: per-thread scratch elements = alignUp(filterH*filterW, 8) (F1/F3)
    // v4.0 (shape-routed-chanexcl-v2): det scheme fields, append-only; nondet branch never reads them
    int32_t routeDense = 0; // det: 1 = shape-routed frozen-baseline dense path (f1/f2, 0/1 flag semantics)
    int64_t chunkLen = 0;   // det: idle since v5.0 f3 rollback (round1 strided partition restored); kept append-only,
                            // always 0
    // v6.0 (r1_dense_keyarb_v1): det dense-scatter route fields, append-only; nondet branch never reads them
    int32_t
        routeDenseScatter = 0; // det: 1 = capacity-gated dense-scatter route (elementwise argmax + target ownership)
    int64_t posBufElems = 0;   // det: posBuf elements (= totalElements when the scatter route is active, else 0)
    int64_t perCorePartialsElems = 0; // det: per-core target partials elements (= needCoreNum × filterSize when active)
    // r9 (vec21 route): det vector form for deep-channel 7x3 NHWC shapes, append-only; nondet never reads it
    int32_t routeVec21 = 0;          // det: 1 = vector-form argmax scan + per-core partial reduce
    int32_t routeSingleTapSmall = 0; // det: one filter tap, bounded spatial reduction
    int32_t routeBalancedMerge = 0;  // det: fixed warp tree for deep, small filter planes
    int32_t routeSingleSample = 0;   // det: a single output window with one evaluated tap
    int32_t sampleH = 0;
    int32_t sampleW = 0;
    int32_t sampleTap = 0;
    int32_t routeIndex32 = 0; // proven bounds include padded-coordinate intermediates
    // v2.0: fp32AccumOffset removed (T=float, direct accumulation on yGm)
};

#endif // DILATION2D_BACKPROP_FILTER_TILING_DATA_H_
