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
 * \file deformable_roi_pool_simt.h
 * \brief SIMT kernel for deformable_roi_pool operator
 */

#ifndef DEFORMABLE_ROI_POOL_SIMT_H_
#define DEFORMABLE_ROI_POOL_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "deformable_roi_pool_tiling_data.h"
#include "deformable_roi_pool_tiling_key.h"

#pragma clang fp contract(off)

namespace NsDeformableRoiPool {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 256;
constexpr int32_t ROI_COLS = 5;
constexpr int32_t OFFSET_CHANNELS = 2;
// sqrt(INT32_MAX), keeping the 2-D sample count representable by int32_t.
constexpr float MAX_ADAPTIVE_GRID_SIZE = 46340.0f;

template <typename T>
__simt_callee__ inline float DeformableRoiPoolCastToFloat(T val);

template <>
__simt_callee__ inline float DeformableRoiPoolCastToFloat<float>(float val)
{
    return val;
}

template <>
__simt_callee__ inline float DeformableRoiPoolCastToFloat<half>(half val)
{
    return __half2float(val);
}

template <typename T>
__simt_callee__ inline T DeformableRoiPoolCastFromFloat(float val);

template <>
__simt_callee__ inline float DeformableRoiPoolCastFromFloat<float>(float val)
{
    return val;
}

template <>
__simt_callee__ inline half DeformableRoiPoolCastFromFloat<half>(float val)
{
    return __float2half(val);
}

__simt_callee__ inline int32_t GetAdaptiveGridSize(float binSize)
{
    if (isnan(binSize) || isinf(binSize) || binSize <= 0.0f || binSize > MAX_ADAPTIVE_GRID_SIZE) {
        return 0;
    }
    return static_cast<int32_t>(ceilf(binSize));
}

template <typename T>
__simt_callee__ inline void ReadRoiMeta(const __gm__ T* roisGm, int32_t n, float spatialScale, int32_t pooledH,
                                        int32_t pooledW, int32_t samplingRatio, int32_t N, int32_t* batchIdx,
                                        float* roiStartW, float* roiStartH, float* roiWidth, float* roiHeight,
                                        float* binSizeW, float* binSizeH, int32_t* roiBinGridH, int32_t* roiBinGridW,
                                        float* gridH, float* gridW)
{
    int64_t roiBase = static_cast<int64_t>(n) * ROI_COLS;
    float roiBatchIdx = DeformableRoiPoolCastToFloat<T>(roisGm[roiBase + 0]);
    float roiX1 = DeformableRoiPoolCastToFloat<T>(roisGm[roiBase + 1]);
    float roiY1 = DeformableRoiPoolCastToFloat<T>(roisGm[roiBase + 2]);
    float roiX2 = DeformableRoiPoolCastToFloat<T>(roisGm[roiBase + 3]);
    float roiY2 = DeformableRoiPoolCastToFloat<T>(roisGm[roiBase + 4]);

    // Sanitize ROI values before integer conversion.
    if (isnan(roiBatchIdx) || isinf(roiBatchIdx)) {
        roiBatchIdx = 0.0f;
    }
    if (isnan(roiX1) || isinf(roiX1)) {
        roiX1 = 0.0f;
    }
    if (isnan(roiY1) || isinf(roiY1)) {
        roiY1 = 0.0f;
    }
    if (isnan(roiX2) || isinf(roiX2)) {
        roiX2 = 0.0f;
    }
    if (isnan(roiY2) || isinf(roiY2)) {
        roiY2 = 0.0f;
    }

    if (N > 0) {
        // Match the reference "truncate toward zero, then clamp to [0, N-1]". The integer
        // conversion is guarded by 2^31, which is exactly representable in fp32 and above every
        // possible N-1 (N fits int32); comparing against float(N-1) or float(N) would be wrong
        // because both round to even and can fall below a legal, exactly representable index
        // (e.g. N=16777218: float(N-1)=16777216 < 16777216.0f index boundary; N=33554434:
        // float(N)=33554432 < N-1).
        if (roiBatchIdx <= 0.0f) {
            *batchIdx = 0;
        } else if (roiBatchIdx >= 2147483648.0f) {
            *batchIdx = N - 1;
        } else {
            int32_t truncated = static_cast<int32_t>(roiBatchIdx);
            *batchIdx = (truncated > N - 1) ? N - 1 : truncated;
        }
    } else {
        *batchIdx = 0;
    }

    *roiStartW = roiX1 * spatialScale - 0.5f;
    *roiStartH = roiY1 * spatialScale - 0.5f;
    float roiEndW = roiX2 * spatialScale - 0.5f;
    float roiEndH = roiY2 * spatialScale - 0.5f;
    *roiWidth = roiEndW - *roiStartW;
    *roiHeight = roiEndH - *roiStartH;

    *binSizeW = *roiWidth / static_cast<float>(pooledW);
    *binSizeH = *roiHeight / static_cast<float>(pooledH);

    if (samplingRatio > 0) {
        *roiBinGridH = samplingRatio;
        *roiBinGridW = samplingRatio;
    } else {
        *roiBinGridH = GetAdaptiveGridSize(*binSizeH);
        *roiBinGridW = GetAdaptiveGridSize(*binSizeW);
    }

    *gridH = (*roiBinGridH > 0) ? (*binSizeH / static_cast<float>(*roiBinGridH)) : 0.0f;
    *gridW = (*roiBinGridW > 0) ? (*binSizeW / static_cast<float>(*roiBinGridW)) : 0.0f;
}

template <typename T>
__simt_callee__ inline float AccumulateBin(const __gm__ T* featC, int32_t H, int32_t W, float binStartH,
                                           float binStartW, int32_t roiBinGridH, int32_t roiBinGridW, float gridH,
                                           float gridW, float invCount)
{
    // Stable mean: scale every sample by 1/count so the running value stays bounded by max|sample|
    // and cannot overflow to inf on large legal fp32 inputs, and carry Kahan/Neumaier compensation
    // so that large counts (whose scaled terms are far below the running mean's ulp) do not have
    // their low-order bits swallowed by the addition.
    float val = 0.0f;
    float compensation = 0.0f;
    for (int32_t iy = 0; iy < roiBinGridH; iy++) {
        volatile float sampleOffsetH = (static_cast<float>(iy) + 0.5f) * gridH;
        volatile float roundedRawY = binStartH + sampleOffsetH;
        float rawY = roundedRawY;
        if (isnan(rawY) || isinf(rawY) || rawY < -1.0f || rawY > static_cast<float>(H)) {
            continue;
        }
        for (int32_t ix = 0; ix < roiBinGridW; ix++) {
            volatile float sampleOffsetW = (static_cast<float>(ix) + 0.5f) * gridW;
            volatile float roundedRawX = binStartW + sampleOffsetW;
            float rawX = roundedRawX;
            if (isnan(rawX) || isinf(rawX) || rawX < -1.0f || rawX > static_cast<float>(W)) {
                continue;
            }
            float yClip = (rawY > 0.0f) ? rawY : 0.0f;
            float xClip = (rawX > 0.0f) ? rawX : 0.0f;
            int32_t yLo = static_cast<int32_t>(floorf(yClip));
            int32_t xLo = static_cast<int32_t>(floorf(xClip));
            int32_t yHi = yLo + 1;
            int32_t xHi = xLo + 1;
            if (yLo > H - 1) {
                yLo = H - 1;
            }
            if (yHi > H - 1) {
                yHi = H - 1;
            }
            yClip = (yClip < static_cast<float>(H - 1)) ? yClip : static_cast<float>(H - 1);
            if (xLo > W - 1) {
                xLo = W - 1;
            }
            if (xHi > W - 1) {
                xHi = W - 1;
            }
            xClip = (xClip < static_cast<float>(W - 1)) ? xClip : static_cast<float>(W - 1);
            float ly = yClip - static_cast<float>(yLo);
            float lx = xClip - static_cast<float>(xLo);
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;
            float w1 = hy * hx; // 左上 (yLo, xLo)
            float w2 = hy * lx; // 右上 (yLo, xHi)
            float w3 = ly * hx; // 左下 (yHi, xLo)
            float w4 = ly * lx; // 右下 (yHi, xHi)
            int64_t rowLo = static_cast<int64_t>(yLo) * W;
            int64_t rowHi = static_cast<int64_t>(yHi) * W;
            float v1 = DeformableRoiPoolCastToFloat<T>(featC[rowLo + xLo]);
            float v2 = DeformableRoiPoolCastToFloat<T>(featC[rowLo + xHi]);
            float v3 = DeformableRoiPoolCastToFloat<T>(featC[rowHi + xLo]);
            float v4 = DeformableRoiPoolCastToFloat<T>(featC[rowHi + xHi]);
            volatile float product1 = w1 * v1;
            volatile float product2 = w2 * v2;
            volatile float product3 = w3 * v3;
            volatile float product4 = w4 * v4;
            float sample = product1;
            sample = sample + product2;
            sample = sample + product3;
            sample = sample + product4;
            float term = sample * invCount - compensation;
            float next = val + term;
            compensation = (next - val) - term;
            val = next;
        }
    }
    return val;
}

template <typename T, bool HAS_OFFSET>
__simt_callee__ inline void ProcessChannel(const __gm__ T* featC, const __gm__ T* offGm, int64_t phwSize,
                                           int32_t pooledH, int32_t pooledW, float roiStartH, float roiStartW,
                                           float binSizeH, float binSizeW, int32_t roiBinGridH, int32_t roiBinGridW,
                                           float gridH, float gridW, float gamma, float roiWidth, float roiHeight,
                                           int32_t H, int32_t W, __gm__ T* outC)
{
    for (int32_t ph = 0; ph < pooledH; ph++) {
        for (int32_t pw = 0; pw < pooledW; pw++) {
            volatile float binOffsetH = static_cast<float>(ph) * binSizeH;
            volatile float binOffsetW = static_cast<float>(pw) * binSizeW;
            volatile float roundedBinStartH = roiStartH + binOffsetH;
            volatile float roundedBinStartW = roiStartW + binOffsetW;
            float binStartH = roundedBinStartH;
            float binStartW = roundedBinStartW;
            if constexpr (HAS_OFFSET) {
                int64_t offIdx = static_cast<int64_t>(ph) * pooledW + pw;
                // channel 0 = w_offset, channel 1 = h_offset
                float offW = DeformableRoiPoolCastToFloat<T>(offGm[offIdx]);
                float offH = DeformableRoiPoolCastToFloat<T>(offGm[phwSize + offIdx]);
                if (isnan(offW) || isinf(offW)) {
                    offW = 0.0f;
                }
                if (isnan(offH) || isinf(offH)) {
                    offH = 0.0f;
                }
                volatile float scaledOffW = offW * gamma;
                volatile float scaledOffH = offH * gamma;
                volatile float deltaW = scaledOffW * roiWidth;
                volatile float deltaH = scaledOffH * roiHeight;
                volatile float shiftedBinStartW = binStartW + deltaW;
                volatile float shiftedBinStartH = binStartH + deltaH;
                binStartW = shiftedBinStartW;
                binStartH = shiftedBinStartH;
            }
            int64_t cnt = static_cast<int64_t>(roiBinGridH) * roiBinGridW;
            if (cnt == 0) {
                cnt = 1;
            }
            float invCount = 1.0f / static_cast<float>(cnt);
            float result = AccumulateBin<T>(featC, H, W, binStartH, binStartW, roiBinGridH, roiBinGridW, gridH, gridW,
                                            invCount);
            outC[static_cast<int64_t>(ph) * pooledW + pw] = DeformableRoiPoolCastFromFloat<T>(result);
        }
    }
}

template <typename T, bool HAS_OFFSET>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void DeformableRoiPoolSimtKernel(
    int32_t perCoreRois, int32_t numRois, int32_t N, int32_t C, int32_t H, int32_t W, int32_t pooledH, int32_t pooledW,
    int32_t samplingRatio, float spatialScale, float gamma, const __gm__ T* xGm, const __gm__ T* roisGm,
    const __gm__ T* offsetGm, __gm__ T* yGm)
{
    int32_t coreId = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t stride = blockDim.x;
    int64_t roiStart = static_cast<int64_t>(coreId) * perCoreRois;
    int64_t candidateRoiEnd = roiStart + perCoreRois;
    int64_t roiEnd = (candidateRoiEnd > numRois) ? numRois : candidateRoiEnd;
    int64_t hwSize = static_cast<int64_t>(H) * W;
    int64_t phwSize = static_cast<int64_t>(pooledH) * pooledW;

    // Avoid reading inputs when the feature map is empty.
    if (N <= 0 || H <= 0 || W <= 0) {
        for (int64_t roiIdx = roiStart; roiIdx < roiEnd; roiIdx++) {
            __gm__ T* outGm = yGm + roiIdx * C * phwSize;
            for (int32_t c = tid; c < C; c += stride) {
                __gm__ T* outC = outGm + static_cast<int64_t>(c) * phwSize;
                for (int64_t pooledIdx = 0; pooledIdx < phwSize; pooledIdx++) {
                    outC[pooledIdx] = DeformableRoiPoolCastFromFloat<T>(0.0f);
                }
            }
        }
        return;
    }

    for (int64_t roiIdx = roiStart; roiIdx < roiEnd; roiIdx++) {
        int32_t n = static_cast<int32_t>(roiIdx);
        int32_t batchIdx = 0;
        float roiStartW = 0.0f, roiStartH = 0.0f, roiWidth = 0.0f, roiHeight = 0.0f;
        float binSizeW = 0.0f, binSizeH = 0.0f, gridH = 0.0f, gridW = 0.0f;
        int32_t roiBinGridH = 0, roiBinGridW = 0;
        ReadRoiMeta<T>(roisGm, n, spatialScale, pooledH, pooledW, samplingRatio, N, &batchIdx, &roiStartW, &roiStartH,
                       &roiWidth, &roiHeight, &binSizeW, &binSizeH, &roiBinGridH, &roiBinGridW, &gridH, &gridW);

        const __gm__ T* featGm = xGm + static_cast<int64_t>(batchIdx) * C * hwSize;
        __gm__ T* outGm = yGm + static_cast<int64_t>(n) * C * phwSize;
        // Do not offset the dummy pointer when offset is absent.
        const __gm__ T* offGm = offsetGm;
        if constexpr (HAS_OFFSET) {
            offGm = offsetGm + static_cast<int64_t>(n) * OFFSET_CHANNELS * phwSize;
        }

        for (int32_t c = tid; c < C; c += stride) {
            const __gm__ T* featC = featGm + static_cast<int64_t>(c) * hwSize;
            __gm__ T* outC = outGm + static_cast<int64_t>(c) * phwSize;
            ProcessChannel<T, HAS_OFFSET>(featC, offGm, phwSize, pooledH, pooledW, roiStartH, roiStartW, binSizeH,
                                          binSizeW, roiBinGridH, roiBinGridW, gridH, gridW, gamma, roiWidth, roiHeight,
                                          H, W, outC);
        }
    }
}

template <typename T, bool HAS_OFFSET>
__aicore__ inline void DeformableRoiPoolProcess(const DeformableRoiPoolTilingData* tilingData, const __gm__ T* xGm,
                                                const __gm__ T* roisGm, const __gm__ T* offsetGm, __gm__ T* yGm)
{
    asc_vf_call<DeformableRoiPoolSimtKernel<T, HAS_OFFSET>>(
        dim3(THREAD_NUM), tilingData->perCoreRois, tilingData->numRois, tilingData->N, tilingData->C, tilingData->H,
        tilingData->W, tilingData->pooledHeight, tilingData->pooledWidth, tilingData->samplingRatio,
        tilingData->spatialScale, tilingData->gamma, xGm, roisGm, offsetGm, yGm);
}
} // namespace NsDeformableRoiPool
#endif // DEFORMABLE_ROI_POOL_SIMT_H_
