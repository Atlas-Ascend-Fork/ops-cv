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
 * \file yolo_simt.h
 * \brief SIMT kernel implementation for yolo operator
 *
 * Yolo is the post-processing layer of YOLO v2/v3 detection networks.
 * It decodes convolutional feature maps into detection box coordinates,
 * object confidence, and class probabilities.
 *
 * 4 computation modes (yolo_mode):
 *   MODE_1: obj=sigmoid, classes=sigmoid
 *   MODE_2: obj=sigmoid, classes=softmax
 *   MODE_3: obj=move,    classes=sigmoid
 *   MODE_4: obj+classes combined softmax
 *
 * Performance: IDX_T template for 32/64-bit index paths (R003+R006)
 *
 * TTK round-9: Fixed output stride mismatch. The infershape pads the last
 * dim with CeilX alignment (ceilHW / ceilBoxesHw), but the kernel was using
 * HW as stride — causing valid data to spill into padding regions and padding
 * to contain non-zero values. Fix: use ceilHW for coord stride, ceilBoxesHw
 * for obj/cls stride, and zero-fill padding inside the compute kernel (Plan B)
 * to avoid multi-core sync issues with a separate zero-fill VF.
 */

#ifndef YOLO_SIMT_H_
#define YOLO_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/cpp/kernel_simt_common_intf.h"
#include "yolo_tiling_data.h"
#include "yolo_tiling_key.h"

namespace NsYolo {

using namespace AscendC;

// ==================== Thread count ====================
static constexpr uint32_t THREAD_NUM = 512;

// ==================== Precision Conversion Helpers ====================

template <typename T>
__simt_callee__ inline float ToComputeFloat(T val);

template <>
__simt_callee__ inline float ToComputeFloat<half>(half val)
{
    return __half2float(val);
}

template <>
__simt_callee__ inline float ToComputeFloat<float>(float val)
{
    return val;
}

template <typename T>
__simt_callee__ inline T FromComputeFloat(float val);

template <>
__simt_callee__ inline half FromComputeFloat<half>(float val)
{
    // TTK round-8: Ascend hardware __float2half operates in FTZ (flush-to-zero)
    // mode, flushing float16 subnormal (denormal) values to zero. Golden uses
    // numpy astype(np.float16) which preserves subnormals. When softmax produces
    // outputs in [2^-24, 2^-14), golden retains non-zero subnormals but kernel
    // outputs 0. Fix: manually construct subnormal half via __ushort_as_half.
    constexpr float FP16_MIN_SUBNORMAL = 5.960464477539063e-8f; // 2^-24
    constexpr float FP16_MIN_NORMAL = 6.103515625e-5f;          // 2^-14
    float absVal = (val < 0.0f) ? -val : val;
    if (absVal >= FP16_MIN_SUBNORMAL && absVal < FP16_MIN_NORMAL) {
        unsigned short sign = (val < 0.0f) ? 0x8000U : 0x0000U;
        float scaled = absVal * 16777216.0f; // absVal * 2^24
        uint32_t mantissa = static_cast<uint32_t>(scaled + 0.5f);
        if (mantissa > 1023U) {
            mantissa = 1023U;
        }
        if (mantissa == 0U) {
            return __float2half(0.0f);
        }
        return __ushort_as_half(static_cast<unsigned short>(sign | mantissa));
    }
    return __float2half(val);
}

template <>
__simt_callee__ inline float FromComputeFloat<float>(float val)
{
    return val;
}

// ==================== Sigmoid Function ====================
__simt_callee__ inline float SigmoidFloat(float x) { return 1.0f / (1.0f + expf(-x)); }

// ==================== Coordinate Processing (R003: IDX_T) ====================
// TTK round-9: output uses ceilHW stride (padded), input uses HW stride (no pad)
template <typename T, typename IDX_T>
__simt_callee__ inline void ProcessCoords(IDX_T inputBase, IDX_T coordBase, int32_t b, IDX_T hw, int32_t boxes,
                                          IDX_T HW, IDX_T ceilHW, __gm__ T* input, __gm__ T* coord_data)
{
    IDX_T inBhw = static_cast<IDX_T>(b) * HW;
    IDX_T outBhw = static_cast<IDX_T>(b) * ceilHW;
    // x coordinate: sigmoid
    float xVal = ToComputeFloat<T>(input[inputBase + inBhw + hw]);
    coord_data[coordBase + outBhw + hw] = FromComputeFloat<T>(SigmoidFloat(xVal));
    // y coordinate: sigmoid
    IDX_T yInOff = (static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    IDX_T yOutOff = (static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * ceilHW + hw;
    float yVal = ToComputeFloat<T>(input[inputBase + yInOff]);
    coord_data[coordBase + yOutOff] = FromComputeFloat<T>(SigmoidFloat(yVal));
    // h coordinate: move (from input 3B+b -> output 2B+b, swap!)
    IDX_T hInOff = (static_cast<IDX_T>(3) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    IDX_T hOutOff = (static_cast<IDX_T>(2) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * ceilHW + hw;
    coord_data[coordBase + hOutOff] = input[inputBase + hInOff];
    // w coordinate: move (from input 2B+b -> output 3B+b, swap!)
    IDX_T wInOff = (static_cast<IDX_T>(2) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    IDX_T wOutOff = (static_cast<IDX_T>(3) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * ceilHW + hw;
    coord_data[coordBase + wOutOff] = input[inputBase + wInOff];
}

// ==================== MODE_1: obj=sigmoid, classes=sigmoid (R003) ====================
// TTK round-9: output clsStep uses ceilBoxesHw, input clsStep uses boxes*HW
template <typename T, typename IDX_T>
__simt_callee__ inline void ProcessMode1(IDX_T inputBase, IDX_T objBase, IDX_T clsBase, int32_t b, IDX_T hw,
                                         int32_t boxes, int32_t classes, IDX_T HW, IDX_T ceilBoxesHw, __gm__ T* input,
                                         __gm__ T* obj_prob, __gm__ T* classes_prob)
{
    IDX_T bHwHw = static_cast<IDX_T>(b) * HW + hw;
    // obj: sigmoid
    IDX_T objOff = (static_cast<IDX_T>(4) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    float objVal = ToComputeFloat<T>(input[inputBase + objOff]);
    obj_prob[objBase + bHwHw] = FromComputeFloat<T>(SigmoidFloat(objVal));
    // classes: sigmoid
    IDX_T clsInBase = inputBase + static_cast<IDX_T>(5) * static_cast<IDX_T>(boxes) * HW + bHwHw;
    IDX_T clsOutBase = clsBase + bHwHw;
    IDX_T inClsStep = static_cast<IDX_T>(boxes) * HW;
    for (int32_t k = 0; k < classes; k++) {
        IDX_T inK = static_cast<IDX_T>(k) * inClsStep;
        IDX_T outK = static_cast<IDX_T>(k) * ceilBoxesHw;
        float clsVal = ToComputeFloat<T>(input[clsInBase + inK]);
        classes_prob[clsOutBase + outK] = FromComputeFloat<T>(SigmoidFloat(clsVal));
    }
}

// ==================== MODE_2: obj=sigmoid, classes=softmax (R003) ====================
// TTK round-7: reverted to MDE 3-pass (max -> exp+sum -> normalize).
// TTK round-9: output clsStep uses ceilBoxesHw, input clsStep uses boxes*HW
template <typename T, typename IDX_T>
__simt_callee__ inline void ProcessMode2(IDX_T inputBase, IDX_T objBase, IDX_T clsBase, int32_t b, IDX_T hw,
                                         int32_t boxes, int32_t classes, IDX_T HW, IDX_T ceilBoxesHw, __gm__ T* input,
                                         __gm__ T* obj_prob, __gm__ T* classes_prob)
{
    IDX_T bHwHw = static_cast<IDX_T>(b) * HW + hw;
    IDX_T inClsStep = static_cast<IDX_T>(boxes) * HW;
    // obj: sigmoid
    IDX_T objOff = (static_cast<IDX_T>(4) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    float objVal = ToComputeFloat<T>(input[inputBase + objOff]);
    obj_prob[objBase + bHwHw] = FromComputeFloat<T>(SigmoidFloat(objVal));
    // classes: softmax (3-pass: max -> exp+sum -> normalize), MDE 5.3
    IDX_T clsInBase = inputBase + static_cast<IDX_T>(5) * static_cast<IDX_T>(boxes) * HW + bHwHw;
    IDX_T clsOutBase = clsBase + bHwHw;
    // Pass 1: find max over K classes
    float maxVal = -ASCRT_INF_F;
    for (int32_t k = 0; k < classes; k++) {
        float val = ToComputeFloat<T>(input[clsInBase + static_cast<IDX_T>(k) * inClsStep]);
        if (val > maxVal) {
            maxVal = val;
        }
    }
    // Pass 2: compute exp(x-max) and sum (no write, stay in float32)
    float sumExp = 0.0f;
    for (int32_t k = 0; k < classes; k++) {
        float val = ToComputeFloat<T>(input[clsInBase + static_cast<IDX_T>(k) * inClsStep]);
        sumExp += expf(val - maxVal);
    }
    // Pass 3: re-read input, write output with ceilBoxesHw stride
    float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
    for (int32_t k = 0; k < classes; k++) {
        IDX_T inK = static_cast<IDX_T>(k) * inClsStep;
        IDX_T outK = static_cast<IDX_T>(k) * ceilBoxesHw;
        float val = ToComputeFloat<T>(input[clsInBase + inK]);
        classes_prob[clsOutBase + outK] = FromComputeFloat<T>(expf(val - maxVal) * invSum);
    }
}

// ==================== MODE_3: obj=move, classes=sigmoid (R003) ====================
// TTK round-9: output clsStep uses ceilBoxesHw, input clsStep uses boxes*HW
template <typename T, typename IDX_T>
__simt_callee__ inline void ProcessMode3(IDX_T inputBase, IDX_T objBase, IDX_T clsBase, int32_t b, IDX_T hw,
                                         int32_t boxes, int32_t classes, IDX_T HW, IDX_T ceilBoxesHw, __gm__ T* input,
                                         __gm__ T* obj_prob, __gm__ T* classes_prob)
{
    IDX_T bHwHw = static_cast<IDX_T>(b) * HW + hw;
    IDX_T inClsStep = static_cast<IDX_T>(boxes) * HW;
    // obj: move (direct copy)
    IDX_T objOff = (static_cast<IDX_T>(4) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    obj_prob[objBase + bHwHw] = input[inputBase + objOff];
    // classes: sigmoid
    IDX_T clsInBase = inputBase + static_cast<IDX_T>(5) * static_cast<IDX_T>(boxes) * HW + bHwHw;
    IDX_T clsOutBase = clsBase + bHwHw;
    for (int32_t k = 0; k < classes; k++) {
        IDX_T inK = static_cast<IDX_T>(k) * inClsStep;
        IDX_T outK = static_cast<IDX_T>(k) * ceilBoxesHw;
        float clsVal = ToComputeFloat<T>(input[clsInBase + inK]);
        classes_prob[clsOutBase + outK] = FromComputeFloat<T>(SigmoidFloat(clsVal));
    }
}

// ==================== MODE_4: obj+classes combined softmax (R003) ====================
// TTK round-7: reverted to MDE 3-pass (same rationale as ProcessMode2).
// TTK round-9: output clsStep uses ceilBoxesHw, input clsStep uses boxes*HW
template <typename T, typename IDX_T>
__simt_callee__ inline void ProcessMode4(IDX_T inputBase, IDX_T objBase, IDX_T clsBase, int32_t b, IDX_T hw,
                                         int32_t boxes, int32_t classes, IDX_T HW, IDX_T ceilBoxesHw, __gm__ T* input,
                                         __gm__ T* obj_prob, __gm__ T* classes_prob)
{
    IDX_T bHwHw = static_cast<IDX_T>(b) * HW + hw;
    IDX_T inClsStep = static_cast<IDX_T>(boxes) * HW;
    IDX_T clsInBase = inputBase + static_cast<IDX_T>(5) * static_cast<IDX_T>(boxes) * HW + bHwHw;
    IDX_T clsOutBase = clsBase + bHwHw;
    // Read obj value
    IDX_T objOff = (static_cast<IDX_T>(4) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * HW + hw;
    float objVal = ToComputeFloat<T>(input[inputBase + objOff]);
    // Pass 1: find max over obj + K classes (MDE: init with objVal)
    float maxVal = objVal;
    for (int32_t k = 0; k < classes; k++) {
        float val = ToComputeFloat<T>(input[clsInBase + static_cast<IDX_T>(k) * inClsStep]);
        if (val > maxVal) {
            maxVal = val;
        }
    }
    // Pass 2: compute exp(x-max) and sum over obj + K classes (no write, float32)
    float sumExp = expf(objVal - maxVal);
    for (int32_t k = 0; k < classes; k++) {
        float val = ToComputeFloat<T>(input[clsInBase + static_cast<IDX_T>(k) * inClsStep]);
        sumExp += expf(val - maxVal);
    }
    // Pass 3: re-read, write output with ceilBoxesHw stride
    float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
    obj_prob[objBase + bHwHw] = FromComputeFloat<T>(expf(objVal - maxVal) * invSum);
    for (int32_t k = 0; k < classes; k++) {
        IDX_T inK = static_cast<IDX_T>(k) * inClsStep;
        IDX_T outK = static_cast<IDX_T>(k) * ceilBoxesHw;
        float val = ToComputeFloat<T>(input[clsInBase + inK]);
        classes_prob[clsOutBase + outK] = FromComputeFloat<T>(expf(val - maxVal) * invSum);
    }
}

// ==================== Padding Zero-Fill Helpers (TTK round-9, Plan B) ====================
// Zero coord padding: for hw >= HW, write zero to all 4 coord channels.
// Called inside the compute kernel's grid-stride loop — no multi-core sync needed.
template <typename T, typename IDX_T>
__simt_callee__ inline void ZeroCoordPadding(__gm__ T* coord_data, IDX_T coordBase, int32_t b, int32_t boxes,
                                             IDX_T ceilHW, IDX_T hw)
{
    coord_data[coordBase + static_cast<IDX_T>(b) * ceilHW + hw] = static_cast<T>(0);
    coord_data[coordBase + (static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * ceilHW + hw] = static_cast<T>(0);
    coord_data[coordBase + (static_cast<IDX_T>(2) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * ceilHW +
               hw] = static_cast<T>(0);
    coord_data[coordBase + (static_cast<IDX_T>(3) * static_cast<IDX_T>(boxes) + static_cast<IDX_T>(b)) * ceilHW +
               hw] = static_cast<T>(0);
}

// Zero obj/cls padding: tail of B*HW block up to ceilBoxesHw, per n (and per k for cls).
// Called once per n after the compute loop — each thread zeros its grid-stride portion.
template <typename T, typename IDX_T>
__simt_callee__ inline void ZeroObjClsPadding(__gm__ T* obj_prob, __gm__ T* classes_prob, IDX_T objBase, IDX_T clsBase,
                                              int32_t classes, IDX_T bHwTotal, IDX_T ceilBoxesHw, IDX_T tid,
                                              IDX_T gridStride)
{
    for (IDX_T i = tid + bHwTotal; i < ceilBoxesHw; i += gridStride) {
        obj_prob[objBase + i] = static_cast<T>(0);
    }
    for (int32_t k = 0; k < classes; k++) {
        IDX_T kBase = clsBase + static_cast<IDX_T>(k) * ceilBoxesHw + bHwTotal;
        for (IDX_T i = tid; i < ceilBoxesHw - bHwTotal; i += gridStride) {
            classes_prob[kBase + i] = static_cast<T>(0);
        }
    }
}

// ==================== Main SIMT VF Kernel (R003: IDX_T) ====================
// TTK round-9: output strides use ceilHW/ceilBoxesHw; padding zeroed in-kernel
template <typename T, int YOLO_MODE, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpYoloSimtKernel(
    int32_t N, int32_t boxes, int32_t classes, IDX_T HW, IDX_T ceilHW, IDX_T ceilBoxesHw, __gm__ T* input,
    __gm__ T* coord_data, __gm__ T* obj_prob, __gm__ T* classes_prob)
{
    IDX_T bHwTotal = static_cast<IDX_T>(boxes) * HW;
    IDX_T cHw = static_cast<IDX_T>(boxes) * (5 + classes) * HW;
    IDX_T coordStride = static_cast<IDX_T>(4) * boxes * ceilHW;
    IDX_T clsStride = static_cast<IDX_T>(classes) * ceilBoxesHw;
    uint32_t coreId = Simt::GetBlockIdx();
    uint32_t coreNum = Simt::GetBlockNum();
    IDX_T tid = static_cast<IDX_T>(coreId) * static_cast<IDX_T>(THREAD_NUM) + static_cast<IDX_T>(Simt::GetThreadIdx());
    IDX_T gridStride = static_cast<IDX_T>(coreNum) * static_cast<IDX_T>(THREAD_NUM);

    for (int32_t n = 0; n < N; n++) {
        IDX_T inputBase = static_cast<IDX_T>(n) * cHw;
        IDX_T coordBase = static_cast<IDX_T>(n) * coordStride;
        IDX_T objBase = static_cast<IDX_T>(n) * ceilBoxesHw;
        IDX_T clsBase = static_cast<IDX_T>(n) * clsStride;
        for (int32_t b = 0; b < boxes; b++) {
            for (IDX_T hw = tid; hw < ceilHW; hw += gridStride) {
                if (hw < HW) {
                    ProcessCoords<T, IDX_T>(inputBase, coordBase, b, hw, boxes, HW, ceilHW, input, coord_data);
                    if constexpr (YOLO_MODE == YOLO_MODE_1) {
                        ProcessMode1<T, IDX_T>(inputBase, objBase, clsBase, b, hw, boxes, classes, HW, ceilBoxesHw,
                                               input, obj_prob, classes_prob);
                    } else if constexpr (YOLO_MODE == YOLO_MODE_2) {
                        ProcessMode2<T, IDX_T>(inputBase, objBase, clsBase, b, hw, boxes, classes, HW, ceilBoxesHw,
                                               input, obj_prob, classes_prob);
                    } else if constexpr (YOLO_MODE == YOLO_MODE_3) {
                        ProcessMode3<T, IDX_T>(inputBase, objBase, clsBase, b, hw, boxes, classes, HW, ceilBoxesHw,
                                               input, obj_prob, classes_prob);
                    } else {
                        ProcessMode4<T, IDX_T>(inputBase, objBase, clsBase, b, hw, boxes, classes, HW, ceilBoxesHw,
                                               input, obj_prob, classes_prob);
                    }
                } else {
                    ZeroCoordPadding<T, IDX_T>(coord_data, coordBase, b, boxes, ceilHW, hw);
                }
            }
        }
        ZeroObjClsPadding<T, IDX_T>(obj_prob, classes_prob, objBase, clsBase, classes, bHwTotal, ceilBoxesHw, tid,
                                    gridStride);
    }
}

// ==================== Process Entry (R003: 32/64-bit dispatch) ====================
template <typename T, int YOLO_MODE>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR coord_data, GM_ADDR obj_prob, GM_ADDR classes_prob,
                               const YoloTilingData* tilingData)
{
    __gm__ T* inputGm = (__gm__ T*)x;
    __gm__ T* coordGm = (__gm__ T*)coord_data;
    __gm__ T* objGm = (__gm__ T*)obj_prob;
    __gm__ T* clsGm = (__gm__ T*)classes_prob;

    // R003: Dispatch 32-bit or 64-bit path. The index width must cover the largest
    // buffer that is addressed, not only the input: aligned outputs can be much larger.
    const int64_t n = static_cast<int64_t>(tilingData->N);
    const int64_t b = static_cast<int64_t>(tilingData->boxes);
    const int64_t k = static_cast<int64_t>(tilingData->classes);
    const int64_t inputElements = n * b * (5 + k) * tilingData->HW;
    const int64_t coordElements = n * 4 * b * tilingData->ceilHW;
    const int64_t objElements = n * tilingData->ceilBoxesHw;
    const int64_t clsElements = n * k * tilingData->ceilBoxesHw;
    int64_t maxElements = inputElements;
    if (coordElements > maxElements) {
        maxElements = coordElements;
    }
    if (objElements > maxElements) {
        maxElements = objElements;
    }
    if (clsElements > maxElements) {
        maxElements = clsElements;
    }
    if (maxElements <= static_cast<int64_t>(INT32_MAX)) {
        // 32-bit path: higher throughput index math
        asc_vf_call<OpYoloSimtKernel<T, YOLO_MODE, int32_t>>(
            dim3(THREAD_NUM), tilingData->N, tilingData->boxes, tilingData->classes,
            static_cast<int32_t>(tilingData->HW), static_cast<int32_t>(tilingData->ceilHW),
            static_cast<int32_t>(tilingData->ceilBoxesHw), inputGm, coordGm, objGm, clsGm);
    } else {
        // 64-bit path: correctness for large data
        asc_vf_call<OpYoloSimtKernel<T, YOLO_MODE, int64_t>>(dim3(THREAD_NUM), tilingData->N, tilingData->boxes,
                                                             tilingData->classes, tilingData->HW, tilingData->ceilHW,
                                                             tilingData->ceilBoxesHw, inputGm, coordGm, objGm, clsGm);
    }
}

} // namespace NsYolo
#endif // YOLO_SIMT_H_
