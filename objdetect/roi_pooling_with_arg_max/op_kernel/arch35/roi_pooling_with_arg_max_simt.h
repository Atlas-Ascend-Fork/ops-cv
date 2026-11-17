/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROI_POOLING_WITH_ARG_MAX_H
#define ROI_POOLING_WITH_ARG_MAX_H

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
using namespace AscendC;

constexpr int64_t ROI_BOX_ELEMS = 5;
constexpr int64_t ROI_BATCH_IDX = 0;
constexpr int64_t ROI_X1_IDX = 1;
constexpr int64_t ROI_Y1_IDX = 2;
constexpr int64_t ROI_X2_IDX = 3;
constexpr int64_t ROI_Y2_IDX = 4;
constexpr float ROI_RIGHT_BOTTOM_OFFSET = 1.0f;
constexpr int64_t BIN_CLAMP_LB = 0;
constexpr int64_t INVALID_ARGMAX_OFFSET = -1;
constexpr float ROI_EMPTY_BIN_VAL = 0.0f;
constexpr uint32_t ROI_POOLING_SIMT_LAUNCH_BOUND = 2048;

template <typename dT>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void RoiPoolingBinRange(
    int64_t& bin_x1, int64_t& bin_y1, int64_t& bin_x2, int64_t& bin_y2, bool& is_empty, const __gm__ dT* offset_rois,
    uint32_t ph, uint32_t pw, int64_t poolH, int64_t poolW, float spatial_scale_h, float spatial_scale_w, int64_t fmH,
    int64_t fmW)
{
    float roi_x1 = static_cast<float>(offset_rois[ROI_X1_IDX]) * spatial_scale_w;
    float roi_y1 = static_cast<float>(offset_rois[ROI_Y1_IDX]) * spatial_scale_h;
    float roi_x2 = static_cast<float>(offset_rois[ROI_X2_IDX] + ROI_RIGHT_BOTTOM_OFFSET) * spatial_scale_w;
    float roi_y2 = static_cast<float>(offset_rois[ROI_Y2_IDX] + ROI_RIGHT_BOTTOM_OFFSET) * spatial_scale_h;
    float roi_w = roi_x2 - roi_x1;
    float roi_h = roi_y2 - roi_y1;
    float bin_size_w = roi_w / static_cast<float>(poolW);
    float bin_size_h = roi_h / static_cast<float>(poolH);
    float fx1 = static_cast<float>(pw) * bin_size_w + roi_x1;
    float fy1 = static_cast<float>(ph) * bin_size_h + roi_y1;
    float fx2 = static_cast<float>(pw + 1) * bin_size_w + roi_x1;
    float fy2 = static_cast<float>(ph + 1) * bin_size_h + roi_y1;
    bin_x1 = static_cast<int64_t>(floorf(fx1));
    bin_y1 = static_cast<int64_t>(floorf(fy1));
    bin_x2 = static_cast<int64_t>(ceilf(fx2));
    bin_y2 = static_cast<int64_t>(ceilf(fy2));
    bin_x1 = min(max(bin_x1, BIN_CLAMP_LB), fmW);
    bin_y1 = min(max(bin_y1, BIN_CLAMP_LB), fmH);
    bin_x2 = min(max(bin_x2, BIN_CLAMP_LB), fmW);
    bin_y2 = min(max(bin_y2, BIN_CLAMP_LB), fmH);
    is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);
}

template <typename dT>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void RoiPoolingBinMax(float& max_val, int64_t& max_idx,
                                                                                       const __gm__ dT* offset_input,
                                                                                       int64_t bin_x1, int64_t bin_y1,
                                                                                       int64_t bin_x2, int64_t bin_y2,
                                                                                       int64_t fmW)
{
    // Initialize with the first element of the (non-empty) bin so that an all--inf
    // window keeps a valid first index (doc: only geometric empty bins use -1).
    // Caller must ensure !is_empty (bin_x2 > bin_x1 && bin_y2 > bin_y1).
    const int64_t first_offset = bin_y1 * fmW + bin_x1;
    max_val = static_cast<float>(offset_input[first_offset]);
    max_idx = first_offset;
    for (int64_t h = bin_y1; h < bin_y2; ++h) {
        for (int64_t w = bin_x1; w < bin_x2; ++w) {
            const int64_t offset = h * fmW + w;
            const float val = static_cast<float>(offset_input[offset]);
            if (val > max_val) {
                max_val = val;
                max_idx = offset;
            }
        }
    }
}

template <typename dT>
__simt_vf__ LAUNCH_BOUND(ROI_POOLING_SIMT_LAUNCH_BOUND) inline void RoiPoolingWithArgMaxCompute(
    __gm__ dT* x, __gm__ dT* rois, __gm__ dT* roi_actual_num, __gm__ dT* y, __gm__ int32_t* argmax,
    const int32_t channels, const int32_t fm_height, const int32_t fm_width, const int32_t roi_number,
    const int32_t pooled_h, const int32_t pooled_w, const float spatial_scale_h, const float spatial_scale_w,
    const uint32_t mPoolW, const uint32_t shiftPoolW, const uint32_t mPoolH, const uint32_t shiftPoolH,
    const uint32_t mCH, const uint32_t shiftCH)
{
    const int64_t poolH = static_cast<int64_t>(pooled_h), poolW = static_cast<int64_t>(pooled_w);
    const int64_t fmW = static_cast<int64_t>(fm_width), fmH = static_cast<int64_t>(fm_height);
    const uint32_t upoolW = static_cast<uint32_t>(poolW), upoolH = static_cast<uint32_t>(poolH),
                   uchan = static_cast<uint32_t>(channels);
    const int64_t count = static_cast<int64_t>(roi_number) * static_cast<int64_t>(channels) * poolH * poolW;
    for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x * blockDim.x) {
        const uint32_t uidx = static_cast<uint32_t>(idx);
        const uint32_t divW = Simt::UintDiv(uidx, mPoolW, shiftPoolW);
        const uint32_t pw = uidx - divW * upoolW;
        const uint32_t divH = Simt::UintDiv(divW, mPoolH, shiftPoolH);
        const uint32_t ph = divW - divH * upoolH;
        const uint32_t n = Simt::UintDiv(divH, mCH, shiftCH);
        const uint32_t c = divH - n * uchan;

        const __gm__ dT* offset_rois = rois + n * ROI_BOX_ELEMS;
        const int64_t roi_batch_ind = static_cast<int64_t>(static_cast<float>(offset_rois[ROI_BATCH_IDX]));
        float roi_w = static_cast<float>(offset_rois[ROI_X2_IDX] - offset_rois[ROI_X1_IDX] + ROI_RIGHT_BOTTOM_OFFSET) *
                      spatial_scale_w;
        float roi_h = static_cast<float>(offset_rois[ROI_Y2_IDX] - offset_rois[ROI_Y1_IDX] + ROI_RIGHT_BOTTOM_OFFSET) *
                      spatial_scale_h;
        if (roi_w <= 0 || roi_h <= 0) {
            y[idx] = static_cast<dT>(ROI_EMPTY_BIN_VAL);
            if (argmax != nullptr) {
                argmax[idx] = static_cast<int32_t>(INVALID_ARGMAX_OFFSET);
            }
            continue;
        }

        int64_t bin_x1 = 0, bin_y1 = 0, bin_x2 = 0, bin_y2 = 0;
        bool is_empty = false;
        RoiPoolingBinRange<dT>(bin_x1, bin_y1, bin_x2, bin_y2, is_empty, offset_rois, ph, pw, poolH, poolW,
                               spatial_scale_h, spatial_scale_w, fmH, fmW);
        const __gm__ dT* offset_input = x + (roi_batch_ind * static_cast<int64_t>(channels) + c) * fmH * fmW;
        float max_val = ROI_EMPTY_BIN_VAL;
        int64_t max_idx = INVALID_ARGMAX_OFFSET;
        if (!is_empty) {
            RoiPoolingBinMax<dT>(max_val, max_idx, offset_input, bin_x1, bin_y1, bin_x2, bin_y2, fmW);
        }
        y[idx] = static_cast<dT>(max_val);
        if (argmax != nullptr) {
            argmax[idx] = static_cast<int32_t>(max_idx);
        }
    }
}
#endif
