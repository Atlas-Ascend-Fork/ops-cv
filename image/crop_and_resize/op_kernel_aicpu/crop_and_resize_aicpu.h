/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_CROP_AND_RESIZE_H_
#define AICPU_KERNELS_NORMALIZED_CROP_AND_RESIZE_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
constexpr int kDimNumFour = 4;
constexpr int kDimNumTwo = 2;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;

namespace {

inline int64_t MapV2Coordinate(float coord, int64_t imgSize)
{
    int64_t pixel = static_cast<int64_t>(std::floor(coord * imgSize));
    return coord == 1.0f ? imgSize - 1 : pixel;
}

inline void CalcCropWindow(float y1, float x1, float y2, float x2, int64_t imgH, int64_t imgW, int64_t& y1o,
                           int64_t& x1o, int64_t& y2o, int64_t& x2o, int64_t& wo, int64_t& ho)
{
    y1o = MapV2Coordinate(y1, imgH);
    x1o = MapV2Coordinate(x1, imgW);
    y2o = MapV2Coordinate(y2, imgH);
    x2o = MapV2Coordinate(x2, imgW);
    const int64_t xDistance = (x2o >= x1o) ? (x2o - x1o) : (x1o - x2o);
    const int64_t yDistance = (y2o >= y1o) ? (y2o - y1o) : (y1o - y2o);
    wo = xDistance + 1;
    ho = yDistance + 1;
}

inline void GetV2Indices(int64_t pos, int64_t cropSz, int64_t winSz, int64_t& lower, int64_t& upper, float& ratio,
                         float& windowPos)
{
    windowPos = (pos + 0.5f) * (winSz / static_cast<float>(cropSz)) - 0.5f;
    lower = static_cast<int64_t>(std::floor(windowPos));
    lower = std::max(static_cast<int64_t>(0), lower);
    lower = std::min(lower, winSz - 1);
    upper = static_cast<int64_t>(std::ceil(windowPos));
    upper = std::max(static_cast<int64_t>(0), upper);
    upper = std::min(upper, winSz - 1);
    ratio = windowPos - lower;
}

inline void GetInterpolateIndices(float inPos, int64_t imgSize, int64_t& lower, int64_t& upper, float& ratio)
{
    lower = static_cast<int64_t>(std::floor(inPos));
    upper = static_cast<int64_t>(std::ceil(inPos));
    ratio = inPos - lower;
}

inline void RoundToNearest(float inY, float inX, int64_t& outY, int64_t& outX)
{
    outY = static_cast<int64_t>(std::round(inY));
    outX = static_cast<int64_t>(std::round(inX));
}

inline void CalcScaleRatio(float coord1, float coord2, int64_t cropSz, int64_t imgSz, float& ratio)
{
    ratio = (cropSz > 1) ? (coord2 - coord1) * (imgSz - 1) / (cropSz - 1) : 0;
}

inline void MapOutputToInput(float coord1, float coord2, int64_t cropSz, int64_t imgSz, int64_t outPos,
                             float scaleRatio, float& inPos)
{
    inPos = (cropSz > 1) ? coord1 * (imgSz - 1) + outPos * scaleRatio : 0.5f * (coord1 + coord2) * (imgSz - 1);
}

template <typename T>
inline void FillWithExtrapolation(typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int64_t y,
                                  int64_t cropWidth, int64_t depth, float extrapolation_value);

template <typename T>
inline void FillWithExtrapolationSingle(typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int64_t y,
                                        int64_t x, int64_t depth, float extrapolation_value);

template <typename T>
inline void ComputeBilinearV2(typename TTypes<T, kDimNumFour>::Tensor image,
                              typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int32_t b_in, int64_t y1,
                              int64_t x1, int64_t cropHeight, int64_t cropWidth, int64_t depth, int64_t w, int64_t h,
                              int64_t imageHeight, int64_t imageWidth, bool extrapolateTop, bool extrapolateLeft,
                              bool extrapolateBottom, bool extrapolateRight, float extrapolation_value, int64_t xStep,
                              int64_t yStep)
{
    for (int64_t y = 0; y < cropHeight; ++y) {
        int64_t y_base, y_top;
        float y_ratio, y_window_pos;
        GetV2Indices(y, cropHeight, h, y_base, y_top, y_ratio, y_window_pos);
        int64_t y_base_index = y1 + yStep * y_base;
        int64_t y_top_index = y1 + yStep * y_top;
        float y_image_pos = static_cast<float>(y1) + yStep * y_window_pos;
        bool y_index_out_of_bounds = y_base_index < 0 || y_top_index < 0 || y_base_index >= imageHeight ||
                                     y_top_index >= imageHeight;
        if (y_index_out_of_bounds ||
            ((y_image_pos < 0 && extrapolateTop) || (y_image_pos > imageHeight - 1 && extrapolateBottom))) {
            FillWithExtrapolation<T>(crops, b, y, cropWidth, depth, extrapolation_value);
            continue;
        }
        for (int64_t x = 0; x < cropWidth; ++x) {
            int64_t x_base, x_top;
            float x_ratio, x_window_pos;
            GetV2Indices(x, cropWidth, w, x_base, x_top, x_ratio, x_window_pos);
            int64_t x_base_index = x1 + xStep * x_base;
            int64_t x_top_index = x1 + xStep * x_top;
            float x_image_pos = static_cast<float>(x1) + xStep * x_window_pos;
            bool x_index_out_of_bounds = x_base_index < 0 || x_top_index < 0 || x_base_index >= imageWidth ||
                                         x_top_index >= imageWidth;
            if (x_index_out_of_bounds ||
                ((x_image_pos < 0 && extrapolateLeft) || (x_image_pos > imageWidth - 1 && extrapolateRight))) {
                FillWithExtrapolationSingle<T>(crops, b, y, x, depth, extrapolation_value);
                continue;
            }
            for (int64_t d = 0; d < depth; ++d) {
                float v00 = static_cast<float>(image(b_in, y_base_index, x_base_index, d));
                float v01 = static_cast<float>(image(b_in, y_base_index, x_top_index, d));
                float v10 = static_cast<float>(image(b_in, y_top_index, x_base_index, d));
                float v11 = static_cast<float>(image(b_in, y_top_index, x_top_index, d));
                float result = v00 * (1 - y_ratio) * (1 - x_ratio) + v11 * y_ratio * x_ratio +
                               v01 * (1 - y_ratio) * x_ratio + v10 * y_ratio * (1 - x_ratio);
                crops(b, y, x, d) = result;
            }
        }
    }
}

template <typename T>
inline void FillWithExtrapolation(typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int64_t y,
                                  int64_t cropWidth, int64_t depth, float extrapolation_value)
{
    for (int64_t x = 0; x < cropWidth; ++x) {
        for (int64_t d = 0; d < depth; ++d) {
            crops(b, y, x, d) = extrapolation_value;
        }
    }
}

template <typename T>
inline void FillWithExtrapolationSingle(typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int64_t y,
                                        int64_t x, int64_t depth, float extrapolation_value)
{
    for (int64_t d = 0; d < depth; ++d) {
        crops(b, y, x, d) = extrapolation_value;
    }
}

template <typename T>
inline void ProcessCropRow(typename TTypes<T, kDimNumFour>::Tensor image,
                           typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int32_t b_in,
                           int64_t cropWidth, int64_t depth, int64_t imageHeight, int64_t imageWidth,
                           float extrapolation_value, float x1, float x2, float widthScale, int64_t y, float yPos,
                           bool isBilinear)
{
    for (int64_t x = 0; x < cropWidth; ++x) {
        float xPos;
        MapOutputToInput(x1, x2, cropWidth, imageWidth, x, widthScale, xPos);
        if (xPos < 0 || xPos > imageWidth - 1) {
            FillWithExtrapolationSingle<T>(crops, b, y, x, depth, extrapolation_value);
            continue;
        }
        if (isBilinear) {
            int64_t topY = static_cast<int64_t>(std::floor(yPos));
            int64_t bottomY = static_cast<int64_t>(std::ceil(yPos));
            int64_t leftX = static_cast<int64_t>(std::floor(xPos));
            int64_t rightX = static_cast<int64_t>(std::ceil(xPos));
            float yRatio = yPos - topY;
            float xRatio = xPos - leftX;
            for (int64_t d = 0; d < depth; ++d) {
                float v00 = static_cast<float>(image(b_in, topY, leftX, d));
                float v01 = static_cast<float>(image(b_in, topY, rightX, d));
                float v10 = static_cast<float>(image(b_in, bottomY, leftX, d));
                float v11 = static_cast<float>(image(b_in, bottomY, rightX, d));
                float row0 = v00 + (v01 - v00) * xRatio;
                float row1 = v10 + (v11 - v10) * xRatio;
                crops(b, y, x, d) = row0 + (row1 - row0) * yRatio;
            }
        } else {
            int64_t rndX = static_cast<int64_t>(std::round(xPos));
            int64_t rndY = static_cast<int64_t>(std::round(yPos));
            for (int64_t d = 0; d < depth; ++d) {
                crops(b, y, x, d) = static_cast<float>(image(b_in, rndY, rndX, d));
            }
        }
    }
}

template <typename T>
inline void ComputeCropByMethod(typename TTypes<T, kDimNumFour>::Tensor image,
                                typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int32_t b_in,
                                int64_t cropHeight, int64_t cropWidth, int64_t depth, int64_t imageHeight,
                                int64_t imageWidth, float extrapolation_value, float y1, float x1, float y2, float x2,
                                float heightScale, float widthScale, bool isBilinear)
{
    for (int64_t y = 0; y < cropHeight; ++y) {
        float yPos;
        MapOutputToInput(y1, y2, cropHeight, imageHeight, y, heightScale, yPos);
        if (yPos < 0 || yPos > imageHeight - 1) {
            FillWithExtrapolation<T>(crops, b, y, cropWidth, depth, extrapolation_value);
            continue;
        }
        ProcessCropRow<T>(image, crops, b, b_in, cropWidth, depth, imageHeight, imageWidth, extrapolation_value, x1, x2,
                          widthScale, y, yPos, isBilinear);
    }
}

template <typename T>
inline void ComputeBilinear(typename TTypes<T, kDimNumFour>::Tensor image,
                            typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int32_t b_in,
                            int64_t cropHeight, int64_t cropWidth, int64_t depth, int64_t imageHeight,
                            int64_t imageWidth, float extrapolation_value, float y1, float x1, float y2, float x2,
                            float heightScale, float widthScale)
{
    ComputeCropByMethod<T>(image, crops, b, b_in, cropHeight, cropWidth, depth, imageHeight, imageWidth,
                           extrapolation_value, y1, x1, y2, x2, heightScale, widthScale, true);
}

template <typename T>
inline void ComputeNearest(typename TTypes<T, kDimNumFour>::Tensor image,
                           typename TTypes<float, kDimNumFour>::Tensor crops, int64_t b, int32_t b_in,
                           int64_t cropHeight, int64_t cropWidth, int64_t depth, int64_t imageHeight,
                           int64_t imageWidth, float extrapolation_value, float y1, float x1, float y2, float x2,
                           float heightScale, float widthScale)
{
    ComputeCropByMethod<T>(image, crops, b, b_in, cropHeight, cropWidth, depth, imageHeight, imageWidth,
                           extrapolation_value, y1, x1, y2, x2, heightScale, widthScale, false);
}

} // namespace

class CropAndResizeMsCpuKernel : public CpuKernel {
public:
    ~CropAndResizeMsCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    uint32_t GetMethodAndAttr(const CpuKernelContext& ctx);
    uint32_t GetInputIndexX(const CpuKernelContext& ctx);
    uint32_t GetInputBox(const CpuKernelContext& ctx);
    uint32_t GetInputCropSize(const CpuKernelContext& ctx);
    uint32_t GetInputAndCheck(CpuKernelContext& ctx);
    std::vector<Tensor*> inputs_;
    std::vector<Tensor*> outputs_;

    std::string method_;
    float extrapolation_value_ = 0;

    std::vector<int64_t> x_shape_;
    std::vector<int64_t> crop_size_shape_;
    std::vector<int64_t> boxes_shape_;
    std::vector<int64_t> box_index_shape_;
    DataType x_dtype_ = DT_INT32;

    template <typename T>
    struct CropAndResize {
        bool operator()(typename TTypes<T, kDimNumFour>::Tensor image, typename TTypes<float, kDimNumTwo>::Tensor boxes,
                        typename TTypes<int32_t, 1>::Tensor box_index, const std::string& method_name,
                        float extrapolation_value, typename TTypes<float, kDimNumFour>::Tensor crops,
                        const CpuKernelContext& ctx)
        {
            const int64_t imageHeight = image.dimension(1);
            const int64_t imageWidth = image.dimension(2);
            const int64_t numBoxes = crops.dimension(0);
            const int64_t cropHeight = crops.dimension(1);
            const int64_t cropWidth = crops.dimension(2);
            const int64_t depth = crops.dimension(3);

            auto CropAndResizePerBox = [&](int64_t start_box, int64_t limit_box) {
                for (int64_t b = start_box; b < limit_box; ++b) {
                    if (method_name == "bilinear_v2") {
                        int64_t y1, x1, y2, x2, w, h;
                        CalcCropWindow(boxes(b, 0), boxes(b, 1), boxes(b, 2), boxes(b, 3), imageHeight, imageWidth, y1,
                                       x1, y2, x2, w, h);
                        const int64_t xStep = (x2 >= x1) ? 1 : -1;
                        const int64_t yStep = (y2 >= y1) ? 1 : -1;
                        bool extrapolateTop = std::min(boxes(b, 0), boxes(b, 2)) < 0;
                        bool extrapolateBottom = std::max(boxes(b, 0), boxes(b, 2)) > 1;
                        bool extrapolateLeft = std::min(boxes(b, 1), boxes(b, 3)) < 0;
                        bool extrapolateRight = std::max(boxes(b, 1), boxes(b, 3)) > 1;
                        ComputeBilinearV2<T>(image, crops, b, box_index(b), y1, x1, cropHeight, cropWidth, depth, w, h,
                                             imageHeight, imageWidth, extrapolateTop, extrapolateLeft,
                                             extrapolateBottom, extrapolateRight, extrapolation_value, xStep, yStep);
                    } else {
                        const float y1 = boxes(b, 0);
                        const float x1 = boxes(b, 1);
                        const float y2 = boxes(b, 2);
                        const float x2 = boxes(b, 3);
                        float heightScale, widthScale;
                        CalcScaleRatio(y1, y2, cropHeight, imageHeight, heightScale);
                        CalcScaleRatio(x1, x2, cropWidth, imageWidth, widthScale);
                        if (method_name == "bilinear") {
                            ComputeBilinear<T>(image, crops, b, box_index(b), cropHeight, cropWidth, depth, imageHeight,
                                               imageWidth, extrapolation_value, y1, x1, y2, x2, heightScale,
                                               widthScale);
                        } else {
                            ComputeNearest<T>(image, crops, b, box_index(b), cropHeight, cropWidth, depth, imageHeight,
                                              imageWidth, extrapolation_value, y1, x1, y2, x2, heightScale, widthScale);
                        }
                    }
                }
            };
            if (numBoxes != 0) {
                KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, numBoxes, 1, CropAndResizePerBox),
                                    "CropAndResize Compute failed.");
            }
            return true;
        }
    };

    template <typename T>
    static uint32_t CalCropAndResize(std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs,
                                     const std::vector<int64_t>& x_shape, const std::vector<int64_t>& boxes_shape,
                                     const std::string& method, float extrapolation_value, const CpuKernelContext& ctx)
    {
        EigenTensor image(inputs[0], inputs[0]->GetData());
        EigenTensor boxes(inputs[1], inputs[1]->GetData());
        EigenTensor box_index(inputs[kInputIndex2], inputs[kInputIndex2]->GetData());
        EigenTensor crop_size(inputs[kInputIndex3], inputs[kInputIndex3]->GetData());

        auto batch_size = x_shape[0];
        auto numBoxes = boxes_shape[0];
        auto crop_size_vec = crop_size.vec<int32_t>();
        int64_t cropHeight = crop_size_vec(0);
        int64_t cropWidth = crop_size_vec(1);
        if (!(cropHeight > 0 && cropWidth > 0)) {
            KERNEL_LOG_ERROR("The value of cropHeight: [%ld] and cropWidth: [%ld] should > 0", cropHeight, cropWidth);
            return KERNEL_STATUS_PARAM_INVALID;
        }

        const int kOutputY = 0;
        EigenTensor output(outputs[kOutputY], outputs[kOutputY]->GetData());

        typename TTypes<int32_t, 1>::Tensor box_index_t = box_index.tensor<int32_t, 1>();
        for (int64_t b = 0; b < numBoxes; ++b) {
            if (!(box_index_t(b) >= 0 && box_index_t(b) < static_cast<int>(batch_size))) {
                KERNEL_LOG_ERROR("Invalid box_index[%ld] value: [%d], should be in [0, %ld)!", b, box_index_t(b),
                                 batch_size);
                return KERNEL_STATUS_PARAM_INVALID;
            }
        }

        if (CropAndResize<T>()(image.tensor<T, kDimNumFour>(), boxes.tensor<float, kDimNumTwo>(),
                               box_index.tensor<int32_t, 1>(), method, extrapolation_value,
                               output.tensor<float, kDimNumFour>(), ctx)) {
            return KERNEL_STATUS_OK;
        }

        return KERNEL_STATUS_PARAM_INVALID;
    }
};
} // namespace aicpu

#endif // AICPU_CROP_AND_RESIZE_H_
