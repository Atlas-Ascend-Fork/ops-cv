/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file blend_images_custom.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

static const size_t FRAME_INDEX = 2;
static const size_t OUT_INDEX = 0;
static const size_t RGB_INDEX = 0;
static const size_t ALPHA_INDEX = 1;
static const size_t H_INDEX = 0;
static const size_t W_INDEX = 1;
static const size_t CHANNEL_INDEX = 2;
static const int64_t RGB_CHANNEL_NUM = 3;
static const int64_t ALPHA_CHANNEL_NUM = 1;

using namespace ge;
namespace ge {
namespace {
bool IsHwcShape(const gert::Shape& shape, int64_t channels)
{
    return shape.GetDimNum() == 3U &&
           (shape.GetDim(CHANNEL_INDEX) == ge::UNKNOWN_DIM || shape.GetDim(CHANNEL_INDEX) == channels);
}

bool IsSameKnownDim(const gert::Shape& lhs, const gert::Shape& rhs, size_t index)
{
    return lhs.GetDim(index) == ge::UNKNOWN_DIM || rhs.GetDim(index) == ge::UNKNOWN_DIM ||
           lhs.GetDim(index) == rhs.GetDim(index);
}

bool IsBlendImagesCustomShapeValid(const gert::Shape& rgb_shape, const gert::Shape& alpha_shape,
                                   const gert::Shape& frame_shape)
{
    if (Ops::Base::IsUnknownRank(rgb_shape) || Ops::Base::IsUnknownRank(alpha_shape) ||
        Ops::Base::IsUnknownRank(frame_shape)) {
        return true;
    }
    if (!IsHwcShape(rgb_shape, RGB_CHANNEL_NUM) || !IsHwcShape(alpha_shape, ALPHA_CHANNEL_NUM) ||
        !IsHwcShape(frame_shape, RGB_CHANNEL_NUM)) {
        return false;
    }
    for (size_t index = H_INDEX; index <= W_INDEX; ++index) {
        if ((rgb_shape.GetDim(index) < 0 && rgb_shape.GetDim(index) != ge::UNKNOWN_DIM) ||
            (alpha_shape.GetDim(index) < 0 && alpha_shape.GetDim(index) != ge::UNKNOWN_DIM) ||
            (frame_shape.GetDim(index) < 0 && frame_shape.GetDim(index) != ge::UNKNOWN_DIM)) {
            return false;
        }
        if (!IsSameKnownDim(rgb_shape, alpha_shape, index) || !IsSameKnownDim(rgb_shape, frame_shape, index)) {
            return false;
        }
    }
    return true;
}
} // namespace

static ge::graphStatus InferShape4BlendImagesCustom(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // infer shape
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4BlendImagesCustom.");
    const gert::Shape* rgb_shape = context->GetInputShape(RGB_INDEX);
    const gert::Shape* alpha_shape = context->GetInputShape(ALPHA_INDEX);
    const gert::Shape* frame_shape = context->GetInputShape(FRAME_INDEX);
    gert::Shape* out_shape = context->GetOutputShape(OUT_INDEX);
    if (rgb_shape == nullptr || alpha_shape == nullptr || frame_shape == nullptr || out_shape == nullptr ||
        !IsBlendImagesCustomShapeValid(*rgb_shape, *alpha_shape, *frame_shape)) {
        return ge::GRAPH_FAILED;
    }
    *out_shape = *frame_shape;
    OP_LOGD(context->GetNodeName(), "End to do InferShape4BlendImagesCustom.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlendImagesCustom).InferShape(InferShape4BlendImagesCustom);
} // namespace ge
