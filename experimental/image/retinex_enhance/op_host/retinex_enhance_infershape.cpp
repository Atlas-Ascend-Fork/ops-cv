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
 * \file retinex_enhance_infershape.cpp
 * \brief Shape and data type inference for RetinexEnhance.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_IMAGE = 0;
constexpr size_t OUTPUT_ENHANCED = 0;
constexpr size_t IMAGE_RANK = 3;
constexpr int64_t GRAYSCALE_CHANNELS = 1;
constexpr int64_t RGB_CHANNELS = 3;
constexpr int64_t RGBA_CHANNELS = 4;

bool IsSupportedChannels(int64_t channels)
{
    return channels == GRAYSCALE_CHANNELS || channels == RGB_CHANNELS || channels == RGBA_CHANNELS;
}
} // namespace

static ge::graphStatus InferShapeRetinexEnhance(gert::InferShapeContext* context)
{
    const gert::Shape* imageShape = context->GetInputShape(INPUT_IMAGE);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageShape);
    gert::Shape* enhancedShape = context->GetOutputShape(OUTPUT_ENHANCED);
    OP_CHECK_NULL_WITH_CONTEXT(context, enhancedShape);

    const size_t dimNum = imageShape->GetDimNum();
    OP_CHECK_IF(dimNum != IMAGE_RANK,
                OP_LOGE(context->GetNodeName(), "RetinexEnhance expects rank 3 HWC input, got rank %zu", dimNum),
                return ge::GRAPH_FAILED);
    const int64_t height = imageShape->GetDim(0);
    const int64_t width = imageShape->GetDim(1);
    OP_CHECK_IF((height != ge::UNKNOWN_DIM && height <= 0) || (width != ge::UNKNOWN_DIM && width <= 0),
                OP_LOGE(context->GetNodeName(), "RetinexEnhance requires positive or dynamic H and W"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(imageShape->GetDim(2) != ge::UNKNOWN_DIM && !IsSupportedChannels(imageShape->GetDim(2)),
                OP_LOGE(context->GetNodeName(), "RetinexEnhance requires the last dimension to be 1, 3, or 4"),
                return ge::GRAPH_FAILED);

    *enhancedShape = *imageShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRetinexEnhance(gert::InferDataTypeContext* context)
{
    const auto imageDtype = context->GetInputDataType(INPUT_IMAGE);
    OP_CHECK_IF(imageDtype != ge::DT_FLOAT, OP_LOGE(context->GetNodeName(), "RetinexEnhance only supports FLOAT32"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_ENHANCED, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RetinexEnhance).InferShape(InferShapeRetinexEnhance).InferDataType(InferDataTypeRetinexEnhance);
} // namespace ops
