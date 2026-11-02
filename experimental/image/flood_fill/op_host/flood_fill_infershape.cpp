/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill_infershape.cpp
 * \brief FloodFill shape inference.
 */
#include <cstdint>

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
int64_t AddMaskBorder(int64_t dim)
{
    // Unknown dimensions are represented by negative values and must remain
    // unknown. Only a known image dimension can be expanded by the mask border.
    return dim < 0 ? dim : dim + 2;
}
} // namespace

static ge::graphStatus InferShapeFloodFill(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;

    // Optional mask_out: [H+2, W+2] derived from the rank-2 input.
    gert::Shape* maskShape = context->GetOutputShape(1);
    if (maskShape != nullptr) {
        OP_CHECK_IF(
            xShape->GetDimNum() != 2,
            OP_LOGE(context, "[FloodFill] mask_out requires rank-2 input, but got rank %zu", xShape->GetDimNum()),
            return ge::GRAPH_FAILED);

        gert::Shape maskDim({AddMaskBorder(xShape->GetDim(0)), AddMaskBorder(xShape->GetDim(1))});
        *maskShape = maskDim;
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFloodFill(gert::InferDataTypeContext* context)
{
    const auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    context->SetOutputDataType(1, ge::DT_UINT8);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FloodFill).InferShape(InferShapeFloodFill).InferDataType(InferDataTypeFloodFill);
} // namespace ops
