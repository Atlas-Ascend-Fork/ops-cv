/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace {
constexpr int64_t INPUT_BOXES = 0;
constexpr int64_t OUTPUT_SELECTED_INDICES = 0;
constexpr int64_t OUTPUT_RANK = 1;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForSortedNMS(gert::InferShapeContext* context)
{
    auto boxesShape = context->GetInputShape(INPUT_BOXES);
    auto selectedShape = context->GetOutputShape(OUTPUT_SELECTED_INDICES);
    if (boxesShape == nullptr || selectedShape == nullptr) {
        OP_LOGE(context, "boxes shape or selected_indices shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    // Shape inference does not require a known input rank. Concrete input
    // constraints belong to tiling, as in the legacy runtime callback.
    selectedShape->SetDimNum(OUTPUT_RANK);
    selectedShape->SetDim(0, ge::UNKNOWN_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForSortedNMS(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SortedNMS", "InferShapeRangeContext is nullptr"), return ge::GRAPH_FAILED);

    auto boxesRange = context->GetInputShapeRange(INPUT_BOXES);
    auto selectedRange = context->GetOutputShapeRange(OUTPUT_SELECTED_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedRange->GetMax());

    OP_CHECK_IF(boxesRange->GetMax()->GetDimNum() == 0,
                OP_LOGE(context, "boxes shape range must contain a first dimension"), return ge::GRAPH_FAILED);
    const int64_t maxBoxesNum = boxesRange->GetMax()->GetDim(0);

    selectedRange->GetMin()->SetDimNum(OUTPUT_RANK);
    selectedRange->GetMin()->SetDim(0, 0);
    selectedRange->GetMax()->SetDimNum(OUTPUT_RANK);
    selectedRange->GetMax()->SetDim(0, maxBoxesNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSortedNMS(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_SELECTED_INDICES, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SortedNMS)
    .InferShape(InferShapeForSortedNMS)
    .InferShapeRange(InferShapeRangeForSortedNMS)
    .InferDataType(InferDataTypeForSortedNMS)
    .OutputShapeDependOnCompute({OUTPUT_SELECTED_INDICES});
} // namespace ops
