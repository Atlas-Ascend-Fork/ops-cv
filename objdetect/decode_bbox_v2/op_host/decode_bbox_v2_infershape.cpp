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

#include <cstdint>
#include <vector>

#include "graph/types.h"

using namespace ge;

namespace ops {

namespace {

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

} // namespace

/**
 * InferShapeForDecodeBboxV2: GE shape inference callback.
 *
 * y.shape = boxes.shape (identity).  Validates that both inputs are
 * non-null, rank == 2, last dim == 4, and the two input shapes are equal.
 * Empty tensor (N==0) is supported — the identity copy propagates (0, 4)
 * to the output.
 */
ge::graphStatus InferShapeForDecodeBboxV2(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(0);
    const gert::Shape* anchorsShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (boxesShape == nullptr || anchorsShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsUnknownRank(boxesShape) || IsUnknownRank(anchorsShape)) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    const size_t rank = boxesShape->GetDimNum();
    if (rank != 2 || anchorsShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (boxesShape->GetDim(0) != 4 && boxesShape->GetDim(1) != 4) {
        return GRAPH_FAILED;
    }
    if (anchorsShape->GetDim(0) != 4 && anchorsShape->GetDim(1) != 4) {
        return GRAPH_FAILED;
    }

    // Validate boxes.shape == anchors.shape (no broadcast, full per-dim equality)
    for (size_t i = 0; i < rank; ++i) {
        const int64_t boxesDim = boxesShape->GetDim(i);
        const int64_t anchorsDim = anchorsShape->GetDim(i);
        if (boxesDim != anchorsDim && boxesDim != ge::UNKNOWN_DIM && anchorsDim != ge::UNKNOWN_DIM) {
            return GRAPH_FAILED;
        }
    }

    // y.shape = boxes.shape (identity copy)
    yShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        yShape->SetDim(i, boxesShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DecodeBboxV2).InferShape(InferShapeForDecodeBboxV2);

} // namespace ops
