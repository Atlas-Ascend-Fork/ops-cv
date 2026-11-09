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
 * \file rotated_box_encode_infershape.cpp
 * \brief RotatedBoxEncode shape inference: y.shape == anchor_box.shape
 */

// op_impl_registry.h: provides IMPL_OP_INFERSHAPE macro, gert::InferShapeContext.
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "op_common/log/log.h"

#include <cstdint>

#include "graph/types.h"

using namespace ge;

namespace ops {

// InferShapeForRotatedBoxEncode: GE shape inference callback.
//   Per proto.md §4: both inputs and the output are identical 3-D (B, 5, N)
//   tensors; broadcasting is not supported.
static bool IsUnknownRank(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

static bool IsUnknownDim(int64_t dim)
{
    // -1 (ge::UNKNOWN_DIM) is the only valid dynamic-dimension marker;
    // other negative placeholders must not be propagated.
    return dim == ge::UNKNOWN_DIM;
}

// ValidateBoxShapeContract: rank must be 3 and dim 1 must be 5 (or unknown).
// Shared by the fully-known path and the one-side-unknown-rank path so the
// (B, 5, N) contract is never silently skipped.
static ge::graphStatus ValidateBoxShapeContract(gert::InferShapeContext* context, const gert::Shape& shape,
                                                const char* name)
{
    if (shape.GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "%s rank must be 3 (B, 5, N), but got %zu.", name, shape.GetDimNum());
        return GRAPH_FAILED;
    }
    if (!IsUnknownDim(shape.GetDim(1)) && shape.GetDim(1) != 5) {
        OP_LOGE(context->GetNodeName(), "%s dim 1 must be 5, but got %ld.", name, shape.GetDim(1));
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForRotatedBoxEncode(gert::InferShapeContext* context)
{
    const gert::Shape* anchorShape = context->GetInputShape(0);
    const gert::Shape* gtShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (anchorShape == nullptr || gtShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    // weight is optional; when present its length is part of the public contract.
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto* weight = attrs->GetAttrPointer<gert::TypedContinuousVector<float>>(0);
        if (weight != nullptr && weight->GetSize() != static_cast<size_t>(5)) {
            OP_LOGE(context->GetNodeName(), "weight length must be 5, but got %zu.", weight->GetSize());
            return GRAPH_FAILED;
        }
    }

    // Unknown rank must remain propagate-able: both dynamic-rank inputs yield a
    // dynamic-rank output.  When only one side is unknown-rank, the other side
    // is still statically known and must satisfy the (B, 5, N) contract;
    // rejecting it here avoids deferring an impossible shape to runtime.
    const bool anchorUnknownRank = IsUnknownRank(*anchorShape);
    const bool gtUnknownRank = IsUnknownRank(*gtShape);
    if (anchorUnknownRank || gtUnknownRank) {
        if (!anchorUnknownRank && ValidateBoxShapeContract(context, *anchorShape, "anchor_box") != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (!gtUnknownRank && ValidateBoxShapeContract(context, *gtShape, "gt_box") != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        yShape->SetDimNum(1);
        yShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    if (ValidateBoxShapeContract(context, *anchorShape, "anchor_box") != GRAPH_SUCCESS ||
        ValidateBoxShapeContract(context, *gtShape, "gt_box") != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    yShape->SetDimNum(3);
    for (size_t i = 0; i < 3; ++i) {
        const int64_t anchorDim = anchorShape->GetDim(i);
        const int64_t gtDim = gtShape->GetDim(i);
        if (!IsUnknownDim(anchorDim) && !IsUnknownDim(gtDim) && anchorDim != gtDim) {
            OP_LOGE(context->GetNodeName(), "anchor_box and gt_box must have identical shapes, dim[%zu]: %ld vs %ld.",
                    i, anchorDim, gtDim);
            return GRAPH_FAILED;
        }
        yShape->SetDim(i, IsUnknownDim(anchorDim) ? gtDim : anchorDim);
    }
    return GRAPH_SUCCESS;
}

// [REF_SAMPLE] — original AddCustom sample InferShape logic (rank matching,
// UNKNOWN_DIM / UNKNOWN_RANK resolution).  Preserved verbatim for reference; NOT compiled.
//
// namespace {
// bool IsUnknownRank(const gert::Shape* shape)
// {
//     return shape != nullptr && shape->GetDimNum() == 1 &&
//            shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
// }
// } // namespace
//
// static ge::graphStatus InferShapeForRotatedBoxEncode(gert::InferShapeContext* context)
// {
//     const gert::Shape* xShape = context->GetInputShape(0);
//     const gert::Shape* yShape = context->GetInputShape(1);
//     gert::Shape* zShape = context->GetOutputShape(0);
//     if (xShape == nullptr || yShape == nullptr || zShape == nullptr) {
//         return GRAPH_FAILED;
//     }
//     if (IsUnknownRank(xShape) || IsUnknownRank(yShape)) {
//         zShape->SetDimNum(1);
//         zShape->SetDim(0, ge::UNKNOWN_DIM_NUM);
//         return GRAPH_SUCCESS;
//     }
//     const size_t rank = xShape->GetDimNum();
//     if (rank != yShape->GetDimNum()) {
//         return GRAPH_FAILED;
//     }
//     std::vector<int64_t> outputDims(rank);
//     for (size_t i = 0; i < rank; ++i) {
//         const int64_t xDim = xShape->GetDim(i);
//         const int64_t yDim = yShape->GetDim(i);
//         if (xDim == yDim) {
//             outputDims[i] = xDim;
//         } else if (xDim == ge::UNKNOWN_DIM) {
//             outputDims[i] = yDim;
//         } else if (yDim == ge::UNKNOWN_DIM) {
//             outputDims[i] = xDim;
//         } else {
//             return GRAPH_FAILED;
//         }
//     }
//     zShape->SetDimNum(rank);
//     for (size_t i = 0; i < rank; ++i) {
//         zShape->SetDim(i, outputDims[i]);
//     }
//     return GRAPH_SUCCESS;
// }

IMPL_OP_INFERSHAPE(RotatedBoxEncode).InferShape(InferShapeForRotatedBoxEncode);

} // namespace ops
