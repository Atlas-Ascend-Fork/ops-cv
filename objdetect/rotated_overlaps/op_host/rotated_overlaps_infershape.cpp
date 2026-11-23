/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include <string>

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

namespace {
constexpr int64_t kBoxesIndex = 0;
constexpr int64_t kQueryBoxesIndex = 1;
constexpr int64_t kOutputIndex = 0;
constexpr int64_t kCoordinateCount = 5;
constexpr int64_t kMaxQueries = 2000;

bool IsKnown(int64_t dim) { return dim != ge::UNKNOWN_DIM; }

bool IsKnownPositive(int64_t dim) { return !IsKnown(dim) || dim > 0; }

bool ProductFitsInt64(int64_t a, int64_t b, int64_t c)
{
    if (!IsKnown(a) || !IsKnown(b) || !IsKnown(c)) {
        return true;
    }
    if (a <= 0 || b <= 0 || c <= 0) {
        return false;
    }
    const uint64_t ua = static_cast<uint64_t>(a);
    const uint64_t ub = static_cast<uint64_t>(b);
    const uint64_t uc = static_cast<uint64_t>(c);
    return ua <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / ub &&
           ua * ub <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / uc;
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeForRotatedOverlaps(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(kBoxesIndex);
    const gert::Shape* queryBoxesShape = context->GetInputShape(kQueryBoxesIndex);
    gert::Shape* outputShape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryBoxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    if (Ops::Base::IsUnknownRank(*boxesShape) || Ops::Base::IsUnknownRank(*queryBoxesShape)) {
        Ops::Base::SetUnknownRank(*outputShape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        boxesShape->GetDimNum() != 3 || queryBoxesShape->GetDimNum() != 3,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "boxes and query_boxes",
            (std::to_string(boxesShape->GetDimNum()) + " and " + std::to_string(queryBoxesShape->GetDimNum())).c_str(),
            "boxes and query_boxes must both be rank 3"),
        return ge::GRAPH_FAILED);

    const int64_t batch = boxesShape->GetDim(0);
    const int64_t queryBatch = queryBoxesShape->GetDim(0);
    const int64_t numBoxes = boxesShape->GetDim(2);
    const int64_t numQueries = queryBoxesShape->GetDim(2);
    OP_CHECK_IF((IsKnown(boxesShape->GetDim(1)) && boxesShape->GetDim(1) != kCoordinateCount) ||
                    (IsKnown(queryBoxesShape->GetDim(1)) && queryBoxesShape->GetDim(1) != kCoordinateCount),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(*boxesShape) + " and " + Ops::Base::ToString(*queryBoxesShape)).c_str(),
                    "the channel dimension of boxes and query_boxes must be 5"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsKnownPositive(batch) || !IsKnownPositive(queryBatch) || !IsKnownPositive(numBoxes) ||
                    !IsKnownPositive(numQueries),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(*boxesShape) + " and " + Ops::Base::ToString(*queryBoxesShape)).c_str(),
                    "the known dimensions B, N and K must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsKnown(numQueries) && numQueries > kMaxQueries,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "query_boxes", std::to_string(numQueries).c_str(),
                    ("RotatedOverlaps supports at most " + std::to_string(kMaxQueries) + " query boxes").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        IsKnown(batch) && IsKnown(queryBatch) && batch != queryBatch,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "boxes and query_boxes",
                                               (std::to_string(batch) + " and " + std::to_string(queryBatch)).c_str(),
                                               "the batch dimensions of boxes and query_boxes must match"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ProductFitsInt64(batch, numBoxes, numQueries),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(*boxesShape) + " and " + Ops::Base::ToString(*queryBoxesShape)).c_str(),
                    "the output element count B * N * K overflows int64"),
                return ge::GRAPH_FAILED);

    outputShape->SetDimNum(3);
    outputShape->SetDim(0, batch);
    outputShape->SetDim(1, numBoxes);
    outputShape->SetDim(2, numQueries);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRotatedOverlaps(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutputIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotatedOverlaps)
    .InferShape(InferShapeForRotatedOverlaps)
    .InferDataType(InferDataTypeForRotatedOverlaps);
} // namespace ops
