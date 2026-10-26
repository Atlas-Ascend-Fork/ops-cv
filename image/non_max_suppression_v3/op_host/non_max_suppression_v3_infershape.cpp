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
 * \file non_max_suppression_v3_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kBoxesIndex = 0U;
constexpr size_t kScoresIndex = 1U;
constexpr size_t kMaxOutputSizeIndex = 2U;
constexpr size_t kIouThresholdIndex = 3U;
constexpr size_t kScoreThresholdIndex = 4U;
constexpr size_t kSelectedIndicesIndex = 0U;
constexpr size_t kBoxesRank = 2U;
constexpr size_t kScoresRank = 1U;
constexpr size_t kOutputRank = 1U;
constexpr size_t kBoxesCoordinateDim = 1U;
constexpr int64_t kCoordinateNum = 4;

bool IsRankInvalid(const gert::Shape& shape, size_t expectedRank)
{
    return !Ops::Base::IsUnknownRank(shape) && shape.GetDimNum() != expectedRank;
}

bool IsScalarOrSingleElement(const gert::Shape& shape)
{
    if (Ops::Base::IsUnknownRank(shape) || shape.IsScalar()) {
        return true;
    }
    return shape.GetDimNum() == 1U && (shape.GetDim(0U) == 1 || shape.GetDim(0U) == ge::UNKNOWN_DIM);
}

int64_t GetDimOrUnknown(const gert::Shape& shape, size_t index)
{
    return Ops::Base::IsUnknownRank(shape) ? ge::UNKNOWN_DIM : shape.GetDim(index);
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeForNonMaxSuppressionV3(gert::InferShapeContext* context)
{
    const gert::Shape* boxesShape = context->GetInputShape(kBoxesIndex);
    const gert::Shape* scoresShape = context->GetInputShape(kScoresIndex);
    const gert::Shape* maxOutputSizeShape = context->GetInputShape(kMaxOutputSizeIndex);
    const gert::Shape* iouThresholdShape = context->GetInputShape(kIouThresholdIndex);
    const gert::Shape* scoreThresholdShape = context->GetInputShape(kScoreThresholdIndex);
    gert::Shape* selectedIndicesShape = context->GetOutputShape(kSelectedIndicesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxOutputSizeShape);
    const auto* boxesDesc = context->GetInputDesc(kBoxesIndex);
    const auto* scoresDesc = context->GetInputDesc(kScoresIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresDesc);
    OP_CHECK_IF(boxesDesc->GetDataType() != scoresDesc->GetDataType(),
                OP_LOGE(context, "boxes and scores must have the same data type"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, iouThresholdShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreThresholdShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedIndicesShape);

    OP_CHECK_IF(IsRankInvalid(*boxesShape, kBoxesRank),
                OP_LOGE(context, "boxes must be rank 2, but got %zu", boxesShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsRankInvalid(*scoresShape, kScoresRank),
                OP_LOGE(context, "scores must be rank 1, but got %zu", scoresShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScalarOrSingleElement(*maxOutputSizeShape),
                OP_LOGE(context, "max_output_size must be a scalar or a single-element tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScalarOrSingleElement(*iouThresholdShape),
                OP_LOGE(context, "iou_threshold must be a scalar or a single-element tensor"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsScalarOrSingleElement(*scoreThresholdShape),
                OP_LOGE(context, "score_threshold must be a scalar or a single-element tensor"),
                return ge::GRAPH_FAILED);

    const int64_t boxesNum = GetDimOrUnknown(*boxesShape, 0U);
    const int64_t scoresNum = GetDimOrUnknown(*scoresShape, 0U);
    OP_CHECK_IF(
        boxesNum != ge::UNKNOWN_DIM && scoresNum != ge::UNKNOWN_DIM && boxesNum != scoresNum,
        OP_LOGE(context, "boxes and scores first dimensions are incompatible: %ld and %ld", boxesNum, scoresNum),
        return ge::GRAPH_FAILED);

    const int64_t coordinateNum = GetDimOrUnknown(*boxesShape, kBoxesCoordinateDim);
    OP_CHECK_IF(coordinateNum != ge::UNKNOWN_DIM && coordinateNum != kCoordinateNum,
                OP_LOGE(context, "boxes second dimension must be 4, but got %ld", coordinateNum),
                return ge::GRAPH_FAILED);

    selectedIndicesShape->SetDimNum(kOutputRank);
    selectedIndicesShape->SetDim(0U, ge::UNKNOWN_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForNonMaxSuppressionV3(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("NonMaxSuppressionV3", "InferShapeRangeContext is nullptr"),
                return ge::GRAPH_FAILED);
    auto* selectedIndicesRange = context->GetOutputShapeRange(kSelectedIndicesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedIndicesRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedIndicesRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedIndicesRange->GetMax());

    selectedIndicesRange->GetMin()->SetDimNum(kOutputRank);
    selectedIndicesRange->GetMin()->SetDim(0U, 0);
    selectedIndicesRange->GetMax()->SetDimNum(kOutputRank);
    selectedIndicesRange->GetMax()->SetDim(0U, ge::UNKNOWN_DIM);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NonMaxSuppressionV3)
    .InferShape(InferShapeForNonMaxSuppressionV3)
    .InferShapeRange(InferShapeRangeForNonMaxSuppressionV3)
    .OutputShapeDependOnCompute({kSelectedIndicesIndex});
} // namespace ops
