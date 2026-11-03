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
 * \file grid_unnormal_infershape.cpp
 * \brief GridUnnormal 形状/类型推导：
 *   diff/position 的 shape 与 grid 完全一致；
 *   diff dtype = grid dtype，position dtype = int32。
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "op_common/log/log.h"

#include <string>

using namespace ge;

namespace {
bool IsSupportedGridDtype(ge::DataType dtype) { return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT; }
} // namespace

namespace ops {

static constexpr size_t kInputGridIdx = 0;
static constexpr size_t kInputAssistIdx = 1;
static constexpr size_t kOutputDiffIdx = 0;
static constexpr size_t kOutputPosIdx = 1;
static constexpr size_t kGridRank = 4;
static constexpr size_t kLastDimIdx = 3;
static constexpr int64_t kCoordDim = 2;
static constexpr int64_t kUnknownRankDim = -2;

static bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == kUnknownRankDim;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext* context, const gert::Shape* gridShape,
                                       const gert::Shape* assistShape)
{
    if (IsUnknownRank(gridShape)) {
        if (IsUnknownRank(assistShape)) {
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(assistShape->GetDimNum() != kGridRank,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "assist",
                        ("assist rank=" + std::to_string(assistShape->GetDimNum()) + ", grid rank=unknown"),
                        "assist rank must be 4 or unknown when grid rank is unknown"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        gridShape->GetDimNum() != kGridRank,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid", "rank is not 4", "grid rank must be 4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape->GetDim(kLastDimIdx) != ge::UNKNOWN_DIM && gridShape->GetDim(kLastDimIdx) != kCoordDim,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid", "last dim is not 2",
                                                       "grid last dim must be 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(assistShape->GetDimNum() != gridShape->GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "assist",
                                                       ("assist rank=" + std::to_string(assistShape->GetDimNum()) +
                                                        ", grid rank=" + std::to_string(gridShape->GetDimNum())),
                                                       "grid and assist rank must be equal"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GridUnnormal(gert::InferShapeContext* context)
{
    const gert::Shape* gridShape = context->GetInputShape(kInputGridIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridShape);
    const gert::Shape* assistShape = context->GetInputShape(kInputAssistIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, assistShape);
    if (CheckInputShape(context, gridShape, assistShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* diffShape = context->GetOutputShape(kOutputDiffIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, diffShape);
    gert::Shape* posShape = context->GetOutputShape(kOutputPosIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, posShape);

    *diffShape = *gridShape;
    *posShape = *gridShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4GridUnnormal(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_WITHOUT_REPORT("GridUnnormal", "InferShapeRangeContext is nullptr"),
                return ge::GRAPH_FAILED);
    auto gridRange = context->GetInputShapeRange(kInputGridIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, gridRange->GetMax());
    auto assistRange = context->GetInputShapeRange(kInputAssistIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, assistRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, assistRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, assistRange->GetMax());

    const auto checkRangeRank = [context](const gert::Shape* shape, const char* inputName) -> ge::graphStatus {
        if (IsUnknownRank(shape)) {
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(shape->GetDimNum() != kGridRank,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), inputName, "range rank is not 4",
                                                           "shape range rank must be 4 or unknown"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };
    OP_CHECK_IF(checkRangeRank(gridRange->GetMin(), "grid") != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkRangeRank(gridRange->GetMax(), "grid") != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkRangeRank(assistRange->GetMin(), "assist") != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkRangeRank(assistRange->GetMax(), "assist") != ge::GRAPH_SUCCESS, (void)0, return ge::GRAPH_FAILED);
    if (!IsUnknownRank(gridRange->GetMin())) {
        const int64_t minLastDim = gridRange->GetMin()->GetDim(kLastDimIdx);
        OP_CHECK_IF(minLastDim != ge::UNKNOWN_DIM && minLastDim != kCoordDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid",
                                                           ("range min last dim=" + std::to_string(minLastDim)),
                                                           "grid shape range last dim must be 2"),
                    return ge::GRAPH_FAILED);
    }
    if (!IsUnknownRank(gridRange->GetMax())) {
        const int64_t maxLastDim = gridRange->GetMax()->GetDim(kLastDimIdx);
        OP_CHECK_IF(maxLastDim != ge::UNKNOWN_DIM && maxLastDim != kCoordDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid",
                                                           ("range max last dim=" + std::to_string(maxLastDim)),
                                                           "grid shape range last dim must be 2"),
                    return ge::GRAPH_FAILED);
    }

    auto diffRange = context->GetOutputShapeRange(kOutputDiffIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, diffRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, diffRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, diffRange->GetMax());
    auto posRange = context->GetOutputShapeRange(kOutputPosIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, posRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, posRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, posRange->GetMax());

    *diffRange->GetMin() = *gridRange->GetMin();
    *diffRange->GetMax() = *gridRange->GetMax();
    *posRange->GetMin() = *gridRange->GetMin();
    *posRange->GetMax() = *gridRange->GetMax();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4GridUnnormal(gert::InferDataTypeContext* context)
{
    const ge::DataType gridDtype = context->GetInputDataType(kInputGridIdx);
    const ge::DataType assistDtype = context->GetInputDataType(kInputAssistIdx);
    OP_CHECK_IF(gridDtype != assistDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "grid and assist",
                    ("grid ge::DataType=" + std::to_string(static_cast<int32_t>(gridDtype)) +
                     ", assist ge::DataType=" + std::to_string(static_cast<int32_t>(assistDtype))),
                    "input dtypes must be equal"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !IsSupportedGridDtype(gridDtype),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "grid",
                                  std::to_string(static_cast<int32_t>(gridDtype)).c_str(), "DT_FLOAT16 or DT_FLOAT"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(kOutputDiffIdx, gridDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "diff", std::to_string(gridDtype).c_str(),
                                                      "SetOutputDataType failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(kOutputPosIdx, ge::DT_INT32) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "position",
                                                      std::to_string(ge::DT_INT32).c_str(), "SetOutputDataType failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridUnnormal)
    .InferShape(InferShape4GridUnnormal)
    .InferShapeRange(InferShapeRange4GridUnnormal)
    .InferDataType(InferDtype4GridUnnormal);

} // namespace ops
