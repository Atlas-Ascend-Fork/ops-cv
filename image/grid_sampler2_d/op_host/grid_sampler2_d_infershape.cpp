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
 * \file grid_sampler2_d_infershape.cpp
 * \brief InferShape implementation for grid_sampler2_d operator
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "error_report_util.h"

namespace ops {
static constexpr int64_t X_INDEX = 0;
static constexpr int64_t GRID_INDEX = 1;
static constexpr int64_t Y_INDEX = 0;
static constexpr int64_t N_DIM_INDEX = 0;
static constexpr int64_t C_DIM_INDEX = 1;
static constexpr int64_t H_DIM_INDEX = 2;
static constexpr int64_t W_DIM_INDEX = 3;
static constexpr int64_t GRID_H_DIM_INDEX = 1;
static constexpr int64_t GRID_W_DIM_INDEX = 2;
static constexpr int64_t GRID_COORD_DIM_INDEX = 3;
static constexpr int64_t DIM_NUM_2D = 4;
static constexpr int64_t GRID_COORD_DIM = 2;

static ge::graphStatus InferShapeGridSampler2D(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                error_report::ReportInferShapeStageError(
                    "GridSampler2D",
                    "InferShapeContext is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeGridSampler2D");

    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(xShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "xShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    const gert::Shape* gridShape = context->GetInputShape(GRID_INDEX);
    OP_CHECK_IF(gridShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "gridShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(yShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "yShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*gridShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        xShape->GetDimNum() != DIM_NUM_2D,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input",
                                              std::to_string(xShape->GetDimNum()).c_str(), "x must be 4D, got dims"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape->GetDimNum() != DIM_NUM_2D,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "grid",
                                                      std::to_string(gridShape->GetDimNum()).c_str(),
                                                      "grid must be 4D, got dims"),
                return ge::GRAPH_FAILED);

    const int64_t gridCoordDim = gridShape->GetDim(GRID_COORD_DIM_INDEX);
    OP_CHECK_IF(
        gridCoordDim != ge::UNKNOWN_DIM && gridCoordDim != GRID_COORD_DIM,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "grid", std::to_string(gridCoordDim).c_str(),
                                              "grid last dim must be 2, got"),
        return ge::GRAPH_FAILED);

    const int64_t xBatch = xShape->GetDim(N_DIM_INDEX);
    const int64_t gridBatch = gridShape->GetDim(N_DIM_INDEX);
    OP_CHECK_IF(xBatch != ge::UNKNOWN_DIM && gridBatch != ge::UNKNOWN_DIM && xBatch != gridBatch,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "grid", (std::to_string(xBatch) + ", " + std::to_string(gridBatch)).c_str(),
                    "x N() != grid N()"),
                return ge::GRAPH_FAILED);

    const int64_t inputHeight = xShape->GetDim(H_DIM_INDEX);
    const int64_t inputWidth = xShape->GetDim(W_DIM_INDEX);
    OP_CHECK_IF(
        (inputHeight != ge::UNKNOWN_DIM && inputHeight <= 0) || (inputWidth != ge::UNKNOWN_DIM && inputWidth <= 0),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "input", std::to_string(inputWidth).c_str(),
                                  std::to_string(inputHeight).c_str()),
        return ge::GRAPH_FAILED);

    const int64_t gridHeight = gridShape->GetDim(GRID_H_DIM_INDEX);
    const int64_t gridWidth = gridShape->GetDim(GRID_W_DIM_INDEX);
    OP_CHECK_IF(
        (gridHeight != ge::UNKNOWN_DIM && gridHeight < 0) || (gridWidth != ge::UNKNOWN_DIM && gridWidth < 0),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "grid",
                                               (std::to_string(gridHeight) + ", " + std::to_string(gridWidth)).c_str(),
                                               "grid H/W must be non-negative or unknown, got H[] and W[]"),
        return ge::GRAPH_FAILED);

    const int64_t xChannel = xShape->GetDim(C_DIM_INDEX);
    OP_CHECK_IF((xBatch != ge::UNKNOWN_DIM && xBatch < 0) || (xChannel != ge::UNKNOWN_DIM && xChannel < 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "input", (std::to_string(xBatch) + ", " + std::to_string(xChannel)).c_str(),
                    "x N/C must be non-negative or unknown, got N[] and C[]"),
                return ge::GRAPH_FAILED);

    yShape->SetDimNum(DIM_NUM_2D);
    yShape->SetDim(N_DIM_INDEX, xBatch);
    yShape->SetDim(C_DIM_INDEX, xChannel);
    yShape->SetDim(H_DIM_INDEX, gridHeight);
    yShape->SetDim(W_DIM_INDEX, gridWidth);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeGridSampler2D");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSampler2D).InferShape(InferShapeGridSampler2D);
} // namespace ops
