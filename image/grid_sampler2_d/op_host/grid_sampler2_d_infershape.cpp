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
    OP_CHECK_IF(context == nullptr, OP_LOGE("GridSampler2D", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeGridSampler2D");

    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gridShape = context->GetInputShape(GRID_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridShape);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*gridShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape->GetDimNum() != DIM_NUM_2D, OP_LOGE(context, "x must be 4D, got %zu dims", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape->GetDimNum() != DIM_NUM_2D,
                OP_LOGE(context, "grid must be 4D, got %zu dims", gridShape->GetDimNum()), return ge::GRAPH_FAILED);

    const int64_t gridCoordDim = gridShape->GetDim(GRID_COORD_DIM_INDEX);
    OP_CHECK_IF(gridCoordDim != ge::UNKNOWN_DIM && gridCoordDim != GRID_COORD_DIM,
                OP_LOGE(context, "grid last dim must be 2, got %ld", gridCoordDim), return ge::GRAPH_FAILED);

    const int64_t xBatch = xShape->GetDim(N_DIM_INDEX);
    const int64_t gridBatch = gridShape->GetDim(N_DIM_INDEX);
    OP_CHECK_IF(xBatch != ge::UNKNOWN_DIM && gridBatch != ge::UNKNOWN_DIM && xBatch != gridBatch,
                OP_LOGE(context, "x N(%ld) != grid N(%ld)", xBatch, gridBatch), return ge::GRAPH_FAILED);

    const int64_t inputHeight = xShape->GetDim(H_DIM_INDEX);
    const int64_t inputWidth = xShape->GetDim(W_DIM_INDEX);
    OP_CHECK_IF(
        (inputHeight != ge::UNKNOWN_DIM && inputHeight <= 0) || (inputWidth != ge::UNKNOWN_DIM && inputWidth <= 0),
        OP_LOGE(context, "input H/W must be greater than 0, but got H[%ld] and W[%ld]", inputHeight, inputWidth),
        return ge::GRAPH_FAILED);

    const int64_t gridHeight = gridShape->GetDim(GRID_H_DIM_INDEX);
    const int64_t gridWidth = gridShape->GetDim(GRID_W_DIM_INDEX);
    OP_CHECK_IF(
        (gridHeight != ge::UNKNOWN_DIM && gridHeight < 0) || (gridWidth != ge::UNKNOWN_DIM && gridWidth < 0),
        OP_LOGE(context, "grid H/W must be non-negative or unknown, got H[%ld] and W[%ld]", gridHeight, gridWidth),
        return ge::GRAPH_FAILED);

    const int64_t xChannel = xShape->GetDim(C_DIM_INDEX);
    OP_CHECK_IF((xBatch != ge::UNKNOWN_DIM && xBatch < 0) || (xChannel != ge::UNKNOWN_DIM && xChannel < 0),
                OP_LOGE(context, "x N/C must be non-negative or unknown, got N[%ld] and C[%ld]", xBatch, xChannel),
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
