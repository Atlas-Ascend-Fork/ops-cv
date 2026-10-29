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
 * \file grid_sampler3_d_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"
#include "error_report_util.h"

static constexpr int INPUT_GRAD_INDEX = 0;
static constexpr int INPUT_X_INDEX = 1;
static constexpr int INPUT_GRID_INDEX = 2;
static constexpr int OUTPUT_DX_INDEX = 0;
static constexpr int OUTPUT_DGRID_INDEX = 1;
static constexpr int GRID_SAMPLER3D_GRAD_SHAPE_LIMIT = 5;
using namespace ge;
using namespace Ops::Base;

namespace ops {
static ge::graphStatus InferShape4GridSampler3DGrad(gert::InferShapeContext* context)
{
    // infer shape
    OP_CHECK_IF(context == nullptr,
                error_report::ReportInferShapeStageError(
                    "GridSampler3DGrad",
                    "InferShapeContext is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4GridSampler3DGrad.");

    const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_IF(gradShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "gradShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_IF(xShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "xShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    const gert::Shape* gridShape = context->GetInputShape(INPUT_GRID_INDEX);
    OP_CHECK_IF(gridShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "gridShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((xShape->GetDimNum() != GRID_SAMPLER3D_GRAD_SHAPE_LIMIT ||
                 gridShape->GetDimNum() != GRID_SAMPLER3D_GRAD_SHAPE_LIMIT ||
                 gradShape->GetDimNum() != GRID_SAMPLER3D_GRAD_SHAPE_LIMIT),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "shape",
                    (std::to_string(xShape->GetDimNum()) + ", " + std::to_string(gridShape->GetDimNum()) + ", " +
                     std::to_string(gradShape->GetDimNum()))
                        .c_str(),
                    "shape is invalid, only rank 5 is supported"),
                return ge::GRAPH_FAILED);

    gert::Shape* outDxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    OP_CHECK_IF(outDxShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "outDxShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    gert::Shape* outDgridShape = context->GetOutputShape(OUTPUT_DGRID_INDEX);
    OP_CHECK_IF(outDgridShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "outDgridShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    if (IsUnknownRank(*gridShape)) {
        SetUnknownRank(*outDgridShape);
    } else {
        *outDgridShape = *gridShape;
    }

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*outDxShape);
    } else {
        *outDxShape = *xShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4GridSampler3DGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4GridSampler3DGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4GridSampler3DGrad");
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_DGRID_INDEX, context->GetInputDataType(INPUT_GRID_INDEX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4GridSampler3DGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSampler3DGrad)
    .InferShape(InferShape4GridSampler3DGrad)
    .InferDataType(InferDataType4GridSampler3DGrad);
} // namespace ops
