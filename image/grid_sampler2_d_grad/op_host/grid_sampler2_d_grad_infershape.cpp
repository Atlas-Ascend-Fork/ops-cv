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
 * \file grid_sampler2_d_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

static constexpr int INPUT_GRAD_INDEX = 0;
static constexpr int INPUT_X_INDEX = 1;
static constexpr int INPUT_GRID_INDEX = 2;
static constexpr int OUTPUT_DX_INDEX = 0;
static constexpr int OUTPUT_DGRID_INDEX = 1;
static constexpr int GRID_SAMPLER2D_GRAD_SHAPE_LIMIT = 4;
using namespace ge;
using namespace Ops::Base;
namespace ops {
// InferShape framework contract failures are reported once as EZ1014.
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "GridSampler2DGrad", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

// 参数缺失/非法统一上报EZ0037 (cv opbase暂无ARGUMENT_WITH_REASON宏)
static void ReportMissingArgument(const char* opName, const char* paramName, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(opName, "Parameter %s of operator %s is invalid. Reason: %s.", paramName, opName, reason);
    const std::vector<const char*> msgKey = {"param_name", "op_name", "reason"};
    const std::vector<const char*> msgValue = {paramName, opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ0037", msgKey, msgValue);
}

static ge::graphStatus InferShape4GridSampler2DGrad(gert::InferShapeContext* context)
{
    // infer shape
    OP_CHECK_IF(context == nullptr,
                ReportInferShapeStageError(
                    "GridSampler2DGrad", "the InferShapeContext is null; the infer-shape framework contract is broken"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4GridSampler2DGrad.");

    const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_IF(gradShape == nullptr,
                ReportMissingArgument(context->GetNodeName(), "grad",
                                      "the input shape of required input grad is missing in the infer-shape context; "
                                      "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_IF(xShape == nullptr,
                ReportMissingArgument(context->GetNodeName(), "x",
                                      "the input shape of required input x is missing in the infer-shape context; "
                                      "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    const gert::Shape* gridShape = context->GetInputShape(INPUT_GRID_INDEX);
    OP_CHECK_IF(gridShape == nullptr,
                ReportMissingArgument(context->GetNodeName(), "grid",
                                      "the input shape of required input grid is missing in the infer-shape context; "
                                      "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((xShape->GetDimNum() != GRID_SAMPLER2D_GRAD_SHAPE_LIMIT ||
                 gridShape->GetDimNum() != GRID_SAMPLER2D_GRAD_SHAPE_LIMIT ||
                 gradShape->GetDimNum() != GRID_SAMPLER2D_GRAD_SHAPE_LIMIT),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "x, grid and grad",
                    (std::to_string(xShape->GetDimNum()) + ", " + std::to_string(gridShape->GetDimNum()) + " and " +
                     std::to_string(gradShape->GetDimNum()))
                        .c_str(),
                    "the rank of x, grid and grad must be 4"),
                return ge::GRAPH_FAILED);

    gert::Shape* outDxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    OP_CHECK_IF(outDxShape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape of required output dx is missing in the infer-shape "
                                           "context; check that the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);

    gert::Shape* outDgridShape = context->GetOutputShape(OUTPUT_DGRID_INDEX);
    OP_CHECK_IF(outDgridShape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape of required output dgrid is missing in the infer-shape "
                                           "context; check that the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*outDxShape);
    } else {
        *outDxShape = *xShape;
    }

    if (IsUnknownRank(*gridShape)) {
        SetUnknownRank(*outDgridShape);
    } else {
        *outDgridShape = *gridShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4GridSampler2DGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4GridSampler2DGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4GridSampler2DGrad");
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_DGRID_INDEX, context->GetInputDataType(INPUT_GRID_INDEX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4GridSampler2DGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSampler2DGrad)
    .InferShape(InferShape4GridSampler2DGrad)
    .InferDataType(InferDataType4GridSampler2DGrad);
} // namespace ops
