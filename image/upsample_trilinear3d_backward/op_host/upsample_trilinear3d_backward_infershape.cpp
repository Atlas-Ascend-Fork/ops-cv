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
 * \file upsample3dgrad_infershape.cpp
 * \brief
 */

#include <cmath>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_report_util.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t OUT_Y = 0;
static constexpr size_t SUPPORTED_DIM_NUM = 5;
static constexpr size_t SUPPORTED_OUTPUT_DIM_NUM = 3;
static constexpr size_t INDEX_INPUT_SIZE = 0;
static constexpr size_t INDEX_OUTPUT_SIZE = 1;
static constexpr size_t INDEX_SCALES = 2;
static constexpr size_t NOT_CHANGE_DIM = 2;

static ge::graphStatus SetUpsample3dGradInferShape(const gert::InferShapeContext* context,
                                                   const gert::Shape* grad_output_shape, gert::Shape* outy_shape,
                                                   const gert::ContinuousVector* input_size)
{
    auto attrs = context->GetAttrs();
    const gert::ContinuousVector* output_size = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_OUTPUT_SIZE);
    const gert::ContinuousVector* scales = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_SCALES);
    OP_CHECK_IF(input_size == nullptr || output_size == nullptr || scales == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(), "input_size/output_size/scales is null in the infershape context; "
                                            "check the operator registration and the InferShapeContext "
                                            "construction"),
                return ge::GRAPH_FAILED);
    auto input_size_data = reinterpret_cast<const int64_t*>(input_size->GetData());
    auto output_size_data = reinterpret_cast<const int64_t*>(output_size->GetData());
    auto scales_data = reinterpret_cast<const float*>(scales->GetData());

    if (output_size->GetSize() != 0 && scales->GetSize() == 0) {
        OP_CHECK_IF(output_size->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "attr",
                                                              std::to_string(output_size->GetSize()).c_str(),
                                                              "attr::output_size dims must be 3"),
                    return ge::GRAPH_FAILED);

        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            OP_CHECK_IF(output_size_data[i] != grad_output_shape->GetDim(i + NOT_CHANGE_DIM),
                        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                            context->GetNodeName(), "output_size",
                            (std::to_string(output_size_data[i]) + ", " +
                             std::to_string(grad_output_shape->GetDim(i + NOT_CHANGE_DIM)))
                                .c_str(),
                            "output_size[i] should be equal to grad_output_shape[i]"),
                        return ge::GRAPH_FAILED);
        }
    } else if (output_size->GetSize() == 0 && scales->GetSize() != 0) {
        OP_CHECK_IF(scales->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr",
                                                          std::to_string(scales->GetSize()).c_str(),
                                                          "attr::scales dims must be 3"),
                    return ge::GRAPH_FAILED);

        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            int64_t tmp = int64_t(floor(input_size_data[i + NOT_CHANGE_DIM] * scales_data[i]));
            OP_CHECK_IF(tmp != grad_output_shape->GetDim(i + NOT_CHANGE_DIM),
                        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                            context->GetNodeName(), "input_size",
                            (std::to_string(tmp) + ", " + std::to_string(grad_output_shape->GetDim(i + NOT_CHANGE_DIM)))
                                .c_str(),
                            "input_size[i] * scales[i] should be equal to grad_output_shape[i]"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        error_report::ReportInvalidArgument(
            context->GetNodeName(), "output_size,scales",
            "only one of attr::output_size or attr::scales should be defined as a non-empty value");
        return ge::GRAPH_FAILED;
    }

    *outy_shape = *grad_output_shape;
    for (size_t i = 0; i < SUPPORTED_DIM_NUM; i++) {
        outy_shape->SetDim(i, input_size_data[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Upsample3dGradInferShapeImpl(gert::InferShapeContext* context,
                                                    const gert::Shape* grad_output_shape, gert::Shape* outy_shape)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "attrs is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    const gert::ContinuousVector* input_size = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_INPUT_SIZE);
    OP_CHECK_IF(
        input_size == nullptr,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "Get attr::input_size failed!; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(input_size->GetSize() != SUPPORTED_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "attr",
                                                          std::to_string(input_size->GetSize()).c_str(),
                                                          "attr::input_size dims must be 5"),
                return ge::GRAPH_FAILED);

    return SetUpsample3dGradInferShape(context, grad_output_shape, outy_shape, input_size);
}

static ge::graphStatus InferShape4Upsample3dGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "begin to do InferShape4UpsampleTrilinear3dGrad");
    const gert::Shape* grad_output_shape = context->GetInputShape(IN_X);
    OP_CHECK_IF(grad_output_shape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "grad_output_shape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    gert::Shape* outy_shape = context->GetOutputShape(OUT_Y);
    OP_CHECK_IF(outy_shape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "outy_shape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    auto grad_output_dim = grad_output_shape->GetDimNum();
    OP_CHECK_IF(
        grad_output_dim != SUPPORTED_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "dim", std::to_string(grad_output_dim).c_str(),
                                                 "expected dim of input x should be 5"),
        return ge::GRAPH_FAILED);

    return Upsample3dGradInferShapeImpl(context, grad_output_shape, outy_shape);
}

IMPL_OP_INFERSHAPE(UpsampleTrilinear3dGrad).InferShape(InferShape4Upsample3dGrad);
} // namespace ops
