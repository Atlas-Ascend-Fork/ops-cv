/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_with_arg_max_infershape.cpp
 * \brief roi_pooling_with_arg_max infer shape, logic reused from canndev
 */

#include <string>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_report_util.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;

namespace ops {
// Index constants, aligned with canndev
static constexpr size_t INPUT_FEATURES_IDX = 0;
static constexpr size_t INPUT_ROIS_IDX = 1;
static constexpr size_t ATTR_POOLED_HEIGHT_IDX = 0;
static constexpr size_t ATTR_POOLED_WIDTH_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;
static constexpr size_t OUTPUT_ARGMAX_IDX = 1;
static constexpr size_t ROIPOOL_DIM_SIZE = 4;

static constexpr size_t OUTPUT_DIM0 = 0;
static constexpr size_t OUTPUT_DIM1 = 1;
static constexpr size_t OUTPUT_DIM2 = 2;
static constexpr size_t OUTPUT_DIM3 = 3;

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_FOUR = 4;

static constexpr int64_t BATCH_SIZE_MAX_LIMIT = 1024;

static ge::graphStatus CheckInputShapeValid(gert::InferShapeContext* context, const gert::Shape* rois_shape,
                                            const gert::Shape* x_shape)
{
    if (!Ops::Base::IsUnknownRank(*rois_shape)) {
        int64_t length_rois = static_cast<int64_t>(rois_shape->GetDimNum());
        if (length_rois == 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "rois", "0", "2");
            return ge::GRAPH_FAILED;
        }
        if (length_rois != static_cast<int64_t>(DIM_TWO)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "rois", std::to_string(length_rois).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        if (rois_shape->GetDim(DIM_ZERO) > BATCH_SIZE_MAX_LIMIT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "rois.shape[0]",
                                                  std::to_string(rois_shape->GetDim(DIM_ZERO)).c_str(),
                                                  "the first dim of rois shape exceeds 1024.");
            return ge::GRAPH_FAILED;
        }
    }
    if (!Ops::Base::IsUnknownRank(*x_shape)) {
        int64_t length_features = static_cast<int64_t>(x_shape->GetDimNum());
        if (length_features < 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "x", std::to_string(length_features).c_str(), "x shape dim number is invalid.");
            return ge::GRAPH_FAILED;
        }
        if (length_features != static_cast<int64_t>(DIM_FOUR)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(length_features).c_str(), "4");
            return ge::GRAPH_FAILED;
        }
        if (x_shape->GetDim(DIM_ZERO) > BATCH_SIZE_MAX_LIMIT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x.shape[0]",
                                                  std::to_string(x_shape->GetDim(DIM_ZERO)).c_str(),
                                                  "the first dim of x shape exceeds 1024.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4RoiPoolingWithArgMax(gert::InferShapeContext* context)
{
    auto x_shape = context->GetInputShape(INPUT_FEATURES_IDX);
    OP_CHECK_IF(x_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    auto rois_shape = context->GetInputShape(INPUT_ROIS_IDX);
    OP_CHECK_IF(rois_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "rois"),
                return ge::GRAPH_FAILED);
    auto output_shape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_IF(output_shape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "output_shape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    auto argmax_shape = context->GetOutputShape(OUTPUT_ARGMAX_IDX);
    OP_CHECK_IF(argmax_shape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "argmax_shape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "attrs is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    const int64_t* pool_h_ptr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT_IDX);
    const int64_t* pool_w_ptr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH_IDX);

    const int64_t pool_h_shape = pool_h_ptr == nullptr ? ge::UNKNOWN_DIM : *pool_h_ptr;
    const int64_t pool_w_shape = pool_w_ptr == nullptr ? ge::UNKNOWN_DIM : *pool_w_ptr;

    if (CheckInputShapeValid(context, rois_shape, x_shape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Input shape check failed; propagate the first error");
        return ge::GRAPH_FAILED;
    }

    int64_t output_dim0;
    int64_t output_dim1;
    if (Ops::Base::IsUnknownRank(*rois_shape)) {
        output_dim0 = ge::UNKNOWN_DIM;
    } else {
        output_dim0 = rois_shape->GetDim(0);
    }

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        output_dim1 = ge::UNKNOWN_DIM;
    } else {
        output_dim1 = x_shape->GetDim(1);
    }

    output_shape->SetDimNum(ROIPOOL_DIM_SIZE);
    output_shape->SetDim(OUTPUT_DIM0, output_dim0);
    output_shape->SetDim(OUTPUT_DIM1, output_dim1);
    output_shape->SetDim(OUTPUT_DIM2, pool_h_shape);
    output_shape->SetDim(OUTPUT_DIM3, pool_w_shape);

    argmax_shape->SetDimNum(ROIPOOL_DIM_SIZE);
    argmax_shape->SetDim(OUTPUT_DIM0, output_dim0);
    argmax_shape->SetDim(OUTPUT_DIM1, output_dim1);
    argmax_shape->SetDim(OUTPUT_DIM2, pool_h_shape);
    argmax_shape->SetDim(OUTPUT_DIM3, pool_w_shape);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RoiPoolingWithArgMax(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4RoiPoolingWithArgMax start");
    auto input_x_dtype = context->GetInputDataType(INPUT_FEATURES_IDX);
    context->SetOutputDataType(OUTPUT_Y_IDX, input_x_dtype);
    context->SetOutputDataType(OUTPUT_ARGMAX_IDX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "InferDataType4RoiPoolingWithArgMax end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiPoolingWithArgMax)
    .InferShape(InferShape4RoiPoolingWithArgMax)
    .InferDataType(InferDataType4RoiPoolingWithArgMax);
} // namespace ops
