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
 * \file roi_pooling_grad_with_arg_max_infershape.cpp
 * \brief
 */
#include <string>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_report_util.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;

namespace {
constexpr int64_t DIMS_ZERO = 0;
constexpr int64_t DIMS_ONE = 1;
constexpr int64_t DIMS_TWO = 2;
constexpr int64_t DIMS_THREE = 3;
constexpr int64_t DIMS_FOUR = 4;
constexpr int64_t BATCH_SIZE_MAX_LIMIT = 1024;
constexpr int64_t GRAD_IDX = 0;
constexpr int64_t X_IDX = 1;
constexpr int64_t ROIS_IDX = 2;
constexpr int64_t ARGMAX_IDX = 4;
constexpr int64_t Y_IDX = 0;
} // namespace

namespace ops {

static bool checkInputShape(gert::InferShapeContext* context, const gert::Shape* gradShape, const gert::Shape* xShape,
                            const gert::Shape* roisShape, const gert::Shape* argmaxShape)
{
    if (roisShape->GetDimNum() != DIMS_TWO) {
        OP_LOGD(context->GetNodeName(), "roisShape (%s) dim number is not two",
                Ops::Base::ToString(*roisShape).c_str());
        return false;
    }
    if (roisShape->GetDim(DIMS_ZERO) > BATCH_SIZE_MAX_LIMIT) {
        OP_LOGD(context->GetNodeName(), "The batch size (rois_shape[0]) cannot exceed 1024!");
        return false;
    }
    if (gradShape->GetDimNum() != DIMS_FOUR) {
        OP_LOGD(context->GetNodeName(), "grad_shape (%s) dim number is not four",
                Ops::Base::ToString(*gradShape).c_str());
        return false;
    }
    if (argmaxShape->GetDimNum() != DIMS_FOUR) {
        OP_LOGD(context->GetNodeName(), "argmax_shape (%s) dim number is not four",
                Ops::Base::ToString(*argmaxShape).c_str());
        return false;
    }
    if (xShape->GetDimNum() != DIMS_FOUR) {
        OP_LOGD(context->GetNodeName(), "xShape (%s) dim number is not four", Ops::Base::ToString(*xShape).c_str());
        return false;
    }
    if (xShape->GetDim(DIMS_ZERO) > BATCH_SIZE_MAX_LIMIT) {
        OP_LOGD(context->GetNodeName(), "The batch size (x_shape[0]) cannot exceed 1024!");
        return false;
    }
    return true;
}

static ge::graphStatus InferShape4RoiPoolingGradWithArgMax(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "begin to do InferShape4RoiPoolingGradWithArgMax.");
    auto gradShape = context->GetInputShape(GRAD_IDX);
    OP_CHECK_IF(gradShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "grad"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "input grad shape = %s", Ops::Base::ToString(*gradShape).c_str());
    auto xShape = context->GetInputShape(X_IDX);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "input x shape = %s", Ops::Base::ToString(*xShape).c_str());
    auto roisShape = context->GetInputShape(ROIS_IDX);
    OP_CHECK_IF(roisShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "rois"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "input rois shape = %s", Ops::Base::ToString(*roisShape).c_str());
    auto argmaxShape = context->GetInputShape(ARGMAX_IDX);
    OP_CHECK_IF(argmaxShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "argmax"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "input argmax shape = %s", Ops::Base::ToString(*argmaxShape).c_str());

    auto yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_IF(yShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "yShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*gradShape) || Ops::Base::IsUnknownRank(*xShape) ||
        Ops::Base::IsUnknownRank(*roisShape) || Ops::Base::IsUnknownRank(*argmaxShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank.");
        OP_LOGI(context->GetNodeName(), "Do InferShape4RoiPoolingGradWithArgMax rt2.0 success.");
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    OP_CHECK_IF((!checkInputShape(context, gradShape, xShape, roisShape, argmaxShape)),
                error_report::ReportInvalidArgument(context->GetNodeName(), "input", "Input shape check failed."),
                return ge::GRAPH_FAILED);

    auto xShapeSize = xShape->GetDimNum();
    OP_CHECK_IF((xShapeSize != DIMS_FOUR),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShapeSize).c_str(), "4"),
                return ge::GRAPH_FAILED);
    yShape->SetDimNum(xShapeSize);

    yShape->SetDim(DIMS_ZERO, xShape->GetDim(DIMS_ZERO));
    yShape->SetDim(DIMS_ONE, xShape->GetDim(DIMS_ONE));
    yShape->SetDim(DIMS_TWO, xShape->GetDim(DIMS_TWO));
    yShape->SetDim(DIMS_THREE, xShape->GetDim(DIMS_THREE));
    OP_LOGD(context->GetNodeName(), "output y shape = %s", Ops::Base::ToString(*yShape).c_str());

    OP_LOGI(context->GetNodeName(), "Do InferShape4RoiPoolingGradWithArgMax success.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4RoiPoolingGradWithArgMax(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "begin to do InferDtype4RoiPoolingGradWithArgMax.");
    const auto xDTypeInfer = context->GetInputDataType(X_IDX);
    context->SetOutputDataType(Y_IDX, xDTypeInfer);
    OP_LOGI(context->GetNodeName(), "Do InferDtype4RoiPoolingGradWithArgMax success.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiPoolingGradWithArgMax)
    .InferShape(InferShape4RoiPoolingGradWithArgMax)
    .InferDataType(InferDtype4RoiPoolingGradWithArgMax);
} // namespace ops
