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
 * \file resize_bicubic_v2_grad_infershape.cpp
 * \brief resize_bicubic_v2_grad_infershape
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "error_report_util.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_GRADS = 0;
static constexpr size_t IN_ORIGINAL_IMAGE = 1;
static constexpr size_t OUT_Y = 0;
static constexpr size_t IN_GRADS_DIMS = 4;
static constexpr size_t IN_SIZE_NUM = 2;

ge::graphStatus ResizeBicubicV2GradInferShape(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeBicubicV2GradInferShape runtime 2.0");

    auto nodeName = context->GetNodeName();

    auto yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_IF(yShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "yShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    auto imageShape = context->GetInputShape(IN_ORIGINAL_IMAGE);
    OP_CHECK_IF(imageShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "imageShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    auto gradsDesc = context->GetInputDesc(IN_GRADS);
    OP_CHECK_IF(gradsDesc == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "gradsDesc is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    auto gradsFormat = gradsDesc->GetFormat().GetStorageFormat();
    OP_CHECK_IF((gradsFormat != FORMAT_NCHW) && (gradsFormat != FORMAT_NHWC),
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, "grads", Ops::Base::ToString(gradsFormat).c_str(),
                                                        "grads format only supports NCHW and NHWC"),
                return GRAPH_FAILED);
    const size_t hIdx = gradsFormat == FORMAT_NCHW ? 2 : 1;

    if (Ops::Base::IsUnknownRank(*imageShape)) {
        Ops::Base::SetUnknownShape(IN_GRADS_DIMS, *yShape);
    } else {
        OP_CHECK_IF(imageShape->GetDimNum() != IN_GRADS_DIMS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "original_image",
                                                          std::to_string(imageShape->GetDimNum()).c_str(),
                                                          "original_image shape only supports 4D"),
                    return GRAPH_FAILED);
        OP_CHECK_IF((imageShape->GetDim(hIdx) == 0) || (imageShape->GetDim(hIdx + 1) == 0),
                    error_report::ReportInvalidArgument(nodeName, "original_image",
                                                        "original_image size should be greater than 0"),
                    return GRAPH_FAILED);
        *yShape = *imageShape;
    }

    OP_LOGI(context->GetNodeName(), "End to do ResizeBicubicV2GradInferShape runtime 2.0");
    return ge::GRAPH_SUCCESS;
}

graphStatus ResizeBicubicV2GradInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeBicubicV2GradInferDtype");

    auto gradsDtype = context->GetInputDataType(IN_GRADS);
    OP_CHECK_IF((gradsDtype != ge::DT_FLOAT) && (gradsDtype != ge::DT_FLOAT16) && (gradsDtype != ge::DT_BF16),
                error_report::ReportInvalidArgument(context->GetNodeName(), "grads",
                                                    "grads dtype only supports float, float16 and bfloat16"),
                return GRAPH_FAILED);

    context->SetOutputDataType(OUT_Y, gradsDtype);

    OP_LOGI(context->GetNodeName(), "End to do ResizeBicubicV2GradInferDtype");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ResizeBicubicV2Grad)
    .InferShape(ResizeBicubicV2GradInferShape)
    .InferDataType(ResizeBicubicV2GradInferDtype);
} // namespace ops
