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
 * \file resize_bicubic_v2_infershape.cpp
 * \brief resize_bicubic_v2_infershape
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"
#include "error_report_util.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t IN_SIZE = 1;
static constexpr size_t OUT_Y = 0;
static constexpr size_t IN_X_DIMS = 4;
static constexpr int64_t IN_SIZE_NUM = 2;

ge::graphStatus ResizeBicubicV2InferShape(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeBicubicV2InferShape runtime 2.0");

    auto nodeName = context->GetNodeName();

    auto xShape = context->GetInputShape(IN_X);
    OP_CHECK_IF(xShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "xShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    auto yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_IF(yShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "yShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    auto xDesc = context->GetInputDesc(IN_X);
    OP_CHECK_IF(xDesc == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "xDesc is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    OP_CHECK_IF((xFormat != FORMAT_NCHW) && (xFormat != FORMAT_NHWC) && (xFormat != FORMAT_ND),
                error_report::ReportInvalidArgument(nodeName, "format", "x format only support nchw or nhwc or nd"),
                return GRAPH_FAILED);
    const size_t hIdx = xFormat == FORMAT_NHWC ? 1 : 2;

    auto sizeTensor = context->GetInputTensor(IN_SIZE);
    OP_CHECK_IF(sizeTensor == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "sizeTensor is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    auto sizeShape = context->GetInputShape(IN_SIZE);
    OP_CHECK_IF(sizeShape == nullptr,
                error_report::ReportInferShapeStageError(
                    context->GetNodeName(),
                    "sizeShape is null in the infershape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sizeShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    nodeName, "input", std::to_string(sizeShape->GetDimNum()).c_str(), "size must be 1D, but rank is"),
                return GRAPH_FAILED);
    auto sizeDtype = sizeTensor->GetDataType();
    OP_CHECK_IF(sizeDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "input", std::to_string(sizeDtype).c_str(),
                                                      "size dtype only support int32"),
                return GRAPH_FAILED);
    int32_t outH = ge::UNKNOWN_DIM;
    int32_t outW = ge::UNKNOWN_DIM;
    if (Ops::Cv::IsConstTensor(sizeTensor)) {
        const int32_t* sizeValue = sizeTensor->GetData<int32_t>();
        auto sizeNum = sizeTensor->GetShapeSize();
        OP_CHECK_IF(sizeNum != IN_SIZE_NUM,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(nodeName, "input", std::to_string(sizeNum).c_str(),
                                                              "the element number of size should be 2, but is"),
                    return GRAPH_FAILED);
        outH = sizeValue[0];
        outW = sizeValue[1];
        OP_CHECK_IF((outH <= 0) || (outW <= 0),
                    error_report::ReportInvalidArgument(nodeName, "output", "output size should be greater than 0"),
                    return GRAPH_FAILED);
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownShape(IN_X_DIMS, *yShape);
    } else {
        OP_CHECK_IF(xShape->GetDimNum() != IN_X_DIMS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "shape", std::to_string(xShape->GetDimNum()).c_str(), "x shape only support 4D"),
                    return GRAPH_FAILED);
        OP_CHECK_IF((xShape->GetDim(hIdx) == 0) || (xShape->GetDim(hIdx + 1) == 0),
                    error_report::ReportInvalidArgument(nodeName, "input", "input size should be greater than 0"),
                    return GRAPH_FAILED);
        *yShape = *xShape;
    }

    yShape->SetDim(hIdx, outH);
    yShape->SetDim(hIdx + 1, outW);

    OP_LOGI(context->GetNodeName(), "End to do ResizeBicubicV2InferShape runtime 2.0");
    return ge::GRAPH_SUCCESS;
}

graphStatus ResizeBicubicV2InferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do ResizeBicubicV2InferDtype");

    auto xDtype = context->GetInputDataType(IN_X);
    OP_CHECK_IF((xDtype != ge::DT_FLOAT) && (xDtype != ge::DT_FLOAT16) && (xDtype != ge::DT_BF16),
                error_report::ReportInvalidArgument(context->GetNodeName(), "dtype",
                                                    "x dtype only support float, float16 and bfloat16"),
                return GRAPH_FAILED);

    context->SetOutputDataType(OUT_Y, xDtype);

    OP_LOGI(context->GetNodeName(), "End to do ResizeBicubicV2InferDtype");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ResizeBicubicV2)
    .InferShape(ResizeBicubicV2InferShape)
    .InputsDataDependency({IN_SIZE})
    .InferDataType(ResizeBicubicV2InferDtype);
} // namespace ops
