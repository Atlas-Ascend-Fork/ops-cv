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
 * \file spatial_transformer_infershape.cpp
 * \brief SpatialTransformer infershape
 *
 * Output shape:
 *   NCHW (4D):    y = [x[0], x[1], OH, OW]
 *   NC1HWC0 (5D): y = [x[0], x[1], OH, OW, x[4]]
 *
 * OH/OW come from the output_size attribute ([-1,-1] means use x H/W).
 * Output dtype equals x dtype.
 */

#include <string>

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kXIndex = 0U;
constexpr size_t kThetaIndex = 1U;
constexpr size_t kOutputIndex = 0U;
constexpr size_t kNchwRank = 4U;
constexpr size_t kNc1hwc0Rank = 5U;
constexpr size_t kBatchDim = 0U;
constexpr size_t kHeightDim = 2U;
constexpr size_t kWidthDim = 3U;
constexpr size_t kOutputSizeLen = 2U;
constexpr int64_t kOutputSizeDefault = -1;
constexpr int32_t kThetaRank2 = 2;
constexpr int32_t kThetaRank3 = 3;
constexpr int64_t kThetaDim2 = 2;
constexpr int64_t kThetaDim3 = 3;
} // namespace

namespace ops {

static ge::graphStatus InferShapeSpatialTransformer(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(kXIndex);
    gert::Shape* yShape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Unknown-rank passthrough
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    // Read x format (NCHW or NC1HWC0 only, aligned with aicpu kernel :112-145)
    auto xDesc = context->GetInputDesc(kXIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetOriginFormat()));
    OP_CHECK_IF(xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NC1HWC0,
                OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(),
                                           "NCHW or NC1HWC0"),
                return ge::GRAPH_FAILED);

    const size_t expectedRank = (xFormat == ge::FORMAT_NCHW) ? kNchwRank : kNc1hwc0Rank;
    OP_CHECK_IF(xShape->GetDimNum() != expectedRank,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         (std::to_string(xShape->GetDimNum()) + "D").c_str(),
                                                         (xFormat == ge::FORMAT_NCHW ? "4D (NCHW)" : "5D (NC1HWC0)")),
                return ge::GRAPH_FAILED);

    // Read output_size attribute (default [-1,-1] = use x H/W)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // output_size is a ListInt attr, read via GetListInt (returns TypedContinuousVector*)
    const auto* outputSizeList = attrs->GetListInt(0);
    int64_t outputH = kOutputSizeDefault;
    int64_t outputW = kOutputSizeDefault;
    // Length rules aligned with RT1.0 legacy infershape: empty = use x H/W, single element = invalid,
    // two or more = use the first two elements.
    OP_CHECK_IF(outputSizeList != nullptr && outputSizeList->GetSize() == 1U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "output_size", "size 1",
                    "output_size must be empty (use input H/W) or contain [output_h, output_w]"),
                return ge::GRAPH_FAILED);
    if (outputSizeList != nullptr && outputSizeList->GetSize() >= static_cast<size_t>(kOutputSizeLen)) {
        outputH = outputSizeList->GetData()[0];
        outputW = outputSizeList->GetData()[1];
        OP_CHECK_IF(outputH < kOutputSizeDefault || outputW < kOutputSizeDefault,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "output_size",
                        ("[" + std::to_string(outputH) + ", " + std::to_string(outputW) + "]").c_str(),
                        "output_h and output_w must be -1 (use input size) or non-negative"),
                    return ge::GRAPH_FAILED);
    }

    // Resolve H/W: -1 means passthrough from x
    const int64_t yH = (outputH == kOutputSizeDefault) ? xShape->GetDim(kHeightDim) : outputH;
    const int64_t yW = (outputW == kOutputSizeDefault) ? xShape->GetDim(kWidthDim) : outputW;

    // Optional theta shape validation (aligned with aicpu kernel :177-201)
    const gert::Shape* thetaShape = context->GetOptionalInputShape(kThetaIndex);
    if (thetaShape != nullptr && !Ops::Base::IsUnknownRank(*thetaShape)) {
        const size_t thetaRank = thetaShape->GetDimNum();
        OP_CHECK_IF(thetaRank != static_cast<size_t>(kThetaRank2) && thetaRank != static_cast<size_t>(kThetaRank3),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "theta",
                                                             (std::to_string(thetaRank) + "D").c_str(), "2D or 3D"),
                    return ge::GRAPH_FAILED);

        auto dimOk = [](int64_t dim, int64_t expected) { return dim == expected || dim == ge::UNKNOWN_DIM; };
        bool thetaDimsValid;
        if (thetaRank == static_cast<size_t>(kThetaRank2)) {
            thetaDimsValid = dimOk(thetaShape->GetDim(0), kThetaDim2) && dimOk(thetaShape->GetDim(1), kThetaDim3);
        } else {
            thetaDimsValid = dimOk(thetaShape->GetDim(1), kThetaDim2) && dimOk(thetaShape->GetDim(2), kThetaDim3);
        }
        OP_CHECK_IF(!thetaDimsValid,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "theta",
                                                           Ops::Base::ToString(*thetaShape).c_str(),
                                                           "theta must have shape [2, 3] or [batch, 2, 3]"),
                    return ge::GRAPH_FAILED);

        if (thetaRank == static_cast<size_t>(kThetaRank3)) {
            // Skip batch check when either dim is UNKNOWN_DIM (-1); dynamic batch graphs must pass
            const int64_t thetaBatch = thetaShape->GetDim(0);
            const int64_t xBatch = xShape->GetDim(kBatchDim);
            OP_CHECK_IF(thetaBatch != ge::UNKNOWN_DIM && xBatch != ge::UNKNOWN_DIM && thetaBatch != xBatch,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context->GetNodeName(), "theta, x",
                            (std::to_string(thetaBatch) + ", " + std::to_string(xBatch)).c_str(),
                            "theta batch must match x batch"),
                        return ge::GRAPH_FAILED);
        }
    }

    // Construct output
    if (xFormat == ge::FORMAT_NCHW) {
        yShape->SetDimNum(kNchwRank);
        yShape->SetDim(0, xShape->GetDim(0)); // N
        yShape->SetDim(1, xShape->GetDim(1)); // C
        yShape->SetDim(2, yH);
        yShape->SetDim(3, yW);
    } else {
        yShape->SetDimNum(kNc1hwc0Rank);
        yShape->SetDim(0, xShape->GetDim(0)); // N
        yShape->SetDim(1, xShape->GetDim(1)); // C1
        yShape->SetDim(2, yH);
        yShape->SetDim(3, yW);
        yShape->SetDim(4, xShape->GetDim(4)); // C0
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSpatialTransformer(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Output dtype equals x dtype (aligned with def.cpp Output DataType list = x DataType list)
    const ge::DataType xDtype = context->GetInputDataType(kXIndex);
    context->SetOutputDataType(kOutputIndex, xDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SpatialTransformer)
    .InferShape(InferShapeSpatialTransformer)
    .InferDataType(InferDataTypeSpatialTransformer);
} // namespace ops
