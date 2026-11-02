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
 * \file roi_align_grad_infershape.cpp
 * \brief RoiAlignGrad InferShape implementation.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr size_t kAttrXdiffShape = 0U;
constexpr size_t kAttrPooledWidth = 1U;
constexpr size_t kAttrPooledHeight = 2U;
constexpr size_t kYDiffIndex = 0U;
constexpr size_t kRoisIndex = 1U;
constexpr size_t kXDiffIndex = 0U;
constexpr int64_t kC0Size = 16;

static int64_t CeilDiv(int64_t value, int64_t divisor) { return divisor == 0 ? 0 : (value + divisor - 1) / divisor; }
} // namespace

namespace ops {

static ge::graphStatus InferShapeRoiAlignGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeRoiAlignGrad");

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* xdiffShape = attrs->GetListInt(kAttrXdiffShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xdiffShape);

    gert::Shape* yShape = context->GetOutputShape(kXDiffIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const size_t dimNum = xdiffShape->GetSize();
    const int64_t* dims = xdiffShape->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, dims);

    OP_CHECK_IF(dimNum != 4U, OP_LOGE(context->GetNodeName(), "xdiff_shape must be 4D, but got %zu.", dimNum),
                return GRAPH_FAILED);
    for (size_t i = 0; i < dimNum; ++i) {
        OP_CHECK_IF(dims[i] <= 0,
                    OP_LOGE(context->GetNodeName(), "xdiff_shape[%zu] must be positive, but got %lld.", i,
                            static_cast<long long>(dims[i])),
                    return GRAPH_FAILED);
    }

    const int64_t* pooledWidth = attrs->GetInt(kAttrPooledWidth);
    const int64_t* pooledHeight = attrs->GetInt(kAttrPooledHeight);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledWidth);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledHeight);
    OP_CHECK_IF(
        *pooledWidth <= 0 || *pooledHeight <= 0,
        OP_LOGE(context->GetNodeName(), "pooled_width and pooled_height must be positive, but got %lld and %lld.",
                static_cast<long long>(*pooledWidth), static_cast<long long>(*pooledHeight)),
        return GRAPH_FAILED);

    const gert::Shape* yDiffShape = context->GetInputShape(kYDiffIndex);
    const gert::Shape* roisShape = context->GetInputShape(kRoisIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDiffShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);

    if (Ops::Base::IsUnknownRank(*yDiffShape) || Ops::Base::IsUnknownRank(*roisShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    const size_t yDiffDimNum = yDiffShape->GetDimNum();
    OP_CHECK_IF(yDiffDimNum != 4U && yDiffDimNum != 5U,
                OP_LOGE(context->GetNodeName(), "y_diff must be 4D or 5D, but got %zu.", yDiffDimNum),
                return GRAPH_FAILED);
    OP_CHECK_IF(roisShape->GetDimNum() != 2U,
                OP_LOGE(context->GetNodeName(), "rois must be 2D, but got %zu.", roisShape->GetDimNum()),
                return GRAPH_FAILED);

    for (size_t i = 0; i < yDiffDimNum; ++i) {
        OP_CHECK_IF(yDiffShape->GetDim(i) < 0 && yDiffShape->GetDim(i) != UNKNOWN_DIM,
                    OP_LOGE(context->GetNodeName(), "y_diff[%zu] has invalid dimension %lld.", i,
                            static_cast<long long>(yDiffShape->GetDim(i))),
                    return GRAPH_FAILED);
    }
    for (size_t i = 0; i < roisShape->GetDimNum(); ++i) {
        OP_CHECK_IF(roisShape->GetDim(i) < 0 && roisShape->GetDim(i) != UNKNOWN_DIM,
                    OP_LOGE(context->GetNodeName(), "rois[%zu] has invalid dimension %lld.", i,
                            static_cast<long long>(roisShape->GetDim(i))),
                    return GRAPH_FAILED);
    }

    const int64_t roisNum = roisShape->GetDim(0);
    const int64_t roisRowSize = roisShape->GetDim(1);
    OP_CHECK_IF(roisRowSize != UNKNOWN_DIM && roisRowSize < 5,
                OP_LOGE(context->GetNodeName(), "rois second dimension must be at least 5, but got %lld.",
                        static_cast<long long>(roisRowSize)),
                return GRAPH_FAILED);

    const int64_t yDiffN = yDiffShape->GetDim(0);
    OP_CHECK_IF(yDiffN != UNKNOWN_DIM && roisNum != UNKNOWN_DIM && yDiffN != roisNum,
                OP_LOGE(context->GetNodeName(), "y_diff K (%lld) must equal rois K (%lld).",
                        static_cast<long long>(yDiffN), static_cast<long long>(roisNum)),
                return GRAPH_FAILED);

    if (yDiffDimNum == 4U) {
        OP_CHECK_IF(yDiffShape->GetDim(1) != UNKNOWN_DIM && yDiffShape->GetDim(1) != dims[1],
                    OP_LOGE(context->GetNodeName(), "y_diff C (%lld) must equal xdiff_shape C (%lld).",
                            static_cast<long long>(yDiffShape->GetDim(1)), static_cast<long long>(dims[1])),
                    return GRAPH_FAILED);
        OP_CHECK_IF(yDiffShape->GetDim(2) != UNKNOWN_DIM && yDiffShape->GetDim(2) != *pooledHeight,
                    OP_LOGE(context->GetNodeName(), "y_diff H (%lld) must equal pooled_height (%lld).",
                            static_cast<long long>(yDiffShape->GetDim(2)), static_cast<long long>(*pooledHeight)),
                    return GRAPH_FAILED);
        OP_CHECK_IF(yDiffShape->GetDim(3) != UNKNOWN_DIM && yDiffShape->GetDim(3) != *pooledWidth,
                    OP_LOGE(context->GetNodeName(), "y_diff W (%lld) must equal pooled_width (%lld).",
                            static_cast<long long>(yDiffShape->GetDim(3)), static_cast<long long>(*pooledWidth)),
                    return GRAPH_FAILED);
    } else {
        const int64_t expectedC1 = CeilDiv(dims[1], kC0Size);
        OP_CHECK_IF(yDiffShape->GetDim(1) != UNKNOWN_DIM && yDiffShape->GetDim(1) != expectedC1,
                    OP_LOGE(context->GetNodeName(), "y_diff C1 (%lld) must equal ceil(xdiff_shape C / 16) (%lld).",
                            static_cast<long long>(yDiffShape->GetDim(1)), static_cast<long long>(expectedC1)),
                    return GRAPH_FAILED);
        OP_CHECK_IF(yDiffShape->GetDim(2) != UNKNOWN_DIM && yDiffShape->GetDim(2) != *pooledHeight,
                    OP_LOGE(context->GetNodeName(), "y_diff H (%lld) must equal pooled_height (%lld).",
                            static_cast<long long>(yDiffShape->GetDim(2)), static_cast<long long>(*pooledHeight)),
                    return GRAPH_FAILED);
        OP_CHECK_IF(yDiffShape->GetDim(3) != UNKNOWN_DIM && yDiffShape->GetDim(3) != *pooledWidth,
                    OP_LOGE(context->GetNodeName(), "y_diff W (%lld) must equal pooled_width (%lld).",
                            static_cast<long long>(yDiffShape->GetDim(3)), static_cast<long long>(*pooledWidth)),
                    return GRAPH_FAILED);
        OP_CHECK_IF(yDiffShape->GetDim(4) != UNKNOWN_DIM && yDiffShape->GetDim(4) != kC0Size,
                    OP_LOGE(context->GetNodeName(), "y_diff C0 must be 16, but got %lld.",
                            static_cast<long long>(yDiffShape->GetDim(4))),
                    return GRAPH_FAILED);
    }

    if (yDiffDimNum == 4U) {
        yShape->SetDimNum(4U);
        yShape->SetDim(0U, dims[0]);
        yShape->SetDim(1U, dims[1]);
        yShape->SetDim(2U, dims[2]);
        yShape->SetDim(3U, dims[3]);
    } else {
        yShape->SetDimNum(5U);
        yShape->SetDim(0U, dims[0]);
        yShape->SetDim(1U, CeilDiv(dims[1], kC0Size));
        yShape->SetDim(2U, dims[2]);
        yShape->SetDim(3U, dims[3]);
        yShape->SetDim(4U, kC0Size);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeRoiAlignGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ROIAlignGrad).InferShape(InferShapeRoiAlignGrad);
} // namespace ops
