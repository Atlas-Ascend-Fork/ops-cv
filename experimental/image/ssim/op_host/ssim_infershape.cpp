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
 * \file ssim_infershape.cpp
 * \brief ssim infer shape
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int64_t SSIM_MIN_WIN_SIZE = 11;
static constexpr int64_t SSIM_IDX_0 = 0;
static constexpr int64_t SSIM_IDX_1 = 1;
static constexpr int64_t SSIM_MIN_RANK = 2;

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4Ssim(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4Ssim");

    const gert::Shape* x1Shape = context->GetInputShape(SSIM_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(SSIM_IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* yOutShape = context->GetOutputShape(SSIM_IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutShape);

    if (IsUnknownRank(*x1Shape) || IsUnknownRank(*x2Shape)) {
        SetUnknownRank(*yOutShape);
        OP_LOGD(context, "End to do InferShape4Ssim with unknown rank.");
        return GRAPH_SUCCESS;
    }

    // x1 and x2 must have identical shape and dtype
    if (x1Shape->GetDimNum() != x2Shape->GetDimNum()) {
        OP_LOGE(context, "x1 and x2 must have the same rank, got %zu and %zu.", x1Shape->GetDimNum(),
                x2Shape->GetDimNum());
        return GRAPH_FAILED;
    }
    // Dynamic dims (UNKNOWN_DIM) stay legal: only two known dims have to agree.
    for (size_t i = 0; i < x1Shape->GetDimNum(); i++) {
        const int64_t x1Dim = x1Shape->GetDim(i);
        const int64_t x2Dim = x2Shape->GetDim(i);
        const bool compatible = (x1Dim == x2Dim) || (x1Dim == UNKNOWN_DIM) || (x2Dim == UNKNOWN_DIM);
        if (!compatible) {
            OP_LOGE(context, "x1 and x2 must have the same shape, mismatch at dim %zu: %lld vs %lld.", i,
                    static_cast<long long>(x1Dim), static_cast<long long>(x2Dim));
            return GRAPH_FAILED;
        }
    }

    // shape is [..., H, W] or [..., H, W, C], rank >= 2
    size_t dimNum = x1Shape->GetDimNum();
    if (dimNum < SSIM_MIN_RANK) {
        OP_LOGE(context, "rank of input must be at least 2, got %zu.", dimNum);
        return GRAPH_FAILED;
    }
    // rank-2: [H, W]; rank >= 3: [..., H, W, C] (C is the last dim)
    int64_t height = 0;
    int64_t width = 0;
    if (dimNum == SSIM_MIN_RANK) {
        height = x1Shape->GetDim(0);
        width = x1Shape->GetDim(1);
    } else {
        height = x1Shape->GetDim(dimNum - 3);
        width = x1Shape->GetDim(dimNum - 2);
    }
    // ssim_def.cpp declares dynamic shape/rank support, so UNKNOWN_DIM must be tolerated here;
    // the runtime range check lives in Tiling4Ssim, which sees the concrete shape.
    const bool heightKnown = (height != UNKNOWN_DIM);
    const bool widthKnown = (width != UNKNOWN_DIM);
    if ((heightKnown && height < SSIM_MIN_WIN_SIZE) || (widthKnown && width < SSIM_MIN_WIN_SIZE)) {
        OP_LOGE(context, "height/width must be >= 11, got H=%lld W=%lld.", static_cast<long long>(height),
                static_cast<long long>(width));
        return GRAPH_FAILED;
    }

    yOutShape->SetDimNum(1);
    yOutShape->SetDim(0, 1);

    OP_LOGD(context, "End to do InferShape4Ssim");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4Ssim(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4Ssim");
    if (context->GetInputDataType(SSIM_IDX_0) != context->GetInputDataType(SSIM_IDX_1)) {
        OP_LOGE(context, "x1 and x2 must have the same dtype.");
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(SSIM_IDX_0, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4Ssim");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Ssim).InferShape(InferShape4Ssim).InferDataType(InferDataType4Ssim);
} // namespace ops
