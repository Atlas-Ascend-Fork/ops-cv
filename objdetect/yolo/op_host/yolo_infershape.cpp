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
 * \file yolo_infershape.cpp
 * \brief Shape and dtype inference for yolo operator
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {

static int64_t CeilX(int64_t size, int64_t alignSize) { return (size + alignSize - 1) / alignSize * alignSize; }

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t IDX_3 = 3;

static ge::graphStatus InferShapeYolo(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeYolo");

    // Get input shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // Unknown-rank input: spatial size unknown, set all outputs as UnknownRank
    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set outputs as UnknownRank.");
        for (int64_t i = 0; i <= IDX_2; i++) {
            gert::Shape* outShape = context->GetOutputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
            Ops::Base::SetUnknownRank(*outShape);
        }
        return GRAPH_SUCCESS;
    }

    // Get attributes
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* boxesPtr = attrs->GetInt(0);
    const int64_t* coordsPtr = attrs->GetInt(1);
    const int64_t* classesPtr = attrs->GetInt(2);
    int64_t boxes = (boxesPtr != nullptr) ? *boxesPtr : 3;
    int64_t coords = (coordsPtr != nullptr) ? *coordsPtr : 4;
    int64_t classes = (classesPtr != nullptr) ? *classesPtr : 80;

    // Validate input dimensions: rank must be >= 4; only dims 0/2/3 are used
    OP_CHECK_IF(xShape->GetDimNum() < 4,
                OP_LOGE(context, "Yolo: input dim num = %zu, should be >= 4", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate attribute constraints
    OP_CHECK_IF(coords != 4, OP_LOGE(context, "Yolo: coords must be 4, got %ld", coords), return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxes <= 0, OP_LOGE(context, "Yolo: boxes must be > 0, got %ld", boxes), return ge::GRAPH_FAILED);
    OP_CHECK_IF(classes <= 0 || classes > 1024,
                OP_LOGE(context, "Yolo: classes must be in [1, 1024], got %ld", classes), return ge::GRAPH_FAILED);

    int64_t N = xShape->GetDim(IDX_0);
    int64_t C = xShape->GetDim(IDX_1);
    int64_t H = xShape->GetDim(IDX_2);
    int64_t W = xShape->GetDim(IDX_3);

    // Validate batch dimension (SE requires N >= 1, no empty tensor)
    OP_CHECK_IF(N != ge::UNKNOWN_DIM && N <= 0, OP_LOGE(context, "Yolo: N must be >= 1, got N=%ld", N),
                return ge::GRAPH_FAILED);

    // Validate spatial dimensions
    OP_CHECK_IF((H != ge::UNKNOWN_DIM && H <= 0) || (W != ge::UNKNOWN_DIM && W <= 0),
                OP_LOGE(context, "Yolo: H and W must be > 0, got H=%ld, W=%ld", H, W), return ge::GRAPH_FAILED);

    // Validate channel consistency
    int64_t expectedC = boxes * (coords + 1 + classes);
    OP_CHECK_IF(C != ge::UNKNOWN_DIM && C != expectedC,
                OP_LOGE(context, "Yolo: input channel C=%ld, expected boxes*(coords+1+classes)=%ld", C, expectedC),
                return ge::GRAPH_FAILED);

    // H/W unknown: dims derived from H*W stay unknown
    const bool hwKnown = (H != ge::UNKNOWN_DIM && W != ge::UNKNOWN_DIM);
    int64_t HW = hwKnown ? H * W : ge::UNKNOWN_DIM;

    // Set coord_data shape: (N, boxes*coords, H*W)
    gert::Shape* coordShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, coordShape);
    coordShape->SetDimNum(3);
    coordShape->SetDim(IDX_0, N);
    coordShape->SetDim(IDX_1, boxes * coords);
    coordShape->SetDim(IDX_2, hwKnown ? CeilX(HW * 2 + 32, 32) / 2 : ge::UNKNOWN_DIM);

    // Set obj_prob shape: (N, boxes*H*W)
    gert::Shape* objShape = context->GetOutputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, objShape);
    objShape->SetDimNum(2);
    objShape->SetDim(IDX_0, N);
    objShape->SetDim(IDX_1, hwKnown ? CeilX(boxes * HW * 2 + 32, 32) / 2 : ge::UNKNOWN_DIM);

    // Set classes_prob shape: (N, classes, boxes*H*W)
    gert::Shape* clsShape = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, clsShape);
    clsShape->SetDimNum(3);
    clsShape->SetDim(IDX_0, N);
    clsShape->SetDim(IDX_1, classes);
    clsShape->SetDim(IDX_2, hwKnown ? CeilX(boxes * HW * 2 + 32, 32) / 2 : ge::UNKNOWN_DIM);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeYolo");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Yolo).InferShape(InferShapeYolo);
} // namespace ops
