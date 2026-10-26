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
 * \file crop_infershape.cpp
 * \brief Infershape implementation for crop operator
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int32_t MAX_DIMS = 8;
static constexpr int32_t INDEX_ATTR_AXIS = 0;
static constexpr int32_t INDEX_ATTR_OFFSETS = 1;

static ge::graphStatus ValidateDtype(gert::InferShapeContext* context)
{
    const auto* xDesc = context->GetInputDesc(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto* sizeDesc = context->GetInputDesc(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeDesc);
    if (xDesc->GetDataType() != sizeDesc->GetDataType()) {
        OP_LOGE(context, "Crop: x.dtype != size.dtype, only same-dtype supported");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus ComputeCropOutputShape(gert::InferShapeContext* context, const gert::Shape* xShape,
                                              const gert::Shape* sizeShape, int64_t axis, size_t offsetsLen,
                                              const int64_t* offsetsData, gert::Shape* yShape)
{
    size_t xRank = xShape->GetDimNum();
    yShape->SetDimNum(xRank);
    for (size_t i = 0; i < xRank; i++) {
        int64_t xDim = xShape->GetDim(i);
        int64_t sizeDim = sizeShape->GetDim(i);
        // axis 之前维度取 x（对齐 910 基线/Caffe：size 前 axis 维不参与裁剪）
        if (static_cast<int64_t>(i) < axis) {
            yShape->SetDim(i, xDim);
        } else {
            // axis 及之后：output 维度 = size 维度，校验 offset+size<=x 且 offset>=0
            yShape->SetDim(i, sizeDim);
            int64_t offset = (offsetsLen == 1) ? offsetsData[0] : offsetsData[i - static_cast<size_t>(axis)];
            if (offset < 0) {
                OP_LOGE(context, "Crop: dim %zu offset=%ld < 0", i, offset);
                return GRAPH_FAILED;
            }
            if (xDim != ge::UNKNOWN_DIM && sizeDim != ge::UNKNOWN_DIM) {
                // 无溢出等价判断：offset > xDim 单值比较；offset<=xDim 时 xDim-offset 非负无溢出
                if (offset > xDim || sizeDim > xDim - offset) {
                    OP_LOGE(context, "Crop: dim %zu offset=%ld + size=%ld > x=%ld", i, offset, sizeDim, xDim);
                    return GRAPH_FAILED;
                }
            }
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCrop(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCrop");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* sizeShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeShape);
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*sizeShape)) {
        OP_LOGD(context->GetNodeName(), "input is UnknownRank, set output as UnknownRank.");
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    size_t xRank = xShape->GetDimNum();
    size_t sizeRank = sizeShape->GetDimNum();
    if (xRank != sizeRank) {
        OP_LOGE(context, "Crop: rank(x)=%zu != rank(size)=%zu", xRank, sizeRank);
        return GRAPH_FAILED;
    }
    if (xRank < 1 || xRank > static_cast<size_t>(MAX_DIMS)) {
        OP_LOGE(context, "Crop: rank=%zu out of range [1, %d]", xRank, MAX_DIMS);
        return GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* axisPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisPtr);
    int64_t axis = *axisPtr;
    if (axis < 0) {
        axis += static_cast<int64_t>(xRank);
    }
    if (axis < 0 || axis >= static_cast<int64_t>(xRank)) {
        OP_LOGE(context, "Crop: axis=%ld out of range [0, %zu)", axis, xRank);
        return GRAPH_FAILED;
    }

    const auto* offsetsVec = attrs->GetListInt(INDEX_ATTR_OFFSETS);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetsVec);
    size_t offsetsLen = offsetsVec->GetSize();
    if (offsetsLen != 1 && offsetsLen != static_cast<size_t>(static_cast<int64_t>(xRank) - axis)) {
        OP_LOGE(context, "Crop: offsets length=%zu, must be 1 or %ld", offsetsLen, static_cast<int64_t>(xRank) - axis);
        return GRAPH_FAILED;
    }

    if (ValidateDtype(context) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if (ComputeCropOutputShape(context, xShape, sizeShape, axis, offsetsLen, offsetsVec->GetData(), yShape) !=
        GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCrop");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Crop).InferShape(InferShapeCrop);
} // namespace ops
