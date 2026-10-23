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
 * \file grid_sample_infershape.cpp
 * \brief
 */
#include <numeric>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr int64_t DIM_NUM_2D = 4;
constexpr int64_t DIM_NUM_3D = 5;
constexpr int64_t INTERPOLATION_DIM_2D = 2;
constexpr int64_t INTERPOLATION_DIM_3D = 3;
constexpr uint64_t X_IDX_CHANNEL = 3;
constexpr uint64_t X_IDX_CHANNEL_3D = 4;
constexpr uint64_t GRID_DIM_IDX_W = 2;
constexpr uint64_t GRID_DIM_IDX_DIMS = 3;
constexpr uint64_t GRID_3D_DIM_IDX_DIMS = 4;
constexpr uint64_t Y_DIM_IDX_DIMS_START = 2;
constexpr uint64_t Y_DIM_IDX_H = 2;
constexpr uint64_t Y_DIM_IDX_W = 3;
constexpr uint64_t ATTR_IDX_CHANNEL_LAST = 3;
constexpr uint64_t NUM_1 = 1;
constexpr uint64_t NUM_2 = 2;
constexpr uint64_t NUM_3 = 3;
constexpr uint64_t NUM_4 = 4;

static ge::graphStatus InferDataType4GridSample(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GridSample", "InferDataTypeContext is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferDataType4GridSample begin");

    context->SetOutputDataType(0, context->GetInputDataType(0));

    OP_LOGD(context->GetNodeName(), "InferDataType4GridSample end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferGridSampleShape2D(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                              const gert::Shape* gridShape, gert::Shape* yShape)
{
    OP_LOGD(context->GetNodeName(), "InferGridSampleShape2D begin");

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* channelLast = attrs->GetAttrPointer<bool>(ATTR_IDX_CHANNEL_LAST);
    OP_CHECK_NULL_WITH_CONTEXT(context, channelLast);
    OP_LOGD(context->GetNodeName(), "channel_last attribute is :%d", *channelLast);

    int64_t nDim = xShape->GetDim(0U);
    OP_CHECK_IF(nDim == 0, OP_LOGE(context->GetNodeName(), "N must not be 0."), return ge::GRAPH_FAILED);
    if (nDim < 0) {
        nDim = gridShape->GetDim(0U); // if N from input_x is -1, then use N value from input_grid
    }

    int64_t cDim = xShape->GetDim(1U);
    if (*channelLast) {
        cDim = xShape->GetDim(X_IDX_CHANNEL);
    }
    OP_CHECK_IF(cDim == 0, OP_LOGE(context->GetNodeName(), "C must not be 0."), return ge::GRAPH_FAILED);

    int64_t hDim = gridShape->GetDim(1U);
    int64_t wDim = gridShape->GetDim(GRID_DIM_IDX_W);

    yShape->SetDimNum(DIM_NUM_2D);
    yShape->SetDim(0, nDim);
    yShape->SetDim(1, cDim);
    yShape->SetDim(Y_DIM_IDX_H, hDim);
    yShape->SetDim(Y_DIM_IDX_W, wDim);

    OP_LOGD(context->GetNodeName(), "InferGridSampleShape2D end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferGridSampleShape3D(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                              const gert::Shape* gridShape, gert::Shape* yShape, const Format format)
{
    OP_LOGD(context->GetNodeName(), "InferGridSampleShape3D begin");

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    bool channelLast = false;
    if (format == FORMAT_NDHWC) {
        channelLast = true;
    }

    int64_t nDim = xShape->GetDim(0U);
    OP_CHECK_IF(nDim == 0, OP_LOGE(context->GetNodeName(), "N must not be 0."), return ge::GRAPH_FAILED);
    if (nDim < 0) {
        nDim = gridShape->GetDim(0U); // if N from input_x is -1, then use N value from input_grid
    }

    int64_t cDim = xShape->GetDim(1U);
    int64_t dDim = gridShape->GetDim(NUM_1);
    int64_t hDim = gridShape->GetDim(NUM_2);
    int64_t wDim = gridShape->GetDim(NUM_3);
    if (channelLast) {
        cDim = xShape->GetDim(X_IDX_CHANNEL_3D);
    }
    OP_CHECK_IF(cDim == 0, OP_LOGE(context->GetNodeName(), "C must not be 0."), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "cDim = %ld", cDim);
    OP_LOGD(context->GetNodeName(), "dDim = %ld", dDim);
    OP_LOGD(context->GetNodeName(), "hDim = %ld", hDim);
    OP_LOGD(context->GetNodeName(), "wDim = %ld", wDim);

    yShape->SetDimNum(DIM_NUM_3D);
    yShape->SetDim(0, nDim);
    yShape->SetDim(1, cDim);
    yShape->SetDim(NUM_2, dDim);
    yShape->SetDim(NUM_3, hDim);
    yShape->SetDim(NUM_4, wDim);

    OP_LOGD(context->GetNodeName(), "InferGridSampleShape3D end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GridSample(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GridSample", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShape4GridSample begin");

    const gert::Shape* xShape = context->GetInputShape(0U);
    const gert::Shape* gridShape = context->GetInputShape(1U);
    gert::Shape* yShape = context->GetOutputShape(0U);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_LOGD(context->GetNodeName(), "x dim num:%ld, x shape:%s, grid dim num:%ld, grid shape:%s", xShape->GetDimNum(),
            Ops::Base::ToString(*xShape).c_str(), gridShape->GetDimNum(), Ops::Base::ToString(*gridShape).c_str());
    // Shape inference may run before a concrete input tensor is available (for
    // example for a dynamic graph).  The input descriptor already carries the
    // origin format, so do not dereference the optional tensor here.
    const auto* input_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_desc);
    auto format = input_desc->GetOriginFormat();
    OP_LOGD(context->GetNodeName(), "format = %d", format);

    if (IsUnknownRank(*xShape) || IsUnknownRank(*gridShape)) {
        SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }

    OP_CHECK_IF((xShape->GetDimNum() != DIM_NUM_2D || gridShape->GetDimNum() != DIM_NUM_2D) &&
                    (xShape->GetDimNum() != DIM_NUM_3D || gridShape->GetDimNum() != DIM_NUM_3D),
                OP_LOGE(context->GetNodeName(),
                        "shape is invalid, only rank 4 or 5 is supported, but got x rank %zu, grid rank %zu",
                        xShape->GetDimNum(), gridShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetDim(0U) != ge::UNKNOWN_DIM && gridShape->GetDim(0U) != ge::UNKNOWN_DIM &&
                    xShape->GetDim(0U) != gridShape->GetDim(0U),
                OP_LOGE(context->GetNodeName(), "N of x/grid should be same value"), return ge::GRAPH_FAILED);
    if (xShape->GetDimNum() == DIM_NUM_2D) {
        OP_CHECK_IF(
            gridShape->GetDim(GRID_DIM_IDX_DIMS) > 0 && gridShape->GetDim(GRID_DIM_IDX_DIMS) != INTERPOLATION_DIM_2D,
            OP_LOGE(context->GetNodeName(), "grid shape invalid, only support rank is 4"), return ge::GRAPH_FAILED);

        OP_CHECK_IF(InferGridSampleShape2D(context, xShape, gridShape, yShape) != GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to infershape"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(gridShape->GetDim(GRID_3D_DIM_IDX_DIMS) > 0 &&
                        gridShape->GetDim(GRID_3D_DIM_IDX_DIMS) != INTERPOLATION_DIM_3D,
                    OP_LOGE(context->GetNodeName(), "grid shape invalid, only support rank is 5"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(InferGridSampleShape3D(context, xShape, gridShape, yShape, format) != GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to infershape"), return ge::GRAPH_FAILED);
    }

    OP_LOGD(context->GetNodeName(), "InferShape4GridSample end. y shape:%s", Ops::Base::ToString(*yShape).c_str());
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferGridSampleShapeRange(const gert::InferShapeRangeContext* context,
                                                 const gert::Range<gert::Shape>* xRange,
                                                 const gert::Range<gert::Shape>* gridRange,
                                                 gert::Range<gert::Shape>* yRange)
{
    OP_LOGD(context->GetNodeName(), "InferGridSampleShapeRange begin");

    size_t xDimNum = xRange->GetMax()->GetDimNum();
    size_t gridDimNum = gridRange->GetMax()->GetDimNum();
    if (xDimNum == 0 || gridDimNum == 0) {
        yRange->GetMin()->SetDimNum(0);
        yRange->GetMax()->SetDimNum(0);
    } else if (xDimNum == 1) {
        yRange->GetMin()->SetDimNum(1);
        yRange->GetMin()->SetDim(0, xRange->GetMin()->GetDim(0));

        yRange->GetMax()->SetDimNum(1);
        yRange->GetMax()->SetDim(0, xRange->GetMax()->GetDim(0));
    } else if (gridDimNum == 1) {
        yRange->GetMin()->SetDimNum(1);
        yRange->GetMin()->SetDim(0, gridRange->GetMin()->GetDim(0));

        yRange->GetMax()->SetDimNum(1);
        yRange->GetMax()->SetDim(0, gridRange->GetMax()->GetDim(0));
    } else {
        OP_CHECK_IF(xDimNum != gridDimNum, OP_LOGE(context->GetNodeName(), "rank of x and grid should be same"),
                    return ge::GRAPH_FAILED);

        const gert::RuntimeAttrs* attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const bool* channelLast = attrs->GetAttrPointer<bool>(ATTR_IDX_CHANNEL_LAST);
        OP_CHECK_NULL_WITH_CONTEXT(context, channelLast);
        OP_LOGD(context->GetNodeName(), "channel_last attribute is :%d", *channelLast);

        // set range for N
        yRange->GetMin()->SetDimNum(xDimNum);
        yRange->GetMin()->SetDim(0, xRange->GetMin()->GetDim(0));
        yRange->GetMax()->SetDimNum(xDimNum);
        yRange->GetMax()->SetDim(0, xRange->GetMax()->GetDim(0));

        // set range for C
        if (*channelLast) {
            yRange->GetMin()->SetDim(1, xRange->GetMin()->GetDim(xDimNum - 1));
            yRange->GetMax()->SetDim(1, xRange->GetMax()->GetDim(xDimNum - 1));
        } else {
            yRange->GetMin()->SetDim(1, xRange->GetMin()->GetDim(1));
            yRange->GetMax()->SetDim(1, xRange->GetMax()->GetDim(1));
        }

        // For GridSample-2D, set range for H/W
        // For GridSample-3D, set range for D/H/W
        // For GridSample-nD, set range for H/W/....
        for (size_t axis = Y_DIM_IDX_DIMS_START; axis < xDimNum; ++axis) {
            yRange->GetMin()->SetDim(axis, gridRange->GetMin()->GetDim(axis - 1));
            yRange->GetMax()->SetDim(axis, gridRange->GetMax()->GetDim(axis - 1));
        }
    }

    OP_LOGD(context->GetNodeName(), "InferGridSampleShapeRange end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4GridSample(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GridSample", "InferShapeRangeContext is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShapeRange4GridSample begin");

    auto xRange = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMax());

    // if dim num is 0 or 1, maybe unkown rank for GridSample, infer process should not be terminated
    // if rank is known, it should be greater or equal 4
    // that is to say, GridSample op support 4-dim or 5-dim or n-dim ( n >= 4 )
    size_t xDimNum = xRange->GetMax()->GetDimNum();
    OP_CHECK_IF(
        xDimNum == 2 || xDimNum == 3,
        OP_LOGE(context->GetNodeName(), "x range invalid, only unknown rank or rank greater than 3 is supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xRange->GetMin()->GetDimNum() != xDimNum,
                OP_LOGE(context->GetNodeName(), "min value of x range is invalid"), return ge::GRAPH_FAILED);

    auto gridRange = context->GetInputShapeRange(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, gridRange->GetMax());

    size_t gridDimNum = gridRange->GetMax()->GetDimNum(); // the explanation is similar to xDimNum
    OP_CHECK_IF(
        gridDimNum == 2 || gridDimNum == 3,
        OP_LOGE(context->GetNodeName(), "grid range invalid, only unknown rank or rank greater than 3 is supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridRange->GetMin()->GetDimNum() != gridDimNum,
                OP_LOGE(context->GetNodeName(), "min value of grid range is invalid"), return ge::GRAPH_FAILED);

    auto yRange = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMax());

    OP_CHECK_IF(InferGridSampleShapeRange(context, xRange, gridRange, yRange) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to infer shape range"), return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSample)
    .InferDataType(InferDataType4GridSample)
    .InferShape(InferShape4GridSample)
    .InferShapeRange(InferShapeRange4GridSample);
} // namespace ops
