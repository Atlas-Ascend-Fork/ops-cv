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
 * \file deformable_roi_pool_infershape.cpp
 * \brief Infershape implementation for deformable_roi_pool operator
 */

#include <cmath>
#include <limits>
#include <string>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_X = 0;
static constexpr int64_t IDX_ROIS = 1;
static constexpr int64_t IDX_OFFSET = 2;
static constexpr int64_t IDX_Y = 0;

static constexpr size_t OUTPUT_DIM_NUM = 4;
static constexpr int64_t ROIS_DIM1 = 5;
static constexpr int64_t OFFSET_CHANNELS = 2;
static constexpr size_t OFFSET_DIM_NUM = 4;
static constexpr int64_t MAX_INT32_VALUE = std::numeric_limits<int32_t>::max();
static constexpr size_t OUTPUT_SIZE_NUM = 2;
static constexpr size_t ATTR_IDX_OUTPUT_SIZE = 1;
static constexpr size_t ATTR_IDX_GAMMA = 3;

static std::string ShapeToTraceString(const gert::Shape* shape)
{
    if (shape == nullptr) {
        return "<null>";
    }
    if (Ops::Base::IsUnknownRank(*shape)) {
        return "<unknown-rank>";
    }

    std::string result = "[";
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (i != 0) {
            result += ",";
        }
        result += std::to_string(shape->GetDim(i));
    }
    result += "]";
    return result;
}

static ge::graphStatus InferShapeDeformableRoiPool(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeDeformableRoiPool");

    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* roisShape = context->GetInputShape(IDX_ROIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);
    const gert::Shape* offsetShape = context->GetOptionalInputShape(IDX_OFFSET);
    OP_LOGI(context->GetNodeName(), "[DRP_INFERSHAPE2_TRACE] ENTER x=%s, rois=%s, offset=%s",
            ShapeToTraceString(xShape).c_str(), ShapeToTraceString(roisShape).c_str(),
            ShapeToTraceString(offsetShape).c_str());

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto outputSizeVec = attrs->GetListInt(ATTR_IDX_OUTPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSizeVec);
    if (outputSizeVec->GetSize() != OUTPUT_SIZE_NUM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "output_size",
                                     std::to_string(outputSizeVec->GetSize()).c_str(), "2");
        return GRAPH_FAILED;
    }
    const int64_t* outputSizeData = outputSizeVec->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSizeData);
    if (outputSizeData[0] < 1 || outputSizeData[0] > MAX_INT32_VALUE || outputSizeData[1] < 1 ||
        outputSizeData[1] > MAX_INT32_VALUE) {
        std::string outputSizeValue = std::to_string(outputSizeData[0]) + ", " + std::to_string(outputSizeData[1]);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size", outputSizeValue.c_str(),
                                              "both elements must be in [1, INT32_MAX]");
        return GRAPH_FAILED;
    }

    const float* gammaPtr = attrs->GetFloat(ATTR_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaPtr);
    if (!std::isfinite(*gammaPtr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "gamma", std::to_string(*gammaPtr).c_str(),
                                              "gamma must be finite");
        return GRAPH_FAILED;
    }

    // No fixed-rank requirement for x (align with the canndev built-in): the chip-agnostic
    // registration also serves A2/A3, whose execute-stage re-inference sees the 5HD storage
    // shape (rank 5) after GE format assignment. Only two dims are needed to derive the
    // output channels. The strict rank-4 NCHW contract is enforced by the arch35 tiling
    // which only runs on 950/350.
    const bool xUnknownRank = Ops::Base::IsUnknownRank(*xShape);
    if (!xUnknownRank && xShape->GetDimNum() < 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()).c_str(),
                                              "rank must be at least 2 to infer the output channels");
        return GRAPH_FAILED;
    }
    if (!xUnknownRank && xShape->GetDimNum() == OUTPUT_DIM_NUM) {
        const int64_t batch = xShape->GetDim(0);
        const int64_t channels = xShape->GetDim(1);
        const int64_t height = xShape->GetDim(2);
        const int64_t width = xShape->GetDim(3);
        // Empty feature maps (N/C/H/W == 0) are tolerated here: infershape registration is
        // chip-agnostic and A2/A3 built-in kernels accept them. The strict kernel contract
        // (C/H/W >= 1) is enforced by the arch35 tiling which only runs on 950/350.
        if ((batch != ge::UNKNOWN_DIM && (batch < 0 || batch > MAX_INT32_VALUE)) ||
            (channels != ge::UNKNOWN_DIM && (channels < 0 || channels > MAX_INT32_VALUE)) ||
            (height != ge::UNKNOWN_DIM && (height < 0 || height > MAX_INT32_VALUE)) ||
            (width != ge::UNKNOWN_DIM && (width < 0 || width > MAX_INT32_VALUE))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", "N, C, H, W",
                                                  "N/C/H/W must be in [0, INT32_MAX]");
            return GRAPH_FAILED;
        }
    }

    const bool roisUnknownRank = Ops::Base::IsUnknownRank(*roisShape);
    if (!roisUnknownRank && roisShape->GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "rois", std::to_string(roisShape->GetDimNum()).c_str(),
                                     "2");
        return GRAPH_FAILED;
    }
    if (!roisUnknownRank) {
        const int64_t numRois = roisShape->GetDim(0);
        const int64_t roiCols = roisShape->GetDim(1);
        if (numRois != ge::UNKNOWN_DIM && (numRois < 0 || numRois > MAX_INT32_VALUE)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "rois.shape[0]",
                                                  std::to_string(numRois).c_str(),
                                                  "num_rois must be in [0, INT32_MAX]");
            return GRAPH_FAILED;
        }
        // Empty rois (zero-element placeholder, e.g. shape [0, 0]) is tolerated here: the infershape
        // registration is chip-agnostic and the A2/A3 built-in infershape performs no such check.
        // The strict kernel contract (dim[1] == 5) is enforced by the arch35 tiling on 950/350.
        const bool emptyRois = numRois == 0 || roiCols == 0;
        if (!emptyRois && roiCols != ge::UNKNOWN_DIM && roiCols != ROIS_DIM1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rois", std::to_string(roiCols).c_str(),
                                                  "rois dim[1] must be 5");
            return GRAPH_FAILED;
        }
    }

    const bool emptyOffset = offsetShape == nullptr || Ops::Base::IsUnknownRank(*offsetShape) ||
                             offsetShape->GetDimNum() == 0 ||
                             (offsetShape->GetDimNum() == 1 && offsetShape->GetDim(0) == 0);
    if (!emptyOffset) {
        if (offsetShape->GetDimNum() != OFFSET_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "offset",
                                         std::to_string(offsetShape->GetDimNum()).c_str(), "4");
            return GRAPH_FAILED;
        }
        const int64_t offsetNumRois = offsetShape->GetDim(0);
        const int64_t offsetChannels = offsetShape->GetDim(1);
        const int64_t offsetPooledH = offsetShape->GetDim(2);
        const int64_t offsetPooledW = offsetShape->GetDim(3);
        const int64_t numRois = roisUnknownRank ? ge::UNKNOWN_DIM : roisShape->GetDim(0);
        const bool shapeMismatch = (offsetNumRois != ge::UNKNOWN_DIM && numRois != ge::UNKNOWN_DIM &&
                                    offsetNumRois != numRois) ||
                                   (offsetChannels != ge::UNKNOWN_DIM && offsetChannels != OFFSET_CHANNELS) ||
                                   (offsetPooledH != ge::UNKNOWN_DIM && offsetPooledH != outputSizeData[0]) ||
                                   (offsetPooledW != ge::UNKNOWN_DIM && offsetPooledW != outputSizeData[1]);
        if (shapeMismatch) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "offset", "num_rois, 2, pooled_h, pooled_w",
                                                  "offset shape must be [num_rois, 2, output_size[0], output_size[1]]");
            return GRAPH_FAILED;
        }
    }

    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(OUTPUT_DIM_NUM);
    yShape->SetDim(0, roisUnknownRank ? ge::UNKNOWN_DIM : roisShape->GetDim(0));
    yShape->SetDim(1, xUnknownRank ? ge::UNKNOWN_DIM : xShape->GetDim(1));
    yShape->SetDim(2, outputSizeData[0]);
    yShape->SetDim(3, outputSizeData[1]);

    OP_LOGI(context->GetNodeName(), "[DRP_INFERSHAPE2_TRACE] EXIT y=%s", ShapeToTraceString(yShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeDeformableRoiPool");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DeformableRoiPool).InferShape(InferShapeDeformableRoiPool);
} // namespace ops
