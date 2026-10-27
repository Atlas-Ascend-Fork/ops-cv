/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stack_group_points_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include <string>

using namespace ge;
using namespace std;

namespace {
const int64_t INPUT_INDEX_BBOXES = 0;
const int64_t INPUT_INDEX_GTBOXES = 1;
const int64_t OUTPUT_INDEX_OVERLAP = 0;
const int64_t IOUS_DIM = 2;
const int64_t ALIGNED_INFO_IDX = 2;
const int64_t INPUT_DIM_NUM = 2;
const int64_t BOX_COORDINATE_DIM = 4;

const char* OpNameOf(gert::InferShapeContext* context)
{
    const char* name = context->GetNodeName();
    return name == nullptr ? "IouV2" : name;
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeForIouV2(gert::InferShapeContext* context)
{
    auto overlapShape = context->GetOutputShape(OUTPUT_INDEX_OVERLAP);
    auto const bboxesShape = context->GetInputShape(INPUT_INDEX_BBOXES);
    auto const gtboxesShape = context->GetInputShape(INPUT_INDEX_GTBOXES);

    auto attrs = context->GetAttrs();

    if (overlapShape == nullptr || bboxesShape == nullptr || gtboxesShape == nullptr || attrs == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "Input, output shape, or attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const bool* aligned = attrs->GetAttrPointer<bool>(ALIGNED_INFO_IDX);
    if (aligned == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "aligned attr is nullptr");
        return ge::GRAPH_FAILED;
    }

    if (Ops::Base::IsUnknownRank(*bboxesShape) || Ops::Base::IsUnknownRank(*gtboxesShape)) {
        Ops::Base::SetUnknownRank(*overlapShape);
        return ge::GRAPH_SUCCESS;
    }
    const bool bboxesRankOk = bboxesShape->GetDimNum() == INPUT_DIM_NUM;
    const bool gtboxesRankOk = gtboxesShape->GetDimNum() == INPUT_DIM_NUM;
    if (!bboxesRankOk && !gtboxesRankOk) {
        const std::string dims = std::to_string(bboxesShape->GetDimNum()) + "D and " +
                                 std::to_string(gtboxesShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(OpNameOf(context), "bboxes, gtboxes", dims.c_str(),
                                                  "bboxes and gtboxes must be 2D");
        return ge::GRAPH_FAILED;
    }
    if (!bboxesRankOk) {
        const std::string dim = std::to_string(bboxesShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(OpNameOf(context), "bboxes", dim.c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (!gtboxesRankOk) {
        const std::string dim = std::to_string(gtboxesShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(OpNameOf(context), "gtboxes", dim.c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    const int64_t coordinateDim = *aligned ? bboxesShape->GetDim(0) : bboxesShape->GetDim(1);
    const int64_t gtCoordinateDim = *aligned ? gtboxesShape->GetDim(0) : gtboxesShape->GetDim(1);
    if (coordinateDim != ge::UNKNOWN_DIM && coordinateDim != BOX_COORDINATE_DIM) {
        const std::string actual = std::to_string(coordinateDim);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OpNameOf(context), "bboxes", actual.c_str(),
                                              "the coordinate dimension must be 4");
        return ge::GRAPH_FAILED;
    }
    if (gtCoordinateDim != ge::UNKNOWN_DIM && gtCoordinateDim != BOX_COORDINATE_DIM) {
        const std::string actual = std::to_string(gtCoordinateDim);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OpNameOf(context), "gtboxes", actual.c_str(),
                                              "the coordinate dimension must be 4");
        return ge::GRAPH_FAILED;
    }

    // update output shape.
    overlapShape->SetDimNum(IOUS_DIM); // the output dimensions are 2.
    const int64_t bboxesNum = *aligned ? bboxesShape->GetDim(1) : bboxesShape->GetDim(0);
    const int64_t gtboxesNum = *aligned ? gtboxesShape->GetDim(1) : gtboxesShape->GetDim(0);
    const int64_t outputDim1 = *aligned ? 1 : bboxesNum;
    if (*aligned && bboxesNum != ge::UNKNOWN_DIM && gtboxesNum != ge::UNKNOWN_DIM && bboxesNum != gtboxesNum) {
        const std::string actual = "[" + std::to_string(bboxesNum) + "] and [" + std::to_string(gtboxesNum) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OpNameOf(context), "bboxes, gtboxes", actual.c_str(),
            "when aligned is true, the number of bboxes and gtboxes must be the same");
        return ge::GRAPH_FAILED;
    }

    overlapShape->SetDim(0, gtboxesNum);
    overlapShape->SetDim(1, outputDim1);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForIouV2(gert::InferDataTypeContext* context)
{
    const ge::DataType bboxesDtype = context->GetInputDataType(INPUT_INDEX_BBOXES);
    const ge::DataType gtboxesDtype = context->GetInputDataType(INPUT_INDEX_GTBOXES);
    if (gtboxesDtype != bboxesDtype) {
        const char* name = context->GetNodeName();
        if (name == nullptr) {
            name = "IouV2";
        }
        const std::string actual = Ops::Base::ToString(bboxesDtype) + " and " + Ops::Base::ToString(gtboxesDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(name, "bboxes, gtboxes", actual.c_str(),
                                               "bboxes and gtboxes must have the same data type");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_INDEX_OVERLAP, bboxesDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IouV2).InferShape(InferShapeForIouV2).InferDataType(InferDataTypeForIouV2);
} // namespace ops
