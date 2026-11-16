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
 * \file paste_sub_img_infershape.cpp
 * \brief Infershape implementation for paste_sub_img operator
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_PATCH_IMG = 0;
static constexpr size_t IDX_PATCH_COORD = 1;
static constexpr size_t IDX_CORE_AREA_COORD = 2;
static constexpr size_t IDX_COMBINE_IMG = 3;
static constexpr size_t IDX_OUT = 0;
static constexpr int64_t EXPECTED_RANK_3D = 3;
static constexpr int64_t EXPECTED_RANK_1D = 1;
static constexpr int64_t COORD_LEN = 4;
static constexpr int64_t DIM_C = 2;

static ge::graphStatus InferShape4PasteSubImg(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShape4PasteSubImg");

    auto patchShape = context->GetInputShape(IDX_PATCH_IMG);
    OP_CHECK_NULL_WITH_CONTEXT(context, patchShape);
    auto patchCoordShape = context->GetInputShape(IDX_PATCH_COORD);
    OP_CHECK_NULL_WITH_CONTEXT(context, patchCoordShape);
    auto coreAreaCoordShape = context->GetInputShape(IDX_CORE_AREA_COORD);
    OP_CHECK_NULL_WITH_CONTEXT(context, coreAreaCoordShape);
    auto combineShape = context->GetInputShape(IDX_COMBINE_IMG);
    OP_CHECK_NULL_WITH_CONTEXT(context, combineShape);
    auto outShape = context->GetOutputShape(IDX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (Ops::Base::IsUnknownRank(*combineShape)) {
        Ops::Base::SetUnknownRank(*outShape);
        return GRAPH_SUCCESS;
    }

    // 与rt1.0一致：patch为未知rank时跳过其rank校验（输出仅由combine决定）；combine必须3D无条件校验
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*patchShape) && patchShape->GetDimNum() != EXPECTED_RANK_3D,
                OP_LOGE(context->GetNodeName(), "patch_img must be 3D"), return GRAPH_FAILED);
    OP_CHECK_IF(combineShape->GetDimNum() != EXPECTED_RANK_3D,
                OP_LOGE(context->GetNodeName(), "combine_img must be 3D"), return GRAPH_FAILED);
    // 与rt1.0一致：coord为未知rank或长度未知(-1)时跳过校验（rt1.0数据不可得时也跳过CheckCoord）
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*patchCoordShape) && patchCoordShape->GetDimNum() != EXPECTED_RANK_1D,
                OP_LOGE(context->GetNodeName(), "patch_coord must be 1D"), return GRAPH_FAILED);
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*coreAreaCoordShape) && coreAreaCoordShape->GetDimNum() != EXPECTED_RANK_1D,
                OP_LOGE(context->GetNodeName(), "core_area_coord must be 1D"), return GRAPH_FAILED);
    OP_CHECK_IF((!Ops::Base::IsUnknownRank(*patchCoordShape) && patchCoordShape->GetDim(0) != COORD_LEN &&
                 patchCoordShape->GetDim(0) != ge::UNKNOWN_DIM) ||
                    (!Ops::Base::IsUnknownRank(*coreAreaCoordShape) && coreAreaCoordShape->GetDim(0) != COORD_LEN &&
                     coreAreaCoordShape->GetDim(0) != ge::UNKNOWN_DIM),
                OP_LOGE(context->GetNodeName(), "patch_coord and core_area_coord length must be 4"),
                return GRAPH_FAILED);

    // 与rt1.0一致：patch为未知rank时跳过C维校验（输出仅由combine决定）
    if (!Ops::Base::IsUnknownRank(*patchShape)) {
        int64_t patchC = patchShape->GetDim(DIM_C);
        int64_t combineC = combineShape->GetDim(DIM_C);
        OP_CHECK_IF(patchC != ge::UNKNOWN_DIM && combineC != ge::UNKNOWN_DIM && patchC != combineC,
                    OP_LOGE(context->GetNodeName(), "C dimension of patch_img and combine_img must be the same"),
                    return GRAPH_FAILED);
    }

    *outShape = *combineShape;
    OP_LOGI(context->GetNodeName(), "End to do InferShape4PasteSubImg");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4PasteSubImg(gert::InferDataTypeContext* context)
{
    auto patchDtype = context->GetInputDataType(IDX_PATCH_IMG);
    auto combineDtype = context->GetInputDataType(IDX_COMBINE_IMG);
    OP_CHECK_IF(patchDtype != combineDtype,
                OP_LOGE(context->GetNodeName(), "patch_img and combine_img dtype must be the same"),
                return GRAPH_FAILED);
    context->SetOutputDataType(IDX_OUT, combineDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PasteSubImg)
    .InferShape(InferShape4PasteSubImg)
    .InputsDataDependency({IDX_PATCH_COORD, IDX_CORE_AREA_COORD})
    .InferDataType(InferDataType4PasteSubImg);
} // namespace ops
