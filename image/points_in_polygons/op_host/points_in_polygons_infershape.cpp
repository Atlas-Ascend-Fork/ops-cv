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
 * \file points_in_polygons_infershape.cpp
 * \brief PointsInPolygons shape inference: output.shape = (N, M)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {

static constexpr size_t POINTS_INDEX = 0U;
static constexpr size_t POLYGONS_INDEX = 1U;
static constexpr size_t OUTPUT_INDEX = 0U;
static constexpr size_t INPUT_RANK = 2U;

static ge::graphStatus InferShape4PointsInPolygons(gert::InferShapeContext* context)
{
    const gert::Shape* pointsShape = context->GetInputShape(POINTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pointsShape);
    const gert::Shape* polygonsShape = context->GetInputShape(POLYGONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, polygonsShape);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // -2 (unknown rank): skip checks and set output to unknown rank
    if (Ops::Base::IsUnknownRank(*pointsShape) || Ops::Base::IsUnknownRank(*polygonsShape)) {
        Ops::Base::SetUnknownRank(*outputShape);
        return ge::GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownShape(*pointsShape) || Ops::Base::IsUnknownShape(*polygonsShape)) {
        Ops::Base::SetUnknownShape(INPUT_RANK, *outputShape);
        return ge::GRAPH_SUCCESS;
    }

    // output.shape = (points.shape[0], polygons.shape[1]) = (N, M)
    *outputShape = {pointsShape->GetDim(0), polygonsShape->GetDim(1)};

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PointsInPolygons).InferShape(InferShape4PointsInPolygons);

} // namespace ops
