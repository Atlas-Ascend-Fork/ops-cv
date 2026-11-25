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
 * \file points_in_polygons_graph_infer.cpp
 * \brief PointsInPolygons graph-level data type inference
 */

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferDataTypeForPointsInPolygons(gert::InferDataTypeContext* context)
{
    const ge::DataType pointsDataType = context->GetInputDataType(0);
    const ge::DataType polygonsDataType = context->GetInputDataType(1);

    if (pointsDataType != ge::DT_FLOAT || polygonsDataType != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "PointsInPolygons", "points, polygons",
            Ops::Base::ToString(pointsDataType) + ", " + Ops::Base::ToString(polygonsDataType),
            "points and polygons must be DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (pointsDataType != polygonsDataType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "PointsInPolygons", "points, polygons",
            Ops::Base::ToString(pointsDataType) + ", " + Ops::Base::ToString(polygonsDataType),
            "points and polygons must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, pointsDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(PointsInPolygons).InferDataType(InferDataTypeForPointsInPolygons);
} // namespace ops
