/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include <cstdint>
#include <limits>
#include <string>

namespace ops {
ge::graphStatus InferNmsV7(gert::InferShapeContext* c)
{
    auto* b = c->GetInputShape(0);
    auto* s = c->GetInputShape(1);
    auto* a = c->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(c, b);
    OP_CHECK_NULL_WITH_CONTEXT(c, s);
    OP_CHECK_NULL_WITH_CONTEXT(c, a);

    // Shape inference can see unknown dimensions. Validate dimensions that
    // are already known, while leaving the dynamic dimensions for tiling.
    const auto known_or_equal = [](int64_t lhs, int64_t rhs) {
        return lhs == ge::UNKNOWN_DIM || rhs == ge::UNKNOWN_DIM || lhs == rhs;
    };
    OP_CHECK_IF(
        b->GetDimNum() != 3 || s->GetDimNum() != 3 || (b->GetDim(2) != ge::UNKNOWN_DIM && b->GetDim(2) != 4) ||
            !known_or_equal(b->GetDim(0), s->GetDim(0)) || !known_or_equal(b->GetDim(1), s->GetDim(2)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(c->GetNodeName(), "boxes and scores",
                                               (Ops::Base::ToString(*b) + " and " + Ops::Base::ToString(*s)).c_str(),
                                               "boxes must be [B, N, 4] and scores must be [B, C, N]"),
        return ge::GRAPH_FAILED);

    const int64_t* center_point_box = a->GetAttrPointer<int64_t>(0);
    const int64_t* max_boxes_size = a->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(c, center_point_box);
    OP_CHECK_NULL_WITH_CONTEXT(c, max_boxes_size);
    OP_CHECK_IF(*center_point_box != 0 && *center_point_box != 1,
                OP_LOGE_FOR_INVALID_VALUE(c->GetNodeName(), "center_point_box",
                                          std::to_string(*center_point_box).c_str(), "0 or 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*max_boxes_size < 0 || *max_boxes_size > std::numeric_limits<int32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(c->GetNodeName(), "max_boxes_size",
                                                      std::to_string(*max_boxes_size).c_str(),
                                                      "max_boxes_size must be in [0, INT32_MAX]"),
                return ge::GRAPH_FAILED);

    auto* o = c->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(c, o);
    o->SetDimNum(2);
    o->SetDim(0, *max_boxes_size);
    o->SetDim(1, 3);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferNmsV7Type(gert::InferDataTypeContext* c)
{
    OP_CHECK_NULL_WITH_CONTEXT(c, c);
    c->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NonMaxSuppressionV7).InferShape(InferNmsV7).InferDataType(InferNmsV7Type);
} // namespace ops
