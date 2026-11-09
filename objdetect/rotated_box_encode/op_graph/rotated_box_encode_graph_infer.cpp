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
 * \file rotated_box_encode_graph_infer.cpp
 * \brief RotatedBoxEncode graph-level data-type inference
 */

// op_impl_registry.h: provides IMPL_OP macro, gert::InferDataTypeContext.
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "op_common/log/log.h"

// Use GE namespace for graphStatus and GRAPH_SUCCESS.
using namespace ge;

namespace ops {

// InferDataTypeForRotatedBoxEncode: GE data-type inference callback.
//   Per proto.md §2/§4: y.dtype == anchor_box.dtype == gt_box.dtype.  Rejects
//   mismatched input dtypes without setting the output dtype (stays DT_UNDEFINED).
static ge::graphStatus InferDataTypeForRotatedBoxEncode(gert::InferDataTypeContext* context)
{
    // proto.md §2/§4：y.dtype == anchor_box.dtype == gt_box.dtype（无提升，2 种 dtype 组合）。
    const ge::DataType anchorDtype = context->GetInputDataType(0);
    const ge::DataType gtDtype = context->GetInputDataType(1);
    if (anchorDtype != gtDtype) {
        OP_LOGE(context->GetNodeName(), "anchor_box and gt_box dtypes must be identical.");
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto* weight = attrs->GetAttrPointer<gert::TypedContinuousVector<float>>(0);
        if (weight != nullptr && weight->GetSize() != static_cast<size_t>(5)) {
            OP_LOGE(context->GetNodeName(), "weight length must be 5, but got %zu.", weight->GetSize());
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, anchorDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(RotatedBoxEncode).InferDataType(InferDataTypeForRotatedBoxEncode);

} // namespace ops
