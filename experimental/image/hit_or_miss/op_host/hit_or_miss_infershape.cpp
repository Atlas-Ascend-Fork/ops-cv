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
#include "hit_or_miss_utils.h"

namespace ops {
namespace {
ge::graphStatus InferShapeForHitOrMiss(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("HitOrMiss", "context is null");
        return ge::GRAPH_FAILED;
    }
    const auto* inputShape = context->GetInputShape(0);
    const auto* kernelShape = context->GetInputShape(1);
    auto* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || kernelShape == nullptr || outputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "input, kernel, or output shape is null");
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    const bool hasAxes = attrs != nullptr && attrs->GetAttrNum() > 0;
    const bool hasAnchor = attrs != nullptr && attrs->GetAttrNum() > 1;
    const auto* axesAttr = hasAxes ? attrs->GetListInt(0) : nullptr;
    const auto* anchorAttr = hasAnchor ? attrs->GetListInt(1) : nullptr;
    const int64_t defaultAxes[] = {-2, -1};
    const int64_t defaultAnchor[] = {-1, -1};

    hit_or_miss::CanonicalParams params;
    const char* error = hit_or_miss::CanonicalizeParams(
        *inputShape, *kernelShape, hasAxes ? (axesAttr == nullptr ? nullptr : axesAttr->GetData()) : defaultAxes,
        hasAxes ? (axesAttr == nullptr ? 0 : axesAttr->GetSize()) : 2,
        hasAnchor ? (anchorAttr == nullptr ? nullptr : anchorAttr->GetData()) : defaultAnchor,
        hasAnchor ? (anchorAttr == nullptr ? 0 : anchorAttr->GetSize()) : 2, true, params);
    if (error != nullptr) {
        OP_LOGE(context->GetNodeName(), "%s", error);
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForHitOrMiss(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输出数据类型固定为布尔类型，与输入类型无关。
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(HitOrMiss).InferShape(InferShapeForHitOrMiss).InferDataType(InferDataTypeForHitOrMiss);
} // namespace ops
