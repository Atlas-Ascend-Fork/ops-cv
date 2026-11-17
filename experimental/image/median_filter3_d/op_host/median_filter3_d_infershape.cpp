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
#include "op_common/op_host/util/shape_util.h"
#include "median_filter3_d_validation.h"
namespace ops {
static ge::graphStatus InferMedianFilter3D(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputShape = ctx->GetInputShape(0);
    auto* outputShape = ctx->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t k[3], mode;
    float cval;
    if (!MedianFilter3DHost::Parse(ctx->GetAttrs(), k, mode, cval)) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*inputShape)) {
        Ops::Base::SetUnknownRank(*outputShape);
        return ge::GRAPH_SUCCESS;
    }
    if (!MedianFilter3DHost::ValidShape(*inputShape, true)) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferMedianFilter3DDtype(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr || !MedianFilter3DHost::Dtype(ctx->GetInputDataType(0))) {
        return ge::GRAPH_FAILED;
    }
    ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MedianFilter3D).InferShape(InferMedianFilter3D).InferDataType(InferMedianFilter3DDtype);
} // namespace ops
