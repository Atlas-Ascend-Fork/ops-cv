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
 * \file lbp_feature_infershape.cpp
 * \brief LbpFeature InferShape: output is the input plane cropped by 2*radius on H/W.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeLbpFeature(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeLbpFeature");

    // 获取并校验输入 / 输出 shape 指针
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // 获取 attrs 中的 radius 属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 兼容属性指针获取
    int64_t radius = 0;
    const int64_t* radiusPtr = attrs->GetAttrPointer<int64_t>(0); // 第 0 个属性是 radius
    if (radiusPtr != nullptr) {
        radius = *radiusPtr;
    }

    // 【新增守卫①】radius < 0 会让输出比输入还大（6-2×(-1)=8），必须拒绝
    // 与 tiling 侧的 radius<0 守卫对齐，避免 infershape 放行、tiling 拒绝的割裂
    if (radius < 0) {
        OP_LOGE(context->GetNodeName(), "radius must be >= 0, but got %lld", radius);
        return ge::GRAPH_FAILED;
    }

    // 填充输出 shape 大小
    auto inputShapeSize = inputShape->GetDimNum();
    if (inputShapeSize < 2) {
        OP_LOGE(context->GetNodeName(), "Input dim num must be >= 2, but got %zu", inputShapeSize);
        return ge::GRAPH_FAILED;
    }

    // 【新增守卫②】前导维（batch 部分）任一维 <= 0 → 拒绝
    // 覆盖 batchCount=0 场景（如首维为 0），与 tiling 侧 batchCount<=0 守卫对齐。
    // 注意：无此检查时，{0,3,63,65} 会静默推出 {0,3,61,63} 并返回 SUCCESS
    for (size_t i = 0; i + 2 < inputShapeSize; ++i) {
        if (inputShape->GetDim(i) <= 0) {
            OP_LOGE(context->GetNodeName(), "Batch dim[%zu] must be positive, but got %lld", i, inputShape->GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }

    outputShape->SetDimNum(inputShapeSize);
    for (size_t i = 0; i < inputShapeSize; i++) {
        int64_t dim = inputShape->GetDim(i);
        // 最后两维是 H 和 W，需要减去 2 * radius
        if (i >= inputShapeSize - 2) {
            int64_t newDim = dim - 2 * radius;
            // newDim <= 0（含等号边界）等价于 tiling 的 H <= 2*radius / W <= 2*radius 守卫
            if (newDim <= 0) {
                OP_LOGE(context->GetNodeName(), "Output dim[%zu] is non-positive: %lld (input %lld - 2*radius %lld)", i,
                        newDim, dim, radius);
                return ge::GRAPH_FAILED;
            }
            outputShape->SetDim(i, newDim);
        } else {
            outputShape->SetDim(i, dim);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeLbpFeature");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LbpFeature).InferShape(InferShapeLbpFeature);
} // namespace ops
