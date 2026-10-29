/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {

static constexpr size_t KERNEL_SIZE_ATTR_INDEX = 0;
static constexpr size_t SIGMA_SPACE_ATTR_INDEX = 1;
static constexpr size_t SIGMA_COLOR_ATTR_INDEX = 2;

static bool IsSupportedKernel(int64_t kernel) { return kernel == 3 || kernel == 5 || kernel == 7 || kernel == 9; }

static ge::graphStatus InferShapeBilateralFilter3D(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_IF(inputShape->GetDimNum() < 3U || inputShape->GetDimNum() > 5U,
                OP_LOGE(context, "BilateralFilter3D only supports rank 3/4/5 ND tensors."), return ge::GRAPH_FAILED);
    // A negative dim is an unknown-shape placeholder and must pass through so the
    // operator stays usable in graph mode (DynamicShapeSupportFlag is on); only an
    // explicit 0 denotes an empty tensor, which is unsupported.
    for (size_t i = 0; i < inputShape->GetDimNum(); ++i) {
        OP_CHECK_IF(inputShape->GetDim(i) == 0, OP_LOGE(context, "BilateralFilter3D does not support empty tensors."),
                    return ge::GRAPH_FAILED);
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kernelSize = attrs->GetInt(KERNEL_SIZE_ATTR_INDEX);
    const float* sigmaSpace = attrs->GetFloat(SIGMA_SPACE_ATTR_INDEX);
    const float* sigmaColor = attrs->GetFloat(SIGMA_COLOR_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kernelSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaSpace);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaColor);
    OP_CHECK_IF(!IsSupportedKernel(*kernelSize),
                OP_LOGE(context, "BilateralFilter3D supports kernel_size 3/5/7/9, got %ld.", *kernelSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(std::isfinite(*sigmaSpace) && *sigmaSpace > 0.0F),
                OP_LOGE(context, "sigma_space must be a positive finite float."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(std::isfinite(*sigmaColor) && *sigmaColor > 0.0F),
                OP_LOGE(context, "sigma_color must be a positive finite float."), return ge::GRAPH_FAILED);

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBilateralFilter3D(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BilateralFilter3D)
    .InferShape(InferShapeBilateralFilter3D)
    .InferDataType(InferDataTypeBilateralFilter3D);

} // namespace ops
