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
 * \file gaussian_filter_3_d_infershape.cpp
 * \brief GaussianFilter3D算子shape/dtype推导：y 与 x 完全同 shape 同 dtype
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static constexpr int INPUT_X_INDEX = 0;
static constexpr int OUTPUT_Y_INDEX = 0;
using namespace ge;

static graphStatus InferShape4GaussianFilter3D(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GaussianFilter3D(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4GaussianFilter3D");
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    OP_LOGD(context, "End to do InferDataType4GaussianFilter3D");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GaussianFilter3D)
    .InferShape(InferShape4GaussianFilter3D)
    .InferDataType(InferDataType4GaussianFilter3D);
} // namespace ops
