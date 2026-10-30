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
 * \file resize_bilinear_v2_grad_infershape.cpp
 * \brief resize_bilinear_v2_grad_infershape
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "infershape_utils.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4ResizeBilinearV2Grad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do InferShape4ResizeBilinearV2Grad");
    const gert::Shape* grad_shape = context->GetInputShape(0);
    const gert::Shape* original_image_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, original_image_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*grad_shape) && grad_shape->GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "Input rank must be 4."), return GRAPH_FAILED);
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*original_image_shape) && original_image_shape->GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "Input rank must be 4."), return GRAPH_FAILED);

    auto image_desc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, image_desc);
    ge::Format input_format = image_desc->GetOriginFormat();

    OP_CHECK_IF(input_format != FORMAT_NHWC && input_format != FORMAT_NCHW,
                OP_LOGE(context->GetNodeName(), "input format only support [NHWC, NCHW], but is %s",
                        Ops::Base::ToString(input_format).c_str()),
                return GRAPH_FAILED);

    *y_shape = *original_image_shape;
    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "Do InferShape4ResizeBilinearV2Grad success");

    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDtype4ResizeBilinearV2Grad(gert::InferDataTypeContext* context)
{
    auto originalImageDtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, originalImageDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ResizeBilinearV2Grad)
    .InferShape(InferShape4ResizeBilinearV2Grad)
    .InferDataType(InferDtype4ResizeBilinearV2Grad);
} // namespace ops
