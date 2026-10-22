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
 * \file nms_with_mask_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
// ---------------- NMSWithMask Op-------------------
static graphStatus InferShape4NMSWithMask(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do NMSWithMaskInferShape");
    const gert::Shape* input_scores_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_scores_shape);

    constexpr size_t output_num = 3;
    for (size_t i = 0; i < output_num; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
        if (i == 0) {
            constexpr int OUTPUT_DIM_NUM = 2;
            output_shape->SetDimNum(OUTPUT_DIM_NUM);
            output_shape->SetDim(0, input_scores_shape->GetDim(0));
            constexpr int OUTPUT_DIM_VALUE = 5;
            output_shape->SetDim(1, OUTPUT_DIM_VALUE);
        } else {
            output_shape->SetDimNum(1);
            output_shape->SetDim(0, input_scores_shape->GetDim(0));
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do NMSWithMaskInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NMSWithMask(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do NMSWithMaskInferDataType");
    const ge::DataType inputDtype = context->GetInputDataType(0);
    constexpr size_t outputNum = 3;
    for (size_t i = 0; i < outputNum; i++) {
        if (i == 0) {
            context->SetOutputDataType(i, inputDtype);
        } else if (i == 1) {
            context->SetOutputDataType(i, ge::DT_INT32);
        } else {
            context->SetOutputDataType(i, ge::DT_UINT8);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do NMSWithMaskInferDataType");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NMSWithMask).InferShape(InferShape4NMSWithMask).InferDataType(InferDataType4NMSWithMask);
// ---------------- NMSWithMask Op END---------------------
} //  namespace ops
