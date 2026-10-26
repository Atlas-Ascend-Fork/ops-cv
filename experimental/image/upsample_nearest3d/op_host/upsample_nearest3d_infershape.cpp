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
 * \file upsample_nearest3d_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t OUT_Y = 0;
static constexpr size_t SUPPORTED_DIM_NUM = 5;        // N,C,D,H,W
static constexpr size_t SUPPORTED_OUTPUT_DIM_NUM = 3; // D,H,W
static constexpr size_t INDEX_OUTPUT_SIZE = 0;
static constexpr size_t INDEX_SCALES = 1;
static constexpr size_t NOT_CHANGE_DIM = 2;

static ge::graphStatus InferShape4UpsampleNearest3d(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "begin to do InferShape4UpsampleNearest3d");
    const gert::Shape* input_shape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
    gert::Shape* output_shape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const size_t input_dim = input_shape->GetDimNum();
    OP_CHECK_IF(input_dim != SUPPORTED_DIM_NUM,
                OP_LOGD(context->GetNodeName(), "Expected dim of input x should be 5. but get %zu", input_dim),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto output_size = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(INDEX_OUTPUT_SIZE);
    auto scales = attrs->GetAttrPointer<gert::TypedContinuousVector<float>>(INDEX_SCALES);

    *output_shape = *input_shape;

    if (output_size->GetSize() != 0 && scales->GetSize() == 0) {
        OP_CHECK_IF(
            output_size->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
            OP_LOGD(context->GetNodeName(), "attr::output_size dims must be 3, but get %ld", output_size->GetSize()),
            return ge::GRAPH_FAILED);
        // set output shape D,H,W dim
        auto output_size_data = reinterpret_cast<const int64_t*>(output_size->GetData());
        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            output_shape->SetDim(i + NOT_CHANGE_DIM, output_size_data[i]);
        }
    } else if (output_size->GetSize() == 0 && scales->GetSize() != 0) {
        OP_CHECK_IF(scales->GetSize() != SUPPORTED_OUTPUT_DIM_NUM,
                    OP_LOGD(context->GetNodeName(), "attr::scales dims must be 3, but get %ld", scales->GetSize()),
                    return ge::GRAPH_FAILED);
        // calculate and set output D,H,W dim
        auto scales_data = reinterpret_cast<const float*>(scales->GetData());
        for (size_t i = 0; i < SUPPORTED_OUTPUT_DIM_NUM; i++) {
            const int64_t input_dim_value = input_shape->GetDim(i + NOT_CHANGE_DIM);
            const int64_t output_dim_value = (input_dim_value == ge::UNKNOWN_DIM) ?
                                                 ge::UNKNOWN_DIM :
                                                 static_cast<int64_t>(input_dim_value * scales_data[i]);
            output_shape->SetDim(i + NOT_CHANGE_DIM, output_dim_value);
        }
    } else {
        OP_LOGE(context->GetNodeName(),
                "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus UpsampleNearest3dInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "begin to do UpsampleNearest3dInferDtype");
    auto xDtype = context->GetInputDataType(IN_X);
    context->SetOutputDataType(OUT_Y, xDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UpsampleNearest3d)
    .InferShape(InferShape4UpsampleNearest3d)
    .InferDataType(UpsampleNearest3dInferDtype);
IMPL_OP_INFERSHAPE(UpsampleNearestExact3d)
    .InferShape(InferShape4UpsampleNearest3d)
    .InferDataType(UpsampleNearest3dInferDtype);
} // namespace ops
