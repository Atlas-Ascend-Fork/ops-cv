/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stack_group_points_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace std;

namespace {
const uint8_t FEATURE_INDEX = 0;
const uint8_t FEATURE_BATCH_CNT_INDEX = 1;
const uint8_t INDICES_INDEX = 2;
const uint8_t INDICES_BATCH_CNT_INDEX = 3;
const uint8_t OUTPUT_INDEX = 0;

const uint8_t N_INDEX = 0;
const uint8_t C_INDEX = 1;
const uint8_t M_INDEX = 0;
const uint8_t NSAMPLE_INDEX = 1;

const uint8_t FIRST_DIM_INDEX = 0;
const uint8_t SECOND_DIM_INDEX = 1;
const uint8_t THIRD_DIM_INDEX = 2;
const uint8_t OUTPUT_DIM_NUM = 3;
const uint8_t TWO_NUM_DIM = 2;

static const char* OpNameOf(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetNodeName() == nullptr) {
        return "StackGroupPoints";
    }
    return context->GetNodeName();
}
} // namespace

namespace ops {
static ge::graphStatus ValidateAndFillStackGroupPointsOutput(
    gert::InferShapeContext* context, const gert::Shape* feture_shape, const gert::Shape* featuresBatchCntShape,
    const gert::Shape* indices_shape, const gert::Shape* indicesBatchCntShape, gert::Shape* output_shape)
{
    // 维度校验
    if (feture_shape->GetDimNum() != TWO_NUM_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "features",
                                     (std::to_string(feture_shape->GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }
    if (indices_shape->GetDimNum() != TWO_NUM_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices",
                                     (std::to_string(indices_shape->GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }
    if (featuresBatchCntShape->GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "features_batch_cnt",
                                     (std::to_string(featuresBatchCntShape->GetDimNum()) + "D").c_str(), "1D");
        return GRAPH_FAILED;
    }
    if (indicesBatchCntShape->GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices_batch_cnt",
                                     (std::to_string(indicesBatchCntShape->GetDimNum()) + "D").c_str(), "1D");
        return GRAPH_FAILED;
    }

    int32_t m = indices_shape->GetDim(M_INDEX);
    int32_t nsample = indices_shape->GetDim(NSAMPLE_INDEX);
    int32_t c = feture_shape->GetDim(C_INDEX);

    output_shape->SetDimNum(OUTPUT_DIM_NUM);
    output_shape->SetDim(FIRST_DIM_INDEX, m);
    output_shape->SetDim(SECOND_DIM_INDEX, c);
    output_shape->SetDim(THIRD_DIM_INDEX, nsample);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForStackGroupPoints(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE_WITHOUT_REPORT("StackGroupPoints", "InferShapeContext is nullptr");
        return GRAPH_FAILED;
    }
    const char* opName = OpNameOf(context);

    const gert::Shape* feture_shape = context->GetInputShape(FEATURE_INDEX);
    if (feture_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "features");
        return GRAPH_FAILED;
    }
    const gert::Shape* featuresBatchCntShape = context->GetInputShape(FEATURE_BATCH_CNT_INDEX);
    if (featuresBatchCntShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "features_batch_cnt");
        return GRAPH_FAILED;
    }
    const gert::Shape* indices_shape = context->GetInputShape(INDICES_INDEX);
    if (indices_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "indices");
        return GRAPH_FAILED;
    }
    const gert::Shape* indicesBatchCntShape = context->GetInputShape(INDICES_BATCH_CNT_INDEX);
    if (indicesBatchCntShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "indices_batch_cnt");
        return GRAPH_FAILED;
    }
    gert::Shape* output_shape = context->GetOutputShape(OUTPUT_INDEX);
    if (output_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "y");
        return GRAPH_FAILED;
    }

    // unknown rank：输出置 unknown rank 后直接返回
    if (Ops::Base::IsUnknownRank(*feture_shape) || Ops::Base::IsUnknownRank(*indices_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }
    // unknown shape：维度数已知但维度值未知，输出置 unknown shape 后返回
    if (Ops::Base::IsUnknownShape(*feture_shape) || Ops::Base::IsUnknownShape(*indices_shape)) {
        Ops::Base::SetUnknownShape(OUTPUT_DIM_NUM, *output_shape);
        return ge::GRAPH_SUCCESS;
    }

    return ValidateAndFillStackGroupPointsOutput(context, feture_shape, featuresBatchCntShape, indices_shape,
                                                 indicesBatchCntShape, output_shape);
}

IMPL_OP_INFERSHAPE(StackGroupPoints).InferShape(InferShapeForStackGroupPoints);
} // namespace ops
