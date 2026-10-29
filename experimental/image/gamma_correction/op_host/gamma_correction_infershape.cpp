/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file gamma_correction_infershape.cpp
 *  \brief Shape and dtype inference for GammaCorrection.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t IN_X = 0;
constexpr size_t OUT_Y = 0;
constexpr size_t MIN_RANK = 2;
constexpr size_t MAX_RANK = 4;
} // namespace

static ge::graphStatus InferShapeGammaCorrection(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    gert::Shape* output = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);

    const size_t rank = input->GetDimNum();
    OP_CHECK_IF(rank < MIN_RANK || rank > MAX_RANK, OP_LOGE(context, "x rank must be 2, 3 or 4, got %zu", rank),
                return ge::GRAPH_FAILED);

    output->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        output->SetDim(i, input->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGammaCorrection(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUT_Y, context->GetInputDataType(IN_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GammaCorrection).InferShape(InferShapeGammaCorrection).InferDataType(InferDataTypeGammaCorrection);
} // namespace ops
