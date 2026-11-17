/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext* c)
{
    if (!c || !c->GetInputShape(0) || !c->GetOutputShape(0))
        return ge::GRAPH_FAILED;
    *c->GetOutputShape(0) = *c->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext* c)
{
    if (!c)
        return ge::GRAPH_FAILED;
    c->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(LaplacianOfGaussian).InferShape(Infer).InferDataType(Dtype);
} // namespace ops
