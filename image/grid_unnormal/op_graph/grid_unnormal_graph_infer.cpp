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
 * \file grid_unnormal_graph_infer.cpp
 * \brief GridUnnormal graph dtype infer resource.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "grid_unnormal_proto.h"

namespace {
bool IsSupportedGridDtype(ge::DataType dtype) { return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT; }
} // namespace

namespace ge {
IMPLEMT_VERIFIER(GridUnnormal, VerifyGridUnnormal)
{
    const DataType gridDtype = op.GetInputDesc(0).GetDataType();
    const DataType assistDtype = op.GetInputDesc(1).GetDataType();
    if (gridDtype != assistDtype || !IsSupportedGridDtype(gridDtype)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(GridUnnormal, VerifyGridUnnormal);
} // namespace ge

namespace ops {
using namespace ge;

static constexpr int64_t kInputGridIdx = 0;
static constexpr int64_t kInputAssistIdx = 1;
static constexpr int64_t kOutputDiffIdx = 0;
static constexpr int64_t kOutputPosIdx = 1;

static ge::graphStatus InferDataTypeGridUnnormal(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeGridUnnormal");

    const ge::DataType gridDtype = context->GetInputDataType(kInputGridIdx);
    const ge::DataType assistDtype = context->GetInputDataType(kInputAssistIdx);
    if (gridDtype != assistDtype) {
        OP_LOGE(context->GetNodeName(), "grid and assist must have the same dtype, grid=%d, assist=%d",
                static_cast<int32_t>(gridDtype), static_cast<int32_t>(assistDtype));
        return GRAPH_FAILED;
    }
    if (!IsSupportedGridDtype(gridDtype)) {
        OP_LOGE(context->GetNodeName(), "grid dtype must be float16 or float32, grid=%d",
                static_cast<int32_t>(gridDtype));
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(kOutputDiffIdx, gridDtype) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "failed to set diff dtype");
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(kOutputPosIdx, ge::DT_INT32) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "failed to set position dtype");
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeGridUnnormal");
    return GRAPH_SUCCESS;
}

IMPL_OP(GridUnnormal).InferDataType(InferDataTypeGridUnnormal);

}; // namespace ops
