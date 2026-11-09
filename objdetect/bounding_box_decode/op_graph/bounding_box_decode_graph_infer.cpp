/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

#include <string>

#include "log/log.h"

using namespace ge;

namespace ops {

/**
 * InferDataTypeForBoundingBoxDecode: GE data-type inference callback.
 *
 * boxes.dtype = anchor_box.dtype (same_as_first_input).  Validates that both
 * inputs have the same dtype (DESIGN §3.2, no type promotion).
 */
static ge::graphStatus InferDataTypeForBoundingBoxDecode(gert::InferDataTypeContext* context)
{
    const ge::DataType anchorDtype = context->GetInputDataType(0);
    const ge::DataType deltasDtype = context->GetInputDataType(1);
    if (anchorDtype != deltasDtype) {
        const std::string dtypeMsg = Ops::Base::ToString(anchorDtype) + " and " + Ops::Base::ToString(deltasDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "rois and deltas", dtypeMsg.c_str(),
                                               "rois and deltas must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, anchorDtype);
    return ge::GRAPH_SUCCESS;
}

/**
 * IMPL_OP(BoundingBoxDecode).InferDataType(...): registers the dtype inference
 *   function for the operator named "BoundingBoxDecode" at static init time.
 *   When GE encounters a BoundingBoxDecode node during graph compilation, it
 *   calls InferDataTypeForBoundingBoxDecode to determine the output type.
 *   This static registration is required by the GE runtime to locate the op
 *   implementation via the op_impl_registry.
 */
IMPL_OP(BoundingBoxDecode).InferDataType(InferDataTypeForBoundingBoxDecode);

} // namespace ops
