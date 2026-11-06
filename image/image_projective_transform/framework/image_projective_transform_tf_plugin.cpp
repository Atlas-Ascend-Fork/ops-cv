/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/register.h"
#include "cv_plugin_util.h"

#include "log/log.h"

#include <iostream>
namespace domi {
static Status ParseImageProjectiveTransform(const ge::Operator& op_src, ge::Operator& op)
{
    if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    ge::TensorDesc input_tensor = op.GetInputDesc("images");
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    if (op.UpdateInputDesc("images", input_tensor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input format failed.");
        return FAILED;
    }
    ge::TensorDesc output_tensor = op.GetOutputDesc("transformed_images");
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    if (op.UpdateOutputDesc("transformed_images", output_tensor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
        return FAILED;
    }
    return SUCCESS;
}
// register ImageProjectiveTransform op to GE
REGISTER_CUSTOM_OP("ImageProjectiveTransform")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ImageProjectiveTransformV2")
    .ParseParamsByOperatorFn(ParseImageProjectiveTransform)
    .ImplyType(ImplyType::TVM);
} // namespace domi
