/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>

#include "cv_plugin_util.h"

namespace domi {
namespace {
const char* const kDecodeClip = "decode_clip";
const char* const kPortX2 = "x2";

template <typename T>
bool ReadConst(const ge::Operator& node, const char* port, T& value)
{
    ge::Tensor data;
    if (node.GetInputConstData(port, data) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* buf = data.GetData();
    if ((buf == nullptr) || (data.GetSize() < sizeof(T))) {
        return false;
    }
    return memcpy_s(&value, sizeof(T), buf, sizeof(T)) == EOK;
}
} // namespace

static Status DecodeBboxParams(const ge::Operator& op_src, ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter DecodeBbox fusion parser.");

    float param = 0.0F;
    if (!ReadConst(op_src, kPortX2, param)) {
        OP_LOGE(GetOpName(op).c_str(), "Convert decode_clip data failed.");
        return PARAM_INVALID;
    }
    (void)op.SetAttr(kDecodeClip, param);
    OP_LOGI(GetOpName(op).c_str(), "decode_clip=%.1f", param);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DecodeBbox")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBbox")
    .ParseParamsByOperatorFn(DecodeBboxParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi
