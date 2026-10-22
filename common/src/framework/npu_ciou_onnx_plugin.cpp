/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuCiou(const ge::Operator& op_src, ge::Operator& op_dest)
{
    bool trans = false;
    bool is_cross = true;
    std::string mode = "iou";
    bool atan_sub_flag = false;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i when the int is zero. is_cross defaults to true, so a missing i must become false.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "trans" && type == ATTR_TYPE_INT) {
                        trans = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "is_cross" && type == ATTR_TYPE_INT) {
                        is_cross = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "mode" && type == ATTR_TYPE_INT) {
                        mode = (attr.contains("i") && attr["i"].get<int64_t>() == 1) ? "iof" : "iou";
                    } else if (name == "atan_sub_flag" && type == ATTR_TYPE_INT) {
                        atan_sub_flag = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    }
                }
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }

    op_dest.SetAttr("trans", trans);
    op_dest.SetAttr("is_cross", is_cross);
    op_dest.SetAttr("mode", mode);
    op_dest.SetAttr("atan_sub_flag", atan_sub_flag);
    return SUCCESS;
}

// register npu_ciou op info to GE
REGISTER_CUSTOM_OP("CIoU")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUCiou"), ge::AscendString("ai.onnx::11::NPUCiou"),
                   ge::AscendString("ai.onnx::12::NPUCiou"), ge::AscendString("ai.onnx::13::NPUCiou"),
                   ge::AscendString("ai.onnx::14::NPUCiou"), ge::AscendString("ai.onnx::15::NPUCiou"),
                   ge::AscendString("ai.onnx::16::NPUCiou"), ge::AscendString("ai.onnx::17::NPUCiou"),
                   ge::AscendString("ai.onnx::18::NPUCiou")})
    .ParseParamsByOperatorFn(ParseParamsNpuCiou)
    .ImplyType(ImplyType::TVM);
} // namespace domi
