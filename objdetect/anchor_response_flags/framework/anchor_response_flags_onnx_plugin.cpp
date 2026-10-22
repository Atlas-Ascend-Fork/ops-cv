/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

// Required attr judge
static const int REQ_ATTR_NUM = 1;
static Status ParseParamsNpuAnchorResponseFlags(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::vector<int> v_featmap_size = {};
    std::vector<int> v_strides = {};
    int num_base_anchors = 0;
    // The initialization of required attribute count
    int req_attr_num = 0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "featmap_sizes" && attr.contains("ints")) {
                        for (auto& v : attr["ints"]) {
                            v_featmap_size.push_back(static_cast<int>(v.get<int64_t>()));
                        }
                    } else if (name == "strides" && attr.contains("ints")) {
                        for (auto& v : attr["ints"]) {
                            v_strides.push_back(static_cast<int>(v.get<int64_t>()));
                        }
                    } else if (name == "num_base_anchors" && attr.value("type", -1) == ATTR_TYPE_INT) {
                        // GE omits i when the int is zero. Still count the required attribute.
                        num_base_anchors = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("num_base_anchors", num_base_anchors);
                        ++req_attr_num;
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

    if (v_featmap_size.empty() || v_strides.empty() || req_attr_num != REQ_ATTR_NUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr featmap_size, strides, num_base_anchors");
        return FAILED;
    }

    op_dest.SetAttr("featmap_size", v_featmap_size);
    op_dest.SetAttr("strides", v_strides);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("AnchorResponseFlags")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::11::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::12::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::13::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::14::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::15::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::16::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::17::NPUAnchorResponseFlags"),
                   ge::AscendString("ai.onnx::18::NPUAnchorResponseFlags")})
    .ParseParamsByOperatorFn(ParseParamsNpuAnchorResponseFlags)
    .ImplyType(ImplyType::TVM);
} // namespace domi
