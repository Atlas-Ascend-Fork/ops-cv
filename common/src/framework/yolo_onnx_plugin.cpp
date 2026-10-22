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

/*!
 * \file yolo_onnx_plugin.cpp
 * \brief
 */

#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsYolo(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int boxes = 3;
    int coords = 4;
    int classes = 80;
    std::string yolo_version = "V3";
    bool softmax = false;
    bool background = false;
    bool softmaxtree = false;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i when the int is zero. Do not keep the non-zero defaults.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "boxes" && type == ATTR_TYPE_INT) {
                        boxes = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "coords" && type == ATTR_TYPE_INT) {
                        coords = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "classes" && type == ATTR_TYPE_INT) {
                        classes = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "yolo_version" && attr.contains("s")) {
                        yolo_version = attr["s"];
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

    op_dest.SetAttr("boxes", boxes);
    op_dest.SetAttr("coords", coords);
    op_dest.SetAttr("classes", classes);
    op_dest.SetAttr("yolo_version", yolo_version);
    op_dest.SetAttr("softmax", softmax);
    op_dest.SetAttr("background", background);
    op_dest.SetAttr("softmaxtree", softmaxtree);

    return SUCCESS;
}

// register Yolo op info to GE
REGISTER_CUSTOM_OP("Yolo")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Yolo"), ge::AscendString("ai.onnx::9::Yolo"),
                   ge::AscendString("ai.onnx::10::Yolo"), ge::AscendString("ai.onnx::11::Yolo"),
                   ge::AscendString("ai.onnx::12::Yolo"), ge::AscendString("ai.onnx::13::Yolo"),
                   ge::AscendString("ai.onnx::14::Yolo"), ge::AscendString("ai.onnx::15::Yolo"),
                   ge::AscendString("ai.onnx::16::Yolo")})
    .ParseParamsByOperatorFn(ParseParamsYolo)
    .ImplyType(ImplyType::TVM);
} // namespace domi
