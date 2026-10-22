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
 * \file col2im_onnx_plugin.cpp
 * \brief
 */

#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

using namespace ge;
using json = nlohmann::json;
namespace domi {
static void GetAttrListFromJson(json& attr, std::vector<int32_t>& val)
{
    if (!attr.contains("ints") || !attr["ints"].is_array()) {
        return;
    }
    for (size_t i = 0; i < attr["ints"].size(); ++i) {
        val.push_back(attr["ints"][i].get<int32_t>());
    }
}

static Status ParseOnnxParamsCol2im(const ge::Operator& op_src, ge::Operator& op_dest)
{
    AscendString attrs_string;
    std::vector<int32_t> kernel_size;
    std::vector<int32_t> dilation;
    std::vector<int32_t> padding;
    std::vector<int32_t> stride;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "kernel_size") {
                        GetAttrListFromJson(attr, kernel_size);
                    } else if (name == "dilation") {
                        GetAttrListFromJson(attr, dilation);
                    } else if (name == "padding") {
                        GetAttrListFromJson(attr, padding);
                    } else if (name == "stride") {
                        GetAttrListFromJson(attr, stride);
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

    if (kernel_size.empty() || dilation.empty() || padding.empty() || stride.empty()) {
        OP_LOGE(GetOpName(op_dest).c_str(), "node must have attr kernel_size/dilation/padding/stride");
        return FAILED;
    }
    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        return FAILED;
    }

    op_dest.SetAttr("kernel_size", kernel_size);
    op_dest.SetAttr("dilation", dilation);
    op_dest.SetAttr("padding", padding);
    op_dest.SetAttr("stride", stride);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Col2im")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Col2im"), ge::AscendString("ai.onnx::9::Col2im"),
                   ge::AscendString("ai.onnx::10::Col2im"), ge::AscendString("ai.onnx::11::Col2im"),
                   ge::AscendString("ai.onnx::12::Col2im"), ge::AscendString("ai.onnx::13::Col2im"),
                   ge::AscendString("ai.onnx::14::Col2im"), ge::AscendString("ai.onnx::15::Col2im"),
                   ge::AscendString("ai.onnx::16::Col2im"), ge::AscendString("ai.onnx::17::Col2im"),
                   ge::AscendString("ai.onnx::18::Col2im")})
    .ParseParamsByOperatorFn(ParseOnnxParamsCol2im)
    .ImplyType(ImplyType::TVM);
} // namespace domi
