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
 * \file psroi_poolingV2_onnx_plugin.cpp
 * \brief
 */

#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "cv_plugin_util.h"
#include "register/register.h"

using namespace ge;
using json = nlohmann::json;
namespace domi {
static const size_t ATTR_NUM = 3;

static Status ParsePSROIPoolingV2Attributes(const ge::Operator& op_src, ge::Operator& op_dest)
{
    AscendString attrs_string;
    float spatial_scale = 0;
    int output_dim = 0;
    int group_size = 0;
    int attr_num = 0;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    const std::string attr_name = attr.value("name", "");
                    if (attr_name == "spatial_scale") {
                        if (attr.contains("f")) {
                            std::string spatial = attr["f"];
                            if (StrToFloat(spatial, spatial_scale) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse spatial_scale failed.");
                                return FAILED;
                            }
                        } else {
                            spatial_scale = 0.0f;
                        }
                        ++attr_num;
                    } else if (attr_name == "output_dim") {
                        output_dim = attr.contains("i") ? attr["i"].get<int>() : 0;
                        ++attr_num;
                    } else if (attr_name == "group_size") {
                        group_size = attr.contains("i") ? attr["i"].get<int>() : 0;
                        ++attr_num;
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

    if (attr_num != ATTR_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetOpName(op_dest).c_str(), "attrs", std::to_string(attr_num).c_str(),
                                              "Node must have attr spatial_scale, output_dim, group_size");
        return FAILED;
    }
    op_dest.SetAttr("spatial_scale", spatial_scale);
    op_dest.SetAttr("output_dim", output_dim);
    op_dest.SetAttr("group_size", group_size);
    return SUCCESS;
}

static Status ParseOnnxParamsPSROIPoolingV2(const ge::Operator& op_src, ge::Operator& op_dest)
{
    if (ParsePSROIPoolingV2Attributes(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }

    if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(op_dest, 1, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        OP_LOGE(GetOpName(op_dest).c_str(), "ChangeFormatFromOnnx failed.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PSROIPoolingV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::PSROIPooling"), ge::AscendString("ai.onnx::9::PSROIPooling"),
                   ge::AscendString("ai.onnx::10::PSROIPooling"), ge::AscendString("ai.onnx::11::PSROIPooling"),
                   ge::AscendString("ai.onnx::12::PSROIPooling"), ge::AscendString("ai.onnx::13::PSROIPooling"),
                   ge::AscendString("ai.onnx::14::PSROIPooling"), ge::AscendString("ai.onnx::15::PSROIPooling"),
                   ge::AscendString("ai.onnx::16::PSROIPooling")})
    .ParseParamsByOperatorFn(ParseOnnxParamsPSROIPoolingV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
