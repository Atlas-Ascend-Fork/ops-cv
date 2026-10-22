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
 * \file npu_ps_roi_pooling_onnx_plugin.cpp
 * \brief
 */
#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuPsRoiPooling(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float spatial_scale = 0.0;
    int output_dim = 1;
    int group_size = 0;
    int attr_num = 0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i/f when the scalar is zero. Count the attr once name/type match.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "spatial_scale" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            spatial_scale = 0.0f;
                        } else {
                            std::string spatial_scale_str = attr["f"];
                            if (StrToFloat(spatial_scale_str, spatial_scale) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse spatial_scale failed.");
                                return FAILED;
                            }
                        }
                        op_dest.SetAttr("spatial_scale", spatial_scale);
                        ++attr_num;
                    } else if (name == "output_dim" && type == ATTR_TYPE_INT) {
                        output_dim = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("output_dim", output_dim);
                        ++attr_num;
                    } else if (name == "group_size" && type == ATTR_TYPE_INT) {
                        group_size = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("group_size", group_size);
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

    // Node must have 3 required attr.
    if (attr_num != 3) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr spatial_scale, output_dim, group_size");
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("PSROIPoolingV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUPsRoiPooling"), ge::AscendString("ai.onnx::11::NPUPsRoiPooling"),
                   ge::AscendString("ai.onnx::12::NPUPsRoiPooling"), ge::AscendString("ai.onnx::13::NPUPsRoiPooling"),
                   ge::AscendString("ai.onnx::14::NPUPsRoiPooling"), ge::AscendString("ai.onnx::15::NPUPsRoiPooling"),
                   ge::AscendString("ai.onnx::16::NPUPsRoiPooling"), ge::AscendString("ai.onnx::17::NPUPsRoiPooling"),
                   ge::AscendString("ai.onnx::18::NPUPsRoiPooling")})
    .ParseParamsByOperatorFn(ParseParamsNpuPsRoiPooling)
    .ImplyType(ImplyType::TVM);
} // namespace domi
