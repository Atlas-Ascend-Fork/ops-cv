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

// required attr judge
static const int req_attr_num = 3;
static Status ParseParamsNpuRoiAlign(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float spatial_scale = 0.0;
    int pooled_height = 0;
    int pooled_width = 0;
    int sample_num = 2;
    int roi_end_mode = 1;
    // The initialization of required attributes count
    int req_attr_count = 0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i/f when the scalar is zero. sample_num defaults to 2 and roi_end_mode to 1.
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
                        ++req_attr_count;
                    } else if (name == "pooled_height" && type == ATTR_TYPE_INT) {
                        pooled_height = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("pooled_height", pooled_height);
                        ++req_attr_count;
                    } else if (name == "pooled_width" && type == ATTR_TYPE_INT) {
                        pooled_width = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("pooled_width", pooled_width);
                        ++req_attr_count;
                    } else if (name == "sample_num" && type == ATTR_TYPE_INT) {
                        sample_num = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "roi_end_mode" && type == ATTR_TYPE_INT) {
                        roi_end_mode = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
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

    // Node must have 3 required attributes.
    if (req_attr_count != req_attr_num) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr spatial_scale, pooled_height, pooled_width");
        return FAILED;
    }
    op_dest.SetAttr("sample_num", sample_num);
    op_dest.SetAttr("roi_end_mode", roi_end_mode);

    return SUCCESS;
}

// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("ROIAlign")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPURoiAlign"), ge::AscendString("ai.onnx::11::NPURoiAlign"),
                   ge::AscendString("ai.onnx::12::NPURoiAlign"), ge::AscendString("ai.onnx::13::NPURoiAlign"),
                   ge::AscendString("ai.onnx::14::NPURoiAlign"), ge::AscendString("ai.onnx::15::NPURoiAlign"),
                   ge::AscendString("ai.onnx::16::NPURoiAlign"), ge::AscendString("ai.onnx::17::NPURoiAlign"),
                   ge::AscendString("ai.onnx::18::NPURoiAlign")})
    .ParseParamsByOperatorFn(ParseParamsNpuRoiAlign)
    .ImplyType(ImplyType::TVM);
} // namespace domi
