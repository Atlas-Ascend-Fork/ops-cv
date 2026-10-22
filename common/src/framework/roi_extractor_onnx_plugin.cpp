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
 * \file roi_extractor_onnx_plugin.cpp
 * \brief
 */

#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static const uint32_t MIN_INPUT_NUM = 2;

static Status ParseFloatsAttr(json& attr, std::vector<float>& values, const ge::Operator& op_dest)
{
    if (!attr.contains("floats") || !attr["floats"].is_array()) {
        return SUCCESS;
    }
    for (auto& item : attr["floats"]) {
        float val = 0.0f;
        if (item.is_string()) {
            if (StrToFloat(item.get<std::string>(), val) != SUCCESS) {
                OP_LOGE(GetOpName(op_dest).c_str(), "parse floats attr failed.");
                return FAILED;
            }
        } else if (item.is_number()) {
            val = item.get<float>();
        } else {
            OP_LOGE(GetOpName(op_dest).c_str(), "parse floats attr failed.");
            return FAILED;
        }
        values.push_back(val);
    }
    return SUCCESS;
}

static Status ParseParamsRoiExtractor(const ge::Operator& op_src, ge::Operator& op_dest)
{
    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i/f when the scalar is zero. Match name/type and still publish 0.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "finest_scale" && type == ATTR_TYPE_INT) {
                        int64_t finest_scale = attr.contains("i") ? attr["i"].get<int64_t>() : 0;
                        op_dest.SetAttr("finest_scale", finest_scale);
                    }
                    if (name == "roi_scale_factor" && type == ATTR_TYPE_FLOAT) {
                        float roi_scale_factor = 0.0f;
                        if (attr.contains("f")) {
                            std::string roi_scale_factor_str = attr["f"];
                            if (StrToFloat(roi_scale_factor_str, roi_scale_factor) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse roi_scale_factor failed.");
                                return FAILED;
                            }
                        }
                        op_dest.SetAttr("roi_scale_factor", roi_scale_factor);
                    }
                    if (name == "spatial_scale") {
                        std::vector<float> spatial_scale;
                        if (ParseFloatsAttr(attr, spatial_scale, op_dest) != SUCCESS) {
                            return FAILED;
                        }
                        if (!spatial_scale.empty()) {
                            op_dest.SetAttr("spatial_scale", spatial_scale);
                        }
                    }
                    if (name == "pooled_height" && type == ATTR_TYPE_INT) {
                        int64_t output_size = attr.contains("i") ? attr["i"].get<int64_t>() : 0;
                        op_dest.SetAttr("pooled_height", output_size);
                    }
                    if (name == "pooled_width" && type == ATTR_TYPE_INT) {
                        int64_t output_size = attr.contains("i") ? attr["i"].get<int64_t>() : 0;
                        op_dest.SetAttr("pooled_width", output_size);
                    }
                    if (name == "sample_num" && type == ATTR_TYPE_INT) {
                        int64_t sample_num = attr.contains("i") ? attr["i"].get<int64_t>() : 0;
                        op_dest.SetAttr("sample_num", sample_num);
                    }
                    if (name == "pool_mode" && attr.contains("s")) {
                        std::string pool_mode = attr["s"];
                        op_dest.SetAttr("pool_mode", pool_mode);
                    }
                    if (name == "aligned" && type == ATTR_TYPE_INT) {
                        bool aligned = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                        op_dest.SetAttr("aligned", aligned);
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

    uint32_t input_num = static_cast<uint32_t>(op_src.GetInputsSize());
    if (input_num < MIN_INPUT_NUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "input num must be greater than or equal to 2, but got %u", input_num);
        return FAILED;
    }
    op_dest.DynamicInputRegister("features", input_num - 1, false);
    return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("RoiExtractor")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::RoiExtractor"), ge::AscendString("ai.onnx::9::RoiExtractor"),
                   ge::AscendString("ai.onnx::10::RoiExtractor"), ge::AscendString("ai.onnx::11::RoiExtractor"),
                   ge::AscendString("ai.onnx::12::RoiExtractor"), ge::AscendString("ai.onnx::13::RoiExtractor"),
                   ge::AscendString("ai.onnx::14::RoiExtractor"), ge::AscendString("ai.onnx::15::RoiExtractor"),
                   ge::AscendString("ai.onnx::16::RoiExtractor")})
    .ParseParamsByOperatorFn(ParseParamsRoiExtractor)
    .ImplyType(ImplyType::TVM);
} // namespace domi
