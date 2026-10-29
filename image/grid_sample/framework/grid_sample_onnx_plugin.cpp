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

static Status ParseParamsGridSample(const ge::Operator& op_src, ge::Operator& op_dest)
{
    bool align_corners = false;
    std::string mode_value = "bilinear";
    std::string padding_mode_value = "zeros";

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i when the int is zero.
                    const std::string name = attr.value("name", "");
                    if (name == "align_corners" && attr.value("type", -1) == ATTR_TYPE_INT) {
                        int64_t align_corners_i = attr.contains("i") ? attr["i"].get<int64_t>() : 0;
                        if (align_corners_i != 0) {
                            if (align_corners_i > 1 || align_corners_i < -1) {
                                OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(),
                                                       "the value of align_corners [%ld] is out of range [-1, 1]",
                                                       align_corners_i);
                            }
                            align_corners = true;
                        }
                    } else if (name == "mode" && attr.contains("s")) {
                        mode_value = attr["s"];
                    } else if (name == "padding_mode" && attr.contains("s")) {
                        padding_mode_value = attr["s"];
                    }
                }
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }

    auto orgTensorX = op_dest.GetInputDescByName("x");
    orgTensorX.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorX.SetFormat(ge::FORMAT_NCHW);
    op_dest.UpdateInputDesc("x", orgTensorX);

    auto orgTensorY = op_dest.GetInputDescByName("y");
    orgTensorY.SetOriginFormat(ge::FORMAT_NCHW);
    orgTensorY.SetFormat(ge::FORMAT_NCHW);
    op_dest.UpdateOutputDesc("y", orgTensorY);

    op_dest.SetAttr("interpolation_mode", mode_value);
    op_dest.SetAttr("align_corners", align_corners);
    op_dest.SetAttr("padding_mode", padding_mode_value);

    return SUCCESS;
}

static Status ParseParamsGridSampleV2(const ge::Operator& op_src, ge::Operator& op_dest)
{
    bool align_corners = false;
    std::string mode_value = "bilinear";
    std::string padding_mode_value = "zeros";
    bool channel_last = false;
    int scheduler_mode = 0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attrv2 : attrs["attribute"]) {
                    // GE omits i when the int is zero.
                    const std::string name = attrv2.value("name", "");
                    const int type = attrv2.value("type", -1);
                    if (name == "align_corners" && type == ATTR_TYPE_INT) {
                        int64_t align_corners_i = attrv2.contains("i") ? attrv2["i"].get<int64_t>() : 0;
                        if (align_corners_i != 0) {
                            if (align_corners_i > 1 || align_corners_i < -1) {
                                OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(),
                                                       "the value of align_corners [%ld] is out of range [-1, 1]",
                                                       align_corners_i);
                            }
                            align_corners = true;
                        }
                    } else if (name == "mode" && attrv2.contains("s")) {
                        mode_value = attrv2["s"];
                    } else if (name == "padding_mode" && attrv2.contains("s")) {
                        padding_mode_value = attrv2["s"];
                    } else if (name == "channel_last" && type == ATTR_TYPE_INT) {
                        int64_t channel_last_i = attrv2.contains("i") ? attrv2["i"].get<int64_t>() : 0;
                        if (channel_last_i != 0) {
                            if (channel_last_i > 1 || channel_last_i < 0) {
                                OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(),
                                                       "the value of channel_last [%ld] is out of range [0, 1]",
                                                       channel_last_i);
                            }
                            channel_last = true;
                        }
                    } else if (name == "scheduler_mode" && type == ATTR_TYPE_INT) {
                        scheduler_mode = attrv2.contains("i") ? static_cast<int>(attrv2["i"].get<int64_t>()) : 0;
                    }
                }
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op_dest).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }

    op_dest.SetAttr("interpolation_mode", mode_value);
    op_dest.SetAttr("align_corners", align_corners);
    op_dest.SetAttr("padding_mode", padding_mode_value);
    op_dest.SetAttr("channel_last", channel_last);
    op_dest.SetAttr("scheduler_mode", scheduler_mode);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("GridSample")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::GridSampleV2"), ge::AscendString("ai.onnx::16::GridSampleV2"),
                   ge::AscendString("ai.onnx::17::GridSampleV2"), ge::AscendString("ai.onnx::18::GridSampleV2")})
    .ParseParamsByOperatorFn(ParseParamsGridSampleV2)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("GridSampler2D")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::GridSample"), ge::AscendString("ai.onnx::16::GridSample"),
                   ge::AscendString("ai.onnx::17::GridSample"), ge::AscendString("ai.onnx::18::GridSample")})
    .ParseParamsByOperatorFn(ParseParamsGridSample)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("GridSampler3D")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::GridSample3D"), ge::AscendString("ai.onnx::16::GridSample3D"),
                   ge::AscendString("ai.onnx::17::GridSample3D"), ge::AscendString("ai.onnx::18::GridSample3D")})
    .ParseParamsByOperatorFn(ParseParamsGridSample)
    .ImplyType(ImplyType::TVM);
} // namespace domi
