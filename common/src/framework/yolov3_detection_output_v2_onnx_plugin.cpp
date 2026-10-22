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
 * \file yolov3_detection_output_v2_onnx_plugin.cpp
 * \brief
 */

#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseFloatsList(json& attr, std::vector<float>& values, const ge::Operator& op_dest,
                              const char* attr_name)
{
    if (!attr.contains("floats") || !attr["floats"].is_array()) {
        return SUCCESS;
    }
    for (auto& item : attr["floats"]) {
        float val = 0.0f;
        if (item.is_string()) {
            if (StrToFloat(item.get<std::string>(), val) != SUCCESS) {
                OP_LOGE(GetOpName(op_dest).c_str(), "parse %s failed.", attr_name);
                return FAILED;
            }
        } else if (item.is_number()) {
            val = item.get<float>();
        } else {
            OP_LOGE(GetOpName(op_dest).c_str(), "parse %s failed.", attr_name);
            return FAILED;
        }
        values.push_back(val);
    }
    return SUCCESS;
}

static Status ParseParamsYolov3detectionoutputv2(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int n = static_cast<int>(op_src.GetInputsSize());
    op_dest.DynamicInputRegister("x", n);

    int N = 10;
    int boxes = 3;
    int coords = 4;
    int classes = 80;
    int post_nms_topn = 512;
    int pre_nms_topn = 512;
    int out_box_dim = 3;
    float obj_threshold = 0.5;
    float score_threshold = 0.5;
    float iou_threshold = 0.45;
    bool relative = true;
    bool resize_origin_img_to_net = false;
    std::vector<float> v_biases = {};

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i/f when the scalar is zero. Do not keep the non-zero defaults.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "boxes" && type == ATTR_TYPE_INT) {
                        boxes = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "coords" && type == ATTR_TYPE_INT) {
                        coords = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "classes" && type == ATTR_TYPE_INT) {
                        classes = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "N" && type == ATTR_TYPE_INT) {
                        N = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "post_nms_topn" && type == ATTR_TYPE_INT) {
                        post_nms_topn = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "pre_nms_topn" && type == ATTR_TYPE_INT) {
                        pre_nms_topn = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "out_box_dim" && type == ATTR_TYPE_INT) {
                        out_box_dim = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "obj_threshold" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            obj_threshold = 0.0f;
                        } else {
                            std::string obj_threshold_str = attr["f"];
                            if (StrToFloat(obj_threshold_str, obj_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse obj_threshold failed.");
                                return FAILED;
                            }
                        }
                    } else if (name == "score_threshold" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            score_threshold = 0.0f;
                        } else {
                            std::string score_threshold_str = attr["f"];
                            if (StrToFloat(score_threshold_str, score_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse score_threshold failed.");
                                return FAILED;
                            }
                        }
                    } else if (name == "iou_threshold" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            iou_threshold = 0.0f;
                        } else {
                            std::string iou_threshold_str = attr["f"];
                            if (StrToFloat(iou_threshold_str, iou_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse iou_threshold failed.");
                                return FAILED;
                            }
                        }
                    } else if (name == "biases") {
                        if (ParseFloatsList(attr, v_biases, op_dest, "biases") != SUCCESS) {
                            return FAILED;
                        }
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

    if (v_biases.empty()) {
        OP_LOGE(GetOpName(op_dest).c_str(), "The attr of biases is required.");
        return FAILED;
    }

    op_dest.SetAttr("N", N);
    op_dest.SetAttr("biases", v_biases);
    op_dest.SetAttr("boxes", boxes);
    op_dest.SetAttr("coords", coords);
    op_dest.SetAttr("classes", classes);
    op_dest.SetAttr("relative", relative);
    op_dest.SetAttr("post_nms_topn", post_nms_topn);
    op_dest.SetAttr("pre_nms_topn", pre_nms_topn);
    op_dest.SetAttr("out_box_dim", out_box_dim);
    op_dest.SetAttr("obj_threshold", obj_threshold);
    op_dest.SetAttr("score_threshold", score_threshold);
    op_dest.SetAttr("iou_threshold", iou_threshold);
    op_dest.SetAttr("resize_origin_img_to_net", resize_origin_img_to_net);
    return SUCCESS;
}

// register YoloV3DetectionOutputV2 op info to GE
REGISTER_CUSTOM_OP("YoloV3DetectionOutputV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::9::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::10::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::11::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::12::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::13::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::14::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::15::YoloV3DetectionOutputV2"),
                   ge::AscendString("ai.onnx::16::YoloV3DetectionOutputV2")})
    .ParseParamsByOperatorFn(ParseParamsYolov3detectionoutputv2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
