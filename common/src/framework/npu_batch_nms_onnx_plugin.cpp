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
 * \file npu_batch_nms_onnx_plugin.cpp
 * \brief
 */
#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsBatchNMS(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float iou_threshold = 0.0;
    float score_threshold = 0.0;
    int max_size_per_class = 0;
    int max_total_size = 0;
    bool change_coordinate_frame = false;
    bool transpose_box = false;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i/f when the scalar is zero. Match name/type and still publish 0.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "iou_threshold" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            iou_threshold = 0.0f;
                        } else {
                            std::string iou_threshold_str = attr["f"];
                            if (StrToFloat(iou_threshold_str, iou_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse iou_threshold failed.");
                                return FAILED;
                            }
                        }
                        op_dest.SetAttr("iou_threshold", iou_threshold);
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
                        op_dest.SetAttr("score_threshold", score_threshold);
                    } else if (name == "max_size_per_class" && type == ATTR_TYPE_INT) {
                        max_size_per_class = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("max_size_per_class", max_size_per_class);
                    } else if (name == "max_total_size" && type == ATTR_TYPE_INT) {
                        max_total_size = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        op_dest.SetAttr("max_total_size", max_total_size);
                    } else if (name == "change_coordinate_frame" && type == ATTR_TYPE_INT) {
                        change_coordinate_frame = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                        op_dest.SetAttr("change_coordinate_frame", change_coordinate_frame);
                    } else if (name == "transpose_box" && type == ATTR_TYPE_INT) {
                        transpose_box = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                        op_dest.SetAttr("transpose_box", transpose_box);
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

    return SUCCESS;
}

// register StrideAdd op info to GE
REGISTER_CUSTOM_OP("BatchMultiClassNonMaxSuppression")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUBatchNms"), ge::AscendString("ai.onnx::11::NPUBatchNms"),
                   ge::AscendString("ai.onnx::12::NPUBatchNms"), ge::AscendString("ai.onnx::13::NPUBatchNms"),
                   ge::AscendString("ai.onnx::14::NPUBatchNms"), ge::AscendString("ai.onnx::15::NPUBatchNms"),
                   ge::AscendString("ai.onnx::16::NPUBatchNms"), ge::AscendString("ai.onnx::17::NPUBatchNms"),
                   ge::AscendString("ai.onnx::18::NPUBatchNms")})
    .ParseParamsByOperatorFn(ParseParamsBatchNMS)
    .ImplyType(ImplyType::TVM);
} // namespace domi
