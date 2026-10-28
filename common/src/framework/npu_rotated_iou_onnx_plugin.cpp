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
 * \file npu_rotated_iou_onnx_plugin.cpp
 * \brief
 */
#include "cv_plugin_util.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "op_cv_proto_extend.h"
#include "register/register.h"
#include "stub_ops.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuRotatedIou(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    bool trans = false;
    std::string mode_str = "iou";
    bool is_cross = false;
    float v_threshold = 0.0;
    float e_threshold = 0.0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i/f when the scalar value is zero. Match name/type; only a fully absent attr keeps its
                    // default.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "trans" && type == ATTR_TYPE_INT) {
                        trans = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "mode" && type == ATTR_TYPE_INT) {
                        mode_str = (attr.contains("i") && attr["i"].get<int64_t>() == 1) ? "iof" : "iou";
                    } else if (name == "is_cross" && type == ATTR_TYPE_INT) {
                        is_cross = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "v_threshold" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            v_threshold = 0.0f;
                        } else {
                            std::string v_threshold_str = attr["f"];
                            if (StrToFloat(v_threshold_str, v_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse v_threshold failed.");
                                return FAILED;
                            }
                        }
                    } else if (name == "e_threshold" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            e_threshold = 0.0f;
                        } else {
                            std::string e_threshold_str = attr["f"];
                            if (StrToFloat(e_threshold_str, e_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse e_threshold failed.");
                                return FAILED;
                            }
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

    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("trans", trans);
    op_dest.SetAttr("mode", mode_str);
    op_dest.SetAttr("is_cross", is_cross);
    op_dest.SetAttr("v_threshold", v_threshold);
    op_dest.SetAttr("e_threshold", e_threshold);
    op_dest.SetAttr("original_type", "npu::1::NPURotatedIou");
    return SUCCESS;
}

static Status GetAttrByName(const ge::Operator& op, bool& trans, std::string& mode, bool& is_cross, float& v_threshold,
                            float& e_threshold)
{
    if (op.GetAttr("trans", trans) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr trans failed.");
        return FAILED;
    }
    if (op.GetAttr("mode", mode) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr mode failed.");
        return FAILED;
    }
    if (op.GetAttr("is_cross", is_cross) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr is_cross failed.");
        return FAILED;
    }
    if (op.GetAttr("v_threshold", v_threshold) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr v_threshold failed.");
        return FAILED;
    }
    if (op.GetAttr("e_threshold", e_threshold) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr e_threshold failed.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseOpToGraphNpuRotatedIou(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    bool trans = false;
    std::string mode = "";
    bool is_cross = false;
    float v_threshold = 0.0f;
    float e_threshold = 0.0f;

    Status ret = GetAttrByName(op, trans, mode, is_cross, v_threshold, e_threshold);
    if (ret != SUCCESS) {
        return FAILED;
    }

    std::vector<int32_t> perm_boxes = {0, 2, 1};
    auto tensor_perm_boxes = Vec2Tensor(perm_boxes, {3}, ge::DT_INT32);
    auto const_perm_boxes = ge::op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_perm_boxes);
    auto transpose_op_0 = ge::op::Transpose((ori_name + "_Transpose_0").c_str())
                              .set_input_x(data0)
                              .set_input_perm(const_perm_boxes);

    std::vector<int32_t> perm_queryboxes = {0, 2, 1};
    auto tensor_perm_queryboxes = Vec2Tensor(perm_queryboxes, {3}, ge::DT_INT32);
    auto const_perm_queryboxes = ge::op::Const((ori_name + "_Const_1").c_str()).set_attr_value(tensor_perm_queryboxes);
    auto transpose_op_1 = ge::op::Transpose((ori_name + "_Transpose_1").c_str())
                              .set_input_x(data1)
                              .set_input_perm(const_perm_queryboxes);

    auto rotated_npu_iou_op = ge::op::RotatedIou((ori_name + "_RotatedIou").c_str())
                                  .set_input_boxes(transpose_op_0)
                                  .set_input_query_boxes(transpose_op_1)
                                  .set_attr_trans(trans)
                                  .set_attr_mode(mode)
                                  .set_attr_is_cross(is_cross)
                                  .set_attr_v_threshold(v_threshold)
                                  .set_attr_e_threshold(e_threshold);

    std::vector<ge::Operator> inputs{data0, data1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(rotated_npu_iou_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_rotated_iou op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPURotatedIou"), ge::AscendString("ai.onnx::11::NPURotatedIou"),
                   ge::AscendString("ai.onnx::12::NPURotatedIou"), ge::AscendString("ai.onnx::13::NPURotatedIou"),
                   ge::AscendString("ai.onnx::14::NPURotatedIou"), ge::AscendString("ai.onnx::15::NPURotatedIou"),
                   ge::AscendString("ai.onnx::16::NPURotatedIou"), ge::AscendString("ai.onnx::17::NPURotatedIou"),
                   ge::AscendString("ai.onnx::18::NPURotatedIou")})
    .ParseParamsByOperatorFn(ParseParamsNpuRotatedIou)
    .ParseOpToGraphFn(ParseOpToGraphNpuRotatedIou)
    .ImplyType(ImplyType::TVM);
} // namespace domi
