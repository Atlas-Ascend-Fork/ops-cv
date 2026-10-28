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
#include "graph/graph.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "op_cv_proto_extend.h"
#include "register/register.h"
#include "stub_ops.h"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuGiou(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    bool trans = false;
    bool is_cross = false;
    std::string mode = "iou";

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i when the int value is zero. Match name/type; only a fully absent attr keeps its
                    // default.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "trans" && type == ATTR_TYPE_INT) {
                        trans = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "is_cross" && type == ATTR_TYPE_INT) {
                        is_cross = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "mode" && type == ATTR_TYPE_INT) {
                        mode = (attr.contains("i") && attr["i"].get<int64_t>() == 1) ? "iof" : "iou";
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
    op_dest.SetAttr("is_cross", is_cross);
    op_dest.SetAttr("mode", mode);
    op_dest.SetAttr("original_type", "npu::1::NPUGiou");
    return SUCCESS;
}

static Status ParseOpToGraphNpuGiou(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    bool trans;
    if (op.GetAttr("trans", trans) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr trans failed.");
        return FAILED;
    }
    bool is_cross;
    if (op.GetAttr("is_cross", is_cross) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr is_cross failed.");
        return FAILED;
    }
    std::string mode;
    if (op.GetAttr("mode", mode) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr mode failed.");
        return FAILED;
    }

    auto giou_op = ge::op::GIoU((ori_name + "_GIoU").c_str())
                       .set_input_bboxes(data0)
                       .set_input_gtboxes(data1)
                       .set_attr_trans(trans)
                       .set_attr_is_cross(is_cross)
                       .set_attr_mode(mode);

    std::vector<int32_t> perm = {1, 0};
    auto tensor_perm = Vec2Tensor(perm, {2}, ge::DT_INT32);
    auto const_perm = ge::op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_perm);

    auto transpose_op = ge::op::Transpose((ori_name + "_Transpose").c_str())
                            .set_input_x(giou_op)
                            .set_input_perm(const_perm);

    std::vector<ge::Operator> inputs{data0, data1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(transpose_op, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_giou op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUGiou"), ge::AscendString("ai.onnx::11::NPUGiou"),
                   ge::AscendString("ai.onnx::12::NPUGiou"), ge::AscendString("ai.onnx::13::NPUGiou"),
                   ge::AscendString("ai.onnx::14::NPUGiou"), ge::AscendString("ai.onnx::15::NPUGiou"),
                   ge::AscendString("ai.onnx::16::NPUGiou"), ge::AscendString("ai.onnx::17::NPUGiou"),
                   ge::AscendString("ai.onnx::18::NPUGiou")})
    .ParseParamsByOperatorFn(ParseParamsNpuGiou)
    .ParseOpToGraphFn(ParseOpToGraphNpuGiou)
    .ImplyType(ImplyType::TVM);
} // namespace domi
