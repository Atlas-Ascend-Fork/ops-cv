/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_rotated_overlaps_onnx_plugin.cpp
 * \brief onnx plugin for custom operator npu_rotated_overlaps
 */

#include "cv_plugin_util.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"
#include "rotated_overlaps_proto.h"
#include "stub_ops.h"

using namespace std;
using namespace ge;

namespace domi {
using json = nlohmann::json;

static Status ParseParamsNpuRotatedOverlaps(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    bool trans = false;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i when trans is 0. Match name/type; only a fully absent attr keeps the default.
                    if (attr.value("name", "") == "trans" && attr.value("type", -1) == ATTR_TYPE_INT) {
                        trans = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
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
    op_dest.SetAttr("original_type", "npu::1::NPURotatedOverlaps");
    return SUCCESS;
}

static Status ParseOpToGraphNpuRotatedOverlaps(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    bool trans;
    if (op.GetAttr("trans", trans) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr trans failed.");
        return FAILED;
    }

    std::vector<int32_t> perm_box1 = {0, 2, 1};
    auto tensor_perm_box1 = Vec2Tensor(perm_box1, {3}, ge::DT_INT32);
    auto const_perm_box1 = op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_perm_box1);
    auto transpose_op_0 = op::Transpose((ori_name + "_Transpose_0").c_str())
                              .set_input_x(data0)
                              .set_input_perm(const_perm_box1);

    std::vector<int32_t> perm_box2 = {0, 2, 1};
    auto tensor_perm_box2 = Vec2Tensor(perm_box2, {3}, ge::DT_INT32);
    auto const_perm_box2 = op::Const((ori_name + "_Const_1").c_str()).set_attr_value(tensor_perm_box2);
    auto transpose_op_1 = op::Transpose((ori_name + "_Transpose_1").c_str())
                              .set_input_x(data1)
                              .set_input_perm(const_perm_box2);

    auto rotated_npu_overlaps_op = op::RotatedOverlaps((ori_name + "_RotatedOverlaps").c_str())
                                       .set_input_boxes(transpose_op_0)
                                       .set_input_query_boxes(transpose_op_1)
                                       .set_attr_trans(trans);

    std::vector<Operator> inputs{data0, data1};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(rotated_npu_overlaps_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_rotated_overlaps op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("npu::1::NPURotatedOverlaps"), ge::AscendString("ai.onnx::11::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::12::NPURotatedOverlaps"), ge::AscendString("ai.onnx::13::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::14::NPURotatedOverlaps"), ge::AscendString("ai.onnx::15::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::16::NPURotatedOverlaps"), ge::AscendString("ai.onnx::17::NPURotatedOverlaps"),
         ge::AscendString("ai.onnx::18::NPURotatedOverlaps")})
    .ParseParamsByOperatorFn(ParseParamsNpuRotatedOverlaps)
    .ParseOpToGraphFn(ParseOpToGraphNpuRotatedOverlaps)
    .ImplyType(ImplyType::TVM);
} // namespace domi
