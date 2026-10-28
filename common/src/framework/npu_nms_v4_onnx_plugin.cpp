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
 * \file npu_nms_v4_onnx_plugin.cpp
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

static Status ParseParamsNmsV4(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int op_input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int op_output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);

    op_dest.SetAttr("original_type", "npu::1::NPUNmsV4");

    bool pad_to_max_output_size = false;
    int max_output_size = 0;

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
                    if (name == "pad_to_max_output_size" && type == ATTR_TYPE_INT) {
                        pad_to_max_output_size = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "max_output_size" && type == ATTR_TYPE_INT) {
                        max_output_size = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
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

    ge::Tensor scalar_const_value = CreateScalar(max_output_size, ge::DT_INT32);

    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("pad_to_max_output_size", pad_to_max_output_size);
    op_dest.SetAttr("max_output_size", scalar_const_value);

    return SUCCESS;
}

static Status ParseOpToGraphNmsV4(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    bool pad_to_max_output_size = false;
    op.GetAttr("pad_to_max_output_size", pad_to_max_output_size);

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
    auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);

    ge::Tensor const_value;
    if (op.GetAttr("max_output_size", const_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get max_output_size from op failed");
        return FAILED;
    }

    auto const_op = ge::op::Const((ori_name + "_const_data").c_str()).set_attr_value(const_value);
    auto NonMaxSuppressionV4 = ge::op::NonMaxSuppressionV4((ori_name + "_NonMaxSuppressionV4").c_str())
                                   .set_input_boxes(data0)
                                   .set_input_scores(data1)
                                   .set_input_max_output_size(const_op)
                                   .set_input_iou_threshold(data2)
                                   .set_input_score_threshold(data3)
                                   .set_attr_pad_to_max_output_size(pad_to_max_output_size);

    std::vector<ge::Operator> inputs{data0, data1, data2, data3};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(NonMaxSuppressionV4, std::vector<std::size_t>{0});
    outputs.emplace_back(NonMaxSuppressionV4, std::vector<std::size_t>{1});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

// register NonMaxSuppressionV4 op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUNmsV4"), ge::AscendString("ai.onnx::11::NPUNmsV4"),
                   ge::AscendString("ai.onnx::12::NPUNmsV4"), ge::AscendString("ai.onnx::13::NPUNmsV4"),
                   ge::AscendString("ai.onnx::14::NPUNmsV4"), ge::AscendString("ai.onnx::15::NPUNmsV4"),
                   ge::AscendString("ai.onnx::16::NPUNmsV4"), ge::AscendString("ai.onnx::17::NPUNmsV4"),
                   ge::AscendString("ai.onnx::18::NPUNmsV4")})
    .ParseParamsByOperatorFn(ParseParamsNmsV4)
    .ParseOpToGraphFn(ParseOpToGraphNmsV4)
    .ImplyType(ImplyType::TVM);
} // namespace domi
