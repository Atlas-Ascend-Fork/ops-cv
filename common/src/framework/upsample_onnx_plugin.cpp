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

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
const int INPUT_NUM = 2;
const int OUTPUT_NUM = 1;
const int PADS_SIZE = 8;
} // namespace

namespace domi {
using json = nlohmann::json;

static Status ParseParamsUpsample(const ge::Operator& op_src, ge::Operator& op_dest)
{
    op_dest.SetAttr("original_type", ge::AscendString("ai.onnx::11::Upsample"));
    std::string mode_value = "nearest";

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "mode" && attr.contains("s")) {
                        mode_value = attr["s"];
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

    if (mode_value != "nearest" && mode_value != "linear") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetOpName(op_dest).c_str(), "mode", mode_value.c_str(),
                                              "Mode attr of Upsample only supports nearest and linear");
        return FAILED;
    }

    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("mode", mode_value);
    op_dest.DynamicInputRegister("x", INPUT_NUM);
    op_dest.DynamicOutputRegister("y", OUTPUT_NUM);
    return SUCCESS;
}

static Status ParseOpToGraphUpsample(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    std::vector<float> pads_vector(PADS_SIZE, 0);
    int64_t len = pads_vector.size();
    const vector<int64_t> dims = {len};
    ge::Tensor const_value = Vec2Tensor(pads_vector, dims, DT_FLOAT, ge::FORMAT_ND);
    auto const_op = op::Const((ori_name + "_const_data").c_str()).set_attr_value(const_value);
    auto ret_x = ChangeFormatFromOnnx(data0, 0, ge::FORMAT_NCHW, false);
    if (ret_x != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update upsample_x format failed.");
        return FAILED;
    }
    std::string mode_value1;
    if (op.GetAttr("mode", mode_value1) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get value of mode from op failed.");
        return FAILED;
    }
    auto resize_op = op::Resize((ori_name + "_resize").c_str())
                         .set_input_x(data0)
                         .set_input_roi(const_op)
                         .set_input_scales(data1)
                         .set_attr_mode(mode_value1);

    ChangeFormatFromOnnx(resize_op, 0, ge::FORMAT_NCHW, true);
    ChangeFormatFromOnnx(resize_op, 0, ge::FORMAT_NCHW, false);

    std::vector<ge::Operator> inputs = {data0, const_op, data1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(resize_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register Upsample op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Upsample"), ge::AscendString("ai.onnx::9::Upsample"),
                   ge::AscendString("ai.onnx::11::Upsample"), ge::AscendString("ai.onnx::12::Upsample"),
                   ge::AscendString("ai.onnx::13::Upsample"), ge::AscendString("ai.onnx::14::Upsample"),
                   ge::AscendString("ai.onnx::15::Upsample"), ge::AscendString("ai.onnx::16::Upsample")})
    .ParseParamsByOperatorFn(ParseParamsUpsample)
    .ParseOpToGraphFn(ParseOpToGraphUpsample)
    .ImplyType(ImplyType::TVM);
} // namespace domi
