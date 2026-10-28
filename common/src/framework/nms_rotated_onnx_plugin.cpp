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
 * \file nms_rotated_onnx_plugin.cpp
 * \brief onnx plugin for NMSRotated
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

namespace domi {
using json = nlohmann::json;

static const int INPUT_NUM_WITHOUT_LABELS = 2;
static const int INPUT_NUM_WITH_LABELS = 3;

static Status ParseParamsNMSRotatedCall(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float iou_threshold = 0;
    int required_attr_num = 0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits f when the float value is zero. Count the attr once name/type match.
                    if (attr.value("name", "") == "iou_threshold" && attr.value("type", -1) == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            iou_threshold = 0.0f;
                        } else {
                            std::string iou_threshold_str = attr["f"];
                            if (StrToFloat(iou_threshold_str, iou_threshold) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse iou_threshold failed.");
                                return FAILED;
                            }
                        }
                        required_attr_num++;
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

    if (required_attr_num != 1) {
        OP_LOGE(GetOpName(op_dest).c_str(), "attr iou_threshold is required.");
        return FAILED;
    }

    op_dest.SetAttr("iou_threshold", iou_threshold);

    int input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    op_dest.SetAttr("N", input_size);
    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("original_type", "ai.onnx::11::NMSRotated");

    return SUCCESS;
}

static Status ParseOpToGraphNMSRotated(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE("ParseOpToGraphNMSRotated", "get name from op failed.");
        return FAILED;
    }

    float iou_threshold = 0;
    op.GetAttr("iou_threshold", iou_threshold);

    int input_size = 0;
    op.GetAttr("N", input_size);
    if (input_size != INPUT_NUM_WITHOUT_LABELS && input_size != INPUT_NUM_WITH_LABELS) {
        OP_LOGE("ParseOpToGraphNMSRotated", "input must contain boxes, scores and labels(optional)");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    std::vector<Operator> inputs{data0, data1};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;

    std::vector<float> angle_values_vec = {1.0, 1.0, 1.0, 1.0, 57.29578};
    int64_t angle_values_Len = angle_values_vec.size();
    const vector<int64_t> angle_values_dim = {angle_values_Len};
    ge::Tensor angle_values_tensor = Vec2Tensor(angle_values_vec, angle_values_dim, ge::DT_FLOAT, ge::FORMAT_ND);
    auto const_values = op::Const((ori_name + "_const_value").c_str()).set_attr_value(angle_values_tensor);
    auto mul_values = op::Mul((ori_name + "_mul").c_str()).set_input_x1(const_values).set_input_x2(data0);

    if (input_size == INPUT_NUM_WITHOUT_LABELS) {
        auto identity = op::Identity((ori_name + "_identity").c_str()).set_input_x(data1);
        auto shape = op::Shape((ori_name + "_shape").c_str()).set_input_x(identity);
        auto empty = op::Empty((ori_name + "_empty").c_str())
                         .set_input_shape(shape)
                         .set_attr_dtype(ge::DT_INT64)
                         .set_attr_init(true);
        auto output = op::RotatedNMS((ori_name + "_rotatedNMS").c_str())
                          .set_input_boxes(mul_values)
                          .set_input_scores(identity)
                          .set_input_labels(empty)
                          .set_attr_iou_threshold(iou_threshold);
        output_indexs.emplace_back(output, std::vector<size_t>{0});
        output_indexs.emplace_back(output, std::vector<size_t>{1});
    } else {
        auto data2 = op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
        auto output = op::RotatedNMS((ori_name + "_rotatedNMS").c_str())
                          .set_input_boxes(mul_values)
                          .set_input_scores(data1)
                          .set_input_labels(data2)
                          .set_attr_iou_threshold(iou_threshold);
        inputs.emplace_back(data2);
        output_indexs.emplace_back(output, std::vector<size_t>{0});
        output_indexs.emplace_back(output, std::vector<size_t>{1});
    }

    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("mmdeploy::1::NMSRotated"), ge::AscendString("ai.onnx::11::NMSRotated"),
                   ge::AscendString("ai.onnx::12::NMSRotated"), ge::AscendString("ai.onnx::13::NMSRotated"),
                   ge::AscendString("ai.onnx::14::NMSRotated"), ge::AscendString("ai.onnx::15::NMSRotated"),
                   ge::AscendString("ai.onnx::16::NMSRotated"), ge::AscendString("ai.onnx::17::NMSRotated"),
                   ge::AscendString("ai.onnx::18::NMSRotated")})
    .ParseParamsByOperatorFn(ParseParamsNMSRotatedCall)
    .ParseOpToGraphFn(ParseOpToGraphNMSRotated)
    .ImplyType(ImplyType::AI_CPU);
} // namespace domi
