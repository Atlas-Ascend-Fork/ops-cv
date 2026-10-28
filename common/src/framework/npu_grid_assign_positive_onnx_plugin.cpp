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
 * \file npu_grid_assign_positive_onnx_plugin.cpp
 * \brief onnx plugin for custom operator npu_grid_assign_positive
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

static Status ParseParamsNpuGridAssignPositive(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);

    float pos_iou_thr = 0.5;
    float min_pos_iou = 0;
    bool gt_max_assign_all = true;
    int num_gts = 128;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits the scalar value field when the protobuf value is zero.
                    // Match name/type; a missing i/f means 0, and only a fully absent attr keeps its default.
                    const std::string name = attr.value("name", "");
                    const int type = attr.value("type", -1);
                    if (name == "pos_iou_thr" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            pos_iou_thr = 0.0f;
                        } else {
                            std::string pos_iou_thr_str = attr["f"];
                            if (StrToFloat(pos_iou_thr_str, pos_iou_thr) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse pos_iou_thr failed.");
                                return FAILED;
                            }
                        }
                    } else if (name == "min_pos_iou" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            min_pos_iou = 0.0f;
                        } else {
                            std::string min_pos_iou_str = attr["f"];
                            if (StrToFloat(min_pos_iou_str, min_pos_iou) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse min_pos_iou failed.");
                                return FAILED;
                            }
                        }
                    } else if (name == "gt_max_assign_all" && type == ATTR_TYPE_INT) {
                        gt_max_assign_all = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "num_gts" && type == ATTR_TYPE_INT) {
                        num_gts = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
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
    op_dest.SetAttr("pos_iou_thr", pos_iou_thr);
    op_dest.SetAttr("min_pos_iou", min_pos_iou);
    op_dest.SetAttr("gt_max_assign_all", gt_max_assign_all);
    op_dest.SetAttr("num_gts", num_gts);
    op_dest.SetAttr("original_type", "npu::1::NPUGridAssignPositive");
    return SUCCESS;
}

static Status ParseOpToGraphNpuGridAssignPositive(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
    auto data3 = op::Data((ori_name + "_data3").c_str()).set_attr_index(3);
    auto data4 = op::Data((ori_name + "_data4").c_str()).set_attr_index(4);
    auto data5 = op::Data((ori_name + "_data5").c_str()).set_attr_index(5);
    auto data6 = op::Data((ori_name + "_data6").c_str()).set_attr_index(6);

    float pos_iou_thr;
    if (op.GetAttr("pos_iou_thr", pos_iou_thr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr pos_iou_thr failed.");
        return FAILED;
    }
    float min_pos_iou;
    if (op.GetAttr("min_pos_iou", min_pos_iou) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr min_pos_iou failed.");
        return FAILED;
    }
    bool gt_max_assign_all;
    if (op.GetAttr("gt_max_assign_all", gt_max_assign_all) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr gt_max_assign_all failed.");
        return FAILED;
    }
    int num_gts;
    if (op.GetAttr("num_gts", num_gts) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr num_gts failed.");
        return FAILED;
    }

    vector<int32_t> vector_num_gts = {num_gts};
    auto tensor_num_gts = Vec2Tensor(vector_num_gts, {1}, ge::DT_INT32);
    auto const_0 = op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_num_gts);

    auto grid_assign_positive_op = op::GridAssignPositive((ori_name + "_GridAssignPositive").c_str())
                                       .set_input_assigned_gt_inds(data0)
                                       .set_input_overlaps(data1)
                                       .set_input_box_responsible_flags(data2)
                                       .set_input_max_overlaps(data3)
                                       .set_input_argmax_overlaps(data4)
                                       .set_input_gt_max_overlaps(data5)
                                       .set_input_gt_argmax_overlaps(data6)
                                       .set_input_num_gts(const_0)
                                       .set_attr_pos_iou_thr(pos_iou_thr)
                                       .set_attr_min_pos_iou(min_pos_iou)
                                       .set_attr_gt_max_assign_all(gt_max_assign_all);

    std::vector<Operator> inputs{data0, data1, data2, data3, data4, data5, data6};
    std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(grid_assign_positive_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_grid_assign_positive op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("npu::1::NPUGridAssignPositive"), ge::AscendString("ai.onnx::11::NPUGridAssignPositive"),
         ge::AscendString("ai.onnx::12::NPUGridAssignPositive"), ge::AscendString("ai.onnx::13::NPUGridAssignPositive"),
         ge::AscendString("ai.onnx::14::NPUGridAssignPositive"), ge::AscendString("ai.onnx::15::NPUGridAssignPositive"),
         ge::AscendString("ai.onnx::16::NPUGridAssignPositive"), ge::AscendString("ai.onnx::17::NPUGridAssignPositive"),
         ge::AscendString("ai.onnx::18::NPUGridAssignPositive")})
    .ParseParamsByOperatorFn(ParseParamsNpuGridAssignPositive)
    .ParseOpToGraphFn(ParseOpToGraphNpuGridAssignPositive)
    .ImplyType(ImplyType::TVM);
} // namespace domi
