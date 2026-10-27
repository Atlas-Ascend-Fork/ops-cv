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
 * \file non_max_suppression_onnx_plugin.cpp
 * \brief
 */
#include "cv_plugin_util.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "non_max_suppression_v6_proto.h"
#include "register/register.h"
#include "stub_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
constexpr int boxes_index = 0;
constexpr int socre_index = 1;
constexpr int max_output_boxes_index = 2;
constexpr int iou_threshold_index = 3;
constexpr int score_threshold_index = 4;
} // namespace

namespace domi {
using json = nlohmann::json;

using NMSOpDesc = std::shared_ptr<ge::OpDesc>;

static Status ParseParamsNonMaxSuppression(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int center_point_box = 0;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    // GE omits i when the int is zero.
                    if (attr.value("name", "") == "center_point_box" && attr.value("type", -1) == ATTR_TYPE_INT) {
                        center_point_box = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
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

    op_dest.SetAttr("center_point_box", center_point_box);

    int input_size = static_cast<int>(op_src.GetInputsSize());
    op_dest.SetAttr("input_size", input_size);
    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("original_type", "ai.onnx::11::NonMaxSuppression");
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

namespace {
static Status GetAttrByName(const ge::Operator& op, int& input_size, int& center_point_box)
{
    if (op.GetAttr("input_size", input_size) != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op).c_str(), "get input_size from op failed.");
        return FAILED;
    }
    if (op.GetAttr("center_point_box", center_point_box) != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op).c_str(), "get center_point_box from op failed.");
        return FAILED;
    }
    return SUCCESS;
}
} // namespace

static Status ParseOpToGraphNonMaxSuppression(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto boxes = op::Data((ori_name + "_boxes").c_str()).set_attr_index(boxes_index);
    auto scores = op::Data((ori_name + "_scores").c_str()).set_attr_index(socre_index);
    auto max_output_boxes = op::Data((ori_name + "_max_output_boxes").c_str()).set_attr_index(max_output_boxes_index);
    auto iou_threshold = op::Data((ori_name + "_iou_threshold").c_str()).set_attr_index(iou_threshold_index);
    auto score_threshold = op::Data((ori_name + "_score_threshold").c_str()).set_attr_index(score_threshold_index);

    int input_size = 0;
    int center_point_box = 0;
    Status ret = GetAttrByName(op, input_size, center_point_box);
    if (ret != SUCCESS) {
        return FAILED;
    }

    auto non_max_suppression = op::NonMaxSuppressionV6((ori_name + "_NonMaxSuppressionV6").c_str());
    std::vector<Operator> inputs{boxes, scores};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    if (input_size == (socre_index + 1)) {
        non_max_suppression.set_input_boxes(boxes).set_input_scores(scores).set_attr_center_point_box(center_point_box);
    } else if (input_size == (max_output_boxes_index + 1)) {
        inputs.push_back(max_output_boxes);
        non_max_suppression.set_input_boxes(boxes)
            .set_input_scores(scores)
            .set_input_max_output_size(max_output_boxes)
            .set_attr_center_point_box(center_point_box);
    } else if (input_size == (iou_threshold_index + 1)) {
        inputs.push_back(max_output_boxes);
        inputs.push_back(iou_threshold);
        non_max_suppression.set_input_boxes(boxes)
            .set_input_scores(scores)
            .set_input_max_output_size(max_output_boxes)
            .set_input_iou_threshold(iou_threshold)
            .set_attr_center_point_box(center_point_box);
    } else if (input_size == (score_threshold_index + 1)) {
        inputs.push_back(max_output_boxes);
        inputs.push_back(iou_threshold);
        inputs.push_back(score_threshold);
        non_max_suppression.set_input_boxes(boxes)
            .set_input_scores(scores)
            .set_input_max_output_size(max_output_boxes)
            .set_input_iou_threshold(iou_threshold)
            .set_input_score_threshold(score_threshold)
            .set_attr_center_point_box(center_point_box);
    } else {
        OP_LOGE_WITHOUT_REPORT(GetOpName(op).c_str(), "The input_size is invalid.");
        return FAILED;
    }

    auto output_int64 = op::Cast((ori_name + "_Cast").c_str())
                            .set_input_x(non_max_suppression)
                            .set_attr_dst_type(DT_INT64);
    output_indexs.emplace_back(output_int64, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::10::NonMaxSuppression"), ge::AscendString("ai.onnx::11::NonMaxSuppression"),
         ge::AscendString("ai.onnx::12::NonMaxSuppression"), ge::AscendString("ai.onnx::13::NonMaxSuppression"),
         ge::AscendString("ai.onnx::14::NonMaxSuppression"), ge::AscendString("ai.onnx::15::NonMaxSuppression"),
         ge::AscendString("ai.onnx::16::NonMaxSuppression"), ge::AscendString("ai.onnx::17::NonMaxSuppression"),
         ge::AscendString("ai.onnx::18::NonMaxSuppression")})
    .ParseParamsByOperatorFn(ParseParamsNonMaxSuppression)
    .ParseOpToGraphFn(ParseOpToGraphNonMaxSuppression)
    .ImplyType(ImplyType::TVM);
} // namespace domi
