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
 * \file roi_align_rotated_onnx_plugin.cpp
 * \brief RoiAlignRotated ONNX 适配（mmdeploy/ai.onnx::RoiAlignRotated）。
 *        平台感知建图：ascend950 上算子为 mmcv 原生布局（x NCHW、rois (R,6)
 * 行主序、 y (R,C,pooled_h,pooled_w)），ONNX 输入直连免转置；其余芯片为
 * NHWC/(6,R) （mmcv NPU 适配器调用约定），插 Transpose 适配。
 */

#include <string>

#include "cv_plugin_util.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "platform/platform_info.h"
#include "register/register.h"
#include "roi_align_rotated_proto.h"
#include "stub_ops.h"

namespace domi {
using json = nlohmann::json;

static constexpr int REQ_ATTR_NUM = 3;
static constexpr int REQ_INPUT_NUM = 2;
static constexpr int REQ_OUTPUT_NUM = 1;

namespace {
static Status SetRoiAlignRotatedByNode(ge::Operator& op_dest, int input_size, int output_size)
{
    if (input_size != REQ_INPUT_NUM || output_size != REQ_OUTPUT_NUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "input num must be 2, output num must be 1.");
        return FAILED;
    }
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);
    return SUCCESS;
}

// ascend950（regbase）平台识别
static bool IsRegbasePlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::GRAPH_SUCCESS) {
        OP_LOGE("RoiAlignRotated", "Get platform info failed, fallback to legacy transposed path.");
        return false;
    }
    const bool isRegbase = (platformInfo.str_info.short_soc_version == "Ascend950");
    OP_LOGD("RoiAlignRotated", "IsRegbasePlatform check: short_soc_version=%s, is_regbase=%d.",
            platformInfo.str_info.short_soc_version.c_str(), isRegbase);
    return isRegbase;
}
} // namespace

static Status ParseParamsRoiAlignRotated(const ge::Operator& op_src, ge::Operator& op_dest)
{
    bool aligned = true;
    bool clockwise = false;
    int output_height = 1;
    int output_width = 1;
    int sampling_ratio = 0;
    float spatial_scale = 0.5;

    int required_attr_num = 0;

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
                    if (name == "aligned" && type == ATTR_TYPE_INT) {
                        aligned = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "clockwise" && type == ATTR_TYPE_INT) {
                        clockwise = attr.contains("i") && (attr["i"].get<int64_t>() != 0);
                    } else if (name == "output_height" && type == ATTR_TYPE_INT) {
                        output_height = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        required_attr_num++;
                    } else if (name == "output_width" && type == ATTR_TYPE_INT) {
                        output_width = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                        required_attr_num++;
                    } else if (name == "sampling_ratio" && type == ATTR_TYPE_INT) {
                        sampling_ratio = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "spatial_scale" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            spatial_scale = 0.0f;
                        } else {
                            std::string spatial_scale_str = attr["f"];
                            if (StrToFloat(spatial_scale_str, spatial_scale) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse spatial_scale failed.");
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

    if (required_attr_num != REQ_ATTR_NUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "attr output_height, output_width, spatial_scale are required.");
        return FAILED;
    }

    int input_size = static_cast<int>(op_src.GetInputsSize());
    int output_size = static_cast<int>(op_src.GetOutputsSize());
    Status ret = SetRoiAlignRotatedByNode(op_dest, input_size, output_size);
    if (ret != SUCCESS) {
        return FAILED;
    }

    op_dest.SetAttr("pooled_h", output_height);
    op_dest.SetAttr("pooled_w", output_width);
    op_dest.SetAttr("spatial_scale", spatial_scale);
    op_dest.SetAttr("sampling_ratio", sampling_ratio);
    op_dest.SetAttr("aligned", aligned);
    op_dest.SetAttr("clockwise", clockwise);
    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("original_type", ge::AscendString("ai.onnx::11::RoiAlignRotated"));

    return SUCCESS;
}

Status ParseOpToGraphRoiAlignRotated(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE("ParseOpToGraphRoiAlignRotated", "get name from op failed.");
        return FAILED;
    }

    int pooled_h;
    int pooled_w;
    float spatial_scale;
    int sampling_ratio;
    bool aligned;
    bool clockwise;
    if (op.GetAttr("pooled_h", pooled_h) != ge::GRAPH_SUCCESS ||
        op.GetAttr("pooled_w", pooled_w) != ge::GRAPH_SUCCESS ||
        op.GetAttr("spatial_scale", spatial_scale) != ge::GRAPH_SUCCESS ||
        op.GetAttr("sampling_ratio", sampling_ratio) != ge::GRAPH_SUCCESS ||
        op.GetAttr("aligned", aligned) != ge::GRAPH_SUCCESS ||
        op.GetAttr("clockwise", clockwise) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    std::vector<ge::Operator> inputs{data0, data1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;

    if (IsRegbasePlatform()) {
        // ascend950：ONNX 原生布局（x NCHW、rois (R,6) 行主序、y
        // (R,C,ph,pw)）即算子契约，直连免转置
        auto roi_align_rotated = ge::op::RoiAlignRotated((ori_name + "_RoiAlignRotated_0").c_str())
                                     .set_input_x(data0)
                                     .set_input_rois(data1)
                                     .set_attr_pooled_h(pooled_h)
                                     .set_attr_pooled_w(pooled_w)
                                     .set_attr_spatial_scale(spatial_scale)
                                     .set_attr_sampling_ratio(sampling_ratio)
                                     .set_attr_aligned(aligned)
                                     .set_attr_clockwise(clockwise);
        outputs.emplace_back(roi_align_rotated, std::vector<size_t>{0});
        graph.SetInputs(inputs).SetOutputs(outputs);
        return SUCCESS;
    }

    // 其余芯片：算子契约为 NHWC/(6,R)（mmcv NPU 适配器调用约定），插 Transpose
    // 适配
    std::vector<int32_t> perm_boxes = {0, 2, 3, 1};
    auto tensor_perm_boxes = Vec2Tensor(perm_boxes, {4}, ge::DT_INT32);
    auto const_perm_boxes = ge::op::Const((ori_name + "_Const_0").c_str()).set_attr_value(tensor_perm_boxes);
    auto transpose_perm_boxes = ge::op::Transpose((ori_name + "_Transpose_0").c_str())
                                    .set_input_x(data0)
                                    .set_input_perm(const_perm_boxes);
    std::vector<int32_t> perm_rois = {1, 0};
    auto tensor_perm_rois = Vec2Tensor(perm_rois, {2}, ge::DT_INT32);
    auto const_perm_rois = ge::op::Const((ori_name + "_Const_1").c_str()).set_attr_value(tensor_perm_rois);
    auto transpose_perm_rois = ge::op::Transpose((ori_name + "_Transpose_1").c_str())
                                   .set_input_x(data1)
                                   .set_input_perm(const_perm_rois);

    auto roi_align_rotated = ge::op::RoiAlignRotated((ori_name + "_RoiAlignRotated_0").c_str())
                                 .set_input_x(transpose_perm_boxes)
                                 .set_input_rois(transpose_perm_rois)
                                 .set_attr_pooled_h(pooled_h)
                                 .set_attr_pooled_w(pooled_w)
                                 .set_attr_spatial_scale(spatial_scale)
                                 .set_attr_sampling_ratio(sampling_ratio)
                                 .set_attr_aligned(aligned)
                                 .set_attr_clockwise(clockwise);

    std::vector<int32_t> perm_out_boxes = {0, 3, 1, 2};
    auto tensor_perm_out_boxes = Vec2Tensor(perm_out_boxes, {4}, ge::DT_INT32);
    auto const_perm_out_boxes = ge::op::Const((ori_name + "_Const_2").c_str()).set_attr_value(tensor_perm_out_boxes);
    auto transpose_out_boxes = ge::op::Transpose((ori_name + "_Transpose_2").c_str())
                                   .set_input_x(roi_align_rotated)
                                   .set_input_perm(const_perm_out_boxes);

    outputs.emplace_back(transpose_out_boxes, std::vector<size_t>{0});

    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("mmdeploy::1::RoiAlignRotated"), ge::AscendString("ai.onnx::11::RoiAlignRotated"),
                   ge::AscendString("ai.onnx::12::RoiAlignRotated"), ge::AscendString("ai.onnx::13::RoiAlignRotated"),
                   ge::AscendString("ai.onnx::14::RoiAlignRotated"), ge::AscendString("ai.onnx::15::RoiAlignRotated"),
                   ge::AscendString("ai.onnx::16::RoiAlignRotated"), ge::AscendString("ai.onnx::17::RoiAlignRotated"),
                   ge::AscendString("ai.onnx::18::RoiAlignRotated")})
    .ParseParamsByOperatorFn(ParseParamsRoiAlignRotated)
    .ParseOpToGraphFn(ParseOpToGraphRoiAlignRotated)
    .ImplyType(ImplyType::TVM);
} // namespace domi
