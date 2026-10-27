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
#include "register/register.h"
#include "roi_align_proto.h"
#include "stub_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using json = nlohmann::json;

static Status OpRoiAlignUpdateInfo(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int output_height_value = 1;
    int output_width_value = 1;
    int sampling_ratio_value = 0;
    float spatial_scale_value = 1.0;

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
                    if (name == "output_height" && type == ATTR_TYPE_INT) {
                        output_height_value = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "output_width" && type == ATTR_TYPE_INT) {
                        output_width_value = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "sampling_ratio" && type == ATTR_TYPE_INT) {
                        sampling_ratio_value = attr.contains("i") ? static_cast<int>(attr["i"].get<int64_t>()) : 0;
                    } else if (name == "spatial_scale" && type == ATTR_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            spatial_scale_value = 0.0f;
                        } else {
                            std::string spatial_scale_str = attr["f"];
                            if (StrToFloat(spatial_scale_str, spatial_scale_value) != SUCCESS) {
                                OP_LOGE(GetOpName(op_dest).c_str(), "parse spatial_scale failed.");
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

    int input_size = static_cast<int>(op_src.GetInputsSize());
    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("pooled_height", output_height_value);
    op_dest.SetAttr("pooled_width", output_width_value);
    op_dest.SetAttr("sample_num", sampling_ratio_value);
    op_dest.SetAttr("spatial_scale", spatial_scale_value);
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static Status ParseParamsRoiAlign(const ge::Operator& op_src, ge::Operator& op_dest)
{
    if (OpRoiAlignUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }

    int default_roi_end_mode_value = 0;
    op_dest.SetAttr("roi_end_mode", default_roi_end_mode_value);
    op_dest.SetAttr("original_type", ge::AscendString("ai.onnx::11::RoiAlign"));
    return SUCCESS;
}

static Status ParseParamsRoiAlignV16(const ge::Operator& op_src, ge::Operator& op_dest)
{
    if (OpRoiAlignUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }

    int default_roi_end_mode_value = 2;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    if (attr.value("name", "") == "coordinate_transformation_mode" && attr.contains("s") &&
                        attr["s"].get<std::string>() == "output_half_pixel") {
                        default_roi_end_mode_value = 0;
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

    op_dest.SetAttr("roi_end_mode", default_roi_end_mode_value);
    op_dest.SetAttr("original_type", ge::AscendString("ai.onnx::16::RoiAlign"));
    return SUCCESS;
}

static Status ParseOpToGraphRoiAlign(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int pooled_height = 1;
    op.GetAttr("pooled_height", pooled_height);
    int pooled_width = 1;
    op.GetAttr("pooled_width", pooled_width);
    int sample_num = 0;
    op.GetAttr("sample_num", sample_num);
    float spatial_scale = 1.0f;
    op.GetAttr("spatial_scale", spatial_scale);
    int roi_end_mode = 0;
    op.GetAttr("roi_end_mode", roi_end_mode);

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = op::Data((ori_name + "_data2").c_str()).set_attr_index(2);

    int32_t concat_dim = 1;
    ge::Tensor scalar_dim = CreateScalar(concat_dim, ge::DT_INT32);
    auto dim_const_op = op::Const((ori_name + "_concat_dim").c_str()).set_attr_value(scalar_dim);

    std::vector<int64_t> axes = {-1};
    auto unsqueeze_op = op::Unsqueeze((ori_name + "_unsqueeze").c_str()).set_input_x(data2).set_attr_axes(axes);
    auto cast_op = op::Cast((ori_name + "_cast").c_str()).set_input_x(unsqueeze_op).set_attr_dst_type(DT_FLOAT);
    auto cast1_op = op::Cast((ori_name + "_cast1").c_str()).set_input_x(data1).set_attr_dst_type(DT_FLOAT);
    auto concat_op = op::Concat((ori_name + "_concat").c_str())
                         .create_dynamic_input_x(2)
                         .set_dynamic_input_x(0, cast_op)
                         .set_dynamic_input_x(1, cast1_op)
                         .set_input_concat_dim(dim_const_op)
                         .set_attr_N(2);

    auto roli_op = op::ROIAlign((ori_name + "_roialign").c_str())
                       .set_input_features(data0)
                       .set_input_rois(concat_op)
                       .set_attr_spatial_scale(spatial_scale)
                       .set_attr_pooled_height(pooled_height)
                       .set_attr_pooled_width(pooled_width)
                       .set_attr_sample_num(sample_num)
                       .set_attr_roi_end_mode(roi_end_mode);
    std::vector<ge::Operator> inputs = {data0, data1, data2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(roli_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::RoiAlign"), ge::AscendString("ai.onnx::9::RoiAlign"),
                   ge::AscendString("ai.onnx::10::RoiAlign"), ge::AscendString("ai.onnx::11::RoiAlign"),
                   ge::AscendString("ai.onnx::12::RoiAlign"), ge::AscendString("ai.onnx::13::RoiAlign"),
                   ge::AscendString("ai.onnx::14::RoiAlign"), ge::AscendString("ai.onnx::15::RoiAlign")})
    .ParseParamsByOperatorFn(ParseParamsRoiAlign)
    .ParseOpToGraphFn(ParseOpToGraphRoiAlign)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::16::RoiAlign"), ge::AscendString("ai.onnx::17::RoiAlign"),
                   ge::AscendString("ai.onnx::18::RoiAlign")})
    .ParseParamsByOperatorFn(ParseParamsRoiAlignV16)
    .ParseOpToGraphFn(ParseOpToGraphRoiAlign)
    .ImplyType(ImplyType::TVM);
} // namespace domi
