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
#include "resize_nearest_neighbor_v2_proto.h"
#include "stub_ops.h"

using namespace ge;
namespace domi {
using json = nlohmann::json;

int INPUT_SIZES_IS_FOUR = 4;
int INPUT_SIZES_IS_THREE = 3;
int INPUT_SIZES_IS_TWO = 2;

static ge::Operator CreateSliceForResize(const std::string& ori_name, ge::Operator& sizes)
{
    int32_t offsets = 2;
    int32_t size_num = 2;
    ge::Tensor scalar_offsets = CreateScalar(offsets, ge::DT_INT32);
    ge::Tensor scalar_size_num = CreateScalar(size_num, ge::DT_INT32);

    auto data_offsets = op::Const((ori_name + "_data_offsets").c_str()).set_attr_value(scalar_offsets);
    auto data_size = op::Const((ori_name + "_data_size").c_str()).set_attr_value(scalar_size_num);

    return op::Slice((ori_name + "_Slice").c_str())
        .set_input_x(sizes)
        .set_input_offsets(data_offsets)
        .set_input_size(data_size);
}

static std::string AscendStringToStd(const ge::AscendString& value)
{
    return (value.GetString() == nullptr) ? "" : value.GetString();
}

// Prefer framework-provided ONNX input name list (attr "input"); fallback to
// GetInputsSize placeholders.
static void ResolveResizeInputs(const ge::Operator& op_src, int& input_size, std::string& input_roi,
                                std::string& input_scales)
{
    input_roi.clear();
    input_scales.clear();
    std::vector<ge::AscendString> onnx_inputs;
    if (op_src.GetAttr("input", onnx_inputs) == ge::GRAPH_SUCCESS && !onnx_inputs.empty()) {
        input_size = static_cast<int>(onnx_inputs.size());
        if (input_size > 1) {
            input_roi = AscendStringToStd(onnx_inputs[1]);
        }
        if (input_size > INPUT_SIZES_IS_TWO) {
            input_scales = AscendStringToStd(onnx_inputs[INPUT_SIZES_IS_TWO]);
        }
        if (input_size == INPUT_SIZES_IS_FOUR && AscendStringToStd(onnx_inputs[INPUT_SIZES_IS_THREE]).empty()) {
            input_size = INPUT_SIZES_IS_THREE;
        }
        return;
    }

    input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    if (input_size >= INPUT_SIZES_IS_TWO) {
        input_roi = "roi";
    }
    if (input_size >= INPUT_SIZES_IS_THREE) {
        input_scales = "scales";
    }
}

static Status ParseParamsResize(const ge::Operator& op_src, ge::Operator& op_dst)
{
    std::string coordinate_transformation_mode_value = "half_pixel";
    std::string mode_value = "nearest";

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "coordinate_transformation_mode" && attr.contains("s")) {
                        coordinate_transformation_mode_value = attr["s"];
                    } else if (name == "mode" && attr.contains("s")) {
                        mode_value = attr["s"];
                    } else if (name == "antialias" && attr.value("type", -1) == ATTR_TYPE_INT) {
                        // GE omits i when antialias is 0. The attribute is unsupported whether the value is present.
                        OP_LOGW(GetOpName(op_dst).c_str(), "Current antialias unsupported; expected type: int.");
                    } else if (name == "axes" && attr.contains("ints")) {
                        OP_LOGW(GetOpName(op_dst).c_str(), "Current axes unsupported; expected type: ints.");
                    } else if (name == "keep_aspect_ratio_policy" && attr.contains("s")) {
                        OP_LOGW(GetOpName(op_dst).c_str(), "Current keep_aspect_ratio_policy unsupported; expected "
                                                           "type: string.");
                    }
                }
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(GetOpName(op_dst).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE(GetOpName(op_dst).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }

    int input_size = 0;
    std::string input_roi;
    std::string input_scales;
    ResolveResizeInputs(op_src, input_size, input_roi, input_scales);
    if (input_size < INPUT_SIZES_IS_THREE) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Input size is less than 3, cannot access input_roi and input_scales.");
        return FAILED;
    }

    op_dst.SetAttr("input_roi", input_roi);
    op_dst.SetAttr("input_scales", input_scales);
    op_dst.SetAttr("name", GetOpName(op_src));
    op_dst.SetAttr("input_size", input_size);
    op_dst.SetAttr("coordinate_transformation_mode", coordinate_transformation_mode_value);
    op_dst.SetAttr("mode", mode_value);

    op_dst.DynamicInputRegister("x", input_size);
    op_dst.DynamicOutputRegister("y", 1);
    op_dst.SetAttr("original_type", "ai.onnx::11::Resize");
    return SUCCESS;
}

static Status BuildNearestResize(const std::string& ori_name, Operator& resize_x, Operator& sizes, Operator& resize_roi,
                                 Operator& resize_scales, const std::string& input_roi, const std::string& input_scales,
                                 int input_size, bool align_corners, bool half_pixel_centers,
                                 std::vector<Operator>& inputs,
                                 std::vector<std::pair<Operator, std::vector<size_t>>>& output_indexs)
{
    auto size = CreateSliceForResize(ori_name, sizes);
    inputs.push_back(size);
    auto ret_resize_x = ChangeFormatFromOnnx(resize_x, 0, ge::FORMAT_NCHW, false);
    if (ret_resize_x != ge::GRAPH_SUCCESS) {
        OP_LOGE(ori_name.c_str(), "update resize_x format failed.");
        return FAILED;
    }
    auto resizeout_1 = op::ResizeNearestNeighborV2((ori_name + "_ResizeNearestNeighborV2").c_str())
                           .set_input_x(resize_x)
                           .set_input_size(size)
                           .set_attr_align_corners(align_corners)
                           .set_attr_half_pixel_centers(half_pixel_centers);
    if (!input_roi.empty()) {
        resizeout_1.AddControlInput(resize_roi);
    }
    if ((input_size == INPUT_SIZES_IS_FOUR) && (!input_scales.empty())) {
        resizeout_1.AddControlInput(resize_scales);
    }
    ChangeFormatFromOnnx(resizeout_1, 0, ge::FORMAT_NCHW, false);
    ChangeFormatFromOnnx(resizeout_1, 0, ge::FORMAT_NCHW, true);
    output_indexs.emplace_back(resizeout_1, vector<std::size_t>{0});
    return SUCCESS;
}

static Status BuildInterpolatingResize(const std::string& ori_name, Operator& resize_x, Operator& sizes,
                                       Operator& resize_roi, Operator& resize_scales, Operator& data1, Operator& data2,
                                       const std::string& input_roi, const std::string& input_scales,
                                       const std::string& coordinate_transformation_mode_value,
                                       const std::string& mode_value, std::vector<Operator>& inputs,
                                       std::vector<std::pair<Operator, std::vector<size_t>>>& output_indexs)
{
    auto resizeout_2 = op::Resize((ori_name + "_Resize").c_str())
                           .set_input_x(resize_x)
                           .set_input_sizes(sizes)
                           .set_attr_coordinate_transformation_mode(coordinate_transformation_mode_value)
                           .set_attr_mode(mode_value);
    inputs.push_back(sizes);
    std::vector<float> empty_vector = {};
    ge::Tensor empty_tensor = Vec2Tensor(empty_vector, {0}, ge::DT_FLOAT, ge::FORMAT_ND);
    if (!input_roi.empty()) {
        inputs.push_back(data1);
        resizeout_2.set_input_roi(resize_roi);
    } else {
        auto roi = op::Const((ori_name + "_roi").c_str()).set_attr_value(empty_tensor);
        inputs.push_back(roi);
        resizeout_2.set_input_roi(roi);
    }
    if (!input_scales.empty()) {
        inputs.push_back(data2);
        resizeout_2.set_input_scales(resize_scales);
    } else {
        auto scales = op::Const((ori_name + "_scales").c_str()).set_attr_value(empty_tensor);
        inputs.push_back(scales);
        resizeout_2.set_input_scales(scales);
    }
    resizeout_2.SetAttr("resize_original_type", "onnx_resize");
    output_indexs.emplace_back(resizeout_2, vector<std::size_t>{0});
    return SUCCESS;
}

static Status ParseResizeModeAttrs(const Operator& op, std::string& coordinate_transformation_mode_value,
                                   bool& half_pixel_centers, bool& align_corners, std::string& mode_value)
{
    coordinate_transformation_mode_value = "pytorch_half_pixel";
    if (op.GetAttr("coordinate_transformation_mode", coordinate_transformation_mode_value) != GRAPH_SUCCESS) {
        OP_LOGW(GetOpName(op).c_str(), "Get attr coordinate transformation mode failed, set to default.");
    }
    half_pixel_centers = false;
    align_corners = false;
    if (coordinate_transformation_mode_value == "pytorch_half_pixel" ||
        coordinate_transformation_mode_value == "half_pixel") {
        half_pixel_centers = true;
    } else if (coordinate_transformation_mode_value == "align_corners") {
        align_corners = true;
    }

    if (op.GetAttr("mode", mode_value) != GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Get attr mode failed, set to default.");
        return FAILED;
    }
    return SUCCESS;
}

static Status BuildResizeSizesFromInputs(const std::string& ori_name, int input_size, ge::Operator& resize_x,
                                         ge::Operator& resize_scales, ge::Operator& resize_sizes, ge::Operator& sizes)
{
    if (input_size == INPUT_SIZES_IS_FOUR) {
        sizes = op::Cast((ori_name + "_Cast0").c_str()).set_input_x(resize_sizes).set_attr_dst_type(ge::DT_INT32);
    } else if (input_size == INPUT_SIZES_IS_THREE) {
        int dtype = 0;      // DT_INT32
        int sizes_type = 3; // e.g., DT_INT64
        auto resize = op::Shape((ori_name + "_Shape").c_str()).set_input_x(resize_x);
        auto resize_cast = op::Cast((ori_name + "_Cast0").c_str()).set_input_x(resize).set_attr_dst_type(dtype);
        auto mul_sizes = op::Mul((ori_name + "_Mul").c_str()).set_input_x1(resize_cast).set_input_x2(resize_scales);
        sizes = op::Cast((ori_name + "_Cast1").c_str()).set_input_x(mul_sizes).set_attr_dst_type(sizes_type);
    } else {
        OP_LOGE("ParseOpToGraphResize", "The input_size is error.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseOpToGraphResize(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    std::string input_roi, input_scales;
    op.GetAttr("input_roi", input_roi); // intentionally unchecked
    op.GetAttr("input_scales", input_scales);
    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto resize_x = op::Identity((ori_name + "_x").c_str()).set_input_x(data0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto resize_roi = op::Identity((ori_name + "_roi").c_str()).set_input_x(data1);
    auto data2 = op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
    auto resize_scales = op::Identity((ori_name + "_scales").c_str()).set_input_x(data2);
    auto data3 = op::Data((ori_name + "_data3").c_str()).set_attr_index(3);
    auto resize_sizes = op::Identity((ori_name + "_sizes").c_str()).set_input_x(data3);
    int input_size = 0;
    if (op.GetAttr("input_size", input_size) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get input_size from op failed");
        return FAILED;
    }
    std::string coordinate_transformation_mode_value;
    bool half_pixel_centers = false, align_corners = false;
    std::string mode_value;
    if (ParseResizeModeAttrs(op, coordinate_transformation_mode_value, half_pixel_centers, align_corners, mode_value) !=
        SUCCESS) {
        return FAILED;
    }
    ge::Operator sizes;
    if (BuildResizeSizesFromInputs(ori_name, input_size, resize_x, resize_scales, resize_sizes, sizes) != SUCCESS) {
        return FAILED;
    }
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    if (mode_value == "nearest") {
        auto status = BuildNearestResize(ori_name, resize_x, sizes, resize_roi, resize_scales, input_roi, input_scales,
                                         input_size, align_corners, half_pixel_centers, inputs, output_indexs);
        if (status != SUCCESS)
            return status;
    } else if (mode_value == "linear" || mode_value == "cubic") {
        auto status = BuildInterpolatingResize(ori_name, resize_x, sizes, resize_roi, resize_scales, data1, data2,
                                               input_roi, input_scales, coordinate_transformation_mode_value,
                                               mode_value, inputs, output_indexs);
        if (status != SUCCESS)
            return status;
    } else {
        OP_LOGE(GetOpName(op).c_str(), "Unsupported interpolation mode.");
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

static Status ParseParamsResizeV10(const ge::Operator& op_src, ge::Operator& op_dst)
{
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
        OP_LOGE(GetOpName(op_dst).c_str(), "JSON parse error: %s", e.what());
        return FAILED;
    } catch (...) {
        OP_LOGE(GetOpName(op_dst).c_str(), "get unknown exception, please check compile info json.");
        return FAILED;
    }

    op_dst.SetAttr("mode", mode_value);
    op_dst.SetAttr("name", GetOpName(op_src));
    op_dst.DynamicInputRegister("x", INPUT_SIZES_IS_TWO);
    op_dst.DynamicOutputRegister("y", 1);
    op_dst.SetAttr("original_type", "ai.onnx::10::Resize");
    return SUCCESS;
}

static Status BuildNearestResizeV10(const std::string& ori_name, Operator& resize_x, Operator& sizes,
                                    std::vector<Operator>& inputs,
                                    std::vector<std::pair<Operator, std::vector<size_t>>>& output_indexs)
{
    auto size = CreateSliceForResize(ori_name, sizes);
    inputs.push_back(size);
    auto ret_resize_x = ChangeFormatFromOnnx(resize_x, 0, ge::FORMAT_NCHW, false);
    if (ret_resize_x != ge::GRAPH_SUCCESS) {
        OP_LOGE(ori_name.c_str(), "update resize_x format failed.");
        return FAILED;
    }
    bool half_pixel_centers = false;
    bool align_corners = false;
    auto resizeout_1 = op::ResizeNearestNeighborV2((ori_name + "_ResizeNearestNeighborV2").c_str())
                           .set_input_x(resize_x)
                           .set_input_size(size)
                           .set_attr_align_corners(align_corners)
                           .set_attr_half_pixel_centers(half_pixel_centers);
    ChangeFormatFromOnnx(resizeout_1, 0, ge::FORMAT_NCHW, false);
    ChangeFormatFromOnnx(resizeout_1, 0, ge::FORMAT_NCHW, true);
    output_indexs.emplace_back(resizeout_1, vector<std::size_t>{0});
    return SUCCESS;
}

static Status BuildLinearResizeV10(const std::string& ori_name, Operator& resize_x, Operator& sizes,
                                   const std::string& mode_value, std::vector<Operator>& inputs,
                                   std::vector<std::pair<Operator, std::vector<size_t>>>& output_indexs)
{
    auto data2 = op::Const((ori_name + "_data2").c_str()).set_attr_value(0);
    auto data3 = op::Const((ori_name + "_data3").c_str()).set_attr_value(0);
    auto resizeout_2 = op::Resize((ori_name + "_Resize").c_str())
                           .set_input_x(resize_x)
                           .set_input_sizes(sizes)
                           .set_input_roi(data2)
                           .set_input_scales(data3)
                           .set_attr_mode(mode_value);
    inputs.push_back(data2);
    inputs.push_back(data3);
    resizeout_2.SetAttr("resize_original_type", "onnx_resize");
    output_indexs.emplace_back(resizeout_2, vector<std::size_t>{0});
    return SUCCESS;
}

static Status ParseOpToGraphResizeV10(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto resize_x = op::Identity((ori_name + "_Identity").c_str()).set_input_x(data0);

    int dtype = 0;
    int sizes_type = 3;
    auto resize = op::Shape((ori_name + "_Shape").c_str()).set_input_x(resize_x);
    auto resize_cast = op::Cast((ori_name + "_Cast0").c_str()).set_input_x(resize).set_attr_dst_type(dtype);
    auto mul_sizes = op::Mul((ori_name + "_Mul").c_str()).set_input_x1(resize_cast).set_input_x2(data1);
    auto sizes = op::Cast((ori_name + "_Cast1").c_str()).set_input_x(mul_sizes).set_attr_dst_type(sizes_type);

    std::string mode_value;
    if (op.GetAttr("mode", mode_value) != GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Get attr mode failed, set to default.");
        return FAILED;
    }

    std::vector<Operator> inputs{data0, data1};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    if (mode_value == "nearest") {
        auto status = BuildNearestResizeV10(ori_name, resize_x, sizes, inputs, output_indexs);
        if (status != SUCCESS)
            return status;
    } else if (mode_value == "linear") {
        auto status = BuildLinearResizeV10(ori_name, resize_x, sizes, mode_value, inputs, output_indexs);
        if (status != SUCCESS)
            return status;
    } else {
        OP_LOGE(GetOpName(op).c_str(), "Unsupported interpolation mode.");
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::Resize"), ge::AscendString("ai.onnx::12::Resize"),
                   ge::AscendString("ai.onnx::13::Resize"), ge::AscendString("ai.onnx::14::Resize"),
                   ge::AscendString("ai.onnx::15::Resize"), ge::AscendString("ai.onnx::16::Resize"),
                   ge::AscendString("ai.onnx::17::Resize"), ge::AscendString("ai.onnx::18::Resize")})
    .ParseParamsByOperatorFn(ParseParamsResize)
    .ParseOpToGraphFn(ParseOpToGraphResize)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::10::Resize")})
    .ParseParamsByOperatorFn(ParseParamsResizeV10)
    .ParseOpToGraphFn(ParseOpToGraphResizeV10)
    .ImplyType(ImplyType::TVM);
} // namespace domi
