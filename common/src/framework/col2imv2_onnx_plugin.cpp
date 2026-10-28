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
 * \file col2imv2_onnx_plugin.cpp
 * \brief
 */

#include "col2_im_v2_proto.h"
#include "cv_plugin_util.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"
#include "stub_ops.h"

using namespace ge;
namespace {
const int ATTR_INTS = 2;
}
namespace domi {
using json = nlohmann::json;

namespace {
static void SetCol2ImV2ByCondition(ge::Operator& op_dest, bool set_dilations_flag, std::vector<int>& v_dilations,
                                   bool set_pads_flag, std::vector<int>& v_pads, bool set_strides_flag,
                                   std::vector<int>& v_strides)
{
    if (set_dilations_flag) {
        op_dest.SetAttr("dilation", v_dilations);
    }
    if (set_pads_flag) {
        op_dest.SetAttr("padding", v_pads);
    }

    if (set_strides_flag) {
        op_dest.SetAttr("stride", v_strides);
    }
}

static void SetCol2ImV2ByOp(ge::Operator& op_dest, const ge::Operator& op_src)
{
    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("original_type", "ai.onnx::18::Col2Im");

    int input_size = static_cast<int>(ResolveOnnxInputNum(op_src));
    int output_size = static_cast<int>(ResolveOnnxOutputNum(op_src));

    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);
}
} // namespace

static Status ParseOnnxParamsCol2ImV2(const ge::Operator& op_src, ge::Operator& op_dest)
{
    std::vector<int> v_dilations = {};
    bool set_dilations_flag = false;
    std::vector<int> v_pads = {};
    bool set_pads_flag = false;
    std::vector<int> v_strides = {};
    bool set_strides_flag = false;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "dilations" && attr.contains("ints")) {
                        if (attr["ints"].size() == ATTR_INTS) {
                            for (auto& v : attr["ints"]) {
                                v_dilations.push_back(static_cast<int>(v.get<int64_t>()));
                            }
                        } else {
                            OP_LOGE(GetOpName(op_dest).c_str(), "length of dilations must be 2.");
                            return FAILED;
                        }
                        set_dilations_flag = true;
                    } else if (name == "pads" && attr.contains("ints")) {
                        if (attr["ints"].size() == ATTR_INTS) {
                            for (auto& v : attr["ints"]) {
                                v_pads.push_back(static_cast<int>(v.get<int64_t>()));
                            }
                        } else {
                            OP_LOGE(GetOpName(op_dest).c_str(), "length of pads must be 2.");
                            return FAILED;
                        }
                        set_pads_flag = true;
                    } else if (name == "strides" && attr.contains("ints")) {
                        if (attr["ints"].size() == ATTR_INTS) {
                            for (auto& v : attr["ints"]) {
                                v_strides.push_back(static_cast<int>(v.get<int64_t>()));
                            }
                        } else {
                            OP_LOGE(GetOpName(op_dest).c_str(), "length of strides must be 2.");
                            return FAILED;
                        }
                        set_strides_flag = true;
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

    SetCol2ImV2ByCondition(op_dest, set_dilations_flag, v_dilations, set_pads_flag, v_pads, set_strides_flag,
                           v_strides);
    SetCol2ImV2ByOp(op_dest, op_src);

    return SUCCESS;
}

static Status ParseOnnxOpToGraphCol2ImV2(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = op::Data((ori_name + "_data2").c_str()).set_attr_index(2);

    std::vector<int32_t> dilation;
    if (op.GetAttr("dilation", dilation) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr dilation failed");
        return FAILED;
    }

    std::vector<int32_t> padding;
    if (op.GetAttr("padding", padding) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr padding failed");
        return FAILED;
    }

    std::vector<int32_t> stride;
    if (op.GetAttr("stride", stride) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr stride failed");
        return FAILED;
    }

    ge::Operator::OpListInt dilation_value = {dilation[0], dilation[1]};
    ge::Operator::OpListInt padding_value = {padding[0], padding[1]};
    ge::Operator::OpListInt stride_value = {stride[0], stride[1]};
    // 3 in set_attr_dst_type indicates DT_INT32
    auto cast_op = op::Cast((ori_name + "_cast").c_str()).set_input_x(data1).set_attr_dst_type(3);
    auto col2im_v2 = op::Col2ImV2((ori_name + "_Col2ImV2").c_str())
                         .set_input_x(data0)
                         .set_input_output_size(cast_op)
                         .set_input_kernel_size(data2)
                         .set_attr_dilation(dilation_value)
                         .set_attr_padding(padding_value)
                         .set_attr_stride(stride_value);
    if (ChangeFormatFromOnnx(col2im_v2, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
        ChangeFormatFromOnnx(col2im_v2, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
        return FAILED;
    }

    std::vector<ge::Operator> inputs = {data0, data1, data2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(col2im_v2, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::18::Col2Im")})
    .ParseParamsByOperatorFn(ParseOnnxParamsCol2ImV2)
    .ParseOpToGraphFn(ParseOnnxOpToGraphCol2ImV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
