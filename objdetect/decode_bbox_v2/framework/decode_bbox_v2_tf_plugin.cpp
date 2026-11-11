/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>

#include <map>
#include <string>
#include <vector>

#include "cv_plugin_util.h"

namespace domi {
namespace {
const char* const kBoxesUnpack = "/unstack";
const char* const kBoxesDiv = "RealDiv";
const size_t kRealDivInputSize = 2;
const size_t kScaleSize = 4;

bool GetNodeName(const ge::Operator& node, std::string& name)
{
    ge::AscendString ascend_name;
    if ((node.GetName(ascend_name) != ge::GRAPH_SUCCESS) || (ascend_name.GetString() == nullptr)) {
        return false;
    }
    name = ascend_name.GetString();
    return true;
}

bool GetNodeType(const ge::Operator& node, std::string& type)
{
    ge::AscendString ascend_type;
    if ((node.GetOpType(ascend_type) != ge::GRAPH_SUCCESS) || (ascend_type.GetString() == nullptr)) {
        return false;
    }
    type = ascend_type.GetString();
    return true;
}

bool GetInputName(const ge::Operator& node, uint32_t index, std::string& name)
{
    ge::TensorDesc input_desc = node.GetInputDesc(index);
    ge::AscendString ascend_name;
    if ((input_desc.GetName(ascend_name) != ge::GRAPH_SUCCESS) || (ascend_name.GetString() == nullptr)) {
        return false;
    }
    name = ascend_name.GetString();
    return true;
}

bool ReadConstValue(const ge::Operator& node, float& value)
{
    ge::Tensor tensor;
    if (node.GetAttr("value", tensor) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* buf = tensor.GetData();
    if ((buf == nullptr) || (tensor.GetSize() < sizeof(float))) {
        return false;
    }
    return memcpy_s(&value, sizeof(float), buf, sizeof(float)) == EOK;
}
} // namespace

// 参照 CANN 内置 TF 插件 decode_bbox_v2_plugin.cc：把 TF 图中 boxes 经 /unstack
// 拆 4 路后逐路除以常量（RealDiv）的融合子图折叠为 scales 属性，decode_clip /
// reversed_box 走默认值。
static Status DecodeBboxV2Params(const std::vector<ge::Operator>& inside_nodes, ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter DecodeBboxV2 fusion parser.");

    std::map<std::string, std::string> scales_const_name_map;
    std::map<std::string, const ge::Operator*> node_map;
    for (const auto& node : inside_nodes) {
        std::string name;
        std::string type;
        if (!GetNodeName(node, name) || !GetNodeType(node, type)) {
            OP_LOGE(GetOpName(op).c_str(), "Get inside node name or type failed.");
            return FAILED;
        }

        if (type == kBoxesDiv) {
            if (node.GetInputsSize() < kRealDivInputSize) {
                OP_LOGE(GetOpName(op).c_str(), "RealDiv input size is less than 2.");
                return FAILED;
            }
            std::string unpack_name;
            std::string const_name;
            if (!GetInputName(node, 0, unpack_name) || !GetInputName(node, 1, const_name)) {
                OP_LOGE(GetOpName(op).c_str(), "Get RealDiv input name failed.");
                return FAILED;
            }
            if (unpack_name.find(kBoxesUnpack) != std::string::npos) {
                scales_const_name_map[name] = const_name;
            }
        }
        node_map[name] = &node;
    }

    std::vector<float> scales_list = {1.0f, 1.0f, 1.0f, 1.0f};
    if (scales_const_name_map.size() == kScaleSize) {
        size_t i = 0;
        for (const auto& name_pair : scales_const_name_map) {
            auto it = node_map.find(name_pair.second);
            if (it == node_map.end()) {
                OP_LOGE(GetOpName(op).c_str(), "Can not find scales const node %s.", name_pair.second.c_str());
                return FAILED;
            }
            if (!ReadConstValue(*it->second, scales_list[i])) {
                OP_LOGE(GetOpName(op).c_str(), "Convert scales const %s failed.", name_pair.second.c_str());
                return PARAM_INVALID;
            }
            ++i;
        }
    } else {
        OP_LOGI(GetOpName(op).c_str(), "Boxes don't need to scale, use default scales [1,1,1,1].");
    }

    (void)op.SetAttr("scales", scales_list);
    OP_LOGI(GetOpName(op).c_str(), "DecodeBboxV2 scales=[%.1f, %.1f, %.1f, %.1f].", scales_list[0], scales_list[1],
            scales_list[2], scales_list[3]);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DecodeBboxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBboxV2")
    .FusionParseParamsFn(DecodeBboxV2Params)
    .ImplyType(ImplyType::TVM);
} // namespace domi
