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

#include "cv_plugin_util.h"

namespace domi {
namespace {
const char* const kNmsV2 = "NonMaxSuppressionV2";
const char* const kNmsV3 = "NonMaxSuppressionV3";
const char* const kGreater = "Greater";
const char* const kGreaterEqual = "GreaterEqual";
const char* const kMinimum = "Minimum";

// 原插件按下标取常量，下标与算子原型的输入端口一一对应：
//   NonMaxSuppressionV2/V3 : input(2)=max_output_size, input(3)=iou_threshold
//   Greater/GreaterEqual/Minimum : input(0)=x1, input(1)=x2
const char* const kPortIouThreshold = "iou_threshold";
const char* const kPortMaxOutputSize = "max_output_size";
const char* const kPortX1 = "x1";
const char* const kPortX2 = "x2";

const char* const kScoreConstKey = "map/while/MultiClassNonMaxSuppression/FilterGreaterThan/Greater";
const char* const kSizeConstKey = "map/while/MultiClassNonMaxSuppression/Minimum";
const char* const kSizeConstKeySec = "map/while/MultiClassNonMaxSuppression/Minimum_";
const char* const kScoreRetinanetConstKey = "map/while/Greater";
const char* const kSizeRetinanetConstKey = "map/while/Minimum";
const char* const kRetinanetOutNum = "map/while/non_max_suppression_5/NonMaxSuppressionV3";
const char* const kSfScoreConstKey = "map/while/MultiClassNonMaxSuppression/GreaterEqual";
const char* const kScoreFaceboxConstKey = "map/while/GreaterEqual";
const char* const kSizeFaceboxConstKey = "map/while/non_max_suppression/NonMaxSuppressionV3";
const char* const kFaceboxOutNum = "map/while/non_max_suppression/NonMaxSuppressionV3";
const char* const kChangeCoordinateFrame = "ChangeCoordinateFrame";

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

template <typename T>
bool ReadConst(const ge::Operator& node, const char* port, T& value)
{
    ge::Tensor data;
    if (node.GetInputConstData(port, data) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* buf = data.GetData();
    if ((buf == nullptr) || (data.GetSize() < sizeof(T))) {
        return false;
    }
    return memcpy_s(&value, sizeof(T), buf, sizeof(T)) == EOK;
}

bool Contains(const std::string& haystack, const char* needle) { return haystack.find(needle) != std::string::npos; }

struct NmsAttrs {
    float iou_threshold = 0.0F;
    float score_threshold = 0.0F;
    int32_t max_size_per_class = 0;
    int32_t max_total_size = 0;
    bool change_coordinate_frame = false;
    bool has_iou = false;
    bool has_score = false;
    bool has_size_class = false;
    bool has_size_total = false;
};

void CollectFromNms(const ge::Operator& node, const std::string& name, const std::string& type, NmsAttrs& attrs)
{
    if ((type != kNmsV2) && (type != kNmsV3)) {
        return;
    }
    if (!attrs.has_iou) {
        attrs.has_iou = ReadConst(node, kPortIouThreshold, attrs.iou_threshold);
    }
    // facebox 路径：max_size_per_class 取自 NMS 的 max_output_size
    if ((!attrs.has_size_class) && (type == kNmsV3) && Contains(name, kSizeFaceboxConstKey)) {
        attrs.has_size_class = ReadConst(node, kPortMaxOutputSize, attrs.max_size_per_class);
    }
    // retinanet / facebox 路径：max_total_size 取自 NMS 的 max_output_size
    if ((!attrs.has_size_total) && (type == kNmsV3) &&
        (Contains(name, kRetinanetOutNum) || Contains(name, kFaceboxOutNum))) {
        attrs.has_size_total = ReadConst(node, kPortMaxOutputSize, attrs.max_total_size);
    }
}

void CollectFromCompare(const ge::Operator& node, const std::string& name, const std::string& type, NmsAttrs& attrs)
{
    if (attrs.has_score) {
        return;
    }
    const bool is_greater = (type == kGreater) &&
                            (Contains(name, kScoreRetinanetConstKey) || Contains(name, kScoreConstKey) ||
                             Contains(name, kScoreFaceboxConstKey));
    const bool is_greater_equal = (type == kGreaterEqual) &&
                                  (Contains(name, kScoreFaceboxConstKey) || Contains(name, kSfScoreConstKey));
    if (is_greater || is_greater_equal) {
        attrs.has_score = ReadConst(node, kPortX2, attrs.score_threshold);
    }
}

void CollectFromMinimum(const ge::Operator& node, const std::string& name, const std::string& type,
                        std::string& size_output_node, NmsAttrs& attrs)
{
    if ((type != kMinimum) || !(Contains(name, kSizeConstKey) || Contains(name, kSizeRetinanetConstKey))) {
        return;
    }
    if (!Contains(name, kSizeConstKeySec)) {
        attrs.has_size_class = ReadConst(node, kPortX1, attrs.max_size_per_class) || attrs.has_size_class;
    }
    // 原插件取名字最大的 Minimum 节点作为 max_total_size 来源
    if (size_output_node < name) {
        int32_t total = 0;
        if (ReadConst(node, kPortX1, total)) {
            size_output_node = name;
            attrs.max_total_size = total;
            attrs.has_size_total = true;
        }
    }
}
} // namespace

static Status BatchMultiClassNonMaxSuppressionParams(const std::vector<ge::Operator>& inside_nodes, ge::Operator& op)
{
    NmsAttrs attrs;
    std::string size_output_node;

    for (const auto& node : inside_nodes) {
        std::string name;
        std::string type;
        if (!GetNodeName(node, name) || !GetNodeType(node, type)) {
            OP_LOGE(GetOpName(op).c_str(), "Get inside node name or type failed.");
            return FAILED;
        }
        CollectFromNms(node, name, type, attrs);
        CollectFromCompare(node, name, type, attrs);
        CollectFromMinimum(node, name, type, size_output_node, attrs);
        if ((!attrs.change_coordinate_frame) && Contains(name, kChangeCoordinateFrame)) {
            attrs.change_coordinate_frame = true;
        }
    }

    if (!attrs.has_iou) {
        OP_LOGE(GetOpName(op).c_str(), "Can not find iou_threshold for BatchMultiClassNonMaxSuppression.");
        return FAILED;
    }
    if (!attrs.has_score) {
        OP_LOGE(GetOpName(op).c_str(), "Can not find score_threshold for BatchMultiClassNonMaxSuppression.");
        return FAILED;
    }
    if (!attrs.has_size_class) {
        OP_LOGE(GetOpName(op).c_str(), "Can not find max_size_per_class for BatchMultiClassNonMaxSuppression.");
        return FAILED;
    }
    if (!attrs.has_size_total) {
        OP_LOGE(GetOpName(op).c_str(), "Can not find max_total_size for BatchMultiClassNonMaxSuppression.");
        return FAILED;
    }

    (void)op.SetAttr("iou_threshold", attrs.iou_threshold);
    (void)op.SetAttr("score_threshold", attrs.score_threshold);
    (void)op.SetAttr("max_size_per_class", attrs.max_size_per_class);
    (void)op.SetAttr("max_total_size", attrs.max_total_size);
    (void)op.SetAttr("change_coordinate_frame", attrs.change_coordinate_frame);
    OP_LOGI(GetOpName(op).c_str(),
            "Set attr iou_threshold %1.2f, score_threshold %1.2f, max_size_per_class %d, max_total_size %d.",
            attrs.iou_threshold, attrs.score_threshold, attrs.max_size_per_class, attrs.max_total_size);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMultiClassNonMaxSuppression")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BatchMultiClassNonMaxSuppression")
    .FusionParseParamsFn(BatchMultiClassNonMaxSuppressionParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi
