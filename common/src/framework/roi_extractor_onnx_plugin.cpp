/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
static const uint32_t MIN_INPUT_NUM = 2;

static Status VerifyRoiExtractorByNode(ge::Operator& op_dest, const ge::onnx::NodeProto* node)
{
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    uint32_t input_num = node->input_size();
    if (input_num < MIN_INPUT_NUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "input num must be greater than or equal to 2, but got %u", input_num);
        return FAILED;
    }
    op_dest.DynamicInputRegister("features", input_num - 1, false);
    return SUCCESS;
}

static Status ParseParamsRoiExtractor(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    Status ret = VerifyRoiExtractorByNode(op_dest, node);
    if (ret != SUCCESS) {
        return FAILED;
    }

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "finest_scale" && attr.type() == ge::onnx::AttributeProto::INT) {
            int64_t finest_scale = attr.i();
            op_dest.SetAttr("finest_scale", finest_scale);
        }
        if (attr.name() == "roi_scale_factor" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            float roi_scale_factor = attr.f();
            op_dest.SetAttr("roi_scale_factor", roi_scale_factor);
        }
        if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
            std::vector<float> spatial_scale;
            for (auto s : attr.floats()) {
                spatial_scale.push_back(s);
            }
            op_dest.SetAttr("spatial_scale", spatial_scale);
        }
        if (attr.name() == "pooled_height" && attr.type() == ge::onnx::AttributeProto::INT) {
            int64_t output_size = attr.i();
            op_dest.SetAttr("pooled_height", output_size);
        }
        if (attr.name() == "pooled_width" && attr.type() == ge::onnx::AttributeProto::INT) {
            int64_t output_size = attr.i();
            op_dest.SetAttr("pooled_width", output_size);
        }
        if (attr.name() == "sample_num" && attr.type() == ge::onnx::AttributeProto::INT) {
            int64_t sample_num = attr.i();
            op_dest.SetAttr("sample_num", sample_num);
        }
        if (attr.name() == "pool_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
            std::string pool_mode = attr.s();
            op_dest.SetAttr("pool_mode", pool_mode);
        }
        if (attr.name() == "aligned" && attr.type() == ge::onnx::AttributeProto::INT) {
            bool aligned = attr.i();
            op_dest.SetAttr("aligned", aligned);
        }
    }

    return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("RoiExtractor")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::RoiExtractor"), ge::AscendString("ai.onnx::9::RoiExtractor"),
                   ge::AscendString("ai.onnx::10::RoiExtractor"), ge::AscendString("ai.onnx::11::RoiExtractor"),
                   ge::AscendString("ai.onnx::12::RoiExtractor"), ge::AscendString("ai.onnx::13::RoiExtractor"),
                   ge::AscendString("ai.onnx::14::RoiExtractor"), ge::AscendString("ai.onnx::15::RoiExtractor"),
                   ge::AscendString("ai.onnx::16::RoiExtractor")})
    .ParseParamsFn(ParseParamsRoiExtractor)
    .ImplyType(ImplyType::TVM);
} // namespace domi
