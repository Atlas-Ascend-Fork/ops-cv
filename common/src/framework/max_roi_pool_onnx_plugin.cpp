/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file max_roi_pool_onnx_plugin.cpp
 * \brief
 */
#include "cv_plugin_util.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "register/register.h"

namespace domi {
using json = nlohmann::json;

static const size_t POOLED_SHAPE_SIZE = 2;

static Status ParseParamsMaxRoiPool(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float spatial_scale = 1.0;
    std::vector<int> pooled_shape;

    ge::AscendString attrs_string;
    try {
        if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
            json attrs = json::parse(attrs_string.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (json& attr : attrs["attribute"]) {
                    const std::string name = attr.value("name", "");
                    if (name == "pooled_shape" && attr.contains("ints")) {
                        for (auto& v : attr["ints"]) {
                            pooled_shape.push_back(static_cast<int>(v.get<int64_t>()));
                        }
                    } else if (name == "spatial_scale" && attr.value("type", -1) == ATTR_TYPE_FLOAT) {
                        // GE omits f when spatial_scale is 0. Do not keep the default 1.0.
                        if (!attr.contains("f")) {
                            spatial_scale = 0.0f;
                        } else {
                            std::string spatial_scale_str = attr["f"];
                            if (StrToFloat(spatial_scale_str, spatial_scale) != SUCCESS) {
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

    if (pooled_shape.size() != POOLED_SHAPE_SIZE) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Obtain attr pooled_shape failed.");
        return FAILED;
    }

    op_dest.SetAttr("pooled_h", pooled_shape[0]);
    op_dest.SetAttr("pooled_w", pooled_shape[1]);
    op_dest.SetAttr("spatial_scale_h", spatial_scale);
    op_dest.SetAttr("spatial_scale_w", spatial_scale);
    return SUCCESS;
}

// register MaxRoiPool op info to GE
REGISTER_CUSTOM_OP("ROIPooling")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::1::MaxRoiPool"), ge::AscendString("ai.onnx::8::MaxRoiPool"),
                   ge::AscendString("ai.onnx::9::MaxRoiPool"), ge::AscendString("ai.onnx::10::MaxRoiPool"),
                   ge::AscendString("ai.onnx::11::MaxRoiPool"), ge::AscendString("ai.onnx::12::MaxRoiPool"),
                   ge::AscendString("ai.onnx::13::MaxRoiPool"), ge::AscendString("ai.onnx::14::MaxRoiPool"),
                   ge::AscendString("ai.onnx::15::MaxRoiPool"), ge::AscendString("ai.onnx::16::MaxRoiPool"),
                   ge::AscendString("ai.onnx::17::MaxRoiPool"), ge::AscendString("ai.onnx::18::MaxRoiPool")})
    .ParseParamsByOperatorFn(ParseParamsMaxRoiPool)
    .ImplyType(ImplyType::TVM);
} // namespace domi
