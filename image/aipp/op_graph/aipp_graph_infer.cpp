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
 * \file aipp_graph_infer.cpp
 * \brief Aipp InferDataType. Compile only.
 *        static: keep features dtype already on the output.
 *        dynamic:features dtype follows images dtype.
 */

#include <string>
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"
#include "op_api/error_report_util.h"
#include "register/op_impl_registry.h"

namespace ops {
constexpr size_t kImagesIdx = 0;
constexpr size_t kFeaturesIdx = 0;
constexpr size_t kAttrCfgIdx = 0;
constexpr const char* kAttrAippMode = "aipp_mode";
constexpr const char* kValueAippModeDynamic = "dynamic";
constexpr const char* kValueAippModeStatic = "static";

ge::graphStatus InferDataTypeForAipp(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        error_report::ReportInferShapeStageError("Aipp", "InferDataTypeContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType outDtype = context->GetOutputDataType(kFeaturesIdx);
    try {
        auto attrs = context->GetAttrs();
        if (attrs == nullptr) {
            error_report::ReportInferShapeStageError(context->GetNodeName(), "attrs is null in InferDataType context");
            return ge::GRAPH_FAILED;
        }
        const char* configData = attrs->GetAttrPointer<char>(kAttrCfgIdx);
        if (configData == nullptr) {
            error_report::ReportInvalidArgument(context->GetNodeName(), "aipp_config_path",
                                                "configData is null in InferDataType context");
            return ge::GRAPH_FAILED;
        }
        nlohmann::json cfgJson = nlohmann::json::parse(configData);
        OP_CHECK_IF(!cfgJson.is_object(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "aipp_config_path", configData,
                                                          "aipp_config_path must be a json object"),
                    return ge::GRAPH_FAILED);
        if (!cfgJson.contains(kAttrAippMode) || !cfgJson[kAttrAippMode].is_string()) {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "aipp_mode");
            return ge::GRAPH_FAILED;
        }
        const std::string aippMode = cfgJson[kAttrAippMode].get<std::string>();
        if (aippMode == kValueAippModeDynamic) {
            return context->SetOutputDataType(kFeaturesIdx, context->GetInputDataType(kImagesIdx));
        }
        if (aippMode != kValueAippModeStatic) {
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "aipp_mode", aippMode.c_str(), "static or dynamic");
            return ge::GRAPH_FAILED;
        }
        return context->SetOutputDataType(kFeaturesIdx, outDtype);
    } catch (...) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "aipp_config_path", "invalid json",
                                              "failed to parse aipp_config_path");
        return ge::GRAPH_FAILED;
    }
}

IMPL_OP(Aipp).InferDataType(InferDataTypeForAipp);
} // namespace ops
