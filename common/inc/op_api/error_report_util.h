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
 * \file error_report_util.h
 * \brief 结构化错误码上报工具: tiling/infershape 阶段的框架契约失败与参数缺失上报
 *
 * 分层规则:
 *   - 输入 desc/shape 缺失(图侧未提供输入) ........................ 记 EZ0037 参数错误(ReportInvalidArgument)
 *   - 框架构造的容器/指针为空(attrs/compileInfo/输出desc/平台信息) :
 *       tiling 阶段 ................................................ 记 EZ1008(ReportTilingStageError)
 *       infershape 阶段 ............................................ 记 EZ1014(ReportInferShapeStageError)
 *   - 算子参数值类非法(dtype/format/dim/size/value) ................. 使用 log.h 的 OP_LOGE_FOR_INVALID_* 宏族
 */
#ifndef OP_HOST_ERROR_REPORT_UTIL_H
#define OP_HOST_ERROR_REPORT_UTIL_H

#include <string>
#include <vector>
#include "log/log.h"

namespace error_report {
// EZ0037: 参数非法(单参数, 带原因)
inline void ReportInvalidArgument(const char* opName, const char* paramName, const std::string& reason)
{
    const std::string safeOpName = opName != nullptr ? opName : "nil";
    const std::string safeParamName = paramName != nullptr ? paramName : "nil";
    OP_LOGE_LIBOPAPI_REPORT(safeOpName.c_str(), "Parameter %s of %s is invalid. Reason: %s.", safeParamName.c_str(),
                            safeOpName.c_str(), reason.c_str());
    const std::vector<const char*> msgKey = {"param_name", "op_name", "reason"};
    const std::vector<const char*> msgValue = {safeParamName.c_str(), safeOpName.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0037", msgKey, msgValue);
}

// EZ1008: tiling 阶段框架契约失败
inline void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::string safeOpName = opName != nullptr ? opName : "nil";
    OP_LOGE_LIBOPAPI_REPORT(safeOpName.c_str(), "Tiling stage contract failure. Reason: %s.", reason.c_str());
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {safeOpName.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

// EZ1014: infershape 阶段框架契约失败
inline void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::string safeOpName = opName != nullptr ? opName : "nil";
    OP_LOGE_LIBOPAPI_REPORT(safeOpName.c_str(), "Infer-shape stage contract failure. Reason: %s.", reason.c_str());
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {safeOpName.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}
} // namespace error_report

#endif // OP_HOST_ERROR_REPORT_UTIL_H
