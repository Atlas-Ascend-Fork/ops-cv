/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UPSAMPLE_BICUBIC2D_AA_GRAD_ERROR_H
#define UPSAMPLE_BICUBIC2D_AA_GRAD_ERROR_H

#include <string>
#include <vector>
#include "log/log.h"

namespace optiling {
inline void ReportBicubicGradMissingArgument(const char* opName, const char* paramName, const char* reason)
{
    const std::vector<const char*> keys = {"param_name", "op_name", "reason"};
    const std::vector<const char*> values = {paramName, opName, reason};
    OP_LOGE_WITHOUT_REPORT(opName, "Missing parameter %s: %s", paramName, reason);
    REPORT_PREDEFINED_ERR_MSG("EZ0037", keys, values);
}

inline void ReportBicubicGradTilingError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    OP_LOGE_WITHOUT_REPORT(opName, "Tiling failed: %s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}
} // namespace optiling
#endif
