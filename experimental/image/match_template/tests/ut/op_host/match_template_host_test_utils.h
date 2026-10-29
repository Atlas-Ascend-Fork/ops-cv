/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "infershape_case_executor.h"

namespace match_template_test {

inline void CheckInfer(const gert::Shape& image, const gert::Shape& templateImage, ge::graphStatus expectedStatus,
                       const std::vector<std::vector<int64_t>>& expectedShapes = {},
                       ge::DataType imageType = ge::DT_FLOAT, ge::DataType templateType = ge::DT_FLOAT)
{
    // Use the shared fixture and public SDK builder. Input Tensor objects
    // carry metadata only; no input data or context-layout casts are needed.
    gert::StorageShape imageStorage;
    imageStorage.MutableOriginShape() = image;
    imageStorage.MutableStorageShape() = image;
    gert::StorageShape templateStorage;
    templateStorage.MutableOriginShape() = templateImage;
    templateStorage.MutableStorageShape() = templateImage;
    gert::InfershapeContextPara parameters(
        "MatchTemplate", {{imageStorage, imageType, ge::FORMAT_ND}, {templateStorage, templateType, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT, ge::FORMAT_ND}, {{}, ge::DT_FLOAT, ge::FORMAT_ND}, {{}, ge::DT_INT64, ge::FORMAT_ND}});
    ExecuteTestCase(parameters, expectedStatus, expectedShapes);
}

} // namespace match_template_test
