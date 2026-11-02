/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECK_VALID_STRUCT_H
#define CHECK_VALID_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

#define CHECK_VALID_MODE_FP32 0
#define CHECK_VALID_MODE_FP16 1

// UINT modes encode their list positions; keep FP32 first to retain key 0.
ASCENDC_TPL_ARGS_DECL(CheckValid, ASCENDC_TPL_UINT_DECL(SCH_MODE, 1, ASCENDC_TPL_UI_LIST, CHECK_VALID_MODE_FP32,
                                                        CHECK_VALID_MODE_FP16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, CHECK_VALID_MODE_FP32)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, CHECK_VALID_MODE_FP16)));

#endif
