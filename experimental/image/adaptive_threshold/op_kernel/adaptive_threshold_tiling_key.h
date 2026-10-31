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
 * \file adaptive_threshold_tiling_key.h
 * \brief Tiling key definition for AdaptiveThreshold.
 */
#ifndef ADAPTIVE_THRESHOLD_TILING_KEY_H_
#define ADAPTIVE_THRESHOLD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define ADAPTIVE_THRESHOLD_METHOD_MEAN 0
#define ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN 1

#define ADAPTIVE_THRESHOLD_TPL_UINT8 0
#define ADAPTIVE_THRESHOLD_TPL_FLOAT16 1
#define ADAPTIVE_THRESHOLD_TPL_FLOAT32 2

ASCENDC_TPL_ARGS_DECL(AdaptiveThreshold,
                      ASCENDC_TPL_UINT_DECL(adaptiveMethod, 1, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_METHOD_MEAN,
                                            ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN),
                      ASCENDC_TPL_UINT_DECL(dType, 2, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_TPL_UINT8,
                                            ADAPTIVE_THRESHOLD_TPL_FLOAT16, ADAPTIVE_THRESHOLD_TPL_FLOAT32));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(adaptiveMethod, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_METHOD_MEAN,
                                              ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN),
                         ASCENDC_TPL_UINT_SEL(dType, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_TPL_UINT8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(adaptiveMethod, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_METHOD_MEAN,
                                              ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN),
                         ASCENDC_TPL_UINT_SEL(dType, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_TPL_FLOAT16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(adaptiveMethod, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_METHOD_MEAN,
                                              ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN),
                         ASCENDC_TPL_UINT_SEL(dType, ASCENDC_TPL_UI_LIST, ADAPTIVE_THRESHOLD_TPL_FLOAT32)));

#endif // ADAPTIVE_THRESHOLD_TILING_KEY_H_
