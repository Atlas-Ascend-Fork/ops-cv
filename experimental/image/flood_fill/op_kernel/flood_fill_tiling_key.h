/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill_tiling_key.h
 * \brief Tiling key definition for FloodFill (dtype templates).
 */
#ifndef FLOOD_FILL_TILING_KEY_H_
#define FLOOD_FILL_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define FLOOD_FILL_TPL_UINT8 0
#define FLOOD_FILL_TPL_INT32 1
#define FLOOD_FILL_TPL_FLOAT32 2

ASCENDC_TPL_ARGS_DECL(FloodFill,
                      ASCENDC_TPL_UINT_DECL(dType, 2, ASCENDC_TPL_UI_LIST, FLOOD_FILL_TPL_UINT8, FLOOD_FILL_TPL_INT32,
                                            FLOOD_FILL_TPL_FLOAT32),
                      ASCENDC_TPL_UINT_DECL(maskInput, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                     ASCENDC_TPL_UINT_SEL(dType, ASCENDC_TPL_UI_LIST, FLOOD_FILL_TPL_UINT8),
                                     ASCENDC_TPL_UINT_SEL(maskInput, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                     ASCENDC_TPL_UINT_SEL(dType, ASCENDC_TPL_UI_LIST, FLOOD_FILL_TPL_INT32),
                                     ASCENDC_TPL_UINT_SEL(maskInput, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                     ASCENDC_TPL_UINT_SEL(dType, ASCENDC_TPL_UI_LIST, FLOOD_FILL_TPL_FLOAT32),
                                     ASCENDC_TPL_UINT_SEL(maskInput, ASCENDC_TPL_UI_LIST, 0, 1)));

#endif // FLOOD_FILL_TILING_KEY_H_
