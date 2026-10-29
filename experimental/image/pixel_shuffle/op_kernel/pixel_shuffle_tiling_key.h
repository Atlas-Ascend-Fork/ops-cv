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
 * \file pixel_shuffle_tiling_key.h
 * \brief pixel_shuffle tiling key declare
 */

#ifndef __PIXEL_SHUFFLE_TILING_KEY_H__
#define __PIXEL_SHUFFLE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* 调度模式：当前只有一种实现（单核函数内按 UB 预算自适应分支），
 * 需与 op_host tiling 里的 SetTilingKey 取值一致；不声明未使用的模板组合。 */
#define PIXEL_SHUFFLE_TPL_SCH_MODE_0 0

/* 模板参数 */
ASCENDC_TPL_ARGS_DECL(PixelShuffle,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, PIXEL_SHUFFLE_TPL_SCH_MODE_0));

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, PIXEL_SHUFFLE_TPL_SCH_MODE_0)));

#endif
