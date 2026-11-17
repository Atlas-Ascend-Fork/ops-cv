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
 * \file guided_filter_tiling_key.h
 * \brief guided_filter tiling key declare
 */

#ifndef __GUIDED_FILTER_TILING_KEY_H__
#define __GUIDED_FILTER_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* 调度模式：只有一种实现（kernel 内按 UB 预算自适应分支，见 GfGeo::DeriveGeo），
 * 不声明未使用的模板组合。取值必须与 op_host/guided_filter_tiling.cpp 的 SetTilingKey 一致。 */
#define GUIDED_FILTER_TPL_SCH_MODE_0 0

/* 模板参数 */
ASCENDC_TPL_ARGS_DECL(GuidedFilter,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, GUIDED_FILTER_TPL_SCH_MODE_0));

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, GUIDED_FILTER_TPL_SCH_MODE_0)));

#endif
