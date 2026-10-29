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
 * \file match_template_tiling_key.h
 * \brief match_template tiling key declare
 */

#ifndef __MATCH_TEMPLATE_TILING_KEY_H__
#define __MATCH_TEMPLATE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MATCH_TEMPLATE_SCH_MODE_HALO_TILE 1

ASCENDC_TPL_ARGS_DECL(MatchTemplate,
                      ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, MATCH_TEMPLATE_SCH_MODE_HALO_TILE),
                      ASCENDC_TPL_UINT_DECL(method, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          MATCH_TEMPLATE_SCH_MODE_HALO_TILE),
                                     ASCENDC_TPL_UINT_SEL(method, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5)));

#endif
