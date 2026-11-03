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
 * \file scharr_tiling_key.h
 * \brief Scharr tiling key declare. schMode = direction x border:
 *        1=X+REFLECT_101 2=X+REPLICATE 3=X+CONSTANT
 *        4=Y+REFLECT_101 5=Y+REPLICATE 6=Y+CONSTANT
 */

#ifndef __SCHARR_TILING_KEY_H__
#define __SCHARR_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SCHARR_SCH_MODE_DEFAULT 1

ASCENDC_TPL_ARGS_DECL(Scharr,
                      ASCENDC_TPL_UINT_DECL(schMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, SCHARR_SCH_MODE_DEFAULT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SCHARR_SCH_MODE_DEFAULT)));

#endif // __SCHARR_TILING_KEY_H__
