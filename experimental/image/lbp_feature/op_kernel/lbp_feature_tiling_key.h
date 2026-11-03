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
 * \file lbp_feature_tiling_key.h
 * \brief lbp_feature tiling key declare
 */

#ifndef __LBP_FEATURE_TILING_KEY_H__
#define __LBP_FEATURE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define LBP_FEATURE_SCH_MODE_ROW_SPLIT 1
#define LBP_FEATURE_SCH_MODE_TILE_2D 2

ASCENDC_TPL_ARGS_DECL(LbpFeature, ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, LBP_FEATURE_SCH_MODE_ROW_SPLIT,
                                                        LBP_FEATURE_SCH_MODE_TILE_2D));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, LBP_FEATURE_SCH_MODE_ROW_SPLIT,
                                                          LBP_FEATURE_SCH_MODE_TILE_2D)));

#endif // __LBP_FEATURE_TILING_KEY_H__
