/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paste_sub_img_tiling_key.h
 * \brief Tiling key declaration for paste_sub_img operator
 *
 * Single template parameter:
 *   KEY (UINT): tiling key = mergedRank - ubAxis
 *     PASTE_SUB_IMG_KEY_UBAXIS_C = 1  -> split C axis
 *     PASTE_SUB_IMG_KEY_UBAXIS_W = 2  -> split W axis
 *     PASTE_SUB_IMG_KEY_UBAXIS_H = 3  -> split H axis
 */
#ifndef PASTE_SUB_IMG_TILING_KEY_H_
#define PASTE_SUB_IMG_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define PASTE_SUB_IMG_KEY_UBAXIS_C 1
#define PASTE_SUB_IMG_KEY_UBAXIS_W 2
#define PASTE_SUB_IMG_KEY_UBAXIS_H 3

ASCENDC_TPL_ARGS_DECL(PasteSubImg, ASCENDC_TPL_UINT_DECL(KEY, 3, ASCENDC_TPL_UI_RANGE, 1, 1, 3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(KEY, ASCENDC_TPL_UI_RANGE, 1, 1, 3),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(PasteSubImgTilingData)));

#endif // PASTE_SUB_IMG_TILING_KEY_H_
