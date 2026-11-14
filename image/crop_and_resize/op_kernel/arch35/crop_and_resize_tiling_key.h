/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file crop_and_resize_tiling_key.h
 * \brief Tiling key declare for crop_and_resize operator
 *
 * 单模板参数：
 *   schMode（UINT 2-bit）：0 = bilinear NHWC，1 = bilinear NCHW，
 *                          2 = nearest NHWC，3 = nearest NCHW
 * dtype 不编码进 TilingKey；DTYPE_ 宏自动实例化所有 dtype 组合。
 */

#ifndef CROP_AND_RESIZE_TILING_KEY_H_
#define CROP_AND_RESIZE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// 场景模式：interp × layout（ND/NHWC 语义合并为 NHWC 模式）；TilingKey 仅编码场景模式，禁止枚举 dtype
// NHWC 值 0 与原单值模式相同：既有 ND/NHWC 路径 TilingKey 二进制兼容（零回归）
#define CROP_AND_RESIZE_MODE_BILINEAR_NHWC 0
#define CROP_AND_RESIZE_MODE_BILINEAR_NCHW 1
#define CROP_AND_RESIZE_MODE_NEAREST_NHWC 2
#define CROP_AND_RESIZE_MODE_NEAREST_NCHW 3

ASCENDC_TPL_ARGS_DECL(CropAndResize,
                      ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_BILINEAR_NHWC,
                                            CROP_AND_RESIZE_MODE_BILINEAR_NCHW, CROP_AND_RESIZE_MODE_NEAREST_NHWC,
                                            CROP_AND_RESIZE_MODE_NEAREST_NCHW));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_BILINEAR_NHWC),
                         ASCENDC_TPL_TILING_STRUCT_SEL(CropAndResizeTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_BILINEAR_NCHW),
                         ASCENDC_TPL_TILING_STRUCT_SEL(CropAndResizeTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_NEAREST_NHWC),
                         ASCENDC_TPL_TILING_STRUCT_SEL(CropAndResizeTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, CROP_AND_RESIZE_MODE_NEAREST_NCHW),
                         ASCENDC_TPL_TILING_STRUCT_SEL(CropAndResizeTilingData)));

#endif // CROP_AND_RESIZE_TILING_KEY_H_
