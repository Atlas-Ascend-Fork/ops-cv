/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// rotated_box_decode_package/op_kernel/arch35/rotated_box_decode_struct.h
// =============================================================================
//
// ROLE: Ascend C template parameter (TPL) declarations for RotatedBoxDecode.
//   TPL 参数与 proto.md §3 kernel 函数签名一致:
//   template<int COPY_MODE, int UB_AXIS_SEL>
//
//   2 active TPL_SEL 组合（DESIGN §6 TPL_SEL 组合表）:
//     key=0: COPY_MODE=NDDMA(0) + UB_AXIS_SEL=UB_AXIS_N(0)   — N-axis multicore
//     key=1: COPY_MODE=NDDMA(0) + UB_AXIS_SEL=UB_AXIS_B(1)   — B-axis fullload
//   外层 dtype × 内层 TPL = 3 × 2 = 6 个 .o（DESIGN §6 binary 实例数）。
//
//   SetTilingKey 取值 {0,1} 与 §8 路由表 / 本节 TPL_SEL 一致:
//     host TilingFunc 据 SelectUbAxis 结果设 key=0 (ubAxis=N) 或 key=1 (ubAxis=B)。
//
// SOURCE: DESIGN.md §6 TilingKey 模板划分.
//
// =============================================================================

#ifndef ROTATED_BOX_DECODE_STRUCT_H_
#define ROTATED_BOX_DECODE_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

// ---------------------------------------------------------------------------
// TPL param value constants (DESIGN §6)
//   COPY_MODE: 0 = NDDMA (按通道 DataCopyPad + NDDMA transpose/split)
//   UB_AXIS_SEL: 0 = UB_AXIS_N (大 N 多核), 1 = UB_AXIS_B (小 N fullload 兜底),
//                2 = UB_AXIS_I (N==1 SIMT 分支, 见 rotated_box_decode_simt.h)
// ---------------------------------------------------------------------------
#define ROTATED_BOX_DECODE_COPY_MODE_NDDMA 0
#define ROTATED_BOX_DECODE_UB_AXIS_SEL_N 0
#define ROTATED_BOX_DECODE_UB_AXIS_SEL_B 1
#define ROTATED_BOX_DECODE_UB_AXIS_SEL_I 2

// ---------------------------------------------------------------------------
// ASCENDC_TPL_ARGS_DECL — declares compile-time template arguments
//   Param 1: COPY_MODE   (uint, bitwidth=8, values {0} — 1 active value NDDMA)
//   Param 2: UB_AXIS_SEL (uint, bitwidth=8, values {0, 1, 2} — 3 active values N/B/I)
// ---------------------------------------------------------------------------
ASCENDC_TPL_ARGS_DECL(RotatedBoxDecode,
                      ASCENDC_TPL_UINT_DECL(COPY_MODE, 8, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_COPY_MODE_NDDMA),
                      ASCENDC_TPL_UINT_DECL(UB_AXIS_SEL, 8, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_UB_AXIS_SEL_N,
                                            ROTATED_BOX_DECODE_UB_AXIS_SEL_B, ROTATED_BOX_DECODE_UB_AXIS_SEL_I));

// ---------------------------------------------------------------------------
// ASCENDC_TPL_SEL — generates template specialization selectors
//   3 active combinations (key=0 / key=1 / key=2):
//     (COPY_MODE=0, UB_AXIS_SEL=0) → key=0 (N-axis multicore, NDDMA)
//     (COPY_MODE=0, UB_AXIS_SEL=1) → key=1 (B-axis fullload, NDDMA)
//     (COPY_MODE=0, UB_AXIS_SEL=2) → key=2 (N==1 SIMT branch)
// ---------------------------------------------------------------------------
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(COPY_MODE, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_COPY_MODE_NDDMA),
                         ASCENDC_TPL_UINT_SEL(UB_AXIS_SEL, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_UB_AXIS_SEL_N)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(COPY_MODE, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_COPY_MODE_NDDMA),
                         ASCENDC_TPL_UINT_SEL(UB_AXIS_SEL, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_UB_AXIS_SEL_B)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(COPY_MODE, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_COPY_MODE_NDDMA),
                         ASCENDC_TPL_UINT_SEL(UB_AXIS_SEL, ASCENDC_TPL_UI_LIST, ROTATED_BOX_DECODE_UB_AXIS_SEL_I)));

#endif
