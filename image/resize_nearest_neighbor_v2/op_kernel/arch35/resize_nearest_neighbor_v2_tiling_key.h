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
 * \file resize_nearest_neighbor_v2_simt.h
 * \brief resize_nearest_neighbor_v2_simt
 */

#ifndef _RESIZE_NEAREST_NEIGHBOR_V2_TILING_KEY_DECL_H_
#define _RESIZE_NEAREST_NEIGHBOR_V2_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCH_MODE_DATA_COPY_SMALL_C 0
#define TPL_SCH_MODE_DATA_COPY_BIG_C 1
#define TPL_SCH_MODE_DATA_COPY_AGGR_C 2
#define TPL_SCH_MODE_SIMT_COMMON 3
#define TPL_SCH_MODE_SIMT_INPUT_EQ_OUTPUT 4
#define TPL_SCH_MODE_SIMT_INPUT_EQ_ONE 5
#define TPL_SCH_MODE_DATA_COPY_ALL_W_OUT 6
#define TPL_SCH_MODE_DATA_COPY_NOT_ALL_W_OUT 7
#define TPL_SCH_MODE_DATA_COPY_CUT_NH 8
#define TPL_SCH_MODE_GATHER_ALL_HW 9
#define TPL_SCH_MODE_GATHER_CUT_H 10

#define TPL_FORMAT_NCHW 0
#define TPL_FORMAT_NHWC 1
#define TPL_FORMAT_ND 2

#define TPL_ALIGN_CORNERS_0 0
#define TPL_ALIGN_CORNERS_1 1

#define TPL_HALF_PIXEL_CENTERS_0 0
#define TPL_HALF_PIXEL_CENTERS_1 1

#define TPL_IDX_INT32_0 0
#define TPL_IDX_INT32_1 1

#define RESIZE_NEAREST_NEIGHBOR_V2_TPL_KEY_DECL()                                                                \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_DATA_COPY_SMALL_C,          \
                          TPL_SCH_MODE_DATA_COPY_BIG_C, TPL_SCH_MODE_DATA_COPY_AGGR_C, TPL_SCH_MODE_SIMT_COMMON, \
                          TPL_SCH_MODE_SIMT_INPUT_EQ_OUTPUT, TPL_SCH_MODE_SIMT_INPUT_EQ_ONE,                     \
                          TPL_SCH_MODE_DATA_COPY_ALL_W_OUT, TPL_SCH_MODE_DATA_COPY_NOT_ALL_W_OUT,                \
                          TPL_SCH_MODE_DATA_COPY_CUT_NH, TPL_SCH_MODE_GATHER_ALL_HW, TPL_SCH_MODE_GATHER_CUT_H), \
        ASCENDC_TPL_UINT_DECL(format, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC,   \
                              TPL_FORMAT_ND),                                                                    \
        ASCENDC_TPL_UINT_DECL(alignCorners, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_0,          \
                              TPL_ALIGN_CORNERS_1),                                                              \
        ASCENDC_TPL_UINT_DECL(halfPixelCenters, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_0, \
                              TPL_HALF_PIXEL_CENTERS_1),                                                         \
        ASCENDC_TPL_UINT_DECL(idxInt32, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32_0, TPL_IDX_INT32_1)

#define RESIZE_NEAREST_NEIGHBOR_V2_TPL_KEY_SEL()                                                                  \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_SIMT_COMMON, TPL_SCH_MODE_SIMT_INPUT_EQ_OUTPUT, \
                         TPL_SCH_MODE_SIMT_INPUT_EQ_ONE),                                                         \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC, TPL_FORMAT_ND),       \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_0, TPL_ALIGN_CORNERS_1),        \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_0,                     \
                             TPL_HALF_PIXEL_CENTERS_1),                                                           \
        ASCENDC_TPL_UINT_SEL(idxInt32, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32_0, TPL_IDX_INT32_1)

#define RESIZE_NEAREST_NEIGHBOR_V2_SIMD_TPL_KEY_SEL()                                                              \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_DATA_COPY_SMALL_C, TPL_SCH_MODE_DATA_COPY_BIG_C, \
                         TPL_SCH_MODE_DATA_COPY_AGGR_C),                                                           \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NHWC),                                        \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_0),                              \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_0,                      \
                             TPL_HALF_PIXEL_CENTERS_1),                                                            \
        ASCENDC_TPL_UINT_SEL(idxInt32, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32_0, TPL_IDX_INT32_1)

#define RESIZE_NEAREST_NEIGHBOR_V2_NHWC_TPL_KEY_SEL()                                          \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_DATA_COPY_ALL_W_OUT,         \
                         TPL_SCH_MODE_DATA_COPY_NOT_ALL_W_OUT, TPL_SCH_MODE_DATA_COPY_CUT_NH), \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NHWC),                    \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_0),          \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_0), \
        ASCENDC_TPL_UINT_SEL(idxInt32, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32_1)

#define RESIZE_NEAREST_NEIGHBOR_V2_GATHER_TPL_KEY_SEL()                                                      \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_GATHER_ALL_HW, TPL_SCH_MODE_GATHER_CUT_H), \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW),                                  \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_0, TPL_ALIGN_CORNERS_1),   \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_0,                \
                             TPL_HALF_PIXEL_CENTERS_1),                                                      \
        ASCENDC_TPL_UINT_SEL(idxInt32, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32_1)

ASCENDC_TPL_ARGS_DECL(ResizeNearestNeighborV2, RESIZE_NEAREST_NEIGHBOR_V2_TPL_KEY_DECL());
#ifdef __CCE_AICORE_SUPPORT_SIMT__
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_TPL_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_SIMD_TPL_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_NHWC_TPL_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_GATHER_TPL_KEY_SEL()));
#else
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_SIMD_TPL_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_NHWC_TPL_KEY_SEL()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_GATHER_TPL_KEY_SEL()));
#endif
#endif
