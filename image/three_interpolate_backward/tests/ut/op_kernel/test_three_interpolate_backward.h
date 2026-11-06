/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TEST_THREE_INTERPOLATE_BACKWARD_
#define _TEST_THREE_INTERPOLATE_BACKWARD_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct ThreeInterpolateBackwardTilingData {
    uint32_t used_core_num;
    uint32_t bs;
    uint32_t c1;
    uint32_t ms;
    uint32_t ns;
    uint32_t each_core_proc_num;
    uint32_t each_core_loop_times;
    uint32_t each_core_each_loop_n_cnt;
    uint32_t each_core_last_loop_n_cnt;
    uint32_t last_core_proc_num;
    uint32_t last_core_loop_times;
    uint32_t last_core_each_loop_n_cnt;
    uint32_t last_core_last_loop_n_cnt;
    uint32_t weight_move_block_size;
    uint32_t idx_move_block_size;
    uint32_t grad_x_move_block_size;
    uint32_t grad_y_move_block_size;
    uint32_t c_move_num;
    uint32_t c_last_loop_move_num;
    uint32_t c_move_loop_times;
    uint32_t mulit_core_mode;
    uint32_t each_core_proc_batch_num;
    uint32_t core_proc_batch_padding_idx;
    uint32_t rsv;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)              \
    __ubuf__ tilingStruct* tilingDataPointer = reinterpret_cast<__ubuf__ tilingStruct*>( \
        (__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, element) \
    (tilingData).element = tilingDataPointer->element

#define GET_TILING_DATA(tilingData, tilingPointer)                                          \
    ThreeInterpolateBackwardTilingData tilingData;                                          \
    INIT_TILING_DATA(ThreeInterpolateBackwardTilingData, tilingDataPointer, tilingPointer); \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, used_core_num);                 \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, bs);                            \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, c1);                            \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, ms);                            \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, ns);                            \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, each_core_proc_num);            \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, each_core_loop_times);          \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, each_core_each_loop_n_cnt);     \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, each_core_last_loop_n_cnt);     \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, last_core_proc_num);            \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, last_core_loop_times);          \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, last_core_each_loop_n_cnt);     \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, last_core_last_loop_n_cnt);     \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, weight_move_block_size);        \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, idx_move_block_size);           \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, grad_x_move_block_size);        \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, grad_y_move_block_size);        \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, c_move_num);                    \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, c_last_loop_move_num);          \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, c_move_loop_times);             \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, mulit_core_mode);               \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, each_core_proc_batch_num);      \
    COPY_TILIGN_DATA_ELEMENT(tilingData, tilingDataPointer, core_proc_batch_padding_idx);
#endif
