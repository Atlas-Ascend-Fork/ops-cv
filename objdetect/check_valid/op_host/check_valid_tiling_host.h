/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECK_VALID_TILING_HOST_H
#define CHECK_VALID_TILING_HOST_H

#include <cstdint>
#include "../op_kernel/arch35/check_valid_tiling_data.h"

namespace optiling {

constexpr int32_t CV_DTYPE_FLOAT = 0;
constexpr int32_t CV_DTYPE_FLOAT16 = 1;
constexpr int32_t CV_DTYPE_INT8 = 2;
constexpr int32_t CV_DTYPE_INT32 = 3;
constexpr int32_t CV_DTYPE_DOUBLE = 11;

constexpr uint32_t CV_STATUS_SUCCESS = 0;
constexpr uint32_t CV_STATUS_FAILED = 0xFFFFFFFF;

constexpr uint32_t CV_KEY_NORMAL = 0;
constexpr uint32_t CV_KEY_EMPTY = 1;

struct CheckValidPublicInputs {
    int32_t bbox_rank;
    int64_t bbox_rows;
    int64_t bbox_cols;
    int32_t bbox_dtype;
    int32_t img_metas_rank;
    int64_t img_metas_numel;
    int32_t img_metas_dtype;
    float H;
    float W;
    float r;
    int64_t available_cores;
};

struct CheckValidPublicResult {
    uint32_t tiling_key;
    uint32_t block_dim;
    CheckValidTilingData td;
};

uint32_t ComputePublicTiling(const CheckValidPublicInputs& in, CheckValidPublicResult& out);

struct CheckValidBranch0Inputs {
    int64_t N;
    int64_t available_cores;
    int64_t ub_available;
    int32_t dtype;
};

struct CheckValidBranch0Result {
    uint32_t block_dim;
    int64_t tile_n;
    int64_t tile_n_tail;
    int64_t num_tiles;
    int64_t num_cores;
    int64_t tiles_main;
    int64_t cores_tail;
    int64_t per_buf_bytes;
};

uint32_t ComputeBranch0Tiling(const CheckValidBranch0Inputs& in, CheckValidBranch0Result& out);

struct CheckValidBranch1Inputs {
    int64_t N;
    int32_t dtype;
    float H;
    float W;
    float r;
};

struct CheckValidBranch1Result {
    uint32_t tiling_key;
    uint32_t block_dim;
    int64_t N;
    int64_t tile_n;
    int64_t tile_n_tail;
    int64_t num_tiles;
    int64_t num_cores;
    int64_t tiles_main;
    int64_t cores_tail;
    int64_t per_buf_bytes;
    float img_width_x;
    float img_height_y;
};

uint32_t ComputeBranch1Tiling(const CheckValidBranch1Inputs& in, CheckValidBranch1Result& out);

} // namespace optiling

#endif // CHECK_VALID_TILING_HOST_H
