/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check_valid_tiling_host.h"

#include <algorithm>
#include <cinttypes>
#include <limits>

#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {

uint32_t ComputePublicTiling(const CheckValidPublicInputs& in, CheckValidPublicResult& out)
{
    bool bbox_dtype_ok = (in.bbox_dtype == CV_DTYPE_FLOAT || in.bbox_dtype == CV_DTYPE_FLOAT16);
    bool dtype_match = (in.bbox_dtype == in.img_metas_dtype);
    if (!bbox_dtype_ok || !dtype_match) {
        OP_LOGE("CheckValid", "bbox_tensor dtype=%s, img_metas dtype=%s; expected matching FLOAT16 or FLOAT inputs.",
                ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(in.bbox_dtype)).c_str(),
                ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(in.img_metas_dtype)).c_str());
        return CV_STATUS_FAILED;
    }

    if (in.bbox_rank != 2) {
        OP_LOGE("CheckValid", "bbox_tensor rank=%d; expected rank=2.", in.bbox_rank);
        return CV_STATUS_FAILED;
    }

    if (in.bbox_cols != 4) {
        OP_LOGE("CheckValid", "bbox_tensor last dim=%" PRId64 "; expected last dim=4.", in.bbox_cols);
        return CV_STATUS_FAILED;
    }

    if (in.bbox_rows < 0) {
        OP_LOGE("CheckValid", "bbox_tensor runtime N=%" PRId64 "; expected a nonnegative row count.", in.bbox_rows);
        return CV_STATUS_FAILED;
    }

    if (in.img_metas_rank != 1) {
        OP_LOGE("CheckValid", "img_metas rank=%d; expected rank=1 (a one-dimensional tensor).", in.img_metas_rank);
        return CV_STATUS_FAILED;
    }

    if (in.img_metas_numel < 3) {
        OP_LOGE("CheckValid", "img_metas element count=%" PRId64 "; expected at least 3 elements (H, W, r).",
                in.img_metas_numel);
        return CV_STATUS_FAILED;
    }

    const uint64_t elementBytes = in.bbox_dtype == CV_DTYPE_FLOAT16 ? 2 : 4;
    const uint64_t maxRows = std::min(static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 4,
                                      std::numeric_limits<uint64_t>::max() / (4 * elementBytes));
    if (in.bbox_rows > static_cast<int64_t>(maxRows)) {
        OP_LOGE("CheckValid",
                "bbox_tensor N=%" PRId64 " exceeds capacity N<=%" PRIu64 "; element size=%" PRIu64
                " bytes; require 4*N<=INT64_MAX and 4*N*element_size<=UINT64_MAX.",
                in.bbox_rows, maxRows, elementBytes);
        return CV_STATUS_FAILED;
    }

    float img_width_x = in.W * in.r - 1.0f;
    float img_height_y = in.H * in.r - 1.0f;

    out.td.N = 0;
    out.td.tile_n = 0;
    out.td.tile_n_tail = 0;
    out.td.num_tiles = 0;
    out.td.num_cores = 0;
    out.td.tiles_main = 0;
    out.td.cores_tail = 0;
    out.td.per_buf_bytes = 0;

    if (in.bbox_rows == 0) {
        out.tiling_key = CV_KEY_EMPTY;
        out.block_dim = 1;
        out.td.N = 0;
        out.td.img_width_x = img_width_x;
        out.td.img_height_y = img_height_y;
    } else {
        out.tiling_key = CV_KEY_NORMAL;
        out.block_dim = 0;
        out.td.N = in.bbox_rows;
        out.td.img_width_x = img_width_x;
        out.td.img_height_y = img_height_y;
    }

    return CV_STATUS_SUCCESS;
}

uint32_t ComputeBranch0Tiling(const CheckValidBranch0Inputs& in, CheckValidBranch0Result& out)
{
    if (in.N <= 0) {
        OP_LOGE("CheckValid", "nonempty tiling N=%" PRId64 "; expected N>0.", in.N);
        return CV_STATUS_FAILED;
    }
    int64_t sizeofT;
    if (in.dtype == CV_DTYPE_FLOAT16) {
        sizeofT = 2;
    } else if (in.dtype == CV_DTYPE_FLOAT) {
        sizeofT = 4;
    } else {
        OP_LOGE("CheckValid", "nonempty tiling dtype=%d; expected FLOAT16(%d) or FLOAT(%d).", in.dtype,
                CV_DTYPE_FLOAT16, CV_DTYPE_FLOAT);
        return CV_STATUS_FAILED;
    }

    constexpr int64_t P = 3;
    constexpr int64_t COLS = 4;
    // VF loads 64 elements per column, including inactive lanes of the final vector.
    constexpr int64_t VECTOR_ROWS = 64;

    int64_t per_buf_bytes = (in.ub_available / P) & ~31;

    int64_t per_buf_elems = per_buf_bytes / sizeofT;
    int64_t raw_tile_n = per_buf_elems / COLS;

    int64_t tile_n = (raw_tile_n / VECTOR_ROWS) * VECTOR_ROWS;

    if (tile_n <= 0) {
        OP_LOGE("CheckValid",
                "UB tiling cannot fit one vector per column: ub_available=%" PRId64 ", per_buf_bytes=%" PRId64
                ", dtype=%d; expected capacity for 64 rows of 4 elements per buffer.",
                in.ub_available, per_buf_bytes, in.dtype);
        return CV_STATUS_FAILED;
    }

    int64_t num_tiles = in.N / tile_n + (in.N % tile_n != 0);
    int64_t tile_n_tail = in.N - (num_tiles - 1) * tile_n;
    if (tile_n_tail == 0 && num_tiles > 0) {
        tile_n_tail = tile_n;
    }

    int64_t num_cores = (in.available_cores < num_tiles) ? in.available_cores : num_tiles;
    if (num_cores < 1) {
        num_cores = 1;
    }
    int64_t tiles_main = num_tiles / num_cores;
    int64_t cores_tail = num_tiles % num_cores;

    out.block_dim = static_cast<uint32_t>(num_cores);
    out.tile_n = tile_n;
    out.tile_n_tail = tile_n_tail;
    out.num_tiles = num_tiles;
    out.num_cores = num_cores;
    out.tiles_main = tiles_main;
    out.cores_tail = cores_tail;
    out.per_buf_bytes = per_buf_bytes;

    return CV_STATUS_SUCCESS;
}

uint32_t ComputeBranch1Tiling(const CheckValidBranch1Inputs& in, CheckValidBranch1Result& out)
{
    if (in.N != 0) {
        OP_LOGE("CheckValid", "empty tiling N=%" PRId64 "; expected N=0.", in.N);
        return CV_STATUS_FAILED;
    }
    if (in.dtype != CV_DTYPE_FLOAT16 && in.dtype != CV_DTYPE_FLOAT) {
        OP_LOGE("CheckValid", "empty tiling dtype=%d; expected FLOAT16(%d) or FLOAT(%d).", in.dtype, CV_DTYPE_FLOAT16,
                CV_DTYPE_FLOAT);
        return CV_STATUS_FAILED;
    }

    out.tile_n = 0;
    out.tile_n_tail = 0;
    out.num_tiles = 0;
    out.num_cores = 0;
    out.tiles_main = 0;
    out.cores_tail = 0;
    out.per_buf_bytes = 0;

    float img_width_x = in.W * in.r - 1.0f;
    float img_height_y = in.H * in.r - 1.0f;

    out.tiling_key = CV_KEY_EMPTY;
    out.block_dim = 1;
    out.N = 0;
    out.img_width_x = img_width_x;
    out.img_height_y = img_height_y;

    return CV_STATUS_SUCCESS;
}

} // namespace optiling
