/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "../../op_kernel/arch35/check_valid_tiling_struct.h"
#include "../../op_kernel/arch35/check_valid_struct.h"
#include "../check_valid_tiling_host.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/types.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus CheckValidTilingFunc(gert::TilingContext* context)
{
    int64_t availableCores = 1;
    int64_t ubAvailable = 0;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC plat(platformInfo);
        availableCores = static_cast<int64_t>(plat.GetCoreNumAiv());
        if (availableCores < 1) {
            availableCores = 1;
        }
        uint64_t ubSize = 0;
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubAvailable = static_cast<int64_t>(ubSize);
    }
    if (ubAvailable <= 0) {
        ubAvailable = 253952;
    }

    const gert::StorageShape* bboxShape = context->GetInputShape(0);
    if (bboxShape == nullptr) {
        OP_LOGE(context, "bbox_tensor input[0] shape is null; expected a required input shape.");
        return ge::GRAPH_FAILED;
    }
    const auto& bboxStorage = bboxShape->GetStorageShape();
    int32_t bboxRank = static_cast<int32_t>(bboxStorage.GetDimNum());
    int64_t bboxRows = (bboxRank >= 1) ? bboxStorage.GetDim(0) : 0;
    int64_t bboxCols = (bboxRank >= 2) ? bboxStorage.GetDim(1) : 0;

    const gert::Tensor* bboxTensor = context->GetInputTensor(0);
    if (bboxTensor == nullptr) {
        OP_LOGE(context, "bbox_tensor input[0] tensor metadata is null; expected dtype metadata.");
        return ge::GRAPH_FAILED;
    }
    int32_t bboxDtype = static_cast<int32_t>(bboxTensor->GetDataType());

    const gert::StorageShape* imgMetasShape = context->GetInputShape(1);
    if (imgMetasShape == nullptr) {
        OP_LOGE(context, "img_metas input[1] shape is null; expected a required input shape.");
        return ge::GRAPH_FAILED;
    }
    const auto& imgMetasStorage = imgMetasShape->GetStorageShape();
    const int32_t imgMetasRank = static_cast<int32_t>(imgMetasStorage.GetDimNum());
    const int64_t imgMetasNumel = imgMetasStorage.GetShapeSize();

    const gert::Tensor* imgMetasTensor = context->GetInputTensor(1);
    if (imgMetasTensor == nullptr) {
        OP_LOGE(context, "img_metas input[1] tensor metadata is null; expected dtype metadata.");
        return ge::GRAPH_FAILED;
    }
    int32_t imgMetasDtype = static_cast<int32_t>(imgMetasTensor->GetDataType());

    float H = 0.0f, W = 0.0f, r = 0.0f;

    CheckValidPublicInputs in{};
    in.bbox_rank = bboxRank;
    in.bbox_rows = bboxRows;
    in.bbox_cols = bboxCols;
    in.bbox_dtype = bboxDtype;
    in.img_metas_rank = imgMetasRank;
    in.img_metas_numel = imgMetasNumel;
    in.img_metas_dtype = imgMetasDtype;
    in.H = H;
    in.W = W;
    in.r = r;
    in.available_cores = availableCores;

    CheckValidPublicResult out{};
    std::memset(&out, 0, sizeof(out));
    uint32_t ret = ComputePublicTiling(in, out);
    if (ret != CV_STATUS_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    CheckValidTilingData* tiling = context->GetTilingData<CheckValidTilingData>();
    if (tiling == nullptr) {
        const auto* rawTiling = context->GetRawTilingData();
        OP_LOGE(context,
                "tiling data buffer=%p, capacity=%zu bytes; expected a nonnull buffer with at least %zu bytes.",
                static_cast<const void*>(rawTiling), rawTiling == nullptr ? 0UL : rawTiling->GetCapacity(),
                sizeof(CheckValidTilingData));
        return ge::GRAPH_FAILED;
    }
    std::memcpy(tiling, &out.td, sizeof(CheckValidTilingData));

    const uint32_t schMode = bboxDtype == CV_DTYPE_FLOAT ? CHECK_VALID_MODE_FP32 : CHECK_VALID_MODE_FP16;
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));

    uint32_t blockDim = 1;
    if (out.td.N > 0) {
        CheckValidBranch0Inputs b0_in{};
        b0_in.N = out.td.N;
        b0_in.available_cores = availableCores;
        b0_in.ub_available = ubAvailable;
        b0_in.dtype = bboxDtype;

        CheckValidBranch0Result b0_out{};
        std::memset(&b0_out, 0, sizeof(b0_out));
        uint32_t b0_ret = ComputeBranch0Tiling(b0_in, b0_out);
        if (b0_ret != CV_STATUS_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        tiling->tile_n = b0_out.tile_n;
        tiling->tile_n_tail = b0_out.tile_n_tail;
        tiling->num_tiles = b0_out.num_tiles;
        tiling->num_cores = b0_out.num_cores;
        tiling->tiles_main = b0_out.tiles_main;
        tiling->cores_tail = b0_out.cores_tail;
        tiling->per_buf_bytes = b0_out.per_buf_bytes;

        blockDim = b0_out.block_dim;
        if (blockDim == 0u) {
            blockDim = 1u;
        }
    } else {
        blockDim = 1u;
    }
    context->SetBlockDim(blockDim);

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCheckValid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CheckValidCompileInfo {};

IMPL_OP_OPTILING(CheckValid)
    .Tiling(CheckValidTilingFunc)
    .TilingParse<CheckValidCompileInfo>(TilingParseForCheckValid)
    .TilingInputsDataDependency({1});

} // namespace optiling
