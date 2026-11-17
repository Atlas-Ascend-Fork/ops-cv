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
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "median_filter3_d_validation.h"
#include "../op_kernel/median_filter3_d_tiling_data.h"
#include "../op_kernel/median_filter3_d_tiling_key.h"
namespace optiling {
struct MedianFilter3DCompileInfo {};
static ge::graphStatus TilingMedianFilter3D(gert::TilingContext* ctx)
{
    if (ctx == nullptr || ctx->GetInputShape(0) == nullptr || ctx->GetOutputShape(0) == nullptr ||
        ctx->GetInputDesc(0) == nullptr || ctx->GetOutputDesc(0) == nullptr || ctx->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& s = ctx->GetInputShape(0)->GetStorageShape();
    auto dt = ctx->GetInputDesc(0)->GetDataType();
    uint32_t k[3], mode;
    float cval;
    if (!MedianFilter3DHost::ValidShape(s, false) || s != ctx->GetOutputShape(0)->GetStorageShape() ||
        !MedianFilter3DHost::Dtype(dt) || dt != ctx->GetOutputDesc(0)->GetDataType() ||
        !MedianFilter3DHost::Parse(ctx->GetAttrs(), k, mode, cval) ||
        (mode == 1 && dt == ge::DT_UINT8 && (!std::isfinite(cval) || cval < 0 || cval > 255))) {
        OP_LOGE(ctx->GetNodeName(), "Invalid shape/dtype/attributes: rank 3-5, positive dims, <=INT32_MAX elements, "
                                    "window product <=1024, REFLECT/CONSTANT; uint8 CONSTANT cval in [0,255].");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    uint32_t cores = platform.GetCoreNumAiv();
    uint32_t count = k[0] * k[1] * k[2], padded = 1;
    while (padded < count) {
        padded *= 2;
    }
    // Planes + comparator temp + fp16 cast temp + input/output queues (one slot each).
    uint32_t perLane = padded * 4 + 4 + 2 + 2 * 4;
    if (cores == 0 || ub <= 8192 + 32 * perLane) {
        return ge::GRAPH_FAILED;
    }
    uint32_t tile = std::min<uint64_t>(256, (ub - 8192) / perLane / 32 * 32);
    auto* t = ctx->GetTilingData<MedianFilter3DTilingData>();
    auto* ws = ctx->GetWorkspaceSizes(1);
    if (t == nullptr || ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *t = {};
    size_t rank = s.GetDimNum();
    t->d = s.GetDim(rank - 3);
    t->h = s.GetDim(rank - 2);
    t->w = s.GetDim(rank - 1);
    t->rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        t->rows *= s.GetDim(i);
    }
    t->kd = k[0];
    t->kh = k[1];
    t->kw = k[2];
    t->count = count;
    t->padded = padded;
    t->tile = tile;
    t->tilesPerRow = (t->w + tile - 1) / tile;
    t->tasks = t->rows * t->tilesPerRow;
    t->cores = std::min<uint64_t>(cores, t->tasks);
    t->mode = mode;
    t->cval = cval;
    ctx->SetBlockDim(t->cores);
    ctx->SetTilingKey(0);
    ws[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus ParseMedianFilter3D(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(MedianFilter3D)
    .Tiling(TilingMedianFilter3D)
    .TilingParse<MedianFilter3DCompileInfo>(ParseMedianFilter3D);
} // namespace optiling
