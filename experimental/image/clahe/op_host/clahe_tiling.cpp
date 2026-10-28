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
#include <cmath>
#include <cstdint>
#include <limits>
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "clahe_validation.h"
#include "../op_kernel/clahe_tiling_data.h"
#include "../op_kernel/clahe_tiling_key.h"

namespace optiling {
struct ClaheCompileInfo {};
namespace {
constexpr uint64_t SYSTEM_WORKSPACE = 16ULL * 1024 * 1024;
constexpr uint64_t ARENA_BYTES = 184320;
constexpr uint64_t UB_RESERVE = 8192;

ge::graphStatus Failure(gert::TilingContext* ctx, const char* message)
{
    OP_LOGE(ctx == nullptr ? "Clahe" : ctx->GetNodeName(), "%s", message);
    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingCLAHE(gert::TilingContext* ctx)
{
    if (ctx == nullptr || ctx->GetInputShape(0) == nullptr || ctx->GetInputDesc(0) == nullptr ||
        ctx->GetOutputShape(0) == nullptr || ctx->GetOutputDesc(0) == nullptr || ctx->GetAttrs() == nullptr) {
        return Failure(ctx, "input, output or attributes are null");
    }
    const auto& shape = ctx->GetInputShape(0)->GetStorageShape();
    const auto& out = ctx->GetOutputShape(0)->GetStorageShape();
    const auto dtype = ctx->GetInputDesc(0)->GetDataType();
    if (!ClaheHost::ValidShape(shape, false) || shape != out || (dtype != ge::DT_UINT8 && dtype != ge::DT_UINT16) ||
        ctx->GetOutputDesc(0)->GetDataType() != dtype) {
        return Failure(ctx, "expected matching uint8/uint16 tensors [H,W], [N,H,W] or [N,1,H,W], N in [1,256]");
    }
    const auto* clip = ctx->GetAttrs()->GetAttrPointer<float>(0);
    const auto* grid = ctx->GetAttrs()->GetListInt(1);
    if (clip == nullptr || !std::isfinite(*clip) || grid == nullptr || grid->GetSize() != 2 ||
        grid->GetData() == nullptr) {
        return Failure(ctx, "clip_limit must be finite and tile_grid_size must have two integers");
    }
    const int64_t gx = grid->GetData()[0];
    const int64_t gy = grid->GetData()[1];
    const auto rank = shape.GetDimNum();
    const int64_t h = shape.GetDim(rank - 2);
    const int64_t w = shape.GetDim(rank - 1);
    const uint64_t n = rank == 2 ? 1 : shape.GetDim(0);
    constexpr int64_t LIMIT = std::numeric_limits<int32_t>::max();
    if (gx < 1 || gy < 1 || gx > LIMIT || gy > LIMIT || gx > LIMIT / gy || h > LIMIT || w > LIMIT) {
        return Failure(ctx, "image or grid dimensions exceed the OpenCV int32 range");
    }
    // 与参考实现一致：进入补边分支后，整除的另一轴也补一个网格单位。
    const bool extend = h % gy != 0 || w % gx != 0;
    const uint64_t eh = h + (extend ? gy - h % gy : 0);
    const uint64_t ew = w + (extend ? gx - w % gx : 0);
    const uint64_t th = eh / gy;
    const uint64_t tw = ew / gx;
    if (eh > LIMIT || ew > LIMIT || th > static_cast<uint64_t>(LIMIT) / tw ||
        static_cast<uint64_t>(h) * w > static_cast<uint64_t>(INT64_MAX) / n) {
        return Failure(ctx, "padded dimensions, tile area or tensor element count overflow");
    }
    if (ctx->GetPlatformInfo() == nullptr) {
        return Failure(ctx, "platform information is null");
    }
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const auto cores = platform.GetCoreNumAiv();
    if (cores == 0 || ub < ARENA_BYTES + UB_RESERVE) {
        return Failure(ctx, "Clahe requires AIV cores and at least 192512 bytes of UB");
    }
    auto* data = ctx->GetTilingData<ClaheTilingData>();
    auto* workspace = ctx->GetWorkspaceSizes(1);
    if (data == nullptr || workspace == nullptr) {
        return Failure(ctx, "tiling data or workspace buffer is null");
    }
    *data = {};
    data->batch = n;
    data->height = h;
    data->width = w;
    data->tilesX = gx;
    data->tilesY = gy;
    data->tileWidth = tw;
    data->tileHeight = th;
    data->bins = dtype == ge::DT_UINT8 ? 256 : 65536;
    const uint64_t area = th * tw;
    if (*clip > 0.0f) {
        const double normalized = static_cast<double>(*clip) * area / data->bins;
        data->clipCount = normalized >= area ? area : std::max<uint32_t>(1, static_cast<uint32_t>(normalized));
    }
    data->imageElements = static_cast<uint64_t>(h) * w;
    data->lutElements = n * gx * gy * data->bins;
    data->regionCount = n * (static_cast<uint64_t>(gx) + 1) * (static_cast<uint64_t>(gy) + 1);
    data->coreNum = std::min<uint64_t>(cores, std::max<uint64_t>(n * gx * gy, data->regionCount));
    data->wideHistogram = dtype == ge::DT_UINT16 && area > 65534 && (data->clipCount == 0 || data->clipCount > 65534);
    data->lutScale = static_cast<float>(data->bins - 1) / static_cast<float>(area);
    data->invTileWidth = 1.0f / static_cast<float>(tw);
    data->invTileHeight = 1.0f / static_cast<float>(th);
    const uint64_t lutBytes = data->lutElements * sizeof(uint16_t);
    const uint64_t histogramBytes = data->wideHistogram ? static_cast<uint64_t>(data->coreNum) * 65536 * 4 : 0;
    if (lutBytes > static_cast<uint64_t>(INT64_MAX) - histogramBytes - SYSTEM_WORKSPACE) {
        return Failure(ctx, "workspace size overflows int64");
    }
    workspace[0] = SYSTEM_WORKSPACE + lutBytes + histogramBytes;
    ctx->SetBlockDim(data->coreNum);
    ctx->SetTilingKey(CLAHE_SCH_MODE);
    return ge::GRAPH_SUCCESS;
}
} // namespace
static ge::graphStatus ParseCLAHE([[maybe_unused]] gert::TilingParseContext* ctx) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(Clahe).Tiling(TilingCLAHE).TilingParse<ClaheCompileInfo>(ParseCLAHE);
} // namespace optiling
