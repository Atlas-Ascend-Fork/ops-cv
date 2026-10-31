/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "arch35/rasterizer_tiling_struct.h"
#include "arch35/rasterizer_kernel.h"

namespace {

constexpr uint64_t kLegacyFieldMask = UINT64_C(0xffffffff);
constexpr uint32_t kLegacyCompatTileLen = 1024U;

__aicore__ inline bool HasValidArch35Tiling(const RasterizerTilingData& data)
{
    return data.numFaces >= 1U && data.numFaces <= kRasterizerMaxFaces && data.height >= 1U && data.height <= 4096U &&
           data.width >= 1U && data.width <= 4096U &&
           data.totalPixels == static_cast<uint64_t>(data.height) * data.width && data.totalPixels >= 1U &&
           data.totalPixels <= UINT64_C(16777216) && data.activeCoreNum >= 1U && data.pixelBlockLen >= 1U &&
           data.pixelTileLen >= RasterizerKernel::kFloatLanes;
}

// The built-in Rasterizer host currently serializes the legacy 24-byte layout:
//   u32 numFaces, u32 numVertices, u32 height, u32 width, f32 truncation, u32 depthPrior.
// If that host wins the same-name registration, the first two pairs are observed by
// the 96-byte Ascend 950 structure as numFaces and totalPixels respectively.
__aicore__ inline bool BuildLegacyCompatTiling(const RasterizerTilingData& raw, RasterizerTilingData& compat)
{
    const uint64_t packedFacesAndVertices = raw.numFaces;
    const uint64_t packedHeightAndWidth = raw.totalPixels;
    const uint32_t legacyNumFaces = static_cast<uint32_t>(packedFacesAndVertices & kLegacyFieldMask);
    const uint32_t legacyHeight = static_cast<uint32_t>(packedHeightAndWidth & kLegacyFieldMask);
    const uint32_t legacyWidth = static_cast<uint32_t>(packedHeightAndWidth >> 32U);
    const uint32_t activeCoreNum = AscendC::GetBlockNum();

    if (legacyNumFaces < 1U || legacyNumFaces > kRasterizerMaxFaces || legacyHeight < 1U || legacyHeight > 4096U ||
        legacyWidth < 1U || legacyWidth > 4096U || activeCoreNum < 1U) {
        return false;
    }

    const uint64_t totalPixels = static_cast<uint64_t>(legacyHeight) * legacyWidth;
    compat = {};
    compat.numFaces = legacyNumFaces;
    compat.totalPixels = totalPixels;
    compat.height = legacyHeight;
    compat.width = legacyWidth;
    compat.activeCoreNum = activeCoreNum;
    compat.mergeCoreNum = 0U;
    compat.pixelBlockLen = (totalPixels + static_cast<uint64_t>(activeCoreNum) - 1U) / activeCoreNum;
    // PixelDirect uses nine float planes plus a 512-byte face packet. 1024 pixels
    // consume about 37 KiB of UB, remain vector aligned, and are safe on Ascend 950.
    compat.pixelTileLen = kLegacyCompatTileLen;
    compat.workspacePixelStride = 0U;
    compat.workspacePlaneStrideBytes = 0U;
    compat.workspaceCoreStrideBytes = 0U;
    compat.userWorkspaceBytes = 0U;
    compat.systemWorkspaceOffsetBytes = 0U;
    compat.systemWorkspaceBytes = 0U;
    return true;
}

__aicore__ inline const RasterizerTilingData* ResolveTiling(const RasterizerTilingData& raw,
                                                            RasterizerTilingData& compat)
{
    if (HasValidArch35Tiling(raw)) {
        return &raw;
    }
    if (BuildLegacyCompatTiling(raw, compat)) {
        return &compat;
    }
    return &raw;
}

} // namespace

extern "C" __global__ __aicore__ void rasterizer(GM_ADDR v, GM_ADDR f, GM_ADDR d, GM_ADDR findices, GM_ADDR barycentric,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RasterizerTilingData);
    GET_TILING_DATA_WITH_STRUCT(RasterizerTilingData, tilingData, tiling);

    RasterizerTilingData compatTiling{};
    const RasterizerTilingData* resolvedTiling = ResolveTiling(tilingData, compatTiling);
    if (TILING_KEY_IS(0)) {
        RasterizerKernel::PixelDirect op;
        op.Init(v, f, findices, barycentric, resolvedTiling);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // The legacy built-in host dispatches entry 1. Keep it on the same
        // low-stack AIV-only implementation after resolving the wire format.
        RasterizerKernel::PixelDirect op;
        op.Init(v, f, findices, barycentric, resolvedTiling);
        op.Process();
    }
}
