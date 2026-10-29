/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ssim_tiling.cpp
 * \brief ssim tiling
 */
#include "ssim_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {
constexpr uint32_t SSIM_MIN_WIN_SIZE = 11;
constexpr uint32_t SSIM_DEFAULT_COL_TILE = 256;
constexpr uint32_t SSIM_WORKSPACE_ALIGN = 32;
constexpr uint32_t SSIM_UB_HEADROOM = 4;

static ge::graphStatus TilingPrepare4Ssim(gert::TilingParseContext* context)
{
    OP_LOGI(context, "TilingPrepare4Ssim running.");
    auto compileInfo = context->GetCompiledInfo<SsimCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Ssim(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin to do Tiling4Ssim");

    auto compileInfo = context->GetCompileInfo<SsimCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape shape = x1Shape->GetStorageShape();
    size_t rank = shape.GetDimNum();
    if (rank < 2) {
        OP_LOGE(context, "rank must be >= 2, got %zu.", rank);
        return ge::GRAPH_FAILED;
    }

    // Tiling fields are 32-bit and the kernel trusts them, so reject shapes that cannot be
    // represented instead of letting the static_cast below wrap around silently.
    constexpr uint64_t kMaxU32 = 0xFFFFFFFFULL;
    for (size_t i = 0; i < rank; i++) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) > kMaxU32) {
            OP_LOGE(context, "dim[%zu]=%lld is out of range [0, UINT32_MAX].", i, static_cast<long long>(dim));
            return ge::GRAPH_FAILED;
        }
    }

    // Leading dims are [..., H, W, C] for rank >= 3, so batch is the product of the
    // first rank-3 dims (none for rank-2/rank-3, i.e. batch = 1).
    uint64_t batch = 1;
    for (size_t i = 0; i + 3 < rank; i++) {
        const uint64_t dim = static_cast<uint64_t>(shape.GetDim(i));
        if (dim != 0 && batch > kMaxU32 / dim) {
            OP_LOGE(context, "batch product overflows uint32 at dim %zu.", i);
            return ge::GRAPH_FAILED;
        }
        batch *= dim;
    }
    uint32_t height = static_cast<uint32_t>(shape.GetDim(rank - 2));
    uint32_t width = static_cast<uint32_t>(shape.GetDim(rank - 1));
    uint32_t channels = 1;
    if (rank >= 3) {
        channels = static_cast<uint32_t>(shape.GetDim(rank - 1));
        width = static_cast<uint32_t>(shape.GetDim(rank - 2));
        height = static_cast<uint32_t>(shape.GetDim(rank - 3));
    }

    // Note: for rank == 3, shape is [H, W, C] (batch = 1).
    // For rank >= 4, shape is [..., H, W, C] and batch is the product of leading dims.

    if (height < SSIM_MIN_WIN_SIZE || width < SSIM_MIN_WIN_SIZE) {
        OP_LOGE(context, "height/width must be >= 11, got H=%u W=%u.", height, width);
        return ge::GRAPH_FAILED;
    }

    SsimTilingData tiling;
    uint32_t outW = width - (SSIM_MIN_WIN_SIZE - 1);
    // With the wide layout each output row covers all channels (outCols*C elements),
    // so channels are NOT folded into the row axis; they only scale totalPixels.
    uint64_t totalPixels = batch * static_cast<uint64_t>(outW) * static_cast<uint64_t>(height - SSIM_MIN_WIN_SIZE + 1) *
                           channels;
    uint64_t totalOutRows = batch * static_cast<uint64_t>(height - SSIM_MIN_WIN_SIZE + 1);
    if (totalOutRows > kMaxU32 || totalPixels > kMaxU32) {
        OP_LOGE(context, "totalOutRows=%llu or totalPixels=%llu exceeds uint32 range.",
                static_cast<unsigned long long>(totalOutRows), static_cast<unsigned long long>(totalPixels));
        return ge::GRAPH_FAILED;
    }

    uint32_t maxCores = compileInfo->totalCoreNum;
    if (maxCores == 0) {
        maxCores = 40;
    }
    uint64_t blockDim64 = std::min<uint64_t>(totalOutRows, maxCores);
    if (blockDim64 == 0) {
        OP_LOGE(context, "totalOutRows must be > 0.");
        return ge::GRAPH_FAILED;
    }
    uint32_t blockDim = static_cast<uint32_t>(blockDim64);
    uint64_t blockRows = (totalOutRows + blockDim - 1) / blockDim;
    // Keep blockRows * (blockDim - 1) <= totalOutRows so the last block's
    // row count never underflows (small totalOutRows vs large blockDim).
    if (blockDim > 1 && blockRows * (blockDim - 1) > totalOutRows) {
        blockRows = totalOutRows / blockDim;
    }
    uint64_t lastBlockRows = totalOutRows - blockRows * (blockDim - 1);

    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    bool isUint8 = (dtype == ge::DT_UINT8);
    float l = isUint8 ? 255.0f : 1.0f;
    float c1 = (0.01f * l) * (0.01f * l);
    float c2 = (0.03f * l) * (0.03f * l);

    tiling.set_totalOutRows(static_cast<uint32_t>(totalOutRows));
    tiling.set_perImageOutRows(static_cast<uint32_t>(static_cast<uint64_t>(height - SSIM_MIN_WIN_SIZE + 1)));
    tiling.set_perImageInRows(static_cast<uint32_t>(static_cast<uint64_t>(height)));
    tiling.set_outW(outW);
    tiling.set_inW(width);
    tiling.set_inH(height);
    tiling.set_channels(channels);
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_blockRows(static_cast<uint32_t>(blockRows));
    tiling.set_lastBlockRows(static_cast<uint32_t>(lastBlockRows));
    // Channels are innermost ([..., H, W, C]); the kernel packs outCols*C elements per
    // tile, so shrink the column tile to keep the UB ring buffer within budget.
    uint32_t colTile = SSIM_DEFAULT_COL_TILE;
    if (channels > 1) {
        colTile = (SSIM_DEFAULT_COL_TILE / channels) & ~7U;
        if (colTile < 8) {
            colTile = 8;
        }
    }
    tiling.set_colTile(colTile);
    tiling.set_totalPixels(static_cast<uint32_t>(totalPixels));
    tiling.set_blockDim(blockDim);
    tiling.set_invTotalPixels(1.0f / static_cast<float>(totalPixels));
    tiling.set_c1(c1);
    tiling.set_c2(c2);
    tiling.set_isUint8(isUint8 ? 1U : 0U);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(isUint8 ? 1U : 2U);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace = user part (per-core partial sums, aligned) + 16MB system workspace,
    // following the CANN9 pattern used by mse_loss / add_rms_norm
    constexpr size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    uint64_t userWorkspace = 2ULL * static_cast<uint64_t>(maxCores) * 32ULL;
    userWorkspace = (userWorkspace + SSIM_WORKSPACE_ALIGN - 1) / SSIM_WORKSPACE_ALIGN * SSIM_WORKSPACE_ALIGN;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(userWorkspace) + sysWorkspaceSize;

    OP_LOGI(context, "Ssim tiling: batch=%u H=%u W=%u C=%u outW=%u totalOutRows=%u totalPixels=%u blockDim=%u",
            static_cast<uint32_t>(batch), height, width, channels, outW, static_cast<uint32_t>(totalOutRows),
            static_cast<uint32_t>(totalPixels), blockDim);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Ssim).Tiling(Tiling4Ssim).TilingParse<SsimCompileInfo>(TilingPrepare4Ssim);

} // namespace optiling
