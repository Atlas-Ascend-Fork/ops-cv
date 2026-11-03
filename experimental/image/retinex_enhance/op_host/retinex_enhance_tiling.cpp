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
 * \file retinex_enhance_tiling.cpp
 * \brief Host tiling for RetinexEnhance.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "experimental/image/retinex_enhance/op_kernel/retinex_enhance_tiling_data.h"
#include "experimental/image/retinex_enhance/op_kernel/retinex_enhance_tiling_key.h"

namespace optiling {
namespace {
constexpr uint32_t GRAYSCALE_CHANNELS = 1;
constexpr uint32_t RGB_CHANNELS = 3;
constexpr uint32_t RGBA_CHANNELS = 4;
constexpr uint32_t DMA_ALIGN_ELEMENTS = 8;
constexpr uint32_t GATHER_MIN_ELEMENTS = 64;
constexpr uint32_t DEFAULT_TILE_PIXELS = 32;
constexpr uint32_t MAX_VERTICAL_TILE_ROWS = 32;
constexpr uint32_t MAX_RADIUS = 750;
constexpr uint32_t MAX_WEIGHT_ELEMENTS = MAX_RADIUS + 1;
constexpr uint64_t UB_RESERVE_BYTES = 16U * 1024U;
constexpr uint64_t WORKSPACE_SYSTEM_BYTES = 16U * 1024U * 1024U;
// Gather stores FP32 byte offsets in uint32_t; the last element must fit as well.
constexpr uint64_t MAX_GATHER_ELEMENTS = (static_cast<uint64_t>(UINT32_MAX) + 1U) / sizeof(float);
constexpr double SIGMAS[RETINEX_SCALE_COUNT] = {15.0, 80.0, 250.0};
constexpr uint32_t RADII[RETINEX_SCALE_COUNT] = {45, 240, 750};
constexpr uint32_t GROUP_OFFSETS[RETINEX_SCALE_COUNT] = {0, 2, 10};
constexpr uint32_t GROUP_COUNTS[RETINEX_SCALE_COUNT] = {2, 8, 24};

uint32_t AlignElements(uint64_t elements)
{
    return static_cast<uint32_t>((elements + DMA_ALIGN_ELEMENTS - 1) / DMA_ALIGN_ELEMENTS * DMA_ALIGN_ELEMENTS);
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum == 0 || ubSize == 0, OP_LOGE(context, "RetinexEnhance failed to query AIV cores or UB size"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool IsSupportedChannels(int64_t channels)
{
    return channels == GRAYSCALE_CHANNELS || channels == RGB_CHANNELS || channels == RGBA_CHANNELS;
}

ge::graphStatus ValidateShape(gert::TilingContext* context, uint32_t& height, uint32_t& width, uint32_t& channels,
                              uint64_t& totalElements)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = inputShape->GetStorageShape();
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::Shape& output = outputShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 3, OP_LOGE(context, "RetinexEnhance expects rank 3 HWC input"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(output.GetDimNum() != 3, OP_LOGE(context, "RetinexEnhance expects rank 3 HWC output"),
                return ge::GRAPH_FAILED);

    const int64_t h = shape.GetDim(0);
    const int64_t w = shape.GetDim(1);
    const int64_t c = shape.GetDim(2);
    OP_CHECK_IF(h <= 0 || w <= 0 || !IsSupportedChannels(c),
                OP_LOGE(context, "RetinexEnhance requires shape [H,W,C] with positive H/W and C in {1,3,4}"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(output.GetDim(0) != h || output.GetDim(1) != w || output.GetDim(2) != c,
                OP_LOGE(context, "RetinexEnhance output shape must equal input shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(h) > std::numeric_limits<uint32_t>::max() ||
                    static_cast<uint64_t>(w) > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "RetinexEnhance H or W exceeds UINT32_MAX"), return ge::GRAPH_FAILED);

    const uint64_t pixelCount = static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
    OP_CHECK_IF(pixelCount > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(c),
                OP_LOGE(context, "RetinexEnhance element count overflows uint64"), return ge::GRAPH_FAILED);
    totalElements = pixelCount * static_cast<uint64_t>(c);
    OP_CHECK_IF(totalElements > MAX_GATHER_ELEMENTS,
                OP_LOGE(context, "RetinexEnhance supports at most 2^30 FP32 elements for uint32 Gather byte offsets"),
                return ge::GRAPH_FAILED);

    height = static_cast<uint32_t>(h);
    width = static_cast<uint32_t>(w);
    channels = static_cast<uint32_t>(c);
    return ge::GRAPH_SUCCESS;
}

void FillGaussianAnchors(RetinexEnhanceTilingData* tiling)
{
    for (uint32_t scale = 0; scale < RETINEX_SCALE_COUNT; ++scale) {
        const double sigma = SIGMAS[scale];
        const uint32_t radius = RADII[scale];
        const double denom = 2.0 * sigma * sigma;
        double kernelSum = 1.0;
        for (uint32_t tap = 1; tap <= radius; ++tap) {
            kernelSum += 2.0 * std::exp(-static_cast<double>(tap * tap) / denom);
        }

        tiling->radii[scale] = radius;
        tiling->groupOffsets[scale] = GROUP_OFFSETS[scale];
        tiling->groupCounts[scale] = GROUP_COUNTS[scale];
        tiling->ratioSteps[scale] = static_cast<float>(std::exp(-1.0 / (sigma * sigma)));
        for (uint32_t group = 0; group < GROUP_COUNTS[scale]; ++group) {
            const uint32_t tap = group * RETINEX_WEIGHT_GROUP_SIZE;
            const uint32_t index = GROUP_OFFSETS[scale] + group;
            tiling->anchorWeights[index] = static_cast<float>(std::exp(-static_cast<double>(tap * tap) / denom) /
                                                              kernelSum);
            tiling->anchorRatios[index] = static_cast<float>(std::exp(-static_cast<double>(2 * tap + 1) / denom));
        }
    }
}

ge::graphStatus CalculateUbLayout(gert::TilingContext* context, uint64_t ubSize, uint32_t height, uint32_t width,
                                  uint32_t channels, RetinexEnhanceTilingData* tiling)
{
    const uint32_t tilePixels = std::min(DEFAULT_TILE_PIXELS, width);
    const uint64_t rowElements = static_cast<uint64_t>(width) * channels;
    const uint64_t tileElements = static_cast<uint64_t>(tilePixels) * channels;
    const uint64_t maxHaloElements = static_cast<uint64_t>(tilePixels + 2 * MAX_RADIUS) * channels;
    const uint64_t verticalIndexElements = static_cast<uint64_t>(height) + 2 * MAX_RADIUS;
    const uint64_t maxAlignableElements = std::numeric_limits<uint32_t>::max() - (DMA_ALIGN_ELEMENTS - 1U);
    OP_CHECK_IF(rowElements > maxAlignableElements || maxHaloElements > maxAlignableElements ||
                    verticalIndexElements > maxAlignableElements,
                OP_LOGE(context, "RetinexEnhance UB layout exceeds UINT32_MAX elements"), return ge::GRAPH_FAILED);

    // Gather may issue full vector reads for masked repeats on small rows.
    // Keep source and index backing buffers large enough for the halo operation.
    const uint32_t rowAligned = std::max(AlignElements(rowElements), AlignElements(maxHaloElements));
    const uint32_t tileAligned = std::max(GATHER_MIN_ELEMENTS, AlignElements(tileElements));
    const uint32_t haloAligned = AlignElements(maxHaloElements);
    const uint32_t verticalIndexAligned = AlignElements(verticalIndexElements);
    const uint32_t weightAligned = AlignElements(MAX_WEIGHT_ELEMENTS);
    const uint64_t baseElements = static_cast<uint64_t>(rowAligned) + static_cast<uint64_t>(tileAligned) * 5U +
                                  static_cast<uint64_t>(haloAligned) * 2U + verticalIndexAligned + weightAligned;
    OP_CHECK_IF(ubSize <= UB_RESERVE_BYTES,
                OP_LOGE(context, "RetinexEnhance UB size is below the reserved runtime space"),
                return ge::GRAPH_FAILED);
    const uint64_t availableElements = (ubSize - UB_RESERVE_BYTES) / sizeof(float);
    OP_CHECK_IF(baseElements >= availableElements,
                OP_LOGE(context, "RetinexEnhance UB has no space for vertical accumulators"), return ge::GRAPH_FAILED);
    const uint64_t rowsByUb = (availableElements - baseElements) / tileAligned;
    const uint64_t wantedRows = std::min<uint64_t>(MAX_VERTICAL_TILE_ROWS, height);
    const uint64_t verticalTileRows = std::min(wantedRows, rowsByUb);
    OP_CHECK_IF(verticalTileRows == 0 || verticalTileRows > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "RetinexEnhance cannot fit one vertical accumulation row in UB"),
                return ge::GRAPH_FAILED);
    const uint64_t verticalTileElements = verticalTileRows * static_cast<uint64_t>(tileAligned);
    OP_CHECK_IF(verticalTileElements > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "RetinexEnhance vertical accumulation layout exceeds UINT32_MAX elements"),
                return ge::GRAPH_FAILED);
    const uint64_t requiredElements = baseElements + verticalTileElements;
    OP_CHECK_IF(requiredElements > (std::numeric_limits<uint64_t>::max() - UB_RESERVE_BYTES) / sizeof(float),
                OP_LOGE(context, "RetinexEnhance UB layout byte size overflows uint64"), return ge::GRAPH_FAILED);
    const uint64_t requiredUb = UB_RESERVE_BYTES + requiredElements * sizeof(float);
    OP_CHECK_IF(requiredUb > ubSize,
                OP_LOGE(context, "RetinexEnhance requires %lu UB bytes for width %u, platform provides %lu", requiredUb,
                        width, ubSize),
                return ge::GRAPH_FAILED);

    tiling->tilePixels = tilePixels;
    const uint64_t stripeCount = (static_cast<uint64_t>(width) + static_cast<uint64_t>(tilePixels) - 1U) / tilePixels;
    OP_CHECK_IF(stripeCount > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "RetinexEnhance stripe count exceeds UINT32_MAX"), return ge::GRAPH_FAILED);
    tiling->stripeCount = static_cast<uint32_t>(stripeCount);
    tiling->rowElements = static_cast<uint32_t>(rowElements);
    tiling->rowElementsAligned = rowAligned;
    tiling->tileElementsAligned = tileAligned;
    tiling->verticalTileRows = static_cast<uint32_t>(verticalTileRows);
    tiling->verticalTileElementsAligned = static_cast<uint32_t>(verticalTileElements);
    tiling->maxHaloElementsAligned = haloAligned;
    tiling->verticalIndexElementsAligned = verticalIndexAligned;
    return ge::GRAPH_SUCCESS;
}
} // namespace

struct RetinexEnhanceCompileInfo {};

static ge::graphStatus RetinexEnhanceTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<RetinexEnhanceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t channels = 0;
    uint64_t totalElements = 0;
    OP_CHECK_IF(ValidateShape(context, height, width, channels, totalElements) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "RetinexEnhance shape validation failed"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "RetinexEnhance platform query failed"), return ge::GRAPH_FAILED);

    tiling->height = height;
    tiling->width = width;
    tiling->channels = channels;
    OP_CHECK_IF(CalculateUbLayout(context, ubSize, height, width, channels, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "RetinexEnhance UB tiling failed"), return ge::GRAPH_FAILED);
    FillGaussianAnchors(tiling);

    context->SetBlockDim(std::min(coreNum, tiling->stripeCount));
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    const uint64_t intermediateBytes = totalElements * sizeof(float);
    OP_CHECK_IF(intermediateBytes > std::numeric_limits<size_t>::max() - WORKSPACE_SYSTEM_BYTES,
                OP_LOGE(context, "RetinexEnhance workspace size overflows size_t"), return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(WORKSPACE_SYSTEM_BYTES + intermediateBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRetinexEnhance([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RetinexEnhance)
    .Tiling(RetinexEnhanceTilingFunc)
    .TilingParse<RetinexEnhanceCompileInfo>(TilingParseForRetinexEnhance);
} // namespace optiling
