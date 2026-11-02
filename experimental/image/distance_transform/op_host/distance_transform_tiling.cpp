/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <limits>

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "experimental/image/distance_transform/op_kernel/distance_transform_tiling_data.h"
#include "experimental/image/distance_transform/op_kernel/distance_transform_tiling_key.h"

namespace optiling {
namespace {
constexpr uint32_t SYSTEM_WORKSPACE_BYTES = 16U * 1024U * 1024U;
constexpr uint32_t DMA_ALIGN_BYTES = 32U;
constexpr uint32_t DEFAULT_VECTOR_TILE_ELEMS = 4096U;
constexpr uint32_t DEFAULT_TILE_ROWS = 64U;
constexpr uint32_t MIN_VECTOR_TILE_ELEMS = 8U;
constexpr uint32_t VECTOR_MIN_WIDTH = 64U;
constexpr uint32_t VECTOR_ROW_BUFFER = 2U;
constexpr uint32_t CHAMFER_VECTOR_ROW_BUFFER = 3U;
constexpr uint32_t EXACT_L2_MAX_TILE_COLUMNS = 64U;
// Keep float32 row gaps representable in uint32 byte strides.
constexpr uint32_t MAX_IMAGE_WIDTH = 1U << 30U;
constexpr uint32_t DISTANCE_TYPE_ATTR_INDEX = 0U;
constexpr uint32_t MASK_SIZE_ATTR_INDEX = 1U;

enum DistanceType : uint32_t { DISTANCE_L1 = 0U, DISTANCE_L2 = 1U, DISTANCE_CHESSBOARD = 2U };
struct DistanceTransformCompileInfo {};

inline uint64_t AlignUp(uint64_t value, uint64_t align) { return (value + align - 1U) / align * align; }

ge::graphStatus ParseShape(gert::TilingContext* context, uint32_t& imageCount, uint32_t& height, uint32_t& width,
                           uint32_t& imageStride)
{
    const auto* input = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    const gert::Shape& shape = input->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 2U || rank > 4U,
                OP_LOGE(context, "DistanceTransform: x rank must be 2, 3, or 4, but got %zu", rank),
                return ge::GRAPH_FAILED);

    int64_t n = 1;
    int64_t h = 0;
    int64_t w = 0;
    if (rank == 2U) {
        h = shape.GetDim(0);
        w = shape.GetDim(1);
    } else if (rank == 3U) {
        n = shape.GetDim(0);
        h = shape.GetDim(1);
        w = shape.GetDim(2);
    } else {
        n = shape.GetDim(0);
        OP_CHECK_IF(shape.GetDim(1) != 1, OP_LOGE(context, "DistanceTransform: rank-4 x must have shape (N, 1, H, W)"),
                    return ge::GRAPH_FAILED);
        h = shape.GetDim(2);
        w = shape.GetDim(3);
    }

    OP_CHECK_IF(n <= 0 || h <= 0 || w <= 0, OP_LOGE(context, "DistanceTransform: N, H, and W must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(h > std::numeric_limits<int32_t>::max() || w > std::numeric_limits<int32_t>::max(),
                OP_LOGE(context, "DistanceTransform: H and W must fit in int32"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(w > MAX_IMAGE_WIDTH,
                OP_LOGE(context, "DistanceTransform: W must not exceed %u (2^30)", MAX_IMAGE_WIDTH),
                return ge::GRAPH_FAILED);
    const uint64_t stride64 = static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
    const uint64_t total64 = stride64 * static_cast<uint64_t>(n);
    OP_CHECK_IF(stride64 > UINT32_MAX || total64 > UINT32_MAX,
                OP_LOGE(context, "DistanceTransform: tensor contains more than UINT32_MAX elements"),
                return ge::GRAPH_FAILED);

    imageCount = static_cast<uint32_t>(n);
    height = static_cast<uint32_t>(h);
    width = static_cast<uint32_t>(w);
    imageStride = static_cast<uint32_t>(stride64);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseDistanceType(gert::TilingContext* context, uint32_t& distanceType)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* value = attrs->GetAttrPointer<char>(DISTANCE_TYPE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, value);
    if (std::strcmp(value, "L1") == 0) {
        distanceType = DISTANCE_L1;
    } else if (std::strcmp(value, "L2") == 0) {
        distanceType = DISTANCE_L2;
    } else if (std::strcmp(value, "CHESSBOARD") == 0) {
        distanceType = DISTANCE_CHESSBOARD;
    } else {
        OP_LOGE(context, "DistanceTransform: distance_type must be L1, L2, or CHESSBOARD, but got '%s'", value);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseMaskSize(gert::TilingContext* context, uint32_t distanceType, uint32_t& maskSize)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* value = attrs->GetAttrPointer<int64_t>(MASK_SIZE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, value);
    OP_CHECK_IF(*value != 0 && *value != 3 && *value != 5,
                OP_LOGE(context, "DistanceTransform: mask_size must be 0, 3, or 5, but got %ld", *value),
                return ge::GRAPH_FAILED);
    maskSize = distanceType == DISTANCE_L2 ? static_cast<uint32_t>(*value) : 3U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DistanceTransformTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<DistanceTransformTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK,
                OP_LOGE(context, "DistanceTransform: memset tiling failed"), return ge::GRAPH_FAILED);

    uint32_t imageCount = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t imageStride = 0;
    uint32_t distanceType = DISTANCE_L2;
    uint32_t maskSize = 0U;
    OP_CHECK_IF(ParseShape(context, imageCount, height, width, imageStride) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "DistanceTransform: ParseShape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseDistanceType(context, distanceType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "DistanceTransform: ParseDistanceType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseMaskSize(context, distanceType, maskSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "DistanceTransform: ParseMaskSize failed"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(inputDesc->GetDataType() != ge::DT_UINT8, OP_LOGE(context, "DistanceTransform: x must be uint8"),
                return ge::GRAPH_FAILED);

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0U, OP_LOGE(context, "DistanceTransform: no AIV core available"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const bool isPreciseL2 = distanceType == DISTANCE_L2 && maskSize == 0U;
    const uint64_t envelopeUbBytes = isPreciseL2 ?
                                         AlignUp(static_cast<uint64_t>(height) * sizeof(int32_t), DMA_ALIGN_BYTES) +
                                             AlignUp(static_cast<uint64_t>(height + 1U) * sizeof(float),
                                                     DMA_ALIGN_BYTES) +
                                             AlignUp(static_cast<uint64_t>(height) * sizeof(float), DMA_ALIGN_BYTES) :
                                         0ULL;
    const uint64_t minimumExactUbBytes = 2U * MIN_VECTOR_TILE_ELEMS * sizeof(float);
    OP_CHECK_IF(isPreciseL2 && envelopeUbBytes + minimumExactUbBytes > ubSize,
                OP_LOGE(context, "DistanceTransform: H is too large for the exact-L2 UB envelope"),
                return ge::GRAPH_FAILED);

    uint32_t vectorTileElems = DEFAULT_VECTOR_TILE_ELEMS;
    if (isPreciseL2) {
        const uint64_t availableBytes = ubSize - envelopeUbBytes;
        const uint64_t desiredVectorBytes = 2U * DEFAULT_VECTOR_TILE_ELEMS * sizeof(float);
        const uint64_t vectorBytes = desiredVectorBytes < availableBytes ? desiredVectorBytes : availableBytes;
        vectorTileElems = static_cast<uint32_t>(vectorBytes / (2U * sizeof(float)) / MIN_VECTOR_TILE_ELEMS *
                                                MIN_VECTOR_TILE_ELEMS);
    }
    OP_CHECK_IF(vectorTileElems < MIN_VECTOR_TILE_ELEMS, OP_LOGE(context, "DistanceTransform: UB is too small"),
                return ge::GRAPH_FAILED);

    // Directional chamfer scans cannot safely split one image across cores. Cache
    // the current and two dependency rows in UB instead, and fall back to the GM
    // implementation only for exceptionally wide images.
    const uint64_t inputRowBytes = AlignUp(static_cast<uint64_t>(width) * sizeof(uint8_t), DMA_ALIGN_BYTES);
    const uint64_t outputRowBytes = AlignUp(static_cast<uint64_t>(width) * sizeof(float), DMA_ALIGN_BYTES);
    const uint64_t inputBufferCount = distanceType == DISTANCE_L1 ? 3ULL : 1ULL;
    const uint64_t rowBufferUbBytes = inputBufferCount * inputRowBytes + 3U * outputRowBytes;
    const uint64_t chessboardUbBytes = 3U * inputRowBytes + 5U * outputRowBytes;
    const uint64_t halfRowBytes = AlignUp(static_cast<uint64_t>(width) * sizeof(uint16_t), DMA_ALIGN_BYTES);
    const uint64_t vectorUbBytes = 3U * inputRowBytes + 8U * outputRowBytes + halfRowBytes;
    const uint64_t chessVectorUbBytes = vectorUbBytes + outputRowBytes;
    const uint64_t chamferVectorUbBytes = inputRowBytes + 10U * outputRowBytes + halfRowBytes;
    uint32_t useRowBuffer = 0U;
    if (!isPreciseL2) {
        if (distanceType == DISTANCE_L1 && width >= VECTOR_MIN_WIDTH && vectorUbBytes <= ubSize) {
            useRowBuffer = VECTOR_ROW_BUFFER;
        } else if (distanceType == DISTANCE_CHESSBOARD && width >= VECTOR_MIN_WIDTH && chessVectorUbBytes <= ubSize) {
            useRowBuffer = VECTOR_ROW_BUFFER;
        } else if (distanceType == DISTANCE_L2 && width >= VECTOR_MIN_WIDTH && chamferVectorUbBytes <= ubSize) {
            useRowBuffer = CHAMFER_VECTOR_ROW_BUFFER;
        } else if (distanceType == DISTANCE_CHESSBOARD && chessboardUbBytes <= ubSize) {
            useRowBuffer = 1U;
        } else if (distanceType != DISTANCE_CHESSBOARD && rowBufferUbBytes <= ubSize) {
            useRowBuffer = 1U;
        }
    }

    // Each precise-L2 task owns a complete column tile. It redundantly runs the
    // vector horizontal transform, retains only its columns in UB, and then
    // evaluates those columns independently. This removes the need for an
    // unsafe cross-core barrier between the horizontal and vertical stages.
    uint32_t exactTileColumns = 0U;
    if (isPreciseL2 && height <= 4095U) {
        const uint64_t exactBaseBytes = envelopeUbBytes + vectorUbBytes;
        if (exactBaseBytes < ubSize) {
            const uint64_t availableTileBytes = ubSize - exactBaseBytes;
            uint64_t tileColumns = availableTileBytes / (static_cast<uint64_t>(height) * sizeof(float));
            tileColumns = tileColumns < width ? tileColumns : width;
            tileColumns = tileColumns < EXACT_L2_MAX_TILE_COLUMNS ? tileColumns : EXACT_L2_MAX_TILE_COLUMNS;
            exactTileColumns = static_cast<uint32_t>(tileColumns / MIN_VECTOR_TILE_ELEMS * MIN_VECTOR_TILE_ELEMS);
        }
    }

    uint64_t taskCount = imageCount;
    if (exactTileColumns != 0U) {
        const uint64_t tilesPerImage = (static_cast<uint64_t>(width) + exactTileColumns - 1U) / exactTileColumns;
        taskCount *= tilesPerImage;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(taskCount < coreNum ? taskCount : coreNum);

    // ACLNN system workspace is retained; exact-L2 per-column state lives in per-core UB.
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = SYSTEM_WORKSPACE_BYTES;

    *tiling = DistanceTransformTilingData{imageCount,      height,       width,           imageStride,
                                          distanceType,    maskSize,     usedCoreNum,     DEFAULT_TILE_ROWS,
                                          vectorTileElems, useRowBuffer, exactTileColumns};
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(DISTANCE_TRANSFORM_SCH_MODE_DEFAULT));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForDistanceTransform(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
} // namespace

IMPL_OP_OPTILING(DistanceTransform)
    .Tiling(DistanceTransformTilingFunc)
    .TilingParse<DistanceTransformCompileInfo>(TilingParseForDistanceTransform);
} // namespace optiling
