/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flood_fill_tiling.cpp
 * \brief Host tiling for rank-2 FloodFill with exact visited row intervals.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "experimental/image/flood_fill/op_kernel/flood_fill_tiling_data.h"
#include "experimental/image/flood_fill/op_kernel/flood_fill_tiling_key.h"

namespace optiling {

constexpr uint32_t DMA_ALIGN_BYTES = 32U;

constexpr uint32_t ATTR_SEED_INDEX = 0;
constexpr uint32_t ATTR_NEW_VALUE_INDEX = 1;
constexpr uint32_t ATTR_LO_DIFF_INDEX = 2;
constexpr uint32_t ATTR_UP_DIFF_INDEX = 3;
constexpr uint32_t ATTR_CONNECTIVITY_INDEX = 4;
constexpr uint32_t ATTR_DIFF_MODE_INDEX = 5;

constexpr int64_t CONNECTIVITY_4 = 4;
constexpr int64_t CONNECTIVITY_8 = 8;

// Current interface accepts rank-2 single-channel input [H, W].
constexpr size_t SUPPORTED_RANK = 2;

// Upper bound for total pixels: keeps the visited bitmap and the sweep
// round counter inside uint32 range.
constexpr uint64_t MAX_TOTAL_PIXELS = 0x7FFFFFFFULL;

constexpr int32_t INT32_MAX_VALUE = 2147483647;
constexpr int32_t INT32_MIN_VALUE = -2147483647 - 1;

struct FloodFillCompileInfo {};

struct ShapeInfo {
    uint32_t height;
    uint32_t width;
};

struct AttrInfo {
    uint32_t dtypeKey;
    uint32_t seedRow;
    uint32_t seedCol;
    // Integer dtypes (uint8 / int32).
    int32_t newValueI;
    int32_t loDiffI;
    int32_t upDiffI;
    // float32 dtype.
    float newValueF;
    float loDiffF;
    float upDiffF;
    uint32_t connectivity;
    uint32_t diffMode;
};

static uint64_t AlignUpBytes(uint64_t byteCount)
{
    return (byteCount + DMA_ALIGN_BYTES - 1U) / DMA_ALIGN_BYTES * DMA_ALIGN_BYTES;
}

// Clamp in floating point before llround: finite float attributes can exceed
// the long long range. This also keeps half-integer rounding away from zero.
static int32_t QuantizeClamped(float value, int32_t lower, int32_t upper)
{
    const double clamped = std::min(static_cast<double>(upper),
                                    std::max(static_cast<double>(lower), static_cast<double>(value)));
    return static_cast<int32_t>(std::llround(clamped));
}

// uint8 pixel differences never exceed 255, so clamping its tolerances
// to [0, 255] preserves the fill criterion.
static void QuantizeForUint8(float newValue, float loDiff, float upDiff, AttrInfo& attrInfo)
{
    attrInfo.newValueI = QuantizeClamped(newValue, 0, 255);
    attrInfo.loDiffI = QuantizeClamped(loDiff, 0, 255);
    attrInfo.upDiffI = QuantizeClamped(upDiff, 0, 255);
}

static void QuantizeForInt32(float newValue, float loDiff, float upDiff, AttrInfo& attrInfo)
{
    attrInfo.newValueI = QuantizeClamped(newValue, INT32_MIN_VALUE, INT32_MAX_VALUE);
    attrInfo.loDiffI = QuantizeClamped(loDiff, 0, INT32_MAX_VALUE);
    attrInfo.upDiffI = QuantizeClamped(upDiff, 0, INT32_MAX_VALUE);
}

static void QuantizeForFloat32(float newValue, float loDiff, float upDiff, AttrInfo& attrInfo)
{
    attrInfo.newValueF = newValue;
    attrInfo.loDiffF = loDiff;
    attrInfo.upDiffF = upDiff;
}

static ge::graphStatus ParseDtypeKey(gert::TilingContext* context, uint32_t& dtypeKey)
{
    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    switch (inputDesc->GetDataType()) {
        case ge::DT_UINT8:
            dtypeKey = FLOOD_FILL_TPL_UINT8;
            return ge::GRAPH_SUCCESS;
        case ge::DT_INT32:
            dtypeKey = FLOOD_FILL_TPL_INT32;
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            dtypeKey = FLOOD_FILL_TPL_FLOAT32;
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(context, "[FloodFill] unsupported input dtype %d", static_cast<int>(inputDesc->GetDataType()));
            return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus ParseShape(gert::TilingContext* context, ShapeInfo& shapeInfo)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const auto& shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();

    OP_CHECK_IF(rank != SUPPORTED_RANK,
                OP_LOGE(context, "[FloodFill] supports rank-2 input [H, W] only, but got rank %zu", rank),
                return ge::GRAPH_FAILED);

    const int64_t height = shape.GetDim(0);
    const int64_t width = shape.GetDim(1);

    OP_CHECK_IF(height < 0 || width < 0, OP_LOGE(context, "[FloodFill] input dimensions cannot be negative"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(static_cast<uint64_t>(height) * static_cast<uint64_t>(width) > MAX_TOTAL_PIXELS,
                OP_LOGE(context, "[FloodFill] total pixels exceed %llu", MAX_TOTAL_PIXELS), return ge::GRAPH_FAILED);

    // Span stack entries pack row/columns into uint16 fields.
    OP_CHECK_IF(height > 65535 || width > 65535,
                OP_LOGE(context, "[FloodFill] dimensions must not exceed 65535, got [%ld, %ld]", height, width),
                return ge::GRAPH_FAILED);

    shapeInfo.height = static_cast<uint32_t>(height);
    shapeInfo.width = static_cast<uint32_t>(width);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputMask(gert::TilingContext* context, const ShapeInfo& image)
{
    const auto* mask = context->GetOptionalInputShape(1);
    if (mask == nullptr)
        return ge::GRAPH_SUCCESS;
    const auto* desc = context->GetOptionalInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const auto& shape = mask->GetStorageShape();
    OP_CHECK_IF(desc->GetDataType() != ge::DT_UINT8 || shape.GetDimNum() != 2U,
                OP_LOGE(context, "[FloodFill] mask_in must be rank-2 UINT8"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != static_cast<int64_t>(image.height) + 2 ||
                    shape.GetDim(1) != static_cast<int64_t>(image.width) + 2,
                OP_LOGE(context, "[FloodFill] mask_in must have shape [H+2, W+2]"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseAttrs(gert::TilingContext* context, const ShapeInfo& shapeInfo, AttrInfo& attrInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const gert::ContinuousVector* seedVec = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEED_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seedVec);
    OP_CHECK_IF(seedVec->GetSize() != 2,
                OP_LOGE(context, "[FloodFill] seed must have 2 elements (y, x), but got %zu", seedVec->GetSize()),
                return ge::GRAPH_FAILED);

    const int64_t* seedValues = reinterpret_cast<const int64_t*>(seedVec->GetData());

    OP_CHECK_IF(seedValues[0] < 0 || seedValues[0] >= static_cast<int64_t>(shapeInfo.height) || seedValues[1] < 0 ||
                    seedValues[1] >= static_cast<int64_t>(shapeInfo.width),
                OP_LOGE(context, "[FloodFill] seed (%ld, %ld) is outside image [%u, %u]", seedValues[0], seedValues[1],
                        shapeInfo.height, shapeInfo.width),
                return ge::GRAPH_FAILED);

    attrInfo.seedRow = static_cast<uint32_t>(seedValues[0]);
    attrInfo.seedCol = static_cast<uint32_t>(seedValues[1]);

    const float* scalarLists[3] = {nullptr, nullptr, nullptr};
    const uint32_t scalarListIndices[3] = {ATTR_NEW_VALUE_INDEX, ATTR_LO_DIFF_INDEX, ATTR_UP_DIFF_INDEX};

    for (uint32_t listIndex = 0; listIndex < 3; ++listIndex) {
        const gert::ContinuousVector* vec = attrs->GetAttrPointer<gert::ContinuousVector>(scalarListIndices[listIndex]);
        OP_CHECK_NULL_WITH_CONTEXT(context, vec);
        OP_CHECK_IF(vec->GetSize() != 1,
                    OP_LOGE(context,
                            "[FloodFill] requires single-channel (length-1) value lists, attribute %u has %zu elements",
                            scalarListIndices[listIndex], vec->GetSize()),
                    return ge::GRAPH_FAILED);
        scalarLists[listIndex] = reinterpret_cast<const float*>(vec->GetData());
        OP_CHECK_NULL_WITH_CONTEXT(context, scalarLists[listIndex]);
    }

    const float* newValue = scalarLists[0];
    const float* loDiff = scalarLists[1];
    const float* upDiff = scalarLists[2];

    OP_CHECK_IF(!std::isfinite(*newValue) || !std::isfinite(*loDiff) || !std::isfinite(*upDiff),
                OP_LOGE(context, "[FloodFill] new_value, lo_diff and up_diff must be finite"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(*loDiff < 0.0F || *upDiff < 0.0F,
                OP_LOGE(context, "[FloodFill] lo_diff and up_diff must be non-negative"), return ge::GRAPH_FAILED);

    switch (attrInfo.dtypeKey) {
        case FLOOD_FILL_TPL_UINT8:
            QuantizeForUint8(*newValue, *loDiff, *upDiff, attrInfo);
            break;
        case FLOOD_FILL_TPL_INT32:
            QuantizeForInt32(*newValue, *loDiff, *upDiff, attrInfo);
            break;
        case FLOOD_FILL_TPL_FLOAT32:
            QuantizeForFloat32(*newValue, *loDiff, *upDiff, attrInfo);
            break;
        default:
            OP_LOGE(context, "[FloodFill] unknown dtype key %u", attrInfo.dtypeKey);
            return ge::GRAPH_FAILED;
    }

    const int64_t* connectivity = attrs->GetAttrPointer<int64_t>(ATTR_CONNECTIVITY_INDEX);
    if (connectivity == nullptr) {
        attrInfo.connectivity = CONNECTIVITY_4;
    } else {
        OP_CHECK_IF(*connectivity != CONNECTIVITY_4 && *connectivity != CONNECTIVITY_8,
                    OP_LOGE(context, "[FloodFill] connectivity must be 4 or 8, but got %ld", *connectivity),
                    return ge::GRAPH_FAILED);
        attrInfo.connectivity = static_cast<uint32_t>(*connectivity);
    }

    const char* diffMode = attrs->GetAttrPointer<char>(ATTR_DIFF_MODE_INDEX);
    if (diffMode == nullptr) {
        attrInfo.diffMode = FLOODFILL_DIFF_MODE_NEIGHBOR;
    } else if (std::strcmp(diffMode, "NEIGHBOR") == 0) {
        attrInfo.diffMode = FLOODFILL_DIFF_MODE_NEIGHBOR;
    } else if (std::strcmp(diffMode, "FIXED_RANGE") == 0) {
        attrInfo.diffMode = FLOODFILL_DIFF_MODE_FIXED_RANGE;
    } else {
        OP_LOGE(context, "[FloodFill] diff_mode must be NEIGHBOR or FIXED_RANGE, but got %s", diffMode);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context, uint64_t visitedBytes, uint64_t rowFlagsBytes,
                                    uint64_t stackBytes, uint64_t comparisonBytes = 0U)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    // Sys area size is queried from the platform (same pattern as
    // roi_align_v2 / c_io_u); reporting the user region alone is NOT
    // valid (507035, verified).
    // User region layout (see kernel):
    //   [ visited bitmap | row flags (H) | sole intervals (4*H) | span stack ].
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + FLOOD_FILL_CONTROL_BYTES + comparisonBytes +
                   AlignUpBytes(visitedBytes) + AlignUpBytes(rowFlagsBytes) + AlignUpBytes(rowFlagsBytes * 4U) +
                   AlignUpBytes(stackBytes);
    return ge::GRAPH_SUCCESS;
}

// Row flags: one byte per row (records rows touched by the fill).
static uint64_t RowFlagsBytes(const ShapeInfo& shapeInfo) { return shapeInfo.height; }

// Span stack capacity: 2 entries per row + 256 margin; overflow
// falls back to a full sweep closure in the kernel (correct always).
static uint64_t SpanStackBytes(const ShapeInfo& shapeInfo)
{
    constexpr uint64_t spanEntryBytes = 8U;
    constexpr uint64_t marginEntries = 256U;
    const uint64_t maxDim = shapeInfo.height > shapeInfo.width ? shapeInfo.height : shapeInfo.width;
    return (2U * maxDim + marginEntries) * spanEntryBytes;
}

static void FillTilingData(FloodFillTilingData* tiling, const ShapeInfo& shapeInfo, const AttrInfo& attrInfo)
{
    tiling->height = shapeInfo.height;
    tiling->width = shapeInfo.width;

    tiling->seedRow = attrInfo.seedRow;
    tiling->seedCol = attrInfo.seedCol;

    tiling->newValueI = attrInfo.newValueI;
    tiling->loDiffI = attrInfo.loDiffI;
    tiling->upDiffI = attrInfo.upDiffI;

    tiling->newValueF = attrInfo.newValueF;
    tiling->loDiffF = attrInfo.loDiffF;
    tiling->upDiffF = attrInfo.upDiffF;

    tiling->connectivity = attrInfo.connectivity;
    tiling->diffMode = attrInfo.diffMode;
}

static ge::graphStatus FloodFillTilingFunc(gert::TilingContext* context)
{
    FloodFillTilingData* tiling = context->GetTilingData<FloodFillTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    ShapeInfo shapeInfo{};
    OP_CHECK_IF(ParseShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[FloodFill] failed to parse input shape"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateInputMask(context, shapeInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[FloodFill] invalid input mask"), return ge::GRAPH_FAILED);

    AttrInfo attrInfo{};

    // Degenerate images: nothing to fill, run a single idle core.
    // Keep the dtype template key valid so the executor can still
    // resolve the binary.
    OP_CHECK_IF(ParseDtypeKey(context, attrInfo.dtypeKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[FloodFill] failed to parse input dtype"), return ge::GRAPH_FAILED);

    const uint32_t maskInput = context->GetOptionalInputShape(1) != nullptr ? 1U : 0U;
    context->SetTilingKey(GET_TPL_TILING_KEY(attrInfo.dtypeKey, maskInput));

    if (shapeInfo.height == 0 || shapeInfo.width == 0) {
        context->SetBlockDim(1);
        attrInfo.connectivity = CONNECTIVITY_4;
        attrInfo.diffMode = FLOODFILL_DIFF_MODE_NEIGHBOR;
        FillTilingData(tiling, shapeInfo, attrInfo);
        return SetWorkspace(context, 0, RowFlagsBytes(shapeInfo), SpanStackBytes(shapeInfo));
    }

    OP_CHECK_IF(ParseAttrs(context, shapeInfo, attrInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[FloodFill] failed to parse attributes"), return ge::GRAPH_FAILED);

    // Keep small work single-core. Large work partitions independent
    // initialization/output; exact span search still runs on core zero.
    uint32_t cores = 1U;
    if (static_cast<uint64_t>(shapeInfo.height) * shapeInfo.width >= 262144U) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        cores = std::max(1U, std::min(32U, platform.GetCoreNumAiv()));
    }
    context->SetBlockDim(cores);
    if (cores > 1U) {
        OP_CHECK_IF(context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "[FloodFill] failed to enable batch scheduling"), return ge::GRAPH_FAILED);
    }

    FillTilingData(tiling, shapeInfo, attrInfo);

    const uint64_t visitedBytes = static_cast<uint64_t>(shapeInfo.height) * shapeInfo.width;
    const uint64_t comparisonBytes = attrInfo.dtypeKey != FLOOD_FILL_TPL_INT32 && visitedBytes >= 262144U &&
                                             shapeInfo.width >= 512U ?
                                         2U * (((visitedBytes + 7U) / 8U + 63U) / 64U * 64U + 64U) :
                                         0U;
    return SetWorkspace(context, visitedBytes, RowFlagsBytes(shapeInfo), SpanStackBytes(shapeInfo), comparisonBytes);
}

static ge::graphStatus TilingParseForFloodFill([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FloodFill).Tiling(FloodFillTilingFunc).TilingParse<FloodFillCompileInfo>(TilingParseForFloodFill);

} // namespace optiling
