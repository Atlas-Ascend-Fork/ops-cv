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
 * \file adaptive_threshold_tiling.cpp
 * \brief Host tiling implementation for AdaptiveThreshold.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "experimental/image/adaptive_threshold/op_kernel/adaptive_threshold_tiling_data.h"
#include "experimental/image/adaptive_threshold/op_kernel/adaptive_threshold_tiling_key.h"

namespace optiling {

constexpr uint32_t MAX_TILE_WIDTH = 2048;
constexpr uint32_t ALIGN_ELEMS = 32;
constexpr uint64_t UB_RESERVE_BYTES = 16U * 1024U;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

constexpr uint32_t ATTR_MAX_VALUE_INDEX = 0;
constexpr uint32_t ATTR_ADAPTIVE_METHOD_INDEX = 1;
constexpr uint32_t ATTR_THRESHOLD_TYPE_INDEX = 2;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 3;
constexpr uint32_t ATTR_C_INDEX = 4;

constexpr uint32_t THRESHOLD_BINARY = 0;
constexpr uint32_t THRESHOLD_BINARY_INV = 1;

struct AdaptiveThresholdCompileInfo {};

struct ShapeInfo {
    uint32_t imageCount;
    uint32_t height;
    uint32_t width;
    uint32_t totalRows;
};

struct AttrInfo {
    uint32_t adaptiveMethod;
    uint32_t thresholdType;
    uint32_t blockSize;
    uint32_t radius;
    float outputValue;
    float thresholdDelta;
    uint32_t dtypeKey;
    std::array<uint32_t, ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE> gaussianWeights{};
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "[AdaptiveThreshold] AI Vector Core number is 0"),
                return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "[AdaptiveThreshold] UB size is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseShape(gert::TilingContext* context, ShapeInfo& shapeInfo)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const auto& shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();

    int64_t imageCount = 1;
    int64_t channel = 1;
    int64_t height = 0;
    int64_t width = 0;

    if (rank == 2) {
        height = shape.GetDim(0);
        width = shape.GetDim(1);
    } else if (rank == 3) {
        imageCount = shape.GetDim(0);
        height = shape.GetDim(1);
        width = shape.GetDim(2);
    } else if (rank == 4) {
        imageCount = shape.GetDim(0);
        channel = shape.GetDim(1);
        height = shape.GetDim(2);
        width = shape.GetDim(3);
    } else {
        OP_LOGE(context, "[AdaptiveThreshold] input rank must be 2, 3 or 4, but got %zu", rank);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        channel != 1,
        OP_LOGE(context, "[AdaptiveThreshold] rank-4 input must have channel dimension 1, but got %ld", channel),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(imageCount < 0 || height < 0 || width < 0,
                OP_LOGE(context, "[AdaptiveThreshold] input dimensions cannot be negative"), return ge::GRAPH_FAILED);

    constexpr uint64_t UINT32_LIMIT = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

    OP_CHECK_IF(static_cast<uint64_t>(imageCount) > UINT32_LIMIT || static_cast<uint64_t>(height) > UINT32_LIMIT ||
                    static_cast<uint64_t>(width) > UINT32_LIMIT,
                OP_LOGE(context, "[AdaptiveThreshold] an input dimension exceeds UINT32_MAX"), return ge::GRAPH_FAILED);

    const uint64_t totalRows = static_cast<uint64_t>(imageCount) * static_cast<uint64_t>(height);

    OP_CHECK_IF(totalRows > UINT32_LIMIT, OP_LOGE(context, "[AdaptiveThreshold] total image rows exceed UINT32_MAX"),
                return ge::GRAPH_FAILED);

    shapeInfo.imageCount = static_cast<uint32_t>(imageCount);
    shapeInfo.height = static_cast<uint32_t>(height);
    shapeInfo.width = static_cast<uint32_t>(width);
    shapeInfo.totalRows = static_cast<uint32_t>(totalRows);

    return ge::GRAPH_SUCCESS;
}

static double GetGaussianWeight(uint32_t index, uint32_t blockSize, uint32_t radius)
{
    const uint32_t distance = index > radius ? index - radius : radius - index;

    if (blockSize == 3U) {
        return distance == 0U ? 0.5 : 0.25;
    }

    if (blockSize == 5U) {
        if (distance == 0U) {
            return 0.375;
        }
        if (distance == 1U) {
            return 0.25;
        }
        return 0.0625;
    }

    if (blockSize == 7U) {
        if (distance == 0U) {
            return 0.28125;
        }
        if (distance == 1U) {
            return 0.21875;
        }
        if (distance == 2U) {
            return 0.109375;
        }
        return 0.03125;
    }

    if (blockSize == 9U) {
        if (distance == 0U) {
            return 0.234375;
        }
        if (distance == 1U) {
            return 0.19921875;
        }
        if (distance == 2U) {
            return 0.1171875;
        }
        if (distance == 3U) {
            return 0.05078125;
        }
        return 0.015625;
    }

    const double sigma = 0.3 * (static_cast<double>(radius) - 1.0) + 0.8;
    const double coordinate = static_cast<double>(static_cast<int64_t>(index) - static_cast<int64_t>(radius));

    return std::exp(-0.5 * coordinate * coordinate / (sigma * sigma));
}

static bool BuildGaussianWeights(AttrInfo& attrInfo)
{
    attrInfo.gaussianWeights.fill(0U);

    if (attrInfo.adaptiveMethod != ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN) {
        return true;
    }

    double weightSum = 0.0;
    for (uint32_t index = 0; index < attrInfo.blockSize; ++index) {
        weightSum += GetGaussianWeight(index, attrInfo.blockSize, attrInfo.radius);
    }

    if (!std::isfinite(weightSum) || weightSum <= 0.0) {
        return false;
    }

    uint64_t pairSum = 0;
    for (uint32_t index = 0; index < attrInfo.radius; ++index) {
        const double normalizedWeight = GetGaussianWeight(index, attrInfo.blockSize, attrInfo.radius) / weightSum;

        const uint64_t quantizedWeight = static_cast<uint64_t>(
            std::llround(normalizedWeight * ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE));

        if (quantizedWeight > ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE) {
            return false;
        }

        attrInfo.gaussianWeights[index] = static_cast<uint32_t>(quantizedWeight);
        attrInfo.gaussianWeights[attrInfo.blockSize - 1U - index] = static_cast<uint32_t>(quantizedWeight);

        pairSum += 2ULL * quantizedWeight;
    }

    if (pairSum > ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE) {
        return false;
    }

    attrInfo.gaussianWeights[attrInfo.radius] = static_cast<uint32_t>(ADAPTIVE_THRESHOLD_GAUSSIAN_SCALE - pairSum);

    return true;
}

static uint32_t SaturateRoundToUint8(float value)
{
    if (value <= 0.0F) {
        return 0;
    }
    if (value >= 255.0F) {
        return 255;
    }

    const double rounded = std::nearbyint(static_cast<double>(value));
    return static_cast<uint32_t>(rounded);
}

static ge::graphStatus GetInputDtype(gert::TilingContext* context, uint32_t& dtypeKey)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    switch (inputDesc->GetDataType()) {
        case ge::DT_UINT8:
            dtypeKey = ADAPTIVE_THRESHOLD_TPL_UINT8;
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT16:
            dtypeKey = ADAPTIVE_THRESHOLD_TPL_FLOAT16;
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            dtypeKey = ADAPTIVE_THRESHOLD_TPL_FLOAT32;
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(context, "[AdaptiveThreshold] unsupported input dtype");
            return ge::GRAPH_FAILED;
    }
}

static uint32_t DtypeByteSize(uint32_t dtypeKey)
{
    switch (dtypeKey) {
        case ADAPTIVE_THRESHOLD_TPL_UINT8:
            return static_cast<uint32_t>(sizeof(uint8_t));
        case ADAPTIVE_THRESHOLD_TPL_FLOAT16:
            return 2U;
        case ADAPTIVE_THRESHOLD_TPL_FLOAT32:
            return static_cast<uint32_t>(sizeof(float));
        default:
            return 0U;
    }
}

static const char* DtypeName(uint32_t dtypeKey)
{
    switch (dtypeKey) {
        case ADAPTIVE_THRESHOLD_TPL_UINT8:
            return "uint8";
        case ADAPTIVE_THRESHOLD_TPL_FLOAT16:
            return "float16";
        case ADAPTIVE_THRESHOLD_TPL_FLOAT32:
            return "float32";
        default:
            return "unknown";
    }
}

static ge::graphStatus ParseAttrs(gert::TilingContext* context, AttrInfo& attrInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const float* maxValue = attrs->GetAttrPointer<float>(ATTR_MAX_VALUE_INDEX);
    const char* adaptiveMethod = attrs->GetAttrPointer<char>(ATTR_ADAPTIVE_METHOD_INDEX);
    const char* thresholdType = attrs->GetAttrPointer<char>(ATTR_THRESHOLD_TYPE_INDEX);
    const int64_t* blockSize = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX);
    const float* cValue = attrs->GetAttrPointer<float>(ATTR_C_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, maxValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, adaptiveMethod);
    OP_CHECK_NULL_WITH_CONTEXT(context, thresholdType);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, cValue);

    OP_CHECK_IF(!std::isfinite(*maxValue), OP_LOGE(context, "[AdaptiveThreshold] max_value must be finite"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!std::isfinite(*cValue), OP_LOGE(context, "[AdaptiveThreshold] C must be finite"),
                return ge::GRAPH_FAILED);

    if (std::strcmp(adaptiveMethod, "MEAN") == 0) {
        attrInfo.adaptiveMethod = ADAPTIVE_THRESHOLD_METHOD_MEAN;
    } else if (std::strcmp(adaptiveMethod, "GAUSSIAN") == 0) {
        attrInfo.adaptiveMethod = ADAPTIVE_THRESHOLD_METHOD_GAUSSIAN;
    } else {
        OP_LOGE(context, "[AdaptiveThreshold] adaptive_method must be MEAN or GAUSSIAN");
        return ge::GRAPH_FAILED;
    }

    if (std::strcmp(thresholdType, "THRESH_BINARY") == 0) {
        attrInfo.thresholdType = THRESHOLD_BINARY;
    } else if (std::strcmp(thresholdType, "THRESH_BINARY_INV") == 0) {
        attrInfo.thresholdType = THRESHOLD_BINARY_INV;
    } else {
        OP_LOGE(context, "[AdaptiveThreshold] threshold_type must be THRESH_BINARY or THRESH_BINARY_INV");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(*blockSize < 3 || ((*blockSize & 1) == 0),
                OP_LOGE(context, "[AdaptiveThreshold] block_size must be an odd integer greater than or equal to 3"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        static_cast<uint64_t>(*blockSize) > ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE,
        OP_LOGE(context, "[AdaptiveThreshold] block_size must not exceed %u", ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE),
        return ge::GRAPH_FAILED);

    uint32_t dtypeKey = 0;
    OP_CHECK_IF(GetInputDtype(context, dtypeKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[AdaptiveThreshold] failed to get input dtype"), return ge::GRAPH_FAILED);

    attrInfo.blockSize = static_cast<uint32_t>(*blockSize);
    attrInfo.radius = (attrInfo.blockSize - 1U) / 2U;
    attrInfo.dtypeKey = dtypeKey;

    if (dtypeKey == ADAPTIVE_THRESHOLD_TPL_UINT8) {
        const double roundedDelta = attrInfo.thresholdType == THRESHOLD_BINARY ?
                                        std::ceil(static_cast<double>(*cValue)) :
                                        std::floor(static_cast<double>(*cValue));

        OP_CHECK_IF(roundedDelta < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
                        roundedDelta > static_cast<double>(std::numeric_limits<int32_t>::max()),
                    OP_LOGE(context, "[AdaptiveThreshold] rounded C exceeds int32 range"), return ge::GRAPH_FAILED);

        attrInfo.outputValue = static_cast<float>(SaturateRoundToUint8(*maxValue));
        attrInfo.thresholdDelta = static_cast<float>(static_cast<int32_t>(roundedDelta));
    } else {
        attrInfo.outputValue = *maxValue;
        attrInfo.thresholdDelta = *cValue;
    }

    OP_CHECK_IF(!BuildGaussianWeights(attrInfo),
                OP_LOGE(context, "[AdaptiveThreshold] failed to build Gaussian weights"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcTileWidth(gert::TilingContext* context, uint64_t ubSize, uint32_t imageWidth,
                                     const AttrInfo& attrInfo, uint32_t imageHeight, uint32_t& tileWidth)
{
    OP_CHECK_IF(ubSize <= UB_RESERVE_BYTES, OP_LOGE(context, "[AdaptiveThreshold] available UB is too small"),
                return ge::GRAPH_FAILED);

    if (attrInfo.adaptiveMethod == ADAPTIVE_THRESHOLD_METHOD_MEAN &&
        attrInfo.dtypeKey == ADAPTIVE_THRESHOLD_TPL_UINT8 && imageWidth >= ADAPTIVE_THRESHOLD_MEAN_VECTOR_MIN_WIDTH) {
        // Four aligned row buffers (uint8, half, float, float) and one output.
        // Up to 31 extra elements align the real DMA region after the left halo.
        uint32_t candidate = std::min(imageWidth, MAX_TILE_WIDTH) / ALIGN_ELEMS * ALIGN_ELEMS;
        for (; candidate >= ALIGN_ELEMS; candidate -= ALIGN_ELEMS) {
            const uint64_t padded = (candidate + 2ULL * attrInfo.radius + 31ULL + 63ULL) / 64ULL * 64ULL;
            if (padded * 11ULL + candidate <= ubSize - UB_RESERVE_BYTES) {
                tileWidth = candidate;
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(context, "[AdaptiveThreshold] insufficient UB for vector MEAN");
        return ge::GRAPH_FAILED;
    }

    const bool streamingMean = attrInfo.adaptiveMethod == ADAPTIVE_THRESHOLD_METHOD_MEAN &&
                               attrInfo.dtypeKey == ADAPTIVE_THRESHOLD_TPL_UINT8 &&
                               attrInfo.blockSize >= ADAPTIVE_THRESHOLD_MEAN_STREAM_MIN_BLOCK_SIZE &&
                               attrInfo.radius >= imageHeight - 1U;
    const uint64_t inputRows = streamingMean ? 1ULL : attrInfo.blockSize;
    const uint64_t haloWidth = 2ULL * attrInfo.radius;
    const uint64_t elementBytes = DtypeByteSize(attrInfo.dtypeKey);

    // UB usage matches the kernel buffers:
    //   inputWindowBuf = inputRows * (tileWidth + 2 * radius) * elementBytes
    //   outputQueue    = tileWidth * elementBytes
    //   colSumBuf      = (tileWidth + 2 * radius) * sizeof(int64_t)
    // Include both halo buffers and worst-case 32-byte alignment of all three buffers.
    constexpr uint64_t bufferAlignmentBytes = 32;
    constexpr uint64_t alignmentReserve = 3ULL * (bufferAlignmentBytes - 1ULL);
    const uint64_t fixedBytes = haloWidth * (inputRows * elementBytes + sizeof(int64_t)) + alignmentReserve;

    const uint64_t bytesPerColumn = (inputRows + 1ULL) * elementBytes + sizeof(int64_t);

    const uint64_t availableBytes = ubSize - UB_RESERVE_BYTES;

    OP_CHECK_IF(availableBytes <= fixedBytes,
                OP_LOGE(context,
                        "[AdaptiveThreshold] block_size %u is unsupported for dtype %s: "
                        "halo buffer requires %llu bytes but only %llu bytes of UB are "
                        "available, please use a smaller block_size",
                        attrInfo.blockSize, DtypeName(attrInfo.dtypeKey), static_cast<unsigned long long>(fixedBytes),
                        static_cast<unsigned long long>(availableBytes)),
                return ge::GRAPH_FAILED);

    uint64_t maxWidth = (availableBytes - fixedBytes) / bytesPerColumn;

    OP_CHECK_IF(maxWidth == 0, OP_LOGE(context, "[AdaptiveThreshold] calculated tile width is 0"),
                return ge::GRAPH_FAILED);

    maxWidth = std::min<uint64_t>(maxWidth, MAX_TILE_WIDTH);
    maxWidth = std::min<uint64_t>(maxWidth, imageWidth);

    if (maxWidth >= ALIGN_ELEMS) {
        maxWidth = (maxWidth / ALIGN_ELEMS) * ALIGN_ELEMS;
    }

    OP_CHECK_IF(maxWidth == 0, OP_LOGE(context, "[AdaptiveThreshold] aligned tile width is 0"),
                return ge::GRAPH_FAILED);

    tileWidth = static_cast<uint32_t>(maxWidth);
    return ge::GRAPH_SUCCESS;
}

static void FillTilingData(AdaptiveThresholdTilingData* tiling, const ShapeInfo& shapeInfo, const AttrInfo& attrInfo,
                           uint32_t baseRows, uint32_t extraRows, uint32_t tileWidth)
{
    tiling->imageCount = shapeInfo.imageCount;
    tiling->height = shapeInfo.height;
    tiling->width = shapeInfo.width;

    tiling->totalRows = shapeInfo.totalRows;
    tiling->baseRows = baseRows;
    tiling->extraRows = extraRows;

    tiling->tileWidth = tileWidth;
    tiling->blockSize = attrInfo.blockSize;
    tiling->radius = attrInfo.radius;

    tiling->thresholdType = attrInfo.thresholdType;
    tiling->outputValue = attrInfo.outputValue;
    tiling->thresholdDelta = attrInfo.thresholdDelta;

    for (uint32_t index = 0; index < ADAPTIVE_THRESHOLD_MAX_BLOCK_SIZE; ++index) {
        tiling->gaussianWeights[index] = attrInfo.gaussianWeights[index];
    }
}

static void SetTilingKey(gert::TilingContext* context, uint32_t adaptiveMethod, uint32_t dtypeKey)
{
    context->SetTilingKey(GET_TPL_TILING_KEY(adaptiveMethod, dtypeKey));
}

static ge::graphStatus AdaptiveThresholdTilingFunc(gert::TilingContext* context)
{
    AdaptiveThresholdTilingData* tiling = context->GetTilingData<AdaptiveThresholdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    AttrInfo attrInfo{};
    OP_CHECK_IF(ParseAttrs(context, attrInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[AdaptiveThreshold] failed to parse attributes"), return ge::GRAPH_FAILED);

    ShapeInfo shapeInfo{};
    OP_CHECK_IF(ParseShape(context, shapeInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[AdaptiveThreshold] failed to parse input shape"), return ge::GRAPH_FAILED);

    SetTilingKey(context, attrInfo.adaptiveMethod, attrInfo.dtypeKey);

    if (shapeInfo.totalRows == 0 || shapeInfo.width == 0) {
        // The kernel returns before InitBuffer for empty tensors, regardless of block_size.
        context->SetBlockDim(1);
        FillTilingData(tiling, shapeInfo, attrInfo, 0, 0, 1);
        return SetWorkspace(context);
    }

    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "[AdaptiveThreshold] failed to get platform information"), return ge::GRAPH_FAILED);

    uint32_t tileWidth = 0;
    OP_CHECK_IF(
        CalcTileWidth(context, ubSize, shapeInfo.width, attrInfo, shapeInfo.height, tileWidth) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "[AdaptiveThreshold] failed to calculate tile width"), return ge::GRAPH_FAILED);

    const uint32_t usedCoreNum = std::min(coreNum, shapeInfo.totalRows);

    context->SetBlockDim(usedCoreNum);

    FillTilingData(tiling, shapeInfo, attrInfo, shapeInfo.totalRows / usedCoreNum, shapeInfo.totalRows % usedCoreNum,
                   tileWidth);

    return SetWorkspace(context);
}

static ge::graphStatus TilingParseForAdaptiveThreshold([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveThreshold)
    .Tiling(AdaptiveThresholdTilingFunc)
    .TilingParse<AdaptiveThresholdCompileInfo>(TilingParseForAdaptiveThreshold);

} // namespace optiling
