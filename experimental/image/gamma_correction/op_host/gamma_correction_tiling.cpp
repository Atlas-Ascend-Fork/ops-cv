/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file gamma_correction_tiling.cpp
 *  \brief Runtime tiling for the fused element-wise implementation.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "gamma_correction_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t IN_X = 0;
constexpr size_t ATTR_GAMMA = 0;
constexpr size_t ATTR_CORRECTION_MODE = 1;
constexpr size_t ATTR_MAX_VALUE = 2;
constexpr uint64_t MIN_BYTES_PER_CORE = 4096;
constexpr uint64_t BLOCK_ALIGN_ELEMENTS = 512;
// Vector Compare requires a 256-byte aligned element count; aligning ubFactor to
// 256/typeBytes keeps every tile (FP32 -> 64, FP16 -> 128, INT8 -> 256 elements)
// on that boundary so the predicate and Select never fault on a misaligned tail.
constexpr uint64_t UB_ALIGN_BYTES = 256;
constexpr uint64_t UB_FIXED_RESERVE = 256;
constexpr uint64_t DIM_LIMIT_FROM_LAST[4] = {10000, 10000, 1000, 200};

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return (value + divisor - 1U) / divisor; }

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return CeilDiv(value, alignment) * alignment; }

uint64_t AlignDown(uint64_t value, uint64_t alignment) { return value / alignment * alignment; }

ge::DataType GetInputDtype(gert::TilingContext* context)
{
    const auto desc = context->GetInputDesc(IN_X);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool IsSupportedType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_FLOAT:
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_UINT16:
        case ge::DT_INT16:
            return true;
        default:
            return false;
    }
}

uint32_t TypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return sizeof(float); // 4
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_UINT16:
        case ge::DT_INT16:
            return sizeof(uint16_t); // 2
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return sizeof(uint8_t); // 1
        default:
            return sizeof(uint16_t);
    }
}

uint64_t TilingKeyFor(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16:
            return GC_TILING_KEY_BF16;
        case ge::DT_FLOAT:
            return GC_TILING_KEY_FP32;
        case ge::DT_UINT8:
            return GC_TILING_KEY_UINT8;
        case ge::DT_INT8:
            return GC_TILING_KEY_INT8;
        case ge::DT_UINT16:
            return GC_TILING_KEY_UINT16;
        case ge::DT_INT16:
            return GC_TILING_KEY_INT16;
        case ge::DT_FLOAT16:
        default:
            return GC_TILING_KEY_FP16;
    }
}

// Default value-range upper bound when the caller omits maxValue (<=0 or NaN).
float DefaultMaxValue(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 255.0F;
        case ge::DT_UINT16:
        case ge::DT_INT16:
            return 65535.0F;
        default:
            return 1.0F; // float16 / bfloat16 / float32 images are [0, 1]
    }
}

struct RuntimeParams {
    uint64_t totalNum = 0;
    uint64_t blockFactor = 0;
    uint64_t ubFactor = 0;
    uint32_t usedCoreNum = 0;
    uint32_t gammaMode = GC_MODE_GENERIC;
    uint32_t integerExponent = 0;
    uint32_t correctionMode = GC_CORRECTION_PURE;
    float gamma = 0.0F;
    float maxValue = 1.0F;
    float invMax = 1.0F;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

ge::graphStatus ParseShape(gert::TilingContext* context, RuntimeParams& params)
{
    const auto inputShape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto& shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 2 || rank > 4, OP_LOGE(context, "x rank must be 2, 3 or 4, got %zu", rank),
                return ge::GRAPH_FAILED);

    uint64_t totalNum = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        const uint64_t limit = DIM_LIMIT_FROM_LAST[rank - 1U - i];
        OP_CHECK_IF(dim <= 0 || static_cast<uint64_t>(dim) > limit,
                    OP_LOGE(context, "invalid x dimension %zu: %ld, allowed range is [1, %lu]", i, dim, limit),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(totalNum > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim),
                    OP_LOGE(context, "x element count overflows uint64"), return ge::GRAPH_FAILED);
        totalNum *= static_cast<uint64_t>(dim);
    }
    params.totalNum = totalNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseGamma(gert::TilingContext* context, RuntimeParams& params)
{
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* gamma = attrs->GetAttrPointer<float>(ATTR_GAMMA);
    OP_CHECK_IF(gamma == nullptr || !std::isfinite(*gamma) || *gamma <= 0.0F,
                OP_LOGE(context, "gamma must be finite and positive"), return ge::GRAPH_FAILED);
    params.gamma = *gamma;

    const int64_t* correctionMode = attrs->GetAttrPointer<int64_t>(ATTR_CORRECTION_MODE);
    if (correctionMode != nullptr) {
        OP_CHECK_IF(*correctionMode < static_cast<int64_t>(GC_CORRECTION_PURE) ||
                        *correctionMode > static_cast<int64_t>(GC_CORRECTION_SRGB),
                    OP_LOGE(context, "correctionMode must be 0 (pure), 1 (clamp) or 2 (sRGB), got %lld",
                            static_cast<long long>(*correctionMode)),
                    return ge::GRAPH_FAILED);
        params.correctionMode = static_cast<uint32_t>(*correctionMode);
    }

    // maxValue is derived from the dtype when omitted (<=0 or NaN); otherwise it
    // must be finite and strictly positive.
    const float* maxValueAttr = attrs->GetAttrPointer<float>(ATTR_MAX_VALUE);
    if (maxValueAttr != nullptr && std::isfinite(*maxValueAttr) && *maxValueAttr > 0.0F) {
        params.maxValue = *maxValueAttr;
    } else {
        params.maxValue = DefaultMaxValue(params.dtype);
    }
    params.invMax = 1.0F / params.maxValue;

    // The pow fast paths (identity / square / sqrt / integer exponent) only apply
    // to the pure-pow mode; clamp and sRGB modes always run the generic exp(g*ln)
    // path (or the sRGB piecewise curve) regardless of gamma's exact value.
    if (params.correctionMode == GC_CORRECTION_PURE) {
        if (*gamma == 1.0F) {
            params.gammaMode = GC_MODE_IDENTITY;
        } else if (*gamma == 2.0F) {
            params.gammaMode = GC_MODE_SQUARE;
        } else if (*gamma == 0.5F) {
            params.gammaMode = GC_MODE_SQRT;
        } else {
            const double gammaValue = static_cast<double>(*gamma);
            if (std::floor(gammaValue) == gammaValue &&
                gammaValue <= static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                params.gammaMode = GC_MODE_INTEGER;
                params.integerExponent = static_cast<uint32_t>(gammaValue);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseDtype(gert::TilingContext* context, RuntimeParams& params)
{
    params.dtype = GetInputDtype(context);
    OP_CHECK_IF(!IsSupportedType(params.dtype),
                OP_LOGE(context, "x dtype must be float16, bfloat16, float32, "
                                 "uint8, int8, uint16 or int16"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputeTiling(gert::TilingContext* context, platform_ascendc::PlatformAscendC& platform,
                              RuntimeParams& params)
{
    const uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "AIV core num is zero"), return ge::GRAPH_FAILED);
    const uint32_t typeBytes = TypeBytes(params.dtype);
    const uint64_t minElementsPerCore = CeilDiv(MIN_BYTES_PER_CORE, typeBytes);
    const uint64_t usefulCoreNum = std::max<uint64_t>(1, params.totalNum / minElementsPerCore);
    params.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(coreNum, usefulCoreNum));
    params.blockFactor = AlignUp(CeilDiv(params.totalNum, params.usedCoreNum), BLOCK_ALIGN_ELEMENTS);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t budget = ubSize * 4U / 5U;
    // UB budget per element. Typed buffers: double-buffered input queue + single
    // output queue = 3 * typeBytes. FP32 compute buffers cover every stage of the
    // heaviest pipeline (sRGB mode): xFloat, result, base (integer-exponent path),
    // normBuf, linBuf, powBuf = 6 float buffers; plus a 1-byte/element uint8
    // predicate buffer (rounded up to a float slot for safe accounting).
    constexpr uint64_t FP32_BUFFERS = 6U;
    constexpr uint64_t MASK_BYTES_PER_ELEMENT = sizeof(float); // conservative rounding
    const uint64_t bytesPerElement = 3U * typeBytes + FP32_BUFFERS * sizeof(float) + MASK_BYTES_PER_ELEMENT;
    OP_CHECK_IF(ubSize == 0 || budget <= UB_FIXED_RESERVE + bytesPerElement,
                OP_LOGE(context, "UB size %lu is insufficient", ubSize), return ge::GRAPH_FAILED);
    const uint64_t maxTileElements = (budget - UB_FIXED_RESERVE) / bytesPerElement;
    // 256-byte alignment keeps the vector Compare predicate on a legal boundary
    // (required by AscendC) for every dtype.
    const uint64_t alignElements = UB_ALIGN_BYTES / typeBytes;
    params.ubFactor = AlignDown(maxTileElements, alignElements);
    OP_CHECK_IF(params.ubFactor == 0, OP_LOGE(context, "calculated UB factor is zero"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus GammaCorrectionTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    RuntimeParams params;
    OP_CHECK_IF(ParseShape(context, params) != ge::GRAPH_SUCCESS || ParseDtype(context, params) != ge::GRAPH_SUCCESS ||
                    ParseGamma(context, params) != ge::GRAPH_SUCCESS ||
                    ComputeTiling(context, platform, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to calculate GammaCorrection tiling"), return ge::GRAPH_FAILED);

    GammaCorrectionTilingData tiling;
    tiling.set_totalNum(params.totalNum);
    tiling.set_blockFactor(params.blockFactor);
    tiling.set_ubFactor(params.ubFactor);
    tiling.set_usedCoreNum(params.usedCoreNum);
    tiling.set_gammaMode(params.gammaMode);
    tiling.set_integerExponent(params.integerExponent);
    tiling.set_correctionMode(params.correctionMode);
    tiling.set_gamma(params.gamma);
    tiling.set_maxValue(params.maxValue);
    tiling.set_invMax(params.invMax);

    context->SetBlockDim(params.usedCoreNum);
    context->SetTilingKey(TilingKeyFor(params.dtype));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForGammaCorrection(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GammaCorrectionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GammaCorrection)
    .Tiling(GammaCorrectionTilingFunc)
    .TilingParse<GammaCorrectionCompileInfo>(TilingPrepareForGammaCorrection);
} // namespace optiling
