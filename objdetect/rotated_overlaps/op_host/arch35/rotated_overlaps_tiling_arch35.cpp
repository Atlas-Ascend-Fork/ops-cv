/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_api/error_report_util.h"
#include "objdetect/rotated_overlaps/op_kernel/arch35/rotated_overlaps_tiling_data.h"
#include "objdetect/rotated_overlaps/op_kernel/arch35/rotated_overlaps_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <securec.h>

#include "lib/math/sincos_tiling.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kBoxesIndex = 0;
constexpr int64_t kQueryBoxesIndex = 1;
constexpr int64_t kOutputIndex = 0;
constexpr int64_t kTransAttrIndex = 0;
constexpr int64_t kRank = 3;
constexpr int64_t kCoordinateCount = 5;
constexpr uint64_t kMaxQueries = 2000U;
constexpr uint64_t kBytesPerFloat = sizeof(float);
constexpr size_t kWorkspaceCount = 1U;

struct ShapeParams {
    uint64_t batch{0};
    uint64_t numBoxes{0};
    uint64_t numQueries{0};
    uint64_t totalRows{0};
    uint64_t totalPairs{0};
    bool trans{false};
};

bool MulChecked(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

uint64_t CeilDiv(uint64_t numerator, uint64_t denominator)
{
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0U);
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum <= 0,
                error_report::ReportTilingStageError(
                    context->GetNodeName(), "PlatformAscendC::GetCoreNumAiv returned " + std::to_string(aivCoreNum)),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0U,
        error_report::ReportTilingStageError(context->GetNodeName(), "PlatformAscendC::GetCoreMemSize(UB) returned 0"),
        return ge::GRAPH_FAILED);
    coreNum = static_cast<uint64_t>(aivCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseAndValidate(gert::TilingContext* context, ShapeParams& params)
{
    const gert::StorageShape* boxesInput = context->GetInputShape(kBoxesIndex);
    const gert::StorageShape* queryInput = context->GetInputShape(kQueryBoxesIndex);
    const gert::StorageShape* output = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    const gert::Shape& boxesShape = boxesInput->GetStorageShape();
    const gert::Shape& queryShape = queryInput->GetStorageShape();
    const gert::Shape& outputShape = output->GetStorageShape();

    OP_CHECK_IF(boxesShape.GetDimNum() != kRank || queryShape.GetDimNum() != kRank,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (std::to_string(boxesShape.GetDimNum()) + " and " + std::to_string(queryShape.GetDimNum())).c_str(),
                    "boxes and query_boxes must both be rank 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxesShape.GetDim(1) != kCoordinateCount || queryShape.GetDim(1) != kCoordinateCount,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(boxesShape) + " and " + Ops::Base::ToString(queryShape)).c_str(),
                    "the coordinate dimension of boxes and query_boxes must be 5"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxesShape.GetDim(0) <= 0 || boxesShape.GetDim(2) <= 0 || queryShape.GetDim(0) <= 0 ||
                    queryShape.GetDim(2) <= 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(boxesShape) + " and " + Ops::Base::ToString(queryShape)).c_str(),
                    "the dimensions B, N and K must be positive at tiling time"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxesShape.GetDim(0) != queryShape.GetDim(0),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (std::to_string(boxesShape.GetDim(0)) + " and " + std::to_string(queryShape.GetDim(0))).c_str(),
                    "the batch dimensions of boxes and query_boxes must match"),
                return ge::GRAPH_FAILED);

    params.batch = static_cast<uint64_t>(boxesShape.GetDim(0));
    params.numBoxes = static_cast<uint64_t>(boxesShape.GetDim(2));
    params.numQueries = static_cast<uint64_t>(queryShape.GetDim(2));
    OP_CHECK_IF(params.numQueries > kMaxQueries,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "query_boxes", std::to_string(params.numQueries).c_str(),
                    ("K exceeds the first-release limit " + std::to_string(kMaxQueries)).c_str()),
                return ge::GRAPH_FAILED);
    uint64_t inputChannelCount = 0U;
    uint64_t boxesElementCount = 0U;
    uint64_t queryElementCount = 0U;
    OP_CHECK_IF(!MulChecked(params.batch, static_cast<uint64_t>(kCoordinateCount), inputChannelCount) ||
                    !MulChecked(inputChannelCount, params.numBoxes, boxesElementCount) ||
                    !MulChecked(inputChannelCount, params.numQueries, queryElementCount) ||
                    boxesElementCount > std::numeric_limits<size_t>::max() / kBytesPerFloat ||
                    queryElementCount > std::numeric_limits<size_t>::max() / kBytesPerFloat,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(boxesShape) + " and " + Ops::Base::ToString(queryShape)).c_str(),
                    "the input element or byte count overflows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!MulChecked(params.batch, params.numBoxes, params.totalRows) ||
                    !MulChecked(params.totalRows, params.numQueries, params.totalPairs) ||
                    params.totalPairs > std::numeric_limits<size_t>::max() / kBytesPerFloat,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "boxes and query_boxes",
                    (Ops::Base::ToString(boxesShape) + " and " + Ops::Base::ToString(queryShape)).c_str(),
                    "the output element or byte count overflows"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputShape.GetDimNum() != kRank || outputShape.GetDim(0) != boxesShape.GetDim(0) ||
                    outputShape.GetDim(1) != boxesShape.GetDim(2) || outputShape.GetDim(2) != queryShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "overlaps", Ops::Base::ToString(outputShape).c_str(),
                    ("the output shape must be [B, N, K], got B=" + std::to_string(boxesShape.GetDim(0)) +
                     ", N=" + std::to_string(boxesShape.GetDim(2)) + ", K=" + std::to_string(queryShape.GetDim(2)))
                        .c_str()),
                return ge::GRAPH_FAILED);
    const auto* boxesDesc = context->GetInputDesc(kBoxesIndex);
    const auto* queryDesc = context->GetInputDesc(kQueryBoxesIndex);
    const auto* outputDesc = context->GetOutputDesc(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(boxesDesc->GetDataType() != ge::DT_FLOAT || queryDesc->GetDataType() != ge::DT_FLOAT ||
                    outputDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "boxes, query_boxes and overlaps",
                                                       (Ops::Base::ToString(boxesDesc->GetDataType()) + ", " +
                                                        Ops::Base::ToString(queryDesc->GetDataType()) + " and " +
                                                        Ops::Base::ToString(outputDesc->GetDataType()))
                                                           .c_str(),
                                                       "only float32 is supported"),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* trans = attrs->GetAttrPointer<bool>(kTransAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, trans);
    params.trans = *trans;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectTileAndMathWorkspace(gert::TilingContext* context, uint64_t ubSize, uint32_t& tileLen,
                                           uint32_t& mathTmpBytes)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    constexpr uint32_t kCandidates[] = {128U, 96U, 64U, 32U, 8U};
    for (uint32_t candidate : kCandidates) {
        const ge::Shape tileShape({static_cast<int64_t>(candidate)});
        uint32_t sinCosMax = 0U;
        uint32_t sinCosMin = 0U;
        AscendC::GetSinCosMaxMinTmpSize(ascendcPlatform, tileShape, sizeof(float), false, sinCosMax, sinCosMin);
        const uint64_t mathBytes = sinCosMax;
        if (mathBytes == 0U || sinCosMax < sinCosMin || mathBytes > std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        const uint64_t vectorBytes = static_cast<uint64_t>(candidate) * kRotatedOverlapsFloatVectorCount *
                                     kBytesPerFloat;
        const uint64_t requiredBytes = vectorBytes + mathBytes + kRotatedOverlapsMaskReserveBytes;
        if (requiredBytes <= ubSize) {
            tileLen = candidate;
            mathTmpBytes = static_cast<uint32_t>(mathBytes);
            return ge::GRAPH_SUCCESS;
        }
    }
    error_report::ReportTilingStageError(
        context->GetNodeName(),
        "UB cannot hold the minimum vector tile and math workspace: available=" + std::to_string(ubSize));
    return ge::GRAPH_FAILED;
}
} // namespace

namespace optiling {
static ge::graphStatus RotatedOverlapsTiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ShapeParams params;
    // ParseAndValidate already reports the specific shape/dtype cause, so
    // only supplement a plain log here instead of re-reporting a generic
    // parameter error.
    OP_CHECK_IF(ParseAndValidate(context, params) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input contract validation failed"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0U;
    uint64_t availableCoreNum = 0U;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, availableCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "platform query failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ubSize <= kRotatedOverlapsSimtDataCacheReserveBytes,
                error_report::ReportTilingStageError(
                    context->GetNodeName(), "PlatformAscendC::GetCoreMemSize(UB) returned " + std::to_string(ubSize) +
                                                ", which cannot reserve the SIMT data cache of " +
                                                std::to_string(kRotatedOverlapsSimtDataCacheReserveBytes)),
                return ge::GRAPH_FAILED);
    const uint64_t localMemorySize = ubSize - kRotatedOverlapsSimtDataCacheReserveBytes;
    OP_CHECK_IF(localMemorySize > std::numeric_limits<uint32_t>::max() ||
                    context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "failed to reserve the SIMT data cache: localMemorySize=" + std::to_string(localMemorySize)),
                return ge::GRAPH_FAILED);

    uint32_t tileLen = 0U;
    uint32_t mathTmpBytes = 0U;
    OP_CHECK_IF(SelectTileAndMathWorkspace(context, localMemorySize, tileLen, mathTmpBytes) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "tile or math workspace selection failed"),
                return ge::GRAPH_FAILED);

    const uint64_t queryTilesPerBox = CeilDiv(params.numQueries, tileLen);
    const uint64_t boxTilesPerQuery = CeilDiv(params.numBoxes, tileLen);
    uint64_t queryVectorTasks = 0U;
    uint64_t queryRows = 0U;
    uint64_t boxVectorTasks = 0U;
    OP_CHECK_IF(!MulChecked(params.totalRows, queryTilesPerBox, queryVectorTasks) ||
                    !MulChecked(params.batch, params.numQueries, queryRows) ||
                    !MulChecked(queryRows, boxTilesPerQuery, boxVectorTasks),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "batch * num_boxes * num_queries",
                                                      std::to_string(params.totalPairs).c_str(),
                                                      "the tile task count overflows uint64"),
                return ge::GRAPH_FAILED);

    // Keep contiguous [K] output writes when both directions need the same
    // number of core waves.  Vectorising boxes is selected only when it
    // removes at least one full wave; its strided write is then amortised by
    // the saved geometric pass.
    const uint64_t queryVectorWaves = CeilDiv(queryVectorTasks, availableCoreNum);
    const uint64_t boxVectorWaves = CeilDiv(boxVectorTasks, availableCoreNum);
    const bool vectorizeBoxes = boxVectorWaves < queryVectorWaves;
    const uint64_t totalTasks = vectorizeBoxes ? boxVectorTasks : queryVectorTasks;
    const uint64_t tilesPerOuter = vectorizeBoxes ? boxTilesPerQuery : queryTilesPerBox;
    const uint64_t tasksPerCore = CeilDiv(totalTasks, availableCoreNum);
    const bool usePairParallelSimt = params.totalPairs <= std::numeric_limits<uint32_t>::max();
    const uint64_t vectorUsedCoreNum = CeilDiv(totalTasks, tasksPerCore);
    const uint64_t simtUsedCoreNum = std::min<uint64_t>(availableCoreNum,
                                                        CeilDiv(params.totalPairs, kRotatedOverlapsSimtThreadNum));
    const uint64_t usedCoreNum = usePairParallelSimt ? simtUsedCoreNum : vectorUsedCoreNum;
    OP_CHECK_IF(
        totalTasks == 0U || tasksPerCore == 0U || usedCoreNum == 0U ||
            usedCoreNum > std::numeric_limits<uint32_t>::max() || tilesPerOuter > std::numeric_limits<uint32_t>::max(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "used_core_num",
                                              std::to_string(usedCoreNum).c_str(), "invalid tile-task core partition"),
        return ge::GRAPH_FAILED);

    RotatedOverlapsTilingData* tiling = context->GetTilingData<RotatedOverlapsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RotatedOverlapsTilingData), 0, sizeof(RotatedOverlapsTilingData)) != EOK,
        error_report::ReportTilingStageError(context->GetNodeName(), "memset_s failed to initialize the tiling data"),
        return ge::GRAPH_FAILED);
    tiling->batch = params.batch;
    tiling->numBoxes = params.numBoxes;
    tiling->numQueries = params.numQueries;
    tiling->totalRows = params.totalRows;
    tiling->totalPairs = params.totalPairs;
    tiling->totalTasks = totalTasks;
    tiling->tasksPerCore = tasksPerCore;
    tiling->usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    tiling->tileLen = tileLen;
    tiling->tilesPerOuter = static_cast<uint32_t>(tilesPerOuter);
    tiling->mathTmpBytes = mathTmpBytes;
    tiling->trans = params.trans ? 1U : 0U;
    tiling->use32Bit = (totalTasks <= std::numeric_limits<uint32_t>::max() &&
                        params.numQueries <= std::numeric_limits<uint32_t>::max() &&
                        params.totalPairs <= std::numeric_limits<uint32_t>::max()) ?
                           1U :
                           0U;
    tiling->vectorizeBoxes = vectorizeBoxes ? 1U : 0U;
    tiling->usePairParallelSimt = usePairParallelSimt ? 1U : 0U;

    size_t* workspace = context->GetWorkspaceSizes(kWorkspaceCount);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0U;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(
        GET_TPL_TILING_KEY(params.trans ? ROTATED_OVERLAPS_TPL_XYXYT : ROTATED_OVERLAPS_TPL_XYWHT,
                           tiling->use32Bit != 0U ? ROTATED_OVERLAPS_TPL_INDEX_32 : ROTATED_OVERLAPS_TPL_INDEX_64));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RotatedOverlapsTilingParse([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct RotatedOverlapsCompileInfo {};

IMPL_OP_OPTILING(RotatedOverlaps)
    .Tiling(RotatedOverlapsTiling)
    .TilingParse<RotatedOverlapsCompileInfo>(RotatedOverlapsTilingParse);
} // namespace optiling
