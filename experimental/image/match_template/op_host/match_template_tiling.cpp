/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file match_template_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/match_template_tiling_data.h"
#include "../op_kernel/match_template_tiling_key.h"

namespace optiling {

struct MatchTemplateCompileInfo {};

constexpr size_t IMAGE_INDEX = 0;
constexpr size_t TEMPLATE_INDEX = 1;
constexpr size_t METHOD_INDEX = 0;
constexpr size_t EXPECTED_RANK = 4;
struct TileCandidate {
    int64_t height;
    int64_t width;
};

constexpr TileCandidate TILE_CANDIDATES[] = {
    {4, 512}, {8, 256}, {16, 128}, {4, 256}, {8, 128}, {16, 64}, {4, 128}, {8, 64}, {16, 32}, {4, 64},
    {8, 32},  {16, 16}, {4, 32},   {8, 16},  {2, 32},  {4, 16},  {8, 8},   {1, 32}, {2, 16},  {4, 8},
    {1, 16},  {2, 8},   {4, 4},    {1, 8},   {2, 4},   {1, 4},   {2, 2},   {1, 2},  {1, 1},
};
constexpr uint64_t DATA_BLOCK_BYTES = 32;
constexpr uint64_t UB_RESERVE_BYTES = 16 * 1024;
constexpr uint64_t VECTOR_SCRATCH_BYTES = 9 * 512 * sizeof(float);
constexpr uint64_t VECTOR_C3_PLANAR_BYTES = 3 * 576 * sizeof(float);
constexpr int64_t C3_VECTOR_DEFAULT_TILE_H = 4;
constexpr int64_t C3_VECTOR_CCORR_TILE_H = 16;
constexpr int64_t C3_VECTOR_UINT8_CCORR_TILE_H = 16;
constexpr int64_t C3_VECTOR_TILE_W = 512;
constexpr uint64_t UINT8_PREPACK_SCRATCH_BYTES = 16 * 1024;
// Two score accumulators plus three per-channel image-sum accumulators for
// four output rows of a 512-wide tile.
constexpr uint64_t VECTOR_C3_DEFAULT_ACCUMULATOR_PLANES = 2 + 3;
constexpr uint64_t SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024;

static uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

static uint64_t GetElementBytes(ge::DataType dataType)
{
    if (dataType == ge::DT_UINT8) {
        return 1;
    }
    if (dataType == ge::DT_FLOAT16) {
        return 2;
    }
    return 4;
}

static ge::graphStatus ValidateAndFillTiling(gert::TilingContext* context, MatchTemplateTilingData* tiling)
{
    const gert::StorageShape* imageStorage = context->GetInputShape(IMAGE_INDEX);
    const gert::StorageShape* templateStorage = context->GetInputShape(TEMPLATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, templateStorage);
    const gert::Shape& image = imageStorage->GetStorageShape();
    const gert::Shape& templateImage = templateStorage->GetStorageShape();

    OP_CHECK_IF(image.GetDimNum() != EXPECTED_RANK || templateImage.GetDimNum() != EXPECTED_RANK,
                OP_LOGE(context, "image and template must both be rank-4 NHWC tensors"), return ge::GRAPH_FAILED);

    const int64_t n = image.GetDim(0);
    const int64_t imageH = image.GetDim(1);
    const int64_t imageW = image.GetDim(2);
    const int64_t channels = image.GetDim(3);
    const int64_t templateN = templateImage.GetDim(0);
    const int64_t templateH = templateImage.GetDim(1);
    const int64_t templateW = templateImage.GetDim(2);
    const int64_t templateC = templateImage.GetDim(3);
    OP_CHECK_IF(n <= 0 || imageH <= 0 || imageW <= 0 || channels <= 0 || templateN <= 0 || templateH <= 0 ||
                    templateW <= 0 || templateC <= 0,
                OP_LOGE(context, "all runtime dimensions must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(templateN != 1, OP_LOGE(context, "template batch must be 1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(channels != templateC, OP_LOGE(context, "image and template channel counts must match"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(templateH > imageH || templateW > imageW,
                OP_LOGE(context, "template spatial dimensions must not exceed image dimensions"),
                return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* imageDesc = context->GetInputDesc(IMAGE_INDEX);
    const gert::CompileTimeTensorDesc* templateDesc = context->GetInputDesc(TEMPLATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, templateDesc);
    const ge::DataType imageType = imageDesc->GetDataType();
    const ge::DataType templateType = templateDesc->GetDataType();
    OP_CHECK_IF(imageType != templateType, OP_LOGE(context, "image and template dtypes must match"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(imageType != ge::DT_UINT8 && imageType != ge::DT_FLOAT16 && imageType != ge::DT_FLOAT,
                OP_LOGE(context, "supported dtypes are uint8, float16, and float32"), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* method = attrs->GetAttrPointer<int64_t>(METHOD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, method);
    OP_CHECK_IF(*method < 0 || *method > 5, OP_LOGE(context, "method must be in [0, 5]"), return ge::GRAPH_FAILED);

    tiling->batch = n;
    tiling->imageH = imageH;
    tiling->imageW = imageW;
    tiling->channels = channels;
    tiling->templateH = templateH;
    tiling->templateW = templateW;
    tiling->outputH = imageH - templateH + 1;
    tiling->outputW = imageW - templateW + 1;
    tiling->method = *method;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillSchedule(gert::TilingContext* context, MatchTemplateTilingData* tiling,
                                    platform_ascendc::PlatformAscendC& platform, uint32_t availableCores)
{
    const gert::CompileTimeTensorDesc* imageDesc = context->GetInputDesc(IMAGE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageDesc);
    const uint64_t elementBytes = GetElementBytes(imageDesc->GetDataType());

    const uint64_t templateElements = static_cast<uint64_t>(tiling->templateH) * tiling->templateW * tiling->channels;
    const uint64_t templateBytes = templateElements * elementBytes;
    const uint64_t templateBufferBytes = AlignUp(templateBytes, DATA_BLOCK_BYTES);
    const bool isUint8 = imageDesc->GetDataType() == ge::DT_UINT8;
    // Keep FP16 in its native halo buffer; the vector kernel performs one
    // half-to-float cast per gathered plane.  UINT8 C3/CCORR uses a compact
    // FP16 halo as well, while the generic UINT8 path retains its FP32 halo.
    const bool needsFloatInput = isUint8;
    const uint64_t templateFloatBufferBytes = needsFloatInput ?
                                                  AlignUp(templateElements * sizeof(float), DATA_BLOCK_BYTES) :
                                                  0;
    const uint64_t templateStatsBufferBytes = tiling->method >= 4 ?
                                                  AlignUp(static_cast<uint64_t>(tiling->channels) * sizeof(float),
                                                          DATA_BLOCK_BYTES) :
                                                  0;
    const bool vectorScratchRequired = imageDesc->GetDataType() == ge::DT_FLOAT ||
                                       imageDesc->GetDataType() == ge::DT_FLOAT16 ||
                                       imageDesc->GetDataType() == ge::DT_UINT8;

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= UB_RESERVE_BYTES, OP_LOGE(context, "available UB is too small"), return ge::GRAPH_FAILED);
    // The GM fallback still allocates score storage and two local result slots.
    const uint64_t fallbackScoreBytes = 4 * AlignUp(4 * sizeof(float), DATA_BLOCK_BYTES);
    const uint64_t localResultBytes = 2 * DATA_BLOCK_BYTES;
    OP_CHECK_IF(templateStatsBufferBytes + fallbackScoreBytes + localResultBytes > ubSize - UB_RESERVE_BYTES,
                OP_LOGE(context, "fallback buffers do not fit in UB"), return ge::GRAPH_FAILED);

    const bool c3VectorCandidate = tiling->batch > 0 && tiling->imageH == 512 && tiling->imageW == 512 &&
                                   tiling->channels == 3 && tiling->templateH == 16 && tiling->templateW == 16 &&
                                   tiling->method >= 2 && tiling->method <= 5 &&
                                   (imageDesc->GetDataType() == ge::DT_FLOAT16 ||
                                    imageDesc->GetDataType() == ge::DT_UINT8);
    int64_t selectedTileH = 4;
    int64_t selectedTileW = 4;
    uint64_t selectedHaloBufferBytes = 0;
    uint64_t selectedHaloFloatBufferBytes = 0;
    uint64_t selectedPackedImageBytes = 0;
    uint64_t selectedPrepackScratchBytes = 0;
    bool useHaloTile = false;
    bool useC3VectorTile = false;
    if (c3VectorCandidate) {
        const bool isCcorr = tiling->method == 2;
        const bool useCompactUint8C3 = isUint8 && (isCcorr || tiling->method == 5);
        const int64_t c3VectorTileH = isCcorr ?
                                          (useCompactUint8C3 ? C3_VECTOR_UINT8_CCORR_TILE_H : C3_VECTOR_CCORR_TILE_H) :
                                          C3_VECTOR_DEFAULT_TILE_H;
        const int64_t c3VectorTileW = isUint8 && !useCompactUint8C3 ? 256 : C3_VECTOR_TILE_W;
        const uint64_t c3AccumulatorPlanes = isCcorr ? 1 : VECTOR_C3_DEFAULT_ACCUMULATOR_PLANES;
        const uint64_t c3BatchBytes = c3AccumulatorPlanes * static_cast<uint64_t>(c3VectorTileH) * C3_VECTOR_TILE_W *
                                      sizeof(float);
        const uint64_t haloRows = static_cast<uint64_t>(tiling->templateH + c3VectorTileH - 1);
        const uint64_t haloWidth = static_cast<uint64_t>(tiling->templateW + c3VectorTileW - 1);
        const uint64_t haloRowBytes = haloWidth * static_cast<uint64_t>(tiling->channels) * elementBytes;
        const uint64_t rawHaloBufferBytes = haloRows * AlignUp(haloRowBytes, DATA_BLOCK_BYTES);
        const uint64_t fp16HaloRowBytes = haloWidth * static_cast<uint64_t>(tiling->channels) * sizeof(uint16_t);
        const uint64_t fp16HaloBufferBytes = haloRows * AlignUp(fp16HaloRowBytes, DATA_BLOCK_BYTES);
        const uint64_t haloBufferBytes = useCompactUint8C3 ? AlignUp(UINT8_PREPACK_SCRATCH_BYTES, DATA_BLOCK_BYTES) :
                                                             rawHaloBufferBytes;
        const uint64_t haloFloatBufferBytes = useCompactUint8C3 ?
                                                  fp16HaloBufferBytes :
                                                  (needsFloatInput ? rawHaloBufferBytes * sizeof(float) : 0);
        // Match the kernel's full score allocation for non-compact CCORR:
        // its final short tile falls back to the generic Vector path.
        const uint64_t tileScoreBufferBytes = isCcorr && !useCompactUint8C3 ?
                                                  static_cast<uint64_t>(c3VectorTileH) *
                                                      AlignUp(c3VectorTileW * sizeof(float), DATA_BLOCK_BYTES) :
                                                  DATA_BLOCK_BYTES;
        const uint64_t requiredBytes = templateBufferBytes + haloBufferBytes + templateFloatBufferBytes +
                                       haloFloatBufferBytes + templateStatsBufferBytes + VECTOR_SCRATCH_BYTES +
                                       VECTOR_C3_PLANAR_BYTES + c3BatchBytes + tileScoreBufferBytes + localResultBytes;
        if (requiredBytes <= ubSize - UB_RESERVE_BYTES) {
            selectedTileH = c3VectorTileH;
            selectedTileW = c3VectorTileW;
            selectedHaloBufferBytes = haloBufferBytes;
            selectedHaloFloatBufferBytes = haloFloatBufferBytes;
            if (useCompactUint8C3) {
                selectedPackedImageBytes = AlignUp(templateElements > 0 ?
                                                       static_cast<uint64_t>(tiling->batch) * tiling->imageH *
                                                           tiling->imageW * tiling->channels * sizeof(uint16_t) :
                                                       0,
                                                   DATA_BLOCK_BYTES);
                selectedPrepackScratchBytes = haloBufferBytes;
            }
            useHaloTile = true;
            useC3VectorTile = true;
        }
    }
    for (const TileCandidate& candidate : TILE_CANDIDATES) {
        if (useC3VectorTile) {
            break;
        }
        const uint64_t haloRows = static_cast<uint64_t>(tiling->templateH + candidate.height - 1);
        const uint64_t haloWidth = static_cast<uint64_t>(tiling->templateW + candidate.width - 1);
        const uint64_t haloRowBytes = haloWidth * static_cast<uint64_t>(tiling->channels) * elementBytes;
        const uint64_t haloBufferBytes = haloRows * AlignUp(haloRowBytes, DATA_BLOCK_BYTES);
        const uint64_t haloFloatBufferBytes = needsFloatInput ? haloBufferBytes * sizeof(float) : 0;
        const uint64_t c3PlanarBytes = tiling->channels == 3 ? VECTOR_C3_PLANAR_BYTES : 0;
        const uint64_t tileScoreBufferBytes = static_cast<uint64_t>(candidate.height) *
                                              AlignUp(candidate.width * sizeof(float), DATA_BLOCK_BYTES);
        const uint64_t requiredBytes = templateBufferBytes + haloBufferBytes + templateFloatBufferBytes +
                                       haloFloatBufferBytes + templateStatsBufferBytes +
                                       (vectorScratchRequired ? VECTOR_SCRATCH_BYTES + c3PlanarBytes : 0) +
                                       tileScoreBufferBytes + localResultBytes;
        const int64_t tileRows = (tiling->outputH + candidate.height - 1) / candidate.height;
        const int64_t tileCols = (tiling->outputW + candidate.width - 1) / candidate.width;
        const int64_t totalTiles = tiling->batch * tileRows * tileCols;
        const bool enoughParallelism = totalTiles >= static_cast<int64_t>(availableCores);
        if (requiredBytes <= ubSize - UB_RESERVE_BYTES &&
            (enoughParallelism || (candidate.height == 1 && candidate.width == 1))) {
            selectedTileH = candidate.height;
            selectedTileW = candidate.width;
            selectedHaloBufferBytes = haloBufferBytes;
            selectedHaloFloatBufferBytes = haloFloatBufferBytes;
            useHaloTile = true;
            break;
        }
    }

    tiling->tileH = selectedTileH;
    tiling->tileW = selectedTileW;
    tiling->tileRows = (tiling->outputH + selectedTileH - 1) / selectedTileH;
    tiling->tileCols = (tiling->outputW + selectedTileW - 1) / selectedTileW;
    tiling->tilesPerBatch = tiling->tileRows * tiling->tileCols;
    tiling->totalTiles = tiling->batch * tiling->tilesPerBatch;
    tiling->usedCores = std::min<int64_t>(static_cast<int64_t>(availableCores), tiling->totalTiles);
    OP_CHECK_IF(tiling->usedCores <= 0, OP_LOGE(context, "calculated used core count is zero"),
                return ge::GRAPH_FAILED);

    tiling->useHaloTile = useHaloTile ? 1 : 0;
    tiling->useC3VectorTile = useC3VectorTile ? 1 : 0;
    tiling->templateElements = static_cast<int64_t>(templateElements);
    tiling->templateBufferBytes = useHaloTile ? static_cast<int64_t>(templateBufferBytes) : 0;
    tiling->haloBufferBytes = useHaloTile ? static_cast<int64_t>(selectedHaloBufferBytes) : 0;
    tiling->templateFloatBufferBytes = useHaloTile ? static_cast<int64_t>(templateFloatBufferBytes) : 0;
    tiling->haloFloatBufferBytes = useHaloTile ? static_cast<int64_t>(selectedHaloFloatBufferBytes) : 0;
    tiling->templateStatsBufferBytes = static_cast<int64_t>(templateStatsBufferBytes);
    tiling->packedImageBytes = static_cast<int64_t>(selectedPackedImageBytes);
    tiling->prepackScratchBytes = static_cast<int64_t>(selectedPrepackScratchBytes);

    const uint64_t workspaceEntries = static_cast<uint64_t>(tiling->batch) * tiling->usedCores;
    const uint64_t scoreWorkspaceBytes = AlignUp(workspaceEntries * sizeof(float), DATA_BLOCK_BYTES);
    const uint64_t indexWorkspaceBytes = AlignUp(workspaceEntries * sizeof(int64_t), DATA_BLOCK_BYTES);
    tiling->workspaceIndexOffset = static_cast<int64_t>(scoreWorkspaceBytes);
    tiling->workspacePackedImageOffset = static_cast<int64_t>(scoreWorkspaceBytes + indexWorkspaceBytes);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    const uint64_t systemWorkspaceBytes = std::max<uint64_t>(platform.GetLibApiWorkSpaceSize(), SYSTEM_WORKSPACE_BYTES);
    workspace[0] = static_cast<size_t>(systemWorkspaceBytes + scoreWorkspaceBytes + indexWorkspaceBytes +
                                       selectedPackedImageBytes);

    OP_LOGI(
        context->GetNodeName(),
        "MatchTemplate schedule: blockDim=%ld tile=%ldx%ld totalTiles=%ld halo=%ld c3Vector=%ld UB=%lu workspace=%zu",
        tiling->usedCores, tiling->tileH, tiling->tileW, tiling->totalTiles, tiling->useHaloTile,
        tiling->useC3VectorTile, ubSize, workspace[0]);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus MatchTemplateTilingFunc(gert::TilingContext* context)
{
    MatchTemplateTilingData* tiling = context->GetTilingData<MatchTemplateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MatchTemplateTilingData), 0, sizeof(MatchTemplateTilingData)) != EOK,
                OP_LOGE(context, "failed to clear tiling data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAndFillTiling(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "input validation failed"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    OP_CHECK_IF(availableCores == 0, OP_LOGE(context, "no AI Vector Core is available"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillSchedule(context, tiling, platform, availableCores) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to calculate MatchTemplate schedule"), return ge::GRAPH_FAILED);
    context->SetBlockDim(static_cast<uint32_t>(tiling->usedCores));
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to enable batch schedule mode"), return ge::GRAPH_FAILED);
    context->SetTilingKey(GET_TPL_TILING_KEY(MATCH_TEMPLATE_SCH_MODE_HALO_TILE, static_cast<uint32_t>(tiling->method)));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMatchTemplate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(MatchTemplate)
    .Tiling(MatchTemplateTilingFunc)
    .TilingParse<MatchTemplateCompileInfo>(TilingParseForMatchTemplate);
} // namespace optiling
