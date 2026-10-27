/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/rotated_feature_align_tiling_data.h"
#include "../../op_kernel/arch35/rotated_feature_align_tiling_key.h"
#include <cmath>

namespace optiling {

using namespace Ops::Cv::OpTiling;

constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
static constexpr size_t X_DIM_NUM = 4;
static constexpr size_t BBOXES_DIM_NUM = 4;
static constexpr int64_t DIM_N = 0;
static constexpr int64_t DIM_C = 1;
static constexpr int64_t DIM_H = 2;
static constexpr int64_t DIM_W = 3;
static constexpr int64_t POINTS_CENTER_ONLY = 1;
static constexpr int64_t POINTS_CENTER_AND_CORNERS = 5;
static constexpr int64_t MIN_CORE_NUM = 1;
static constexpr uint32_t ATTR_SPATIAL_SCALE_IDX = 0;
static constexpr uint32_t ATTR_POINTS_IDX = 1;
static constexpr int64_t ATTR_POINTS_DEFAULT = 1;

struct RotatedFeatureAlignCompileInfo {};

static constexpr int32_t kXIdx = 0;
static constexpr int32_t kBboxesIdx = 1;

static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(), "float32"),
                return ge::GRAPH_FAILED);
    auto bboxesDesc = context->GetInputDesc(kBboxesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, bboxesDesc);
    const ge::DataType bboxesDtype = bboxesDesc->GetDataType();
    OP_CHECK_IF(bboxesDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "bboxes", Ops::Base::ToString(bboxesDtype).c_str(),
                                          "float32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShape(gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(
        xShape.GetDimNum() != X_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()).c_str(), "4D"),
        return ge::GRAPH_FAILED);
    auto bboxesShapePtr = context->GetInputShape(kBboxesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, bboxesShapePtr);
    auto bboxesShape = bboxesShapePtr->GetStorageShape();
    OP_CHECK_IF(bboxesShape.GetDimNum() != BBOXES_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bboxes",
                                             std::to_string(bboxesShape.GetDimNum()).c_str(), "4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(bboxesShape.GetDim(DIM_C) != BBOX_PARAM_NUM,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "bboxes.shape[1]",
                                          std::to_string(bboxesShape.GetDim(DIM_C)).c_str(), "5"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(DIM_N) != bboxesShape.GetDim(DIM_N),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and bboxes",
                                                       ("x.dim0=" + std::to_string(xShape.GetDim(DIM_N)) +
                                                        ", bboxes.dim0=" + std::to_string(bboxesShape.GetDim(DIM_N)))
                                                           .c_str(),
                                                       "dim0 of x and bboxes must be equal"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(DIM_H) != bboxesShape.GetDim(DIM_H),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and bboxes",
                                                       ("x.dim2=" + std::to_string(xShape.GetDim(DIM_H)) +
                                                        ", bboxes.dim2=" + std::to_string(bboxesShape.GetDim(DIM_H)))
                                                           .c_str(),
                                                       "dim2 of x and bboxes must be equal"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(DIM_W) != bboxesShape.GetDim(DIM_W),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and bboxes",
                                                       ("x.dim3=" + std::to_string(xShape.GetDim(DIM_W)) +
                                                        ", bboxes.dim3=" + std::to_string(bboxesShape.GetDim(DIM_W)))
                                                           .c_str(),
                                                       "dim3 of x and bboxes must be equal"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAttr(gert::TilingContext* context, float spatialScale, int64_t points)
{
    OP_CHECK_IF(
        !std::isfinite(spatialScale),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "spatial_scale",
                                              std::to_string(spatialScale).c_str(), "spatial_scale must be finite"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(points != POINTS_CENTER_ONLY && points != POINTS_CENTER_AND_CORNERS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "points", std::to_string(points).c_str(),
                                                      "points must be 1 (center only) or 5 (center + 4 corners)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputs(gert::TilingContext* context, float spatialScale, int64_t points)
{
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAttr(context, spatialScale, points) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateAttr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillTilingData(gert::TilingContext* context, RotatedFeatureAlignTilingData* tiling,
                                      float spatialScale, int64_t coreNum)
{
    auto xShapePtr = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    const int64_t totalNum = xShape.GetShapeSize();
    int64_t blockFactor = Ops::Base::CeilDiv(totalNum, coreNum);
    if (blockFactor < PER_CORE_MIN_ELEMENTS) {
        blockFactor = PER_CORE_MIN_ELEMENTS;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalNum, blockFactor);
    if (needCoreNum < MIN_CORE_NUM) {
        needCoreNum = MIN_CORE_NUM;
    }
    if (needCoreNum > coreNum) {
        needCoreNum = coreNum;
    }

    tiling->totalNum = totalNum;
    tiling->needCoreNum = needCoreNum;
    tiling->cDim = xShape.GetDim(DIM_C);
    tiling->hDim = xShape.GetDim(DIM_H);
    tiling->wDim = xShape.GetDim(DIM_W);
    tiling->spatialScale = spatialScale;
    return ge::GRAPH_SUCCESS;
}

static void DumpTilingData(gert::TilingContext* context, const RotatedFeatureAlignTilingData* tiling)
{
    OP_LOGD(context,
            "RotatedFeatureAlignTilingData: totalNum=%ld, needCoreNum=%ld, cDim=%ld, "
            "hDim=%ld, wDim=%ld, spatialScale=%.6f",
            tiling->totalNum, tiling->needCoreNum, tiling->cDim, tiling->hDim, tiling->wDim, tiling->spatialScale);
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RotatedFeatureAlignTilingFunc(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* spatialScalePtr = attrs->GetAttrPointer<float>(ATTR_SPATIAL_SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, spatialScalePtr);
    const int64_t* pointsPtr = attrs->GetAttrPointer<int64_t>(ATTR_POINTS_IDX);
    const int64_t points = (pointsPtr == nullptr) ? ATTR_POINTS_DEFAULT : *pointsPtr;

    OP_CHECK_IF(ValidateInputs(context, *spatialScalePtr, points) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputs failed"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    RotatedFeatureAlignTilingData* tiling = context->GetTilingData<RotatedFeatureAlignTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RotatedFeatureAlignTilingData), 0, sizeof(RotatedFeatureAlignTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillTilingData(context, tiling, *spatialScalePtr, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "FillTilingData failed"), return ge::GRAPH_FAILED);
    DumpTilingData(context, tiling);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize failed"),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(tiling->needCoreNum);
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu", ubSize),
                return ge::GRAPH_FAILED);

    const uint32_t pointsMode = (points == POINTS_CENTER_AND_CORNERS) ? RFA_POINTS_MODE_5 : RFA_POINTS_MODE_1;
    context->SetTilingKey(GET_TPL_TILING_KEY(pointsMode));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRotatedFeatureAlign([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotatedFeatureAlign)
    .Tiling(RotatedFeatureAlignTilingFunc)
    .TilingParse<RotatedFeatureAlignCompileInfo>(TilingParseForRotatedFeatureAlign);
} // namespace optiling
