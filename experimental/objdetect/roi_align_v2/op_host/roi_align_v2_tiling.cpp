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
 * \file roi_align_tiling.cpp
 * \brief
 */

#include <cmath>
#include <limits>

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "experimental/objdetect/roi_align_v2/op_kernel/roi_align_v2_tiling_data.h"
#include "experimental/objdetect/roi_align_v2/op_kernel/roi_align_v2_tiling_key.h"

namespace optiling {

using namespace Ops::Cv::OpTiling;

static constexpr int64_t MAX_POOLED_SIZE = std::numeric_limits<int32_t>::max();

struct RoiAlignV2CompileInfo {};

// tiling 分发入口
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiAlignV2TilingFunc(gert::TilingContext* context)
{
    int64_t coreNum;
    uint64_t ubSize;

    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    RoiAlignV2TilingData* tiling = context->GetTilingData<RoiAlignV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(RoiAlignV2TilingData), 0, sizeof(RoiAlignV2TilingData)) != EOK,
                OP_LOGE(context, "Failed to set tiling data"), return ge::GRAPH_FAILED);
    const auto* features_input = context->GetInputShape(0);
    const auto* rois_input = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, features_input);
    OP_CHECK_NULL_WITH_CONTEXT(context, rois_input);
    const auto features_shape = features_input->GetStorageShape();
    const auto rois_shape = rois_input->GetStorageShape();
    uint32_t batch = features_shape.GetDim(0);
    uint32_t channels = features_shape.GetDim(1);
    uint32_t height = features_shape.GetDim(2);
    uint32_t width = features_shape.GetDim(3);
    uint32_t numRois = rois_shape.GetDim(0);
    uint32_t roiLength = rois_shape.GetDim(1);

    auto attrs = context->GetAttrs();
    int32_t pooledHeight = 0;
    int32_t pooledWidth = 0;
    float spatialScale = 0.0f;
    int32_t samplingRatio = 0;

    if (attrs != nullptr) {
        const int64_t* pooledHeightAttr = attrs->GetInt(0);
        if (pooledHeightAttr != nullptr) {
            OP_CHECK_IF(
                *pooledHeightAttr > MAX_POOLED_SIZE,
                OP_LOGE(context, "pooled_height[%ld] exceeds INT32_MAX[%ld]", *pooledHeightAttr, MAX_POOLED_SIZE),
                return ge::GRAPH_FAILED);
            pooledHeight = static_cast<int32_t>(*pooledHeightAttr);
        }
    }
    if (attrs != nullptr) {
        const int64_t* pooledWidthAttr = attrs->GetInt(1);
        if (pooledWidthAttr != nullptr) {
            OP_CHECK_IF(*pooledWidthAttr > MAX_POOLED_SIZE,
                        OP_LOGE(context, "pooled_width[%ld] exceeds INT32_MAX[%ld]", *pooledWidthAttr, MAX_POOLED_SIZE),
                        return ge::GRAPH_FAILED);
            pooledWidth = static_cast<int32_t>(*pooledWidthAttr);
        }
    }
    if (pooledHeight <= 0 || pooledWidth <= 0) {
        OP_LOGE(context, "Invalid pooled height or width");
        return ge::GRAPH_FAILED;
    }
    if (attrs != nullptr) {
        const float* spatialScaleAttr = attrs->GetFloat(2);
        if (spatialScaleAttr != nullptr) {
            spatialScale = *spatialScaleAttr;
        }
    }
    // spatial_scale 为 OPTIONAL 属性，缺失时保持初值 0.0f：kernel 会用该值缩放全部 ROI 坐标
    // （roi_align_v2.h:142-145），坐标被整体缩放为 0 后 ROI 塌缩，输出退化为退化 ROI 的池化值，
    // 全程不报错。此处按 aclnn 侧契约（spatialScale 必须有限且大于 0）显式拒绝。
    OP_CHECK_IF(!std::isfinite(spatialScale) || spatialScale <= 0.0f,
                OP_LOGE(context, "spatial_scale must be finite and greater than 0, but got %f", spatialScale),
                return ge::GRAPH_FAILED);
    if (attrs != nullptr) {
        const int64_t* samplingRatioAttr = attrs->GetInt(3);
        if (samplingRatioAttr != nullptr) {
            // int64 -> int32 窄化前先校验上界，否则 2^32+1 等值回绕为正数后绕过语义约束
            OP_CHECK_IF(
                *samplingRatioAttr > MAX_POOLED_SIZE,
                OP_LOGE(context, "sampling_ratio[%ld] exceeds INT32_MAX[%ld]", *samplingRatioAttr, MAX_POOLED_SIZE),
                return ge::GRAPH_FAILED);
            samplingRatio = static_cast<int32_t>(*samplingRatioAttr);
        }
    }
    // Core allocation: distribute ROIs across cores
    uint32_t nowCoreNum = (coreNum > numRois) ? numRois : coreNum;
    OP_CHECK_IF(nowCoreNum == 0, OP_LOGE(context, "nowCoreNum is 0, invalid core number for ROI align"),
                return ge::GRAPH_FAILED);
    uint32_t baseRoisPerCore = numRois / nowCoreNum;
    uint32_t tailRoiNum = numRois % nowCoreNum;
    uint32_t bigTotalRois = baseRoisPerCore + 1;

    // Calculate buffer sizes
    uint32_t featureTotalSize = batch * channels * height * width;
    uint32_t featureMapSize = channels * height * width;
    uint32_t outRoiSize = channels * pooledHeight * pooledWidth;

    tiling->baseRoisPerCore = baseRoisPerCore;
    tiling->bigTotalRois = bigTotalRois;
    tiling->tailRoiNum = tailRoiNum;
    tiling->featureTotalSize = featureTotalSize;
    tiling->roiLength = roiLength;
    tiling->featureMapSize = featureMapSize;
    tiling->outRoiSize = outRoiSize;
    tiling->numRois = numRois;
    tiling->batch = batch;
    tiling->channels = channels;
    tiling->height = height;
    tiling->width = width;
    tiling->pooledHeight = pooledHeight;
    tiling->pooledWidth = pooledWidth;
    tiling->spatialScale = spatialScale;
    tiling->samplingRatio = samplingRatio;
    context->SetBlockDim(nowCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRoiAlignV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(RoiAlignV2).Tiling(RoiAlignV2TilingFunc).TilingParse<RoiAlignV2CompileInfo>(TilingParseForRoiAlignV2);
} // namespace optiling
