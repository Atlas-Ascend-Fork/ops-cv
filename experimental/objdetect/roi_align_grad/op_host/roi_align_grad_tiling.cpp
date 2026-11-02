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
 * \file roi_align_grad_tiling.cpp
 * \brief RoiAlignGrad host tiling implementation.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/roi_align_grad_tiling_data.h"
#include "../op_kernel/roi_align_grad_tiling_key.h"

namespace optiling {
struct RoiAlignGradCompileInfo {
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
};
namespace {
static const gert::Shape gVec1Shape = {1};

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return gVec1Shape;
    }
    return inShape;
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

constexpr uint32_t kC0Size = 16U;
constexpr uint64_t kReservedUbBytes = 30U * 1024U;
constexpr uint64_t kWorkspaceNum = 1U;
constexpr uint64_t kSyncWorkspaceAlignBytesPerCore = 8U * 32U * sizeof(int32_t);
constexpr uint64_t kDefaultTilingKey = ROI_ALIGN_GRAD_TPL_SCH_DEFAULT;
constexpr uint64_t kMoveOneRowTilingKey = ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW;
constexpr uint64_t kNc1hwc0RoiSplitTilingKey = ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT;
constexpr uint64_t kDefaultKernelEntry = 0U;
constexpr uint64_t kMoveOneRowKernelEntry = kDefaultKernelEntry;
constexpr uint64_t kPerC1BytesDefaultPath = 4096U;
constexpr uint64_t kFloat32Bytes = 4U;
constexpr uint64_t kRoiSplitMinWorkCount = 16U;
constexpr uint64_t kRoiSplitLowCoreMinWorkCount = 2U;
constexpr uint64_t kRoiSplitMinPooledWidth = 16U;
constexpr uint64_t kRoiSplitMinOutputPlane = 4096U;

constexpr size_t kYDiffIndex = 0U;
constexpr size_t kRoisIndex = 1U;
constexpr size_t kXDiffIndex = 0U;

constexpr size_t kAttrXdiffShape = 0U;
constexpr size_t kAttrPooledWidth = 1U;
constexpr size_t kAttrPooledHeight = 2U;
constexpr size_t kAttrSpatialScale = 3U;
constexpr size_t kAttrSampleNum = 4U;
constexpr size_t kAttrRoiEndMode = 5U;

static uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

static bool ParseXdiffShape(const int64_t* dims, size_t dimNum, uint64_t& xDiffN, uint64_t& xDiffC, uint64_t& c1,
                            uint64_t& xDiffH, uint64_t& xDiffW)
{
    if (dims == nullptr || dimNum != 4U) {
        return false;
    }

    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0) {
        return false;
    }

    xDiffN = static_cast<uint64_t>(dims[0]);
    xDiffC = static_cast<uint64_t>(dims[1]);
    c1 = CeilDiv(xDiffC, static_cast<uint64_t>(kC0Size));
    xDiffH = static_cast<uint64_t>(dims[2]);
    xDiffW = static_cast<uint64_t>(dims[3]);
    return c1 > 0U;
}

static uint64_t CalcBlockDim(uint64_t totalNc, int64_t coreNum)
{
    const uint64_t availableCoreNum = coreNum > 0 ? static_cast<uint64_t>(coreNum) : 1U;
    if (totalNc == 0U) {
        return 1U;
    }
    return std::max<uint64_t>(1U, std::min<uint64_t>(availableCoreNum, totalNc));
}

static uint64_t CalcC1BatchMax(uint64_t c1, uint64_t ubSize)
{
    if (c1 == 0U) {
        return 1U;
    }

    const uint64_t usableUb = ubSize > kReservedUbBytes ? (ubSize - kReservedUbBytes) : 0U;
    const uint64_t batchByUb = usableUb / kPerC1BytesDefaultPath;
    return std::max<uint64_t>(1U, std::min<uint64_t>(c1, batchByUb));
}

static uint64_t CalcC1BatchMaxMoveOneRow(uint64_t c1, uint64_t pooledWidth, uint64_t ubSize)
{
    if (c1 == 0U || pooledWidth == 0U) {
        return 1U;
    }

    const uint64_t usableUb = ubSize > kReservedUbBytes ? (ubSize - kReservedUbBytes) : 0U;
    const uint64_t perC1Bytes = pooledWidth * kC0Size * kFloat32Bytes;
    if (perC1Bytes == 0U) {
        return 1U;
    }

    const uint64_t batchByUb = usableUb / perC1Bytes;
    return std::max<uint64_t>(1U, std::min<uint64_t>(c1, batchByUb));
}
} // namespace

static ge::graphStatus RoiAlignGradTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    uint64_t ubSize = 0U;
    int64_t coreNum = 1;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);

    const auto* yDiffShape = context->GetInputShape(kYDiffIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDiffShape);
    const gert::Shape yDiffStorageShape = EnsureNotScalar(yDiffShape->GetStorageShape());
    OP_CHECK_IF(yDiffStorageShape.GetDimNum() != 4U && yDiffStorageShape.GetDimNum() != 5U,
                OP_LOGE(context, "y_diff dim num must be 4 or 5, but got %zu.", yDiffStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    const auto* roisShape = context->GetInputShape(kRoisIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);
    const gert::Shape roisStorageShape = EnsureNotScalar(roisShape->GetStorageShape());
    OP_CHECK_IF(roisStorageShape.GetDimNum() != 2U,
                OP_LOGE(context, "rois dim num must be 2, but got %zu.", roisStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    const int64_t roisNumDim = roisStorageShape.GetDim(0);
    const int64_t roisRowSizeDim = roisStorageShape.GetDim(1);
    OP_CHECK_IF(roisNumDim < 0 || roisRowSizeDim <= 0,
                OP_LOGE(context, "invalid rois shape, rois_num=%lld, rois_row_size=%lld.",
                        static_cast<long long>(roisNumDim), static_cast<long long>(roisRowSizeDim)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        roisRowSizeDim < 5,
        OP_LOGE(context, "rois row size must be at least 5, but got %lld.", static_cast<long long>(roisRowSizeDim)),
        return ge::GRAPH_FAILED);

    auto* yDiffDesc = context->GetInputDesc(kYDiffIndex);
    auto* roisDesc = context->GetInputDesc(kRoisIndex);
    auto* xDiffDesc = context->GetOutputDesc(kXDiffIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDiffDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDiffDesc);

    const ge::DataType yDiffDtype = yDiffDesc->GetDataType();
    const ge::DataType roisDtype = roisDesc->GetDataType();
    const ge::DataType xDiffDtype = xDiffDesc->GetDataType();
    OP_CHECK_IF(yDiffDtype != ge::DT_FLOAT || roisDtype != ge::DT_FLOAT || xDiffDtype != ge::DT_FLOAT,
                OP_LOGE(context, "y_diff, rois and x_diff only support float32."), return ge::GRAPH_FAILED);

    const ge::Format yDiffFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDiffDesc->GetStorageFormat()));
    const ge::Format roisFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(roisDesc->GetStorageFormat()));
    const ge::Format xDiffFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDiffDesc->GetStorageFormat()));
    OP_CHECK_IF(yDiffFormat != ge::FORMAT_ND && yDiffFormat != ge::FORMAT_NCHW && yDiffFormat != ge::FORMAT_NC1HWC0,
                OP_LOGE(context, "y_diff format only supports ND, NCHW or NC1HWC0, but got %d.",
                        static_cast<int32_t>(yDiffFormat)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(roisFormat != ge::FORMAT_ND,
                OP_LOGE(context, "rois format only supports ND, but got %d.", static_cast<int32_t>(roisFormat)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDiffFormat != yDiffFormat,
                OP_LOGE(context, "x_diff format (%d) must be same as y_diff format (%d).",
                        static_cast<int32_t>(xDiffFormat), static_cast<int32_t>(yDiffFormat)),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* xdiffShape = attrs->GetListInt(kAttrXdiffShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xdiffShape);
    const int64_t* xdiffDims = xdiffShape->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, xdiffDims);

    const int64_t* pooledWidthPtr = attrs->GetInt(kAttrPooledWidth);
    const int64_t* pooledHeightPtr = attrs->GetInt(kAttrPooledHeight);
    const int64_t* sampleNumPtr = attrs->GetInt(kAttrSampleNum);
    const int64_t* roiEndModePtr = attrs->GetInt(kAttrRoiEndMode);
    const float* spatialScalePtr = attrs->GetFloat(kAttrSpatialScale);

    OP_CHECK_NULL_WITH_CONTEXT(context, pooledWidthPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledHeightPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampleNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, roiEndModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, spatialScalePtr);

    const int64_t pooledWidth = *pooledWidthPtr;
    const int64_t pooledHeight = *pooledHeightPtr;
    const int64_t sampleNum = *sampleNumPtr;
    const int64_t roiEndMode = *roiEndModePtr;
    const float spatialScale = *spatialScalePtr;

    OP_CHECK_IF(pooledWidth <= 0 || pooledHeight <= 0,
                OP_LOGE(context, "pooled_width and pooled_height must be positive, but got %lld and %lld.",
                        static_cast<long long>(pooledWidth), static_cast<long long>(pooledHeight)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sampleNum < 0,
                OP_LOGE(context, "sample_num must be greater than or equal to 0, but got %lld.",
                        static_cast<long long>(sampleNum)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        roiEndMode < 0 || roiEndMode > 3,
        OP_LOGE(context, "roi_end_mode only supports 0, 1, 2 or 3, but got %lld.", static_cast<long long>(roiEndMode)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(spatialScale <= 0.0F, OP_LOGE(context, "spatial_scale must be positive, but got %f.", spatialScale),
                return ge::GRAPH_FAILED);

    uint64_t xDiffN = 0U;
    uint64_t xDiffC = 0U;
    uint64_t c1 = 0U;
    uint64_t xDiffH = 0U;
    uint64_t xDiffW = 0U;
    OP_CHECK_IF(!ParseXdiffShape(xdiffDims, xdiffShape->GetSize(), xDiffN, xDiffC, c1, xDiffH, xDiffW),
                OP_LOGE(context, "invalid xdiff_shape attr."), return ge::GRAPH_FAILED);

    const bool isNd = yDiffFormat == ge::FORMAT_ND || yDiffFormat == ge::FORMAT_NCHW ||
                      yDiffStorageShape.GetDimNum() == 4U;
    const int64_t yDiffN = yDiffStorageShape.GetDim(0);
    const int64_t yDiffC = yDiffStorageShape.GetDim(1);
    const int64_t yDiffH = yDiffStorageShape.GetDim(2);
    const int64_t yDiffW = yDiffStorageShape.GetDim(3);
    if (isNd) {
        OP_CHECK_IF(yDiffStorageShape.GetDimNum() != 4U, OP_LOGE(context, "ND y_diff dim num must be 4."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(yDiffN < 0 || yDiffC <= 0 || yDiffH < 0 || yDiffW < 0, OP_LOGE(context, "invalid ND y_diff shape."),
                    return ge::GRAPH_FAILED);
    } else {
        const int64_t yDiffC0 = yDiffStorageShape.GetDim(4);
        OP_CHECK_IF(yDiffN < 0 || yDiffC <= 0 || yDiffH < 0 || yDiffW < 0 || yDiffC0 != static_cast<int64_t>(kC0Size),
                    OP_LOGE(context, "invalid NC1HWC0 y_diff shape."), return ge::GRAPH_FAILED);
    }

    const uint64_t roiCount = roisNumDim > 0 ? static_cast<uint64_t>(roisNumDim) : 0U;
    const uint64_t roisRowSize = static_cast<uint64_t>(roisRowSizeDim);
    OP_CHECK_IF(roiCount != static_cast<uint64_t>(yDiffN),
                OP_LOGE(context, "rois num (%llu) must equal y_diff N (%lld).",
                        static_cast<unsigned long long>(roiCount), static_cast<long long>(yDiffN)),
                return ge::GRAPH_FAILED);
    if (isNd) {
        OP_CHECK_IF(static_cast<uint64_t>(yDiffC) != xDiffC,
                    OP_LOGE(context, "ND y_diff C (%lld) must equal xdiff_shape C (%llu).",
                            static_cast<long long>(yDiffC), static_cast<unsigned long long>(xDiffC)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(static_cast<uint64_t>(yDiffC) != c1,
                    OP_LOGE(context, "y_diff C1 (%lld) must equal ceil(xdiff_shape C / 16) (%llu).",
                            static_cast<long long>(yDiffC), static_cast<unsigned long long>(c1)),
                    return ge::GRAPH_FAILED);
    }
    if (isNd) {
        OP_CHECK_IF(
            static_cast<uint64_t>(yDiffH) * static_cast<uint64_t>(yDiffW) !=
                static_cast<uint64_t>(pooledHeight) * static_cast<uint64_t>(pooledWidth),
            OP_LOGE(context,
                    "ND y_diff pooled elems (%lld * %lld) must equal pooled_height * pooled_width (%lld * %lld).",
                    static_cast<long long>(yDiffH), static_cast<long long>(yDiffW),
                    static_cast<long long>(pooledHeight), static_cast<long long>(pooledWidth)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(yDiffH != pooledHeight || yDiffW != pooledWidth,
                    OP_LOGE(context, "y_diff H/W (%lld, %lld) must equal pooled_height/pooled_width (%lld, %lld).",
                            static_cast<long long>(yDiffH), static_cast<long long>(yDiffW),
                            static_cast<long long>(pooledHeight), static_cast<long long>(pooledWidth)),
                    return ge::GRAPH_FAILED);
    }

    const auto* xDiffShape = context->GetOutputShape(kXDiffIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDiffShape);
    const gert::Shape xDiffStorageShape = EnsureNotScalar(xDiffShape->GetStorageShape());
    if (isNd) {
        OP_CHECK_IF(xDiffStorageShape.GetDimNum() != 4U, OP_LOGE(context, "ND x_diff dim num must be 4."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(xDiffStorageShape.GetDim(0) != static_cast<int64_t>(xDiffN) ||
                        xDiffStorageShape.GetDim(1) != static_cast<int64_t>(xDiffC) ||
                        xDiffStorageShape.GetDim(2) != static_cast<int64_t>(xDiffH) ||
                        xDiffStorageShape.GetDim(3) != static_cast<int64_t>(xDiffW),
                    OP_LOGE(context, "ND x_diff shape must be [N, C, H, W]."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(xDiffStorageShape.GetDimNum() != 5U, OP_LOGE(context, "NC1HWC0 x_diff dim num must be 5."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(xDiffStorageShape.GetDim(0) != static_cast<int64_t>(xDiffN) ||
                        xDiffStorageShape.GetDim(1) != static_cast<int64_t>(c1) ||
                        xDiffStorageShape.GetDim(2) != static_cast<int64_t>(xDiffH) ||
                        xDiffStorageShape.GetDim(3) != static_cast<int64_t>(xDiffW) ||
                        xDiffStorageShape.GetDim(4) != static_cast<int64_t>(kC0Size),
                    OP_LOGE(context, "NC1HWC0 x_diff shape must be [N, ceil(C/16), H, W, 16]."),
                    return ge::GRAPH_FAILED);
    }

    const uint64_t workChannelCount = isNd ? xDiffC : c1;
    const uint64_t c1BatchMaxDefault = CalcC1BatchMax(c1, ubSize);
    const uint64_t c1BatchMaxMoveOneRow = CalcC1BatchMaxMoveOneRow(c1, static_cast<uint64_t>(pooledWidth), ubSize);
    const bool useMoveOneRow = !isNd && pooledWidth > 1 && c1BatchMaxMoveOneRow > 0U;
    const uint64_t roiSplitWorkCount = roiCount * c1;
    const uint64_t availableCoreNum = coreNum > 0 ? static_cast<uint64_t>(coreNum) : 1U;
    const uint64_t outputPlane = xDiffH * xDiffW;
    const bool roiBatchShapeSafeForRoiSplit = roiCount <= xDiffN || roiCount > availableCoreNum;
    const bool useNc1hwc0RoiSplit = useMoveOneRow && roiBatchShapeSafeForRoiSplit && c1 < availableCoreNum &&
                                    roiSplitWorkCount >= kRoiSplitMinWorkCount &&
                                    static_cast<uint64_t>(pooledWidth) >= kRoiSplitMinPooledWidth &&
                                    outputPlane >= kRoiSplitMinOutputPlane;
    // 欠核修复：当按 c1 分核会严重欠核（c1 远小于可用核数），且 roi*c1 能提供更多并行、
    // 输出规模足够摊薄原子累加开销时，强制走 roi*c1 满核分核（RoiSplit，含 SetAtomicAdd 解决写冲突），
    // 放宽 pooledWidth / roiBatchSafe 限制。仅在原本欠核时触发，不影响已满核的 case。
    const uint64_t c1BlockDim = CalcBlockDim(c1, coreNum);
    const uint64_t roiSplitBlockDim = CalcBlockDim(roiSplitWorkCount, coreNum);
    const bool lowCoreUtilization = c1BlockDim * 2U <= availableCoreNum;
    const bool forceRoiSplitForLowCore = useMoveOneRow && lowCoreUtilization && roiSplitBlockDim > c1BlockDim &&
                                         roiSplitWorkCount >= kRoiSplitLowCoreMinWorkCount &&
                                         outputPlane >= kRoiSplitMinOutputPlane;
    const bool useNc1hwc0RoiSplitFinal = useNc1hwc0RoiSplit || forceRoiSplitForLowCore;
    uint64_t blockDim = isNd ? CalcBlockDim(xDiffN * workChannelCount, coreNum) :
                               CalcBlockDim(useNc1hwc0RoiSplitFinal ? roiSplitWorkCount : c1, coreNum);
    // Zero-output full-core: ZeroOutput() partitions the whole output plane by
    // runningCoreNum(=blockDim). When RoiSplit blockDim=roi*c1 is low-core, zeroing is
    // stuck on few cores (Test_006: 4 cores clear 51 planes vs TBE full-core MemSet).
    // Raise blockDim to cover output planes; scatter round-robin still uses only the
    // first roi*c1 cores (blockIdx>=totalNc skip), extra cores just zero + SyncAll.
    if (!isNd && useNc1hwc0RoiSplitFinal) {
        const uint64_t zeroCoreDim = CalcBlockDim(xDiffN * c1, coreNum);
        if (zeroCoreDim > blockDim) {
            blockDim = zeroCoreDim;
        }
    }
    const uint64_t tilingKey = useNc1hwc0RoiSplitFinal ? kNc1hwc0RoiSplitTilingKey :
                                                         (useMoveOneRow ? kMoveOneRowTilingKey : kDefaultTilingKey);
    const uint64_t kernelEntry = useNc1hwc0RoiSplitFinal ?
                                     kMoveOneRowKernelEntry :
                                     (useMoveOneRow ? kMoveOneRowKernelEntry : kDefaultKernelEntry);
    const uint64_t c1BatchMax = isNd ? 1U : (useMoveOneRow ? c1BatchMaxMoveOneRow : c1BatchMaxDefault);

    auto* tiling = context->GetTilingData<RoiAlignGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->tilingKey = tilingKey;
    tiling->runningCoreNum = blockDim;
    tiling->roiCount = roiCount;
    tiling->roisRowSize = roisRowSize;
    tiling->xDiffN = xDiffN;
    tiling->xDiffC = xDiffC;
    tiling->c1 = workChannelCount;
    tiling->xDiffH = xDiffH;
    tiling->xDiffW = xDiffW;
    tiling->c1BatchMax = c1BatchMax;
    tiling->pooledWidth = static_cast<int32_t>(pooledWidth);
    tiling->pooledHeight = static_cast<int32_t>(pooledHeight);
    tiling->sampleNum = static_cast<int32_t>(sampleNum);
    tiling->roiEndMode = static_cast<int32_t>(roiEndMode);
    tiling->isNd = isNd ? 1 : 0;
    tiling->spatialScale = spatialScale;
    tiling->pooledWidthReciprocal = 1.0F / static_cast<float>(pooledWidth);
    tiling->pooledHeightReciprocal = 1.0F / static_cast<float>(pooledHeight);
    tiling->sampleNumReciprocal = sampleNum > 0 ? (1.0F / static_cast<float>(sampleNum)) : 0.0F;

    OP_LOGI(context->GetNodeName(),
            "PrintTilingData RoiAlignGrad tilingKey=%lu kernelEntry=%lu blockDim=%lu isNd=%d "
            "useMoveOneRow=%d useNc1hwc0RoiSplit=%d roiCount=%lu c1=%lu xDiffN=%lu xDiffC=%lu "
            "xDiffH=%lu xDiffW=%lu pooledHeight=%ld pooledWidth=%ld sampleNum=%ld c1BatchMax=%lu "
            "roiSplitWorkCount=%lu outputPlane=%lu",
            tilingKey, kernelEntry, blockDim, tiling->isNd, useMoveOneRow ? 1 : 0, useNc1hwc0RoiSplitFinal ? 1 : 0,
            roiCount, workChannelCount, xDiffN, xDiffC, xDiffH, xDiffW, pooledHeight, pooledWidth, sampleNum,
            c1BatchMax, roiSplitWorkCount, outputPlane);

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(kernelEntry);
    context->GetRawTilingData()->SetDataSize(sizeof(RoiAlignGradTilingData));

    size_t* workspaceSizes = context->GetWorkspaceSizes(kWorkspaceNum);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = blockDim * kSyncWorkspaceAlignBytesPerCore;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiAlignGradTilingPrepare(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* compileInfo = context->GetCompiledInfo<RoiAlignGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ROIAlignGrad)
    .Tiling(RoiAlignGradTilingFunc)
    .TilingParse<RoiAlignGradCompileInfo>(RoiAlignGradTilingPrepare);
} // namespace optiling
