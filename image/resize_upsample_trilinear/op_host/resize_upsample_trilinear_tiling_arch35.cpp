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
 * \file resize_upsample_trilinear_tiling_arch35.cpp
 * \brief ResizeUpsampleTrilinear A950 SIMT tiling.
 */

#include <cmath>
#include <limits>
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "resize_upsample_trilinear_tiling.h"
#include "image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_tiling_data.h"
#include "image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_tiling_key.h"
#include "op_api/error_report_util.h"

namespace {
// FULL_3D_SIMD allocates UB for these maxima. Width alignment keeps every
// Gather/DataCopy row naturally aligned for FP16, BF16, and FP32.
constexpr int64_t FULL3D_MIN_IN_D = 4;
constexpr int64_t FULL3D_MAX_IN_D = 8;
constexpr int64_t FULL3D_MIN_IN_H = 64;
constexpr int64_t FULL3D_MAX_IN_H = 128;
constexpr int64_t FULL3D_MIN_IN_W = 64;
constexpr int64_t FULL3D_MAX_IN_W = 128;
constexpr int64_t FULL3D_MIN_OUT_D = 128;
constexpr int64_t FULL3D_MAX_OUT_D = 256;
constexpr int64_t FULL3D_MIN_OUT_H = 128;
constexpr int64_t FULL3D_MAX_OUT_H = 256;
constexpr int64_t FULL3D_MIN_OUT_W = 128;
constexpr int64_t FULL3D_MAX_OUT_W = 256;
constexpr int64_t FULL3D_WIDTH_ALIGN = 16;
constexpr int64_t FULL3D_TARGET_NC = 64;
} // namespace

namespace optiling {
using namespace Ops::Cv::OpTiling;

constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int64_t INPUT_DIMS = 5;
// All A950 schedules operate directly on input/output GM and use only per-core
// UB/register storage.  Reserving a workspace here would allocate device memory
// that the kernel entry explicitly ignores.
constexpr size_t WORKSPACE_SIZE = 0;

constexpr int32_t OUTPUT_SIZE_ATTR = 0;
constexpr int32_t ALIGN_CORNERS_ATTR = 1;
constexpr int32_t SCALE_D_ATTR = 2;
constexpr int32_t SCALE_H_ATTR = 3;
constexpr int32_t SCALE_W_ATTR = 4;

constexpr float MAX_SUPPORT_SCALE = 50.0f;
constexpr int64_t D_ONLY_MIN_ELEMENTS = 1024 * 1024;
constexpr int64_t D_REUSE_MIN_OUTPUT_ELEMENTS = 60 * 1024 * 1024;
constexpr int64_t D_REUSE_MIN_EXPANSION = 4;

struct ResizeUpsampleTrilinearBaseTiling {
    int64_t lenN = 0;
    int64_t lenC = 0;
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t blkProcessNum = 0;
    int32_t tailBlockNum = 0;
    int32_t realCoreNum = 0;
    int32_t coreNum = 0;
    int32_t alignCorners = 0;
    uint64_t isInt32 = 1;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    float scaleD = 0.0f;
    float scaleH = 0.0f;
    float scaleW = 0.0f;
    uint64_t scheduleMode = RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_NCDHW;
};

static bool IsInputDtypeSupported(ge::DataType inputDtype)
{
    return inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16;
}

static bool IsShapeSizeWithinLimit(int64_t shapeSize, uint64_t limit)
{
    return shapeSize > 0 && static_cast<uint64_t>(shapeSize) <= limit;
}

static bool IsDimProductWithinLimit(int64_t dimD, int64_t dimH, int64_t dimW, uint64_t limit)
{
    if (dimD <= 0 || dimH <= 0 || dimW <= 0) {
        return false;
    }
    uint64_t product = static_cast<uint64_t>(dimD);
    uint64_t dimHValue = static_cast<uint64_t>(dimH);
    if (product > limit / dimHValue) {
        return false;
    }
    product *= dimHValue;

    uint64_t dimWValue = static_cast<uint64_t>(dimW);
    return product <= limit / dimWValue;
}

class ResizeUpsampleTrilinearRegbaseTiling {
public:
    explicit ResizeUpsampleTrilinearRegbaseTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScales(float originalScaleD, float originalScaleH, float originalScaleW);
    void CalTilingData();
    bool TryFull3dSimdTiling(int64_t totalNc);
    bool TryDReuseTiling(int64_t totalNc);
    bool TryDOnlyTiling(int64_t totalNc);
    void CalDefaultTiling();
    void FillTilingData();

private:
    ResizeUpsampleTrilinearBaseTiling baseTiling_;
    gert::TilingContext* context_ = nullptr;
    ResizeUpsampleTrilinearRegBaseTilingData* tilingData_ = nullptr;
};

static float ComputeScaleValue(bool alignCorners, int64_t inputSize, int64_t outputSize, float scale)
{
    if (outputSize == inputSize) {
        return 1.0f;
    }
    if (alignCorners) {
        return outputSize > 1 ? static_cast<float>(inputSize - 1) / static_cast<float>(outputSize - 1) : 0.0f;
    }
    return scale > 0.0f ? scale : static_cast<float>(inputSize) / static_cast<float>(outputSize);
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNumAiv <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(coreNumAiv).c_str(),
                                              "coreNum must be greater than 0"),
        return ge::GRAPH_FAILED);
    baseTiling_.coreNum = coreNumAiv;

    tilingData_ = context_->GetTilingData<ResizeUpsampleTrilinearRegBaseTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "get tilingdata ptr failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ResizeUpsampleTrilinearRegBaseTilingData), 0,
                          sizeof(ResizeUpsampleTrilinearRegBaseTilingData)) != EOK),
                OP_LOGE_WITHOUT_REPORT(
                    context_, "memset tilingdata failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::CheckInputParams()
{
    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputDtype failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputShape failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::CheckInputDtype()
{
    auto inputDesc = context_->GetInputDesc(CONST_0);
    OP_CHECK_IF(
        inputDesc == nullptr,
        error_report::ReportInvalidArgument(context_->GetNodeName(), "inputDesc",
                                            "the input desc of required input inputDesc is missing in the tiling "
                                            "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    ge::DataType inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        !IsInputDtypeSupported(inputDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(inputDtype).c_str(),
                                              "input dtype is not supported"),
        return ge::GRAPH_FAILED);
    baseTiling_.inputDtype = inputDtype;

    auto outputDesc = context_->GetOutputDesc(CONST_0);
    OP_CHECK_IF(
        outputDesc == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "outputDesc is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != inputDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input",
                                                      std::to_string(outputDesc->GetDataType()).c_str(),
                                                      "input and output dtype must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::CheckInputShape()
{
    auto inputX = context_->GetInputShape(CONST_0);
    auto outY = context_->GetOutputShape(CONST_0);
    OP_CHECK_IF(
        inputX == nullptr,
        error_report::ReportInvalidArgument(context_->GetNodeName(), "inputX",
                                            "the input desc/shape of required input inputX is missing in the tiling "
                                            "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outY == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "outY is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outputShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outputShape.GetDimNum() != INPUT_DIMS),
                error_report::ReportInvalidArgument(context_->GetNodeName(), "dim",
                                                    "the dim of input or output should be equal to five"),
                return ge::GRAPH_FAILED);

    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outputSize = outputShape.GetShapeSize();
    OP_CHECK_IF(
        inputSize <= 0 || outputSize <= 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "empty", (std::to_string(inputSize) + ", " + std::to_string(outputSize)).c_str(),
            "empty input or output is not supported"),
        return ge::GRAPH_FAILED);
    baseTiling_.lenN = inputShape.GetDim(CONST_0);
    baseTiling_.lenC = inputShape.GetDim(CONST_1);
    baseTiling_.inD = inputShape.GetDim(CONST_2);
    baseTiling_.inH = inputShape.GetDim(CONST_3);
    baseTiling_.inW = inputShape.GetDim(CONST_4);
    baseTiling_.outD = outputShape.GetDim(CONST_2);
    baseTiling_.outH = outputShape.GetDim(CONST_3);
    baseTiling_.outW = outputShape.GetDim(CONST_4);
    baseTiling_.outSize = outputSize;

    OP_CHECK_IF((baseTiling_.lenN != outputShape.GetDim(CONST_0)) || (baseTiling_.lenC != outputShape.GetDim(CONST_1)),
                error_report::ReportInvalidArgument(context_->GetNodeName(), "input",
                                                    "the N and C dimensions of input and output must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(baseTiling_.lenN > 0 && baseTiling_.lenC > 0),
                error_report::ReportInvalidArgument(context_->GetNodeName(), "input",
                                                    "N and C dimensions of input and output must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(baseTiling_.inD > 0 && baseTiling_.inH > 0 && baseTiling_.inW > 0 && baseTiling_.outD > 0 &&
                  baseTiling_.outH > 0 && baseTiling_.outW > 0),
                error_report::ReportInvalidArgument(context_->GetNodeName(), "dimensions",
                                                    "D/H/W dimensions of input and output must be greater than 0"),
                return ge::GRAPH_FAILED);

    uint64_t uint32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    baseTiling_.isInt32 = static_cast<uint64_t>(
        IsShapeSizeWithinLimit(inputSize, uint32Max) && IsShapeSizeWithinLimit(outputSize, uint32Max) &&
        IsDimProductWithinLimit(baseTiling_.inD, baseTiling_.inH, baseTiling_.inW, uint32Max) &&
        IsDimProductWithinLimit(baseTiling_.outD, baseTiling_.outH, baseTiling_.outW, uint32Max));
    return ge::GRAPH_SUCCESS;
}

void ResizeUpsampleTrilinearRegbaseTiling::ComputeScales(float originalScaleD, float originalScaleH,
                                                         float originalScaleW)
{
    bool alignCorners = baseTiling_.alignCorners == 1;
    baseTiling_.scaleD = ComputeScaleValue(alignCorners, baseTiling_.inD, baseTiling_.outD, originalScaleD);
    baseTiling_.scaleH = ComputeScaleValue(alignCorners, baseTiling_.inH, baseTiling_.outH, originalScaleH);
    baseTiling_.scaleW = ComputeScaleValue(alignCorners, baseTiling_.inW, baseTiling_.outW, originalScaleW);
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::CheckInputShapeAndAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    auto outputSizeAttr = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(OUTPUT_SIZE_ATTR);
    int64_t outputSizeNum = outputSizeAttr == nullptr ? 0 : outputSizeAttr->GetSize();
    if (outputSizeAttr != nullptr) {
        OP_CHECK_IF(outputSizeNum > 0 && outputSizeNum != CONST_3,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "output_size",
                                                              std::to_string(outputSizeNum).c_str(),
                                                              "output_size must be empty or have 3 values"),
                    return ge::GRAPH_FAILED);
    }
    if (outputSizeAttr != nullptr && outputSizeNum == CONST_3) {
        const int64_t* outputSizeData = outputSizeAttr->GetData();
        OP_CHECK_IF((baseTiling_.outD != outputSizeData[CONST_0]) || (baseTiling_.outH != outputSizeData[CONST_1]) ||
                        (baseTiling_.outW != outputSizeData[CONST_2]),
                    error_report::ReportInvalidArgument(context_->GetNodeName(), "output",
                                                        "output D/H/W dimensions must be the same as output_size attr"),
                    return ge::GRAPH_FAILED);
    }

    const bool* alignCornersPtr = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    baseTiling_.alignCorners = (alignCornersPtr != nullptr && *alignCornersPtr) ? 1 : 0;

    const float* scaleDPtr = attrs->GetAttrPointer<float>(SCALE_D_ATTR);
    const float* scaleHPtr = attrs->GetAttrPointer<float>(SCALE_H_ATTR);
    const float* scaleWPtr = attrs->GetAttrPointer<float>(SCALE_W_ATTR);
    float originalScaleD = scaleDPtr == nullptr ? 0.0f : *scaleDPtr;
    float originalScaleH = scaleHPtr == nullptr ? 0.0f : *scaleHPtr;
    float originalScaleW = scaleWPtr == nullptr ? 0.0f : *scaleWPtr;
    ComputeScales(originalScaleD, originalScaleH, originalScaleW);
    OP_CHECK_IF((baseTiling_.scaleD > MAX_SUPPORT_SCALE) || (baseTiling_.scaleH > MAX_SUPPORT_SCALE) ||
                    (baseTiling_.scaleW > MAX_SUPPORT_SCALE),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "input",
                    (std::to_string(baseTiling_.scaleD) + ", " + std::to_string(baseTiling_.scaleH) + ", " +
                     std::to_string(baseTiling_.scaleW))
                        .c_str(),
                    "scales should not exceed 50"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ResizeUpsampleTrilinearRegbaseTiling::CalTilingData()
{
    int64_t totalNc = baseTiling_.lenN * baseTiling_.lenC;
    if (TryFull3dSimdTiling(totalNc) || TryDReuseTiling(totalNc) || TryDOnlyTiling(totalNc)) {
        return;
    }
    CalDefaultTiling();
}

bool ResizeUpsampleTrilinearRegbaseTiling::TryFull3dSimdTiling(int64_t totalNc)
{
    bool inputInRange = baseTiling_.inD >= FULL3D_MIN_IN_D && baseTiling_.inD <= FULL3D_MAX_IN_D &&
                        baseTiling_.inH >= FULL3D_MIN_IN_H && baseTiling_.inH <= FULL3D_MAX_IN_H &&
                        baseTiling_.inW >= FULL3D_MIN_IN_W && baseTiling_.inW <= FULL3D_MAX_IN_W;
    bool outputInRange = baseTiling_.outD >= FULL3D_MIN_OUT_D && baseTiling_.outD <= FULL3D_MAX_OUT_D &&
                         baseTiling_.outH >= FULL3D_MIN_OUT_H && baseTiling_.outH <= FULL3D_MAX_OUT_H &&
                         baseTiling_.outW >= FULL3D_MIN_OUT_W && baseTiling_.outW <= FULL3D_MAX_OUT_W;
    bool widthAligned = baseTiling_.inW % FULL3D_WIDTH_ALIGN == 0 && baseTiling_.outW % FULL3D_WIDTH_ALIGN == 0;
    bool isFull3dUpsample = baseTiling_.outD > baseTiling_.inD && baseTiling_.outH > baseTiling_.inH &&
                            baseTiling_.outW > baseTiling_.inW;
    bool scaleValid = baseTiling_.scaleD > 0.0f && baseTiling_.scaleH > 0.0f && baseTiling_.scaleW > 0.0f;
    int64_t targetCoreNum = std::min(static_cast<int64_t>(baseTiling_.coreNum), FULL3D_TARGET_NC);
    bool ncSaturatesCores = totalNc >= targetCoreNum;
    bool addressSafe = baseTiling_.isInt32 == 1;
    bool shapeSupported = inputInRange && outputInRange && widthAligned && isFull3dUpsample;
    if (shapeSupported && scaleValid && ncSaturatesCores && addressSafe) {
        baseTiling_.realCoreNum = static_cast<int32_t>(std::min(totalNc, static_cast<int64_t>(baseTiling_.coreNum)));
        baseTiling_.blkProcessNum = totalNc / baseTiling_.realCoreNum;
        baseTiling_.tailBlockNum = static_cast<int32_t>(totalNc % baseTiling_.realCoreNum);
        baseTiling_.scheduleMode = RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_FULL_3D_SIMD;
        return true;
    }
    return false;
}

bool ResizeUpsampleTrilinearRegbaseTiling::TryDReuseTiling(int64_t totalNc)
{
    uint64_t uint32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    bool ncHwFitsUint32 = IsDimProductWithinLimit(totalNc, baseTiling_.outH, baseTiling_.outW, uint32Max);
    // D_REUSE uses uint32_t task indices for N*C*H*W. Validate that product on
    // the host so the device hot path does not need 64-bit overflow guards.
    bool useDReuseNcHw = baseTiling_.isInt32 == 1 && ncHwFitsUint32 &&
                         baseTiling_.outSize >= D_REUSE_MIN_OUTPUT_ELEMENTS &&
                         baseTiling_.outD >= baseTiling_.inD * D_REUSE_MIN_EXPANSION &&
                         (baseTiling_.outH != baseTiling_.inH || baseTiling_.outW != baseTiling_.inW);
    if (useDReuseNcHw) {
        int64_t ncHwElements = totalNc * baseTiling_.outH * baseTiling_.outW;
        baseTiling_.realCoreNum = static_cast<int32_t>(
            std::min(ncHwElements, static_cast<int64_t>(baseTiling_.coreNum)));
        baseTiling_.blkProcessNum = ncHwElements / baseTiling_.realCoreNum;
        baseTiling_.tailBlockNum = static_cast<int32_t>(ncHwElements % baseTiling_.realCoreNum);
        baseTiling_.scheduleMode = RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_D_REUSE_NC_HW;
        return true;
    }
    return false;
}

bool ResizeUpsampleTrilinearRegbaseTiling::TryDOnlyTiling(int64_t totalNc)
{
    bool scaleDEq = std::abs(baseTiling_.scaleD - 1.0f) < 1e-6f;
    bool scaleHEq = std::abs(baseTiling_.scaleH - 1.0f) < 1e-6f;
    bool scaleWEq = std::abs(baseTiling_.scaleW - 1.0f) < 1e-6f;
    // Half-pixel coordinates use FMA in SIMT. Keep the vector path only where ordinary scalar arithmetic is exact.
    bool dCoordinateSafe = baseTiling_.alignCorners == 1 || std::abs(baseTiling_.scaleD - 0.5f) < 1e-6f;
    bool dOnlyShape = baseTiling_.inD != baseTiling_.outD && baseTiling_.inH == baseTiling_.outH &&
                      baseTiling_.inW == baseTiling_.outW;
    if (dOnlyShape && scaleHEq && scaleWEq && !scaleDEq && dCoordinateSafe &&
        baseTiling_.outSize >= D_ONLY_MIN_ELEMENTS && totalNc >= baseTiling_.coreNum) {
        // The D-only vector path processes complete H*W planes and therefore splits by N*C.
        baseTiling_.realCoreNum = static_cast<int32_t>(std::min(totalNc, static_cast<int64_t>(baseTiling_.coreNum)));
        baseTiling_.blkProcessNum = totalNc / baseTiling_.realCoreNum;
        baseTiling_.tailBlockNum = static_cast<int32_t>(totalNc % baseTiling_.realCoreNum);
        baseTiling_.scheduleMode = RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_D_ONLY;
        return true;
    }
    return false;
}

void ResizeUpsampleTrilinearRegbaseTiling::CalDefaultTiling()
{
    // Generic SIMT and identity-copy paths use a balanced linear element split.
    constexpr int32_t MIN_ELEMENTS_PER_CORE = 64;
    int64_t idealCoreNum = std::max(baseTiling_.outSize / MIN_ELEMENTS_PER_CORE, static_cast<int64_t>(1));
    baseTiling_.realCoreNum = static_cast<int32_t>(std::min(idealCoreNum, static_cast<int64_t>(baseTiling_.coreNum)));
    baseTiling_.realCoreNum = static_cast<int32_t>(
        std::min(static_cast<int64_t>(baseTiling_.realCoreNum), baseTiling_.outSize));
    if (baseTiling_.realCoreNum <= 0) {
        baseTiling_.realCoreNum = 1;
    }
    baseTiling_.blkProcessNum = baseTiling_.outSize / static_cast<int64_t>(baseTiling_.realCoreNum);
    baseTiling_.tailBlockNum = static_cast<int32_t>(baseTiling_.outSize %
                                                    static_cast<int64_t>(baseTiling_.realCoreNum));
    if (std::abs(baseTiling_.scaleD - 1.0f) < 1e-6f && std::abs(baseTiling_.scaleH - 1.0f) < 1e-6f &&
        std::abs(baseTiling_.scaleW - 1.0f) < 1e-6f) {
        baseTiling_.scheduleMode = RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_COPY;
    }
}

void ResizeUpsampleTrilinearRegbaseTiling::FillTilingData()
{
    tilingData_->blkProcessNum = baseTiling_.blkProcessNum;
    tilingData_->lenN = baseTiling_.lenN;
    tilingData_->lenC = baseTiling_.lenC;
    tilingData_->inD = baseTiling_.inD;
    tilingData_->inH = baseTiling_.inH;
    tilingData_->inW = baseTiling_.inW;
    tilingData_->outD = baseTiling_.outD;
    tilingData_->outH = baseTiling_.outH;
    tilingData_->outW = baseTiling_.outW;
    tilingData_->tailBlockNum = baseTiling_.tailBlockNum;
    tilingData_->alignCorners = baseTiling_.alignCorners;
    tilingData_->scaleD = baseTiling_.scaleD;
    tilingData_->scaleH = baseTiling_.scaleH;
    tilingData_->scaleW = baseTiling_.scaleW;
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputParams failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputShapeAndAttr failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();

    const uint64_t tilingKey = GET_TPL_TILING_KEY(baseTiling_.scheduleMode, baseTiling_.isInt32);
    OP_LOGI(context_,
            "ResizeUpsampleTrilinear A950 tilingKey=%lu scheduleMode=%lu isInt32=%lu blockDim=%d "
            "shape=[%ld,%ld,%ld,%ld,%ld]->[%ld,%ld,%ld,%ld,%ld] scale=[%f,%f,%f]",
            tilingKey, baseTiling_.scheduleMode, baseTiling_.isInt32, baseTiling_.realCoreNum, baseTiling_.lenN,
            baseTiling_.lenC, baseTiling_.inD, baseTiling_.inH, baseTiling_.inW, baseTiling_.lenN, baseTiling_.lenC,
            baseTiling_.outD, baseTiling_.outH, baseTiling_.outW, baseTiling_.scaleD, baseTiling_.scaleH,
            baseTiling_.scaleW);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTiling_.realCoreNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "workspaces is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ResizeUpsampleTrilinearRegbase(gert::TilingContext* context)
{
    ResizeUpsampleTrilinearRegbaseTiling tilingImpl(context);
    OP_CHECK_IF(tilingImpl.Init() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "arch35 tiling init failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.DoTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "arch35 tiling do tiling failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
