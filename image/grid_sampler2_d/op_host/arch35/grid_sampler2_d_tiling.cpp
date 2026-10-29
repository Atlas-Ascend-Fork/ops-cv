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
 * \file grid_sampler2_d_tiling.cpp
 * \brief Tiling implementation for grid_sampler2_d operator
 */

#include <string>

#include "../../op_kernel/arch35/grid_sampler2_d_tiling_data.h"
#include "../../op_kernel/arch35/grid_sampler2_d_tiling_key.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_api/error_report_util.h"

namespace optiling {

constexpr int64_t PER_CORE_MIN = 8192;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr int64_t MAX_INT32_VALUE = 2147483647LL;
constexpr int64_t MAX_INT64_VALUE = 9223372036854775807LL;
constexpr int64_t X_INDEX = 0;
constexpr int64_t GRID_INDEX = 1;
constexpr int64_t OUTPUT_INDEX = 0;
constexpr int64_t N_DIM_INDEX = 0;
constexpr int64_t C_DIM_INDEX = 1;
constexpr int64_t H_DIM_INDEX = 2;
constexpr int64_t W_DIM_INDEX = 3;
constexpr int64_t GRID_H_DIM_INDEX = 1;
constexpr int64_t GRID_W_DIM_INDEX = 2;
constexpr int64_t GRID_COORD_DIM_INDEX = 3;
constexpr int64_t DIM_NUM_2D = 4;
constexpr int64_t GRID_COORD_DIM = 2;
constexpr int32_t INTERP_BILINEAR = 0;
constexpr int32_t INTERP_NEAREST = 1;
constexpr int32_t INTERP_BICUBIC = 2;
constexpr int32_t PADDING_ZEROS = 0;
constexpr int32_t PADDING_BORDER = 1;
constexpr int32_t PADDING_REFLECTION = 2;

struct GridSampler2DCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    int64_t isSupportVgather = 0;
    int64_t isSupportMinicihiw = 0;
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input", std::to_string(coreNum).c_str(),
                                                      "coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "input",
                                                          std::to_string(ubSize).c_str(), "ubSize is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtypes(gert::TilingContext* context)
{
    const auto* xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_IF(
        xDesc == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "xDesc is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const auto* gridDesc = context->GetInputDesc(GRID_INDEX);
    OP_CHECK_IF(
        gridDesc == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "gridDesc is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const ge::DataType xDtype = xDesc->GetDataType();
    const ge::DataType gridDtype = gridDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                          "float16 or float32"),
                return ge::GRAPH_FAILED);
    const std::string dtypeMsg = Ops::Base::ToString(xDtype) + " and " + Ops::Base::ToString(gridDtype);
    OP_CHECK_IF(gridDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x and grid", dtypeMsg.c_str(),
                                                       "x and grid must have the same dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputDtype(gert::TilingContext* context)
{
    const auto* xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_IF(
        xDesc == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "xDesc is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const auto* yDesc = context->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_IF(
        yDesc == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "yDesc is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const ge::DataType xDtype = xDesc->GetDataType();
    const ge::DataType yDtype = yDesc->GetDataType();
    OP_CHECK_IF(yDtype != ge::DT_FLOAT16 && yDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", Ops::Base::ToString(yDtype).c_str(),
                                          "float16 or float32"),
                return ge::GRAPH_FAILED);
    const std::string dtypeMsg = Ops::Base::ToString(xDtype) + " and " + Ops::Base::ToString(yDtype);
    OP_CHECK_IF(yDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x and y", dtypeMsg.c_str(),
                                                       "y must have the same dtype as x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndExtractShapes(gert::TilingContext* context, int64_t& N, int64_t& C, int64_t& HIn,
                                             int64_t& WIn, int64_t& HOut, int64_t& WOut)
{
    const auto* xInput = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xInput == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "xInput is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const auto xShape = xInput->GetStorageShape();
    const auto* gridInput = context->GetInputShape(GRID_INDEX);
    OP_CHECK_IF(
        gridInput == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "gridInput is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const auto gridShape = gridInput->GetStorageShape();

    const std::string xDimNum = std::to_string(xShape.GetDimNum()) + "D";
    OP_CHECK_IF(xShape.GetDimNum() != DIM_NUM_2D,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", xDimNum.c_str(), "4D"),
                return ge::GRAPH_FAILED);
    const std::string gridDimNum = std::to_string(gridShape.GetDimNum()) + "D";
    OP_CHECK_IF(gridShape.GetDimNum() != DIM_NUM_2D,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "grid", gridDimNum.c_str(), "4D"),
                return ge::GRAPH_FAILED);

    const int64_t gridCoordDim = gridShape.GetDim(GRID_COORD_DIM_INDEX);
    OP_CHECK_IF(gridShape.GetDim(GRID_COORD_DIM_INDEX) != GRID_COORD_DIM,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "grid", std::to_string(gridCoordDim).c_str(), "2"),
                return ge::GRAPH_FAILED);

    const int64_t xBatch = xShape.GetDim(N_DIM_INDEX);
    const int64_t gridBatch = gridShape.GetDim(N_DIM_INDEX);
    const std::string batchMsg = std::to_string(xBatch) + " and " + std::to_string(gridBatch);
    OP_CHECK_IF(xBatch != gridBatch,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "x and grid", batchMsg.c_str(),
                                                       "x.shape[0] must equal grid.shape[0]"),
                return ge::GRAPH_FAILED);

    N = xBatch;
    C = xShape.GetDim(C_DIM_INDEX);
    HIn = xShape.GetDim(H_DIM_INDEX);
    WIn = xShape.GetDim(W_DIM_INDEX);
    HOut = gridShape.GetDim(GRID_H_DIM_INDEX);
    WOut = gridShape.GetDim(GRID_W_DIM_INDEX);

    const std::string shapeMsg = Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(gridShape);
    OP_CHECK_IF(N < 0 || C < 0 || HIn <= 0 || WIn <= 0 || HOut < 0 || WOut < 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and grid", shapeMsg.c_str(),
                                                       "dimensions must be non-negative and input H/W must be "
                                                       "greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(N > MAX_INT32_VALUE || C > MAX_INT32_VALUE || HIn > MAX_INT32_VALUE || WIn > MAX_INT32_VALUE ||
                    HOut > MAX_INT32_VALUE || WOut > MAX_INT32_VALUE,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and grid", shapeMsg.c_str(),
                                                       "all dimensions must not exceed INT32_MAX"),
                return ge::GRAPH_FAILED);

    const std::string inputSpatialSize = std::to_string(HIn) + " * " + std::to_string(WIn);
    OP_CHECK_IF(HIn > MAX_INT32_VALUE / WIn,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", inputSpatialSize.c_str(),
                                                          "H_in * W_in must not exceed INT32_MAX"),
                return ge::GRAPH_FAILED);

    const std::string outputPixelCount = std::to_string(N) + " * " + std::to_string(HOut) + " * " +
                                         std::to_string(WOut);
    OP_CHECK_IF(N > 0 && HOut > 0 && WOut > 0 && (N > MAX_INT32_VALUE / HOut || N * HOut > MAX_INT32_VALUE / WOut),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "grid", outputPixelCount.c_str(),
                                                          "N * H_out * W_out must not exceed INT32_MAX"),
                return ge::GRAPH_FAILED);

    // The kernel derives x's max address bound from N * C * H_in * W_in in a
    // signed 64-bit product. Each factor alone is bounded by INT32_MAX, so the
    // four-way product can exceed INT64_MAX; constrain it here so the kernel
    // multiplication cannot overflow before its index-width comparison.
    const std::string xElementCount = std::to_string(N) + " * " + std::to_string(C) + " * " + std::to_string(HIn) +
                                      " * " + std::to_string(WIn);
    OP_CHECK_IF(N > 0 && C > 0 && static_cast<int64_t>(N) * C > MAX_INT64_VALUE / (static_cast<int64_t>(HIn) * WIn),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", xElementCount.c_str(),
                                                          "N * C * H_in * W_in must not overflow the 64-bit range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndParseAttrs(gert::TilingContext* context, int32_t& interpolationMode,
                                          int32_t& paddingMode, int32_t& alignCorners)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const char* interpStr = attrs->GetStr(0);
    const std::string interpVal = (interpStr == nullptr) ? "bilinear" : std::string(interpStr);
    if (interpVal == "bilinear") {
        interpolationMode = INTERP_BILINEAR;
    } else if (interpVal == "nearest") {
        interpolationMode = INTERP_NEAREST;
    } else if (interpVal == "bicubic") {
        interpolationMode = INTERP_BICUBIC;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "interpolation_mode", interpVal.c_str(),
                                  "bilinear, nearest or bicubic");
        return ge::GRAPH_FAILED;
    }

    const char* padStr = attrs->GetStr(1);
    const std::string padVal = (padStr == nullptr) ? "zeros" : std::string(padStr);
    if (padVal == "zeros") {
        paddingMode = PADDING_ZEROS;
    } else if (padVal == "border") {
        paddingMode = PADDING_BORDER;
    } else if (padVal == "reflection") {
        paddingMode = PADDING_REFLECTION;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding_mode", padVal.c_str(),
                                  "zeros, border or reflection");
        return ge::GRAPH_FAILED;
    }

    const bool* alignPtr = attrs->GetAttrPointer<bool>(2);
    alignCorners = (alignPtr != nullptr && *alignPtr) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    int64_t userWorkspaceSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "currentWorkspace is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GridSampler2dTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter TilingGridSampler2D");
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "GetPlatformInfo error; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputDtypes(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "CheckInputDtypes error; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckOutputDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "CheckOutputDtype error; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    int64_t N = 0, C = 0, HIn = 0, WIn = 0, HOut = 0, WOut = 0;
    int32_t interpolationMode = 0, paddingMode = 0, alignCorners = 0;
    OP_CHECK_IF(CheckAndExtractShapes(context, N, C, HIn, WIn, HOut, WOut) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "CheckAndExtractShapes error; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckAndParseAttrs(context, interpolationMode, paddingMode, alignCorners) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "CheckAndParseAttrs error; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "GetWorkspaceSize error; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    GridSampler2DTilingData* tiling = context->GetTilingData<GridSampler2DTilingData>();
    OP_CHECK_IF(
        tiling == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "tiling is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(memset_s(tiling, sizeof(GridSampler2DTilingData), 0, sizeof(GridSampler2DTilingData)) != EOK,
                OP_LOGE_WITHOUT_REPORT(
                    context, "Failed to set tiling data; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    int64_t totalPixels = N * HOut * WOut;

    tiling->N = static_cast<int32_t>(N);
    tiling->C = static_cast<int32_t>(C);
    tiling->H_in = static_cast<int32_t>(HIn);
    tiling->W_in = static_cast<int32_t>(WIn);
    tiling->H_out = static_cast<int32_t>(HOut);
    tiling->W_out = static_cast<int32_t>(WOut);
    tiling->interpolationMode = interpolationMode;
    tiling->paddingMode = paddingMode;
    tiling->alignCorners = alignCorners;

    if (totalPixels == 0) {
        context->SetBlockDim(1U);
        context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(interpolationMode)));
        return ge::GRAPH_SUCCESS;
    }

    // Two-step core partitioning
    int64_t perCoreElements = Ops::Base::CeilDiv(totalPixels, coreNum);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalPixels, perCoreElements);

    context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    OP_CHECK_IF(
        ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "input", std::to_string(ubSize).c_str(),
                                                  "ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE"),
        return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context, "SetLocalMemorySize failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    // TilingKey maps interpMode to compile-time template parameter.
    // Values: 0=BILINEAR, 1=NEAREST, 2=BICUBIC.
    // This follows the image_projective_transform pattern exactly.
    uint64_t tilingKeyVal = static_cast<uint64_t>(interpolationMode);
    context->SetTilingKey(GET_TPL_TILING_KEY(tilingKeyVal));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGridSampler2D(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GridSampler2DCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "compileInfo is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "invalid",
                                                      std::to_string(coreNum).c_str(), "coreNum is invalid:"),
                return ge::GRAPH_FAILED);
    compileInfo->coreNum = coreNum;

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "input",
                                                          std::to_string(ubSize).c_str(), "ubSize is 0"),
                return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    // ascend950 supports vgather
    compileInfo->isSupportVgather = 1;
    compileInfo->isSupportMinicihiw = 0;

    OP_LOGD(context->GetNodeName(),
            "TilingParseForGridSampler2D: coreNum=%ld, ubSize=%lu, "
            "isSupportVgather=%ld, isSupportMinicihiw=%ld",
            compileInfo->coreNum, compileInfo->ubSize, compileInfo->isSupportVgather, compileInfo->isSupportMinicihiw);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridSampler2D)
    .Tiling(GridSampler2dTilingFunc)
    .TilingParse<GridSampler2DCompileInfo>(TilingParseForGridSampler2D);

} // namespace optiling
