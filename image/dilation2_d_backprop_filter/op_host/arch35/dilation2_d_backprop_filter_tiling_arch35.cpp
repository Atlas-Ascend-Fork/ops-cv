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
 * \file dilation2_d_backprop_filter_tiling.cpp
 * \brief Tiling implementation for dilation2_d_backprop_filter operator
 *
 * Tiling strategy (MDE v2.1 §3.2):
 *   - Grid-Stride mode, two-step core allocation
 *   - Supports both NHWC and NCHW data formats (v2.5)
 *   - Computes padding (SAME/VALID/CALCULATED) and output dimensions
 *   - TilingKey: single mode NORMAL=0, dtype via DTYPE_ macro
 *   - Workspace: user workspace (per-core buffer) + system workspace
 *
 * v2.1: deterministic accumulation (per-core buffer + final reduce)
 *   - userWorkspaceSize = needCoreNum × perCoreBufElems × sizeof(float)
 *   - perCoreBufElems = alignUp(filterSize, 32) (128B alignment for float)
 *   - Per-core buffer allows deterministic sequential reduce in Phase 3
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "image/dilation2_d_backprop_filter/op_kernel/arch35/dilation2_d_backprop_filter_tiling_data.h"
#include "image/dilation2_d_backprop_filter/op_kernel/arch35/dilation2_d_backprop_filter_tiling_key.h"

#include <string>
#include <algorithm>

namespace optiling {

using namespace Ops::Cv::OpTiling;

constexpr int64_t PER_CORE_MIN = 1024;       // lower bound, aligned to 32
constexpr uint32_t DCACHE_SIZE = 128 * 1024; // 128KB DCache (skill: 32KB~128KB)
constexpr uint32_t STATIC_UB_ESTIMATE = 0;   // no static UB arrays
constexpr int64_t RANK_4D = 4;
constexpr int64_t RANK_3D = 3;
constexpr size_t ATTR_STRIDES_LEN = 4;
constexpr size_t ATTR_RATES_LEN = 4;
constexpr size_t ATTR_PADS_LEN = 4;
constexpr int64_t WS_ALIGN_BYTES = 128;                               // 128B workspace alignment
constexpr int64_t FLOAT_SIZE_BYTES = 4;                               // sizeof(float)
constexpr int64_t WS_ALIGN_ELEMS = WS_ALIGN_BYTES / FLOAT_SIZE_BYTES; // 32 float elements

struct Dilation2DBackpropFilterCompileInfo {
    std::string _pattern;
};

// ============================================================================
// Platform info retrieval
// ============================================================================
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

// ============================================================================
// Workspace allocation (v2.2: per-thread buffer + per-core reduce buffer)
// userWorkspaceSize = needCoreNum × THREAD_NUM × perCoreBufElems × sizeof(float)
// v2.1: userWorkspaceSize = needCoreNum × perCoreBufElems × sizeof(float)
// v2.2: userWorkspaceSize = needCoreNum × 1024 × perCoreBufElems × sizeof(float)
// ============================================================================
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, int32_t needCoreNum, int64_t perCoreBufElems)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    // v2.2: user workspace = perThreadBufs + perCoreBufs
    // perThreadBufs = needCoreNum × THREAD_NUM × perCoreBufElems × sizeof(float)
    // perCoreBufs = needCoreNum × perCoreBufElems × sizeof(float)
    constexpr int64_t THREAD_NUM_TILING = 1024;
    int64_t perThreadBufSize = static_cast<int64_t>(needCoreNum) * THREAD_NUM_TILING * perCoreBufElems *
                               FLOAT_SIZE_BYTES;
    int64_t perCoreBufSize = static_cast<int64_t>(needCoreNum) * perCoreBufElems * FLOAT_SIZE_BYTES;
    int64_t userWorkspaceSize = perThreadBufSize + perCoreBufSize;
    // Align total user workspace to 128B
    userWorkspaceSize = Ops::Base::CeilAlign(userWorkspaceSize, WS_ALIGN_BYTES);

    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

// Non-deterministic mode: system workspace only (no user workspace)
static ge::graphStatus GetWorkspaceSizeNonDet(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Attribute struct and retrieval
// ============================================================================
struct DilBpFilterAttrs {
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t rateH = 0;
    int64_t rateW = 0;
    std::string paddingMode;
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    bool ceilMode = false;
    std::string dataFormat;
};

static ge::graphStatus GetAttrs(gert::TilingContext* context, DilBpFilterAttrs& attrs)
{
    const gert::RuntimeAttrs* runtimeAttrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, runtimeAttrs);

    // Attr 0: strides (ListInt, 4 elements)
    const auto* stridesVec = runtimeAttrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesVec);
    OP_CHECK_IF(stridesVec->GetSize() < ATTR_STRIDES_LEN,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                          std::to_string(stridesVec->GetSize()).c_str(), "4 elements"),
                return ge::GRAPH_FAILED);
    const int64_t* stridesData = stridesVec->GetData();

    // Attr 1: rates (ListInt, 4 elements)
    const auto* ratesVec = runtimeAttrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ratesVec);
    OP_CHECK_IF(ratesVec->GetSize() < ATTR_RATES_LEN,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rates", std::to_string(ratesVec->GetSize()).c_str(),
                                          "4 elements"),
                return ge::GRAPH_FAILED);
    const int64_t* ratesData = ratesVec->GetData();

    // Attr 2: padding_mode (String)
    const char* paddingModePtr = runtimeAttrs->GetStr(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddingModePtr);
    attrs.paddingMode = std::string(paddingModePtr);
    OP_CHECK_IF(attrs.paddingMode != "SAME" && attrs.paddingMode != "VALID" && attrs.paddingMode != "CALCULATED",
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding_mode", attrs.paddingMode.c_str(),
                                          "SAME, VALID or CALCULATED"),
                return ge::GRAPH_FAILED);
    // Attr 3: pads (ListInt, 4 elements)
    const auto* padsVec = runtimeAttrs->GetListInt(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsVec);
    OP_CHECK_IF(padsVec->GetSize() < ATTR_PADS_LEN,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "pads", std::to_string(padsVec->GetSize()).c_str(),
                                          "4 elements"),
                return ge::GRAPH_FAILED);
    const int64_t* padsData = padsVec->GetData();

    // Attr 4: ceil_mode (Bool)
    const bool* ceilModePtr = runtimeAttrs->GetBool(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceilModePtr);
    attrs.ceilMode = *ceilModePtr;

    // Attr 5: data_format (String)
    const char* dataFormatPtr = runtimeAttrs->GetStr(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormatPtr);
    attrs.dataFormat = std::string(dataFormatPtr);

    // Validate data_format: "NHWC" or "NCHW" (v2.5: NCHW support)
    OP_CHECK_IF(
        attrs.dataFormat != "NHWC" && attrs.dataFormat != "NCHW",
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format", attrs.dataFormat.c_str(), "NHWC or NCHW"),
        return ge::GRAPH_FAILED);

    bool isNCHW = (attrs.dataFormat == "NCHW");

    // Validate strides N/C dims must be 1
    // NHWC: strides[0]==1, strides[3]==1; NCHW: strides[0]==1, strides[1]==1
    if (isNCHW) {
        OP_CHECK_IF(stridesData[0] != 1 || stridesData[1] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                              ("strides[0]=" + std::to_string(stridesData[0]) +
                                               ", strides[1]=" + std::to_string(stridesData[1])),
                                              "1"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(stridesData[0] != 1 || stridesData[3] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides",
                                              ("strides[0]=" + std::to_string(stridesData[0]) +
                                               ", strides[3]=" + std::to_string(stridesData[3])),
                                              "1"),
                    return ge::GRAPH_FAILED);
    }

    // Validate rates N/C dims must be 1
    // NHWC: rates[0]==1, rates[3]==1; NCHW: rates[0]==1, rates[1]==1
    if (isNCHW) {
        OP_CHECK_IF(
            ratesData[0] != 1 || ratesData[1] != 1,
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "rates",
                ("rates[0]=" + std::to_string(ratesData[0]) + ", rates[1]=" + std::to_string(ratesData[1])), "1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            ratesData[0] != 1 || ratesData[3] != 1,
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "rates",
                ("rates[0]=" + std::to_string(ratesData[0]) + ", rates[3]=" + std::to_string(ratesData[3])), "1"),
            return ge::GRAPH_FAILED);
    }

    // Extract spatial strides and rates
    // NHWC: [1, strideH, strideW, 1]; NCHW: [1, 1, strideH, strideW]
    if (isNCHW) {
        attrs.strideH = stridesData[2];
        attrs.strideW = stridesData[3];
        attrs.rateH = ratesData[2];
        attrs.rateW = ratesData[3];
    } else {
        attrs.strideH = stridesData[1];
        attrs.strideW = stridesData[2];
        attrs.rateH = ratesData[1];
        attrs.rateW = ratesData[2];
    }

    // Validate strides and rates ranges
    OP_CHECK_IF(
        attrs.strideH < 1,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides", std::to_string(attrs.strideH).c_str(), ">= 1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        attrs.strideW < 1,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides", std::to_string(attrs.strideW).c_str(), ">= 1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs.rateH < 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rates", std::to_string(attrs.rateH).c_str(), ">= 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs.rateW < 1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rates", std::to_string(attrs.rateW).c_str(), ">= 1"),
                return ge::GRAPH_FAILED);

    // Extract pads (CALCULATED mode): [top, bottom, left, right]
    attrs.padTop = padsData[0];
    attrs.padBottom = padsData[1];
    attrs.padLeft = padsData[2];
    attrs.padRight = padsData[3];

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Compute output dimensions and padding (MDE §3.2 step 3)
// ============================================================================
static void ComputeOutputDims(const DilBpFilterAttrs& attrs, int64_t inputH, int64_t inputW, int64_t filterH,
                              int64_t filterW, int64_t& outH, int64_t& outW, int64_t& padTop, int64_t& padLeft)
{
    int64_t windowH = (filterH - 1) * attrs.rateH + 1;
    int64_t windowW = (filterW - 1) * attrs.rateW + 1;

    if (attrs.paddingMode == "SAME") {
        outH = (inputH + attrs.strideH - 1) / attrs.strideH;
        outW = (inputW + attrs.strideW - 1) / attrs.strideW;
        int64_t padRow = std::max((outH - 1) * attrs.strideH + windowH - inputH, static_cast<int64_t>(0));
        int64_t padCol = std::max((outW - 1) * attrs.strideW + windowW - inputW, static_cast<int64_t>(0));
        padTop = std::max(padRow / 2, static_cast<int64_t>(0));
        padLeft = std::max(padCol / 2, static_cast<int64_t>(0));
    } else if (attrs.paddingMode == "CALCULATED") {
        padTop = attrs.padTop;
        padLeft = attrs.padLeft;
        if (attrs.ceilMode) {
            outH = (inputH - windowH + padTop + attrs.padBottom + attrs.strideH - 1) / attrs.strideH + 1;
            outW = (inputW - windowW + padLeft + attrs.padRight + attrs.strideW - 1) / attrs.strideW + 1;
        } else {
            outH = (inputH - windowH + padTop + attrs.padBottom) / attrs.strideH + 1;
            outW = (inputW - windowW + padLeft + attrs.padRight) / attrs.strideW + 1;
        }
    } else { // VALID
        padTop = 0;
        padLeft = 0;
        outH = (inputH - windowH) / attrs.strideH + 1;
        outW = (inputW - windowW) / attrs.strideW + 1;
    }

    // Clamp to non-negative
    outH = std::max(outH, static_cast<int64_t>(0));
    outW = std::max(outW, static_cast<int64_t>(0));
}

struct TilingCommonFields {
    int64_t totalElements;
    int64_t filterSize;
    int32_t needCoreNum;
    int64_t perCoreBufElems;
    int64_t batch;
    int64_t inputH;
    int64_t inputW;
    int64_t depth;
    int64_t filterH;
    int64_t filterW;
    int64_t outH;
    int64_t outW;
    int64_t strideH;
    int64_t strideW;
    int64_t rateH;
    int64_t rateW;
    int64_t padTop;
    int64_t padLeft;
    int64_t padInputH;
    int64_t padInputW;
    bool isNCHW;
    bool isDet;
};

// ========== DumpTilingData (DFX) ==========
static void DumpTilingData(gert::TilingContext* context, const Dilation2DBackpropFilterTilingData* tiling, bool isDet)
{
    OP_LOGD(context,
            "Dilation2DBackpropFilterTilingData: isDet=%d, totalElements=%ld, filterSize=%ld, "
            "needCoreNum=%d, perCoreBufElems=%ld, batch=%ld, inputH=%ld, inputW=%ld, depth=%ld, "
            "filterH=%ld, filterW=%ld, outH=%ld, outW=%ld, strideH=%ld, strideW=%ld, "
            "rateH=%ld, rateW=%ld, padTop=%ld, padLeft=%ld, padInputH=%ld, padInputW=%ld, isNCHW=%d",
            isDet, tiling->totalElements, tiling->filterSize, tiling->needCoreNum, tiling->perCoreBufElems,
            tiling->batch, tiling->inputH, tiling->inputW, tiling->depth, tiling->filterH, tiling->filterW,
            tiling->outH, tiling->outW, tiling->strideH, tiling->strideW, tiling->rateH, tiling->rateW, tiling->padTop,
            tiling->padLeft, tiling->padInputH, tiling->padInputW, tiling->isNCHW);
}

// ============================================================================
static ge::graphStatus FillTilingData(gert::TilingContext* context, const TilingCommonFields& f)
{
    auto* tiling = context->GetTilingData<Dilation2DBackpropFilterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(Dilation2DBackpropFilterTilingData), 0,
                         sizeof(Dilation2DBackpropFilterTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalElements = f.totalElements;
    tiling->filterSize = f.filterSize;
    tiling->needCoreNum = f.needCoreNum;
    tiling->perCoreBufElems = f.perCoreBufElems;
    tiling->batch = f.batch;
    tiling->inputH = f.inputH;
    tiling->inputW = f.inputW;
    tiling->depth = f.depth;
    tiling->filterH = f.filterH;
    tiling->filterW = f.filterW;
    tiling->outH = f.outH;
    tiling->outW = f.outW;
    tiling->strideH = f.strideH;
    tiling->strideW = f.strideW;
    tiling->rateH = f.rateH;
    tiling->rateW = f.rateW;
    tiling->padTop = f.padTop;
    tiling->padLeft = f.padLeft;
    tiling->padInputH = f.padInputH;
    tiling->padInputW = f.padInputW;
    tiling->isNCHW = f.isNCHW ? 1 : 0;

    if (f.isDet) {
        OP_CHECK_IF(GetWorkspaceSize(context, f.needCoreNum, f.perCoreBufElems) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
        uint64_t tilingKey = GET_TPL_TILING_KEY(DILATION2D_BACKPROP_FILTER_MODE_DETERMINISTIC,
                                                DILATION2D_BACKPROP_FILTER_MODE_NORMAL);
        context->SetTilingKey(tilingKey);
    } else {
        OP_CHECK_IF(GetWorkspaceSizeNonDet(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetWorkspaceSizeNonDet error"), return ge::GRAPH_FAILED);
        uint64_t tilingKey = GET_TPL_TILING_KEY(DILATION2D_BACKPROP_FILTER_MODE_NON_DETERMINISTIC,
                                                DILATION2D_BACKPROP_FILTER_MODE_NORMAL);
        context->SetTilingKey(tilingKey);
    }

    DumpTilingData(context, tiling, f.isDet);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Tiling main function
// ============================================================================
static ge::graphStatus Dilation2DBackpropFilterTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get input/output shapes
    // NHWC: x=[N,H,W,C], filter=[fH,fW,C], out_bp=[N,Ho,Wo,C]
    // NCHW: x=[N,C,H,W], filter=[C,fH,fW], out_bp=[N,C,Ho,Wo]
    auto xShapeInput = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapeInput);
    auto xShape = xShapeInput->GetStorageShape();

    auto filterShapeInput = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShapeInput);
    auto filterShape = filterShapeInput->GetStorageShape();

    auto outBpShapeInput = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outBpShapeInput);
    auto outBpShape = outBpShapeInput->GetStorageShape();

    // 3. Validate ranks
    OP_CHECK_IF(
        xShape.GetDimNum() != static_cast<size_t>(RANK_4D),
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()).c_str(), "4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(filterShape.GetDimNum() != static_cast<size_t>(RANK_3D),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "filter",
                                             std::to_string(filterShape.GetDimNum()).c_str(), "3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outBpShape.GetDimNum() != static_cast<size_t>(RANK_4D),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "out_backprop",
                                             std::to_string(outBpShape.GetDimNum()).c_str(), "4"),
                return ge::GRAPH_FAILED);

    // 4. Get attributes (need data_format for dimension extraction)
    DilBpFilterAttrs attrs;
    OP_CHECK_IF(GetAttrs(context, attrs) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrs error"),
                return ge::GRAPH_FAILED);

    bool isNCHW = (attrs.dataFormat == "NCHW");

    // 5. Extract dimensions based on data_format
    // NHWC: x=[N,H,W,C], filter=[fH,fW,C], out_bp=[N,Ho,Wo,C]
    // NCHW: x=[N,C,H,W], filter=[C,fH,fW], out_bp=[N,C,Ho,Wo]
    int64_t batch = 0, inputH = 0, inputW = 0, depth = 0;
    int64_t filterH = 0, filterW = 0;
    if (isNCHW) {
        batch = xShape.GetDim(0);
        depth = xShape.GetDim(1);
        inputH = xShape.GetDim(2);
        inputW = xShape.GetDim(3);
        filterH = filterShape.GetDim(1);
        filterW = filterShape.GetDim(2);
    } else {
        batch = xShape.GetDim(0);
        inputH = xShape.GetDim(1);
        inputW = xShape.GetDim(2);
        depth = xShape.GetDim(3);
        filterH = filterShape.GetDim(0);
        filterW = filterShape.GetDim(1);
    }

    // Validate depth consistency based on data_format
    if (isNCHW) {
        OP_CHECK_IF(depth != filterShape.GetDim(0) || depth != outBpShape.GetDim(1),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop",
                        ("x.C=" + std::to_string(depth) + ", filter.C=" + std::to_string(filterShape.GetDim(0)) +
                         ", out_bp.C=" + std::to_string(outBpShape.GetDim(1))),
                        "depth mismatch: x.C, filter.C and out_bp.C must be the same"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(depth != filterShape.GetDim(2) || depth != outBpShape.GetDim(3),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop",
                        ("x.C=" + std::to_string(depth) + ", filter.C=" + std::to_string(filterShape.GetDim(2)) +
                         ", out_bp.C=" + std::to_string(outBpShape.GetDim(3))),
                        "depth mismatch: x.C, filter.C and out_bp.C must be the same"),
                    return ge::GRAPH_FAILED);
    }

    // Validate dtype: only DT_FLOAT is supported, all inputs/output must have the same dtype
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto filterDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    auto outBpDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outBpDesc);
    auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    ge::DataType filterDtype = filterDesc->GetDataType();
    ge::DataType outBpDtype = outBpDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(filterDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "filter", Ops::Base::ToString(filterDtype).c_str(),
                                          "DT_FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outBpDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "out_backprop",
                                          Ops::Base::ToString(outBpDtype).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", Ops::Base::ToString(yDtype).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDtype != filterDtype || xDtype != outBpDtype || xDtype != yDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, filter, out_backprop, y",
                    (Ops::Base::ToString(xDtype) + ", " + Ops::Base::ToString(filterDtype) + ", " +
                     Ops::Base::ToString(outBpDtype) + ", " + Ops::Base::ToString(yDtype))
                        .c_str(),
                    "all inputs and output must have the same dtype"),
                return ge::GRAPH_FAILED);

    // 6. Compute output dimensions and padding
    int64_t outH = 0, outW = 0, padTop = 0, padLeft = 0;
    ComputeOutputDims(attrs, inputH, inputW, filterH, filterW, outH, outW, padTop, padLeft);

    // v2.3: For CALCULATED padding, kernel needs padded input H/W as boundary
    // (TF golden pads input with zeros, so padded regions are valid with value 0)
    // For SAME/VALID: padInputH = inputH (TF checks original input bounds only)
    int64_t padInputH = inputH;
    int64_t padInputW = inputW;
    if (attrs.paddingMode == "CALCULATED") {
        padInputH = inputH + attrs.padTop + attrs.padBottom;
        padInputW = inputW + attrs.padLeft + attrs.padRight;
    }

    // Validate out_backprop batch/H/W against theoretical forward output geometry.
    // out_backprop is the gradient of the forward output: when both sides are known,
    // its geometry must match the dims derived from x/filter/strides/rates/padding.
    // NHWC: out_bp=[N,Ho,Wo,C]; NCHW: out_bp=[N,C,Ho,Wo]
    int64_t outHActual = 0, outWActual = 0, batchActual = 0;
    if (isNCHW) {
        outHActual = outBpShape.GetDim(2);
        outWActual = outBpShape.GetDim(3);
        batchActual = outBpShape.GetDim(0);
    } else {
        outHActual = outBpShape.GetDim(1);
        outWActual = outBpShape.GetDim(2);
        batchActual = outBpShape.GetDim(0);
    }
    // Batch consistency: fail when both known and mismatched
    OP_CHECK_IF(batch >= 0 && batchActual >= 0 && batchActual != batch,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "x, out_backprop",
                    ("x.N=" + std::to_string(batch) + ", out_bp.N=" + std::to_string(batchActual)),
                    "batch mismatch: out_backprop.N must be the same as x.N"),
                return ge::GRAPH_FAILED);
    if (batch < 0 && batchActual >= 0) {
        // Dynamic x.N: adopt actual out_backprop batch
        batch = batchActual;
    }
    // Spatial consistency: theoretical dims are authoritative when fully derivable (static geometry)
    if (inputH >= 0 && inputW >= 0 && filterH >= 0 && filterW >= 0) {
        OP_CHECK_IF((outHActual >= 0 && outHActual != outH) || (outWActual >= 0 && outWActual != outW),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop",
                        ("out_bp.H=" + std::to_string(outHActual) + ", out_bp.W=" + std::to_string(outWActual) +
                         ", expected H_out=" + std::to_string(outH) + ", W_out=" + std::to_string(outW)),
                        "spatial mismatch: out_backprop H/W must match forward output dims derived from "
                        "x/filter/strides/rates/padding"),
                    return ge::GRAPH_FAILED);
    } else {
        // Dynamic input geometry: adopt actual out_backprop spatial dims when known
        if (outHActual >= 0) {
            outH = outHActual;
        }
        if (outWActual >= 0) {
            outW = outWActual;
        }
    }
    outH = std::max(outH, static_cast<int64_t>(0));
    outW = std::max(outW, static_cast<int64_t>(0));

    // 7. Compute element counts
    int64_t totalElements = batch * outH * outW * depth;
    int64_t filterSize = filterH * filterW * depth;

    // Empty tensor handling
    if (filterSize <= 0 || totalElements <= 0) {
        bool isDet = (context->GetDeterministic() == 1);

        context->SetBlockDim(1);
        context->SetScheduleMode(1);

        TilingCommonFields fields = {totalElements, filterSize,    1,           0,           batch,  inputH,
                                     inputW,        depth,         filterH,     filterW,     outH,   outW,
                                     attrs.strideH, attrs.strideW, attrs.rateH, attrs.rateW, padTop, padLeft,
                                     padInputH,     padInputW,     isNCHW,      isDet};
        OP_CHECK_IF(FillTilingData(context, fields) != ge::GRAPH_SUCCESS, OP_LOGE(context, "FillTilingData error"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize too small"),
                    return ge::GRAPH_FAILED);
        context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
        return ge::GRAPH_SUCCESS;
    }

    // 8. Core allocation (two-step method)
    int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, coreNum);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int32_t needCoreNum = static_cast<int32_t>(Ops::Base::CeilDiv(totalElements, perCoreElements));
    needCoreNum = std::max(needCoreNum, 1); // at least 1 core

    // v2.1: compute per-core buffer size (128B aligned in elements)
    int64_t perCoreBufElems = Ops::Base::CeilAlign(filterSize, WS_ALIGN_ELEMS);

    // 9. Fill TilingData + Workspace + TilingKey (determinism switch)
    bool isDet = (context->GetDeterministic() == 1);

    context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    context->SetScheduleMode(1);

    TilingCommonFields fields = {totalElements, filterSize,    needCoreNum, isDet ? perCoreBufElems : 0,
                                 batch,         inputH,        inputW,      depth,
                                 filterH,       filterW,       outH,        outW,
                                 attrs.strideH, attrs.strideW, attrs.rateH, attrs.rateW,
                                 padTop,        padLeft,       padInputH,   padInputW,
                                 isNCHW,        isDet};
    OP_CHECK_IF(FillTilingData(context, fields) != ge::GRAPH_SUCCESS, OP_LOGE(context, "FillTilingData error"),
                return ge::GRAPH_FAILED);

    // 12. Set LocalMemorySize
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u", ubSize, DCACHE_SIZE),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// TilingParse callback (required by R15 three-part registration)
// Sets _pattern = "Common" in compile_info so that the runtime tiling parse
// can succeed (avoids "compile info not contain [_pattern]" error when the
// framework falls back to DefaultImpl/AutoTilingParser).
// ============================================================================
static ge::graphStatus TilingParseForDilation2DBackpropFilter(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<Dilation2DBackpropFilterCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->_pattern = "Common";
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Tiling registration (three-part: IMPL_OP_OPTILING + .Tiling() + .TilingParse<>())
// ============================================================================
IMPL_OP_OPTILING(Dilation2DBackpropFilter)
    .Tiling(Dilation2DBackpropFilterTilingFunc)
    .TilingParse<Dilation2DBackpropFilterCompileInfo>(TilingParseForDilation2DBackpropFilter);

} // namespace optiling
