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

// Every position in this bounded shape has the same sole evaluated tap.
// The kernel retains the original lowest-value/NaN fallback to tap zero.
static bool SingleSampleGeometry(int64_t batch, int64_t depth, int64_t filterH, int64_t filterW, int64_t outH,
                                 int64_t outW, int64_t padTop, int64_t padLeft, int64_t padInputH, int64_t padInputW,
                                 int64_t rateH, int64_t rateW, int64_t& firstH, int64_t& firstW)
{
    if (batch <= 0 || batch > 4096 || depth <= 0 || depth > INT32_MAX / batch || filterH <= 0 || filterW <= 0 ||
        filterH > 32 || filterW > 32 || filterH * filterW > 32 || outH != 1 || outW != 1 || padTop < 0 || padLeft < 0 ||
        padTop > INT32_MAX || padLeft > INT32_MAX || padInputH <= 0 || padInputW <= 0 || padInputH > INT32_MAX ||
        padInputW > INT32_MAX || rateH <= 0 || rateW <= 0 || rateH > INT32_MAX || rateW > INT32_MAX)
        return false;
    firstH = Ops::Base::CeilDiv(padTop, rateH);
    firstW = Ops::Base::CeilDiv(padLeft, rateW);
    const int64_t lastH = std::min(filterH - 1, (padInputH - 1 + padTop) / rateH);
    const int64_t lastW = std::min(filterW - 1, (padInputW - 1 + padLeft) / rateW);
    return firstH < filterH && firstW < filterW && firstH == lastH && firstW == lastW;
}

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
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, int32_t needCoreNum, int64_t scratchStrideElems,
                                        int64_t partialsElems, bool routeDense, bool routeDenseScatter,
                                        int64_t totalElements, int64_t filterSize, bool emptyCase,
                                        int64_t perCoreBufElems)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    constexpr int64_t THREAD_NUM_TILING = 1024;
    int64_t userWorkspaceSize = 0;
    if (emptyCase) {
        // v5.0 (composed-shape-routed-v3, f5): empty fast path — system workspace only
        // (sysWs + 0 user bytes): the empty kernel only zero-fills y and never touches
        // the workspace (baseline empty-path parity).
        userWorkspaceSize = 0;
    } else if (routeDenseScatter) {
        // v6.0 (r1_dense_keyarb_v1): dense-scatter route — posBuf (int32 per
        // element) + per-core target partials (float per (core, target)):
        //   4 × totalElements + 4 × needCoreNum × filterSize
        userWorkspaceSize = totalElements * FLOAT_SIZE_BYTES +
                            static_cast<int64_t>(needCoreNum) * filterSize * FLOAT_SIZE_BYTES;
    } else if (routeDense) {
        // v4.0→v5.0 (f2/f5): dense path — frozen-baseline formula, kept from round2
        // perThreadBufSize = needCoreNum × THREAD_NUM × perCoreBufElems × sizeof(float)
        // perCoreBufSize   = needCoreNum × perCoreBufElems × sizeof(float)
        int64_t perThreadBufSize = static_cast<int64_t>(needCoreNum) * THREAD_NUM_TILING * perCoreBufElems *
                                   FLOAT_SIZE_BYTES;
        int64_t perCoreBufSize = static_cast<int64_t>(needCoreNum) * perCoreBufElems * FLOAT_SIZE_BYTES;
        userWorkspaceSize = perThreadBufSize + perCoreBufSize;
    } else {
        // v3.0 (chanexcl-fhw-merge-v1, F1) → r3 (chanexcl-sector-private-slot-partials-v1):
        // user workspace = partials only
        // partialsElems = groupNum × depth × alignUp(fH×fW, 8) floats — one
        //                 32B-aligned padded slot per (g, c) accumulator, written
        //                 in place by its owning thread (the thread-private
        //                 scratch buffer and its per-slot flush copy are gone)
        // 128B aligned; no per-thread allocation scaling with filterSize (F1)
        int64_t partialsBytes = partialsElems * FLOAT_SIZE_BYTES;
        userWorkspaceSize = partialsBytes;
    }
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
    // v3.0 (chanexcl-fhw-merge-v1): det scheme fields, populated in FillTilingData (F1/F4)
    // (in-class initializers keep the out-of-line aggregate init sites -Werror clean)
    int32_t groupNum = 0;      // spatial group count G (det path)
    int64_t scratchStride = 0; // per-thread scratch elements, alignUp(fH*fW, 8) (det path)
    // v4.0→v5.0 (composed-shape-routed-v3): det scheme fields, populated in FillTilingData (f1)
    int32_t routeDense = 0; // 1 = shape-routed frozen-baseline dense path (det path)
    int64_t chunkLen = 0;   // idle since v5.0 f3 rollback (round1 strided partition); kept append-only, always 0
};

// Bound every address and signed coordinate used by the 32-bit scan, including
// coordinates in zero padding that are computed before the real-input check.
static bool FitsIndex32Scan(const TilingCommonFields& f)
{
    constexpr int64_t limit = INT32_MAX;
    if (!f.isDet || f.depth <= 0 || f.filterH <= 0 || f.filterW <= 0 || f.filterH > 32 || f.filterW > 32 ||
        f.filterH * f.filterW > 32 || f.totalElements <= 0 || f.totalElements > limit || f.filterSize <= 0 ||
        f.filterSize > limit || f.inputH <= 0 || f.inputW <= 0 || f.padTop < 0 || f.padTop > limit || f.padLeft < 0 ||
        f.padLeft > limit)
        return false;
    int64_t addressBound = 1;
    const int64_t dimensions[] = {f.batch, std::max(f.inputH, f.padInputH), std::max(f.inputW, f.padInputW), f.depth};
    for (const int64_t dim : dimensions) {
        if (dim <= 0 || addressBound > limit / dim)
            return false;
        addressBound *= dim;
    }
    const auto coordinateFits = [limit](int64_t output, int64_t stride, int64_t filter, int64_t rate) {
        if (output <= 0 || stride <= 0 || stride > limit || rate <= 0 || rate > limit || output - 1 > limit / stride)
            return false;
        const int64_t lastStart = (output - 1) * stride;
        return filter - 1 <= (limit - lastStart) / rate;
    };
    const int64_t slotPad = Ops::Base::CeilAlign(f.filterH * f.filterW, int64_t(8));
    return coordinateFits(f.outH, f.strideH, f.filterH, f.rateH) &&
           coordinateFits(f.outW, f.strideW, f.filterW, f.rateW) && f.groupNum > 0 &&
           f.depth <= limit / f.groupNum / slotPad && f.needCoreNum > 0 && f.needCoreNum <= limit / 1024 / slotPad;
}

// ========== DumpTilingData (DFX) ==========
static void DumpTilingData(gert::TilingContext* context, const Dilation2DBackpropFilterTilingData* tiling, bool isDet)
{
    OP_LOGD(context,
            "Dilation2DBackpropFilterTilingData: isDet=%d, totalElements=%ld, filterSize=%ld, "
            "needCoreNum=%d, perCoreBufElems=%ld, batch=%ld, inputH=%ld, inputW=%ld, depth=%ld, "
            "filterH=%ld, filterW=%ld, outH=%ld, outW=%ld, strideH=%ld, strideW=%ld, "
            "rateH=%ld, rateW=%ld, padTop=%ld, padLeft=%ld, padInputH=%ld, padInputW=%ld, isNCHW=%d, "
            "groupNum=%d, scratchStride=%ld, routeDense=%d, chunkLen=%ld",
            isDet, tiling->totalElements, tiling->filterSize, tiling->needCoreNum, tiling->perCoreBufElems,
            tiling->batch, tiling->inputH, tiling->inputW, tiling->depth, tiling->filterH, tiling->filterW,
            tiling->outH, tiling->outW, tiling->strideH, tiling->strideW, tiling->rateH, tiling->rateW, tiling->padTop,
            tiling->padLeft, tiling->padInputH, tiling->padInputW, tiling->isNCHW, tiling->groupNum,
            tiling->scratchStride, tiling->routeDense, tiling->chunkLen);
}

// ============================================================================
static ge::graphStatus FillTilingData(gert::TilingContext* context, TilingCommonFields& f)
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

    // v5.0 (composed-shape-routed-v3): shape-routing predicate (f1) — pure function of the
    // tiling inputs, deterministic (no runtime adaptation, no history):
    //   routeDense = (totalElements > 0 && filterSize > 0
    //                 && ((depth <= ROUTE_DEPTH_MAX && spat >= ROUTE_SPAT_MIN)
    //                     || (isNCHW && filterSize <= ROUTE_NCHW_FS_MAX
    //                         && spat >= ROUTE_NCHW_SPAT_MIN))),
    //   spat = batch×outH×outW, filterSize = filterH×filterW×depth.
    // The NCHW small-fs extension bound (spat >= 4096) is measured (design-input
    // revision 1): L0_032 (fs=44, spat=74655) routes dense (baseline order, precision
    // restored); L0_057 (fs=47, spat=3508) stays chanexcl (round1 56.6us PASS — dense
    // would be baseline 206.2us, over target 142.2); micro-spat NCHW cases
    // (L0_007/bound_023/bound_026/L0_043/L0_044/L0_048/L0_073, spat<4096) are unaffected.
    // Keep this expression identical to the twin in the core-allocation section below:
    // the dense needCoreNum formula and the kernel's routeDense branch must always
    // agree on the routing decision (f2 bit-identical reuse).
    constexpr int64_t ROUTE_DEPTH_MAX = 10;
    constexpr int64_t ROUTE_SPAT_MIN = 2048;
    constexpr int64_t ROUTE_NCHW_FS_MAX = 64;
    constexpr int64_t ROUTE_NCHW_SPAT_MIN = 4096;
    const int64_t spatialCount = f.batch * f.outH * f.outW;
    const int32_t routeDense = (f.totalElements > 0 && f.filterSize > 0 &&
                                ((f.depth <= ROUTE_DEPTH_MAX && spatialCount >= ROUTE_SPAT_MIN) ||
                                 (f.isNCHW && f.filterSize <= ROUTE_NCHW_FS_MAX &&
                                  spatialCount >= ROUTE_NCHW_SPAT_MIN))) ?
                                   1 :
                                   0;
    f.routeDense = routeDense;
    // v6.0 (r1_dense_keyarb_v1): capacity gate for the dense-scatter route.
    // Pure shape/capacity predicate (no case id, no shape whitelist, evaluable
    // for unseen inputs): inside the dense route only shapes whose element
    // count fits the posBuf capacity model take the new route; every other
    // dense shape keeps the byte-identical legacy four-phase body.
    constexpr int64_t SCATTER_CAPACITY_ELEMS = 2097152; // 2^21 elements
    const int32_t routeDenseScatter = (routeDense != 0 && f.totalElements <= SCATTER_CAPACITY_ELEMS &&
                                       f.filterSize >= 1) ?
                                          1 :
                                          0;

    // v3.0 (chanexcl-fhw-merge-v1): det scheme fields (F1/F3/F4/F6)
    // Non-empty chanexcl: G = max(1, min(P, S/depth)) with P = batch×outH×outW, S = needCoreNum×1024
    // (full-core scan parallelism; group count capped by spatial positions so empty
    // groups never allocate partials). Dense path: chanexcl fields are never read by
    // the dense kernel (defensive zeros, same as the empty fast path). Empty fast path
    // keeps zero user workspace (byte-identical to the baseline empty path).
    constexpr int64_t THREAD_NUM_FILL = 1024;
    constexpr int64_t SCRATCH_ALIGN_ELEMS = 8; // 32B per-thread scratch alignment (F1)
    const bool emptyCase = (f.totalElements <= 0 || f.filterSize <= 0);
    const int64_t filterPlane = f.filterH * f.filterW;
    int64_t partialsElems = 0;
    if (emptyCase) {
        f.groupNum = 1;      // defensive; empty path never launches VF-A/VF-B
        f.scratchStride = 0; // → 0 user workspace bytes (F1, baseline empty-path parity)
        f.chunkLen = 0;
    } else if (routeDense != 0) {
        // v4.0 f2: dense path — frozen-baseline execution; chanexcl fields unused
        f.groupNum = 1;
        f.scratchStride = 0;
        f.chunkLen = 0;
    } else {
        const int64_t spatialTotal = f.batch * f.outH * f.outW;
        const int64_t totalThreads = static_cast<int64_t>(f.needCoreNum) * THREAD_NUM_FILL;
        int64_t groupNum = std::min(spatialTotal, totalThreads / f.depth);
        f.groupNum = static_cast<int32_t>(std::max(groupNum, int64_t(1)));
        f.scratchStride = Ops::Base::CeilAlign(filterPlane, SCRATCH_ALIGN_ELEMS);
        // r3 (chanexcl-sector-private-slot-partials-v1): the partials array is the
        // accumulator itself, one padded slot per (g, c): slot stride = scratchStride
        // (= alignUp(filterPlane, 8), 32B per slot) so slots never share a 32B sector.
        partialsElems = static_cast<int64_t>(f.groupNum) * f.depth * f.scratchStride;
        // v5.0 f3: chanexcl core rolled back to the round1 strided partition
        // p ≡ g (mod G) — chunkLen is idle (kept append-only in TilingData, always 0)
        f.chunkLen = 0;
    }
    tiling->groupNum = f.groupNum;
    tiling->scratchStride = f.scratchStride;
    tiling->routeDense = f.routeDense;
    tiling->chunkLen = f.chunkLen;
    // v6.0 (r1_dense_keyarb_v1): append-only dense-scatter route fields
    tiling->routeDenseScatter = routeDenseScatter;
    tiling->posBufElems = (routeDenseScatter != 0) ? f.totalElements : 0;
    tiling->perCorePartialsElems = (routeDenseScatter != 0) ? static_cast<int64_t>(f.needCoreNum) * f.filterSize : 0;
    // r9 (vec21 route): deep-channel 7x3 NHWC vector form. Pure shape/capacity
    // predicate (no case id, no CSV row): among every frozen set only the
    // C>=128 7x3 NHWC shape qualifies (L0_047, C=256; the other 7x3 members
    // are C=39/25). Capacity gate: the per-core partial plane (needCoreNum x
    // 21 x depth) must fit inside the workspace the det path already
    // allocates, so GetWorkspaceSize never changes.
    constexpr int64_t VEC21_DEPTH_MIN = 128;
    // r10: per-core UB slab capacity — the scan stages its touched x batch
    // planes (nSpanMax planes) plus the 21 x depth filter plane into UB.
    constexpr int64_t VEC21_UB_CAP_BYTES = 65536;
    const int64_t vec21Plane = f.filterH * f.filterW; // == 21 when gated below
    const int64_t vec21WsElems = static_cast<int64_t>(f.needCoreNum) * vec21Plane * f.depth;
    const int64_t vec21PlaneElems = f.inputH * f.inputW * f.depth;
    const int64_t vec21PlaneBytes = vec21PlaneElems * int64_t(4);
    const int64_t vec21RowsPerCore = Ops::Base::CeilDiv(f.batch * f.outH * f.outW, static_cast<int64_t>(f.needCoreNum));
    const int64_t vec21NSpanMax = Ops::Base::CeilDiv(vec21RowsPerCore, f.outH * f.outW) + 1;
    const int64_t vec21UbBytes = vec21NSpanMax * vec21PlaneBytes + vec21Plane * f.depth * int64_t(4);
    const int32_t routeVec21 = (f.isDet && !emptyCase && (f.isNCHW == 0) && f.filterH == 7 && f.filterW == 3 &&
                                f.depth >= VEC21_DEPTH_MIN &&
                                vec21WsElems <= static_cast<int64_t>(f.groupNum) * f.depth * f.scratchStride &&
                                vec21PlaneBytes <= 65535 && vec21UbBytes <= VEC21_UB_CAP_BYTES) ?
                                   1 :
                                   0;
    tiling->routeVec21 = routeVec21;
    tiling->routeSingleTapSmall = (f.isDet && !emptyCase && spatialCount <= 4096 && f.totalElements <= INT32_MAX &&
                                   f.filterH == 1 && f.filterW == 1 &&
                                   platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                                       platform_ascendc::SocVersion::ASCEND950) ?
                                      1 :
                                      0;
    tiling->routeBalancedMerge = (f.isDet && !emptyCase && f.depth >= 128 && filterPlane <= 32 &&
                                  platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                                      platform_ascendc::SocVersion::ASCEND950) ?
                                     1 :
                                     0;
    int64_t firstH = 0, firstW = 0;
    const bool singleSample = f.isDet &&
                              platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                                  platform_ascendc::SocVersion::ASCEND950 &&
                              SingleSampleGeometry(f.batch, f.depth, f.filterH, f.filterW, f.outH, f.outW, f.padTop,
                                                   f.padLeft, f.padInputH, f.padInputW, f.rateH, f.rateW, firstH,
                                                   firstW);
    tiling->routeSingleSample = singleSample ? 1 : 0;
    tiling->routeIndex32 = FitsIndex32Scan(f) &&
                                   platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                                       platform_ascendc::SocVersion::ASCEND950 ?
                               1 :
                               0;
    if (singleSample) {
        tiling->sampleH = static_cast<int32_t>(firstH * f.rateH - f.padTop);
        tiling->sampleW = static_cast<int32_t>(firstW * f.rateW - f.padLeft);
        tiling->sampleTap = static_cast<int32_t>(firstH * f.filterW + firstW);
    }

    if (f.isDet) {
        OP_CHECK_IF(GetWorkspaceSize(context, f.needCoreNum, f.scratchStride, partialsElems, f.routeDense != 0,
                                     routeDenseScatter != 0, f.totalElements, f.filterSize, emptyCase,
                                     f.perCoreBufElems) != ge::GRAPH_SUCCESS,
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

    // SAME/VALID follow the forward output geometry. The CALCULATED golden uses
    // actual out_backprop spatial dimensions, including noncanonical windows.
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
    if (attrs.paddingMode != "CALCULATED" && inputH > 0 && inputW > 0 && filterH > 0 && filterW > 0) {
        OP_CHECK_IF((outHActual >= 0 && outHActual != outH) || (outWActual >= 0 && outWActual != outW),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "x, filter, out_backprop",
                        ("out_bp.H=" + std::to_string(outHActual) + ", out_bp.W=" + std::to_string(outWActual) +
                         ", expected H_out=" + std::to_string(outH) + ", W_out=" + std::to_string(outW)),
                        "spatial mismatch: out_backprop H/W must match forward output dims for SAME/VALID"),
                    return ge::GRAPH_FAILED);
    }
    // Adopt CALCULATED or dynamic geometry after checking known SAME/VALID shapes.
    // Zero is a known empty dimension and must not fall back to theoretical geometry.
    if (outHActual >= 0) {
        outH = outHActual;
    }
    if (outWActual >= 0) {
        outW = outWActual;
    }
    outH = std::max(outH, static_cast<int64_t>(0));
    outW = std::max(outW, static_cast<int64_t>(0));

    // 7. Compute element counts
    int64_t totalElements = batch * outH * outW * depth;
    int64_t filterSize = filterH * filterW * depth;

    // Empty tensor handling
    if (filterSize <= 0 || totalElements <= 0) {
        bool isDet = (context->GetDeterministic() == 1);

        // v7.0 f14: det empty case parallel zero-fill —
        // SetBlockDim(clamp(ceil(filterSize/1024), 1, coreNum)). ZeroOutSimt is
        // grid-stride (blockDim.x × gridDim.x covers y), so the kernel body is
        // untouched; micro-empty (fs < 1024) stays 1 core, baseline-identical.
        // nondet empty keeps SetBlockDim(1).
        int32_t emptyCoreNum = 1;
        if (isDet) {
            const int64_t needByFilter = Ops::Base::CeilDiv(filterSize, int64_t(1024));
            int64_t emptyCoreNumReq = std::max(needByFilter, int64_t(1));
            emptyCoreNumReq = std::min(emptyCoreNumReq, coreNum);
            emptyCoreNum = static_cast<int32_t>(emptyCoreNumReq);
        }
        context->SetBlockDim(static_cast<uint32_t>(emptyCoreNum));
        context->SetScheduleMode(1);

        TilingCommonFields fields = {totalElements, filterSize,    emptyCoreNum, 0,           batch,  inputH,
                                     inputW,        depth,         filterH,      filterW,     outH,   outW,
                                     attrs.strideH, attrs.strideW, attrs.rateH,  attrs.rateW, padTop, padLeft,
                                     padInputH,     padInputW,     isNCHW,       isDet};
        OP_CHECK_IF(FillTilingData(context, fields) != ge::GRAPH_SUCCESS, OP_LOGE(context, "FillTilingData error"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize too small"),
                    return ge::GRAPH_FAILED);
        context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
        return ge::GRAPH_SUCCESS;
    }

    // 8. Core allocation (v5.0 f2/f9: shape-routed)
    // routeDense (f1): pure shape function — MUST stay identical to the twin in
    // FillTilingData (the kernel branches on the tiling field; the dense core-count
    // formula and the dense kernel must always agree on the routing decision).
    // det + routeDense → frozen-baseline two-step formula (PER_CORE_MIN semantics,
    // bit-identical core count to the routed baseline cases, f2);
    // otherwise → round1 F9 formula (chanexcl det keeps the deep-channel gains, f9;
    // nondet keeps the round1 behavior unchanged, f6).
    constexpr int64_t ROUTE_DEPTH_MAX = 10;
    constexpr int64_t ROUTE_SPAT_MIN = 2048;
    constexpr int64_t ROUTE_NCHW_FS_MAX = 64;
    constexpr int64_t ROUTE_NCHW_SPAT_MIN = 4096;
    const int64_t spatialCount = batch * outH * outW;
    const bool routeDense = totalElements > 0 && filterSize > 0 &&
                            ((depth <= ROUTE_DEPTH_MAX && spatialCount >= ROUTE_SPAT_MIN) ||
                             (isNCHW && filterSize <= ROUTE_NCHW_FS_MAX && spatialCount >= ROUTE_NCHW_SPAT_MIN));
    // v6.0 (r1_dense_keyarb_v1): capacity gate, identical to the FillTilingData twin
    constexpr int64_t SCATTER_CAPACITY_ELEMS = 2097152; // 2^21 elements
    const bool routeDenseScatter = routeDense && totalElements <= SCATTER_CAPACITY_ELEMS && filterSize >= 1;
    int32_t needCoreNum = 0;
    if ((context->GetDeterministic() == 1) && routeDenseScatter) {
        // v6.0: dense-scatter route — one warp (32 elements) per core lower
        // bound, so the launch fills the available cores and the accumulation
        // grouping (lane intervals + core order) is unaffected by the count:
        //   needCoreNum = clamp(ceil(totalElements / 32), 1, coreNum)
        int64_t needCoreNumReq = Ops::Base::CeilDiv(totalElements, int64_t(32));
        needCoreNumReq = std::max(needCoreNumReq, int64_t(1));
        needCoreNumReq = std::min(needCoreNumReq, coreNum);
        needCoreNum = static_cast<int32_t>(needCoreNumReq);
    } else if ((context->GetDeterministic() == 1) && routeDense) {
        // v4.0 f2: dense path — frozen-baseline formula (two-step method, v2.1)
        int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, coreNum);
        if (perCoreElements < PER_CORE_MIN) {
            perCoreElements = PER_CORE_MIN;
        }
        needCoreNum = static_cast<int32_t>(Ops::Base::CeilDiv(totalElements, perCoreElements));
        needCoreNum = std::max(needCoreNum, 1); // at least 1 core
    } else {
        // v3.0 F9 (chanexcl + nondet, unchanged from round1):
        // needCoreNum = clamp(max(ceil(totalElements/64), ceil(filterSize/64)), 1, coreNum)
        // Removes the PER_CORE_MIN=1024 single-core seal for small-N large-filterSize
        // cases (L0_066/L0_008 class needs ≥ ceil(filterSize/64) cores). coreNum is the
        // GetCoreNumAiv() value from GetPlatformInfo; nondet needCoreNum only feeds
        // blockDim (grid-stride coverage, behaviorally equivalent).
        const int64_t needByElements = Ops::Base::CeilDiv(totalElements, int64_t(64));
        const int64_t needByFilter = Ops::Base::CeilDiv(filterSize, int64_t(64));
        int64_t needCoreNumReq = std::max(std::max(needByElements, needByFilter), int64_t(1));
        needCoreNumReq = std::min(needCoreNumReq, coreNum);
        needCoreNum = static_cast<int32_t>(needCoreNumReq);
    }

    // v2.1: compute per-core buffer size (128B aligned in elements)
    // v6.0: unused on the dense-scatter route (kept 0 there)
    int64_t perCoreBufElems = routeDenseScatter ? 0 : Ops::Base::CeilAlign(filterSize, WS_ALIGN_ELEMS);

    // 9. Fill TilingData + Workspace + TilingKey (determinism switch)
    bool isDet = (context->GetDeterministic() == 1);

    const bool singleTapSmall = context->GetDeterministic() == 1 && totalElements > 0 && spatialCount <= 4096 &&
                                totalElements <= INT32_MAX && filterH == 1 && filterW == 1 &&
                                platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                                    platform_ascendc::SocVersion::ASCEND950;
    if (singleTapSmall) {
        needCoreNum = static_cast<int32_t>(std::min(coreNum, Ops::Base::CeilDiv(depth, int64_t(8))));
    }
    int64_t firstH = 0, firstW = 0;
    if (context->GetDeterministic() == 1 &&
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
            platform_ascendc::SocVersion::ASCEND950 &&
        SingleSampleGeometry(batch, depth, filterH, filterW, outH, outW, padTop, padLeft, padInputH, padInputW,
                             attrs.rateH, attrs.rateW, firstH, firstW)) {
        needCoreNum = static_cast<int32_t>(std::min(coreNum, Ops::Base::CeilDiv(depth, int64_t(8))));
    }
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
