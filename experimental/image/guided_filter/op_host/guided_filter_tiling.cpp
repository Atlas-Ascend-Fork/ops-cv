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
 * \file guided_filter_tiling.cpp
 * \brief
 */

#include <cmath>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "experimental/image/guided_filter/op_kernel/guided_filter_tiling_data.h"
#include "experimental/image/guided_filter/op_kernel/guided_filter_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

struct GuidedFilterCompileInfo {};

namespace {
constexpr size_t kIdxGuide = 0;
constexpr size_t kIdxSrc = 1;
constexpr size_t kOutDst = 0;

constexpr size_t kAttrRadius = 0;
constexpr size_t kAttrEps = 1;
constexpr size_t kAttrDDepth = 2;
constexpr size_t kAttrScale = 3;

// OpenCV 常量：CV_8U=0, CV_32F=5, CV_16F=7；-1 表示跟随 src
constexpr int64_t kDepthFollowSrc = -1;
constexpr int64_t kDepthU8 = 0;
constexpr int64_t kDepthF32 = 5;
constexpr int64_t kDepthF16 = 7;

// 行搬运走 DataCopyPad + DataCopyParams，blockLen 是 **字节** 且为 uint16，上限 65535。
// 单行字节数 = W * max(C, Cs) * elemSize，必须在 tiling 里挡住，否则会被静默截断。
constexpr int64_t kMaxRowBytes = 65535;
// UB 预算内的 src 通道数上限（超过必然放不下一列窗口）
constexpr int64_t kMaxSrcCh = 64;
constexpr int64_t kMaxPlaneElems = 1LL << 30;
// 单维尺寸上界：下游 GfGeo::DeriveGeo 以 int32_t 接收 width/height/batch，超过会静默截断；
// 同时 H*W 与 W*maxCh*maxEs 的 int64 乘法在极端大维下会溢出为负/0，从而绕过 plane / rowBytes 的
// 上界校验。因此先限幅、后相乘（同仓先例：先校验后窄化）。
constexpr int64_t kMaxDim = 0x7FFFFFFFLL; // INT32_MAX

uint32_t ElemeSizeOf(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_UINT8:
            return 1U;
        case ge::DT_FLOAT16:
            return 2U;
        case ge::DT_FLOAT:
            return 4U;
        default:
            return 0U;
    }
}
} // namespace

// tiling 分发入口
static ge::graphStatus GuidedFilterTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin GuidedFilterTilingFunc");

    // ---------------- 形状 ----------------
    const gert::StorageShape* guideShape = context->GetInputShape(kIdxGuide);
    OP_CHECK_NULL_WITH_CONTEXT(context, guideShape);
    const gert::StorageShape* srcShape = context->GetInputShape(kIdxSrc);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcShape);
    const gert::Shape& gs = guideShape->GetOriginShape();
    const gert::Shape& ss = srcShape->GetOriginShape();

    const size_t rank = gs.GetDimNum();
    OP_CHECK_IF(ss.GetDimNum() != rank, OP_LOGE(context, "guide rank %zu != src rank %zu", rank, ss.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rank != 2 && rank != 3 && rank != 4, OP_LOGE(context, "only rank 2/3/4 supported, got %zu", rank),
                return ge::GRAPH_FAILED);

    // rank2=[H,W]；rank3=[H,W,C]；rank4=[N,H,W,C]（通道后置）
    const size_t hIdx = (rank == 4) ? 1 : 0;
    const int64_t batch = (rank == 4) ? gs.GetDim(0) : 1;
    const int64_t height = gs.GetDim(hIdx);
    const int64_t width = gs.GetDim(hIdx + 1);
    const int64_t guideCh = (rank == 2) ? 1 : gs.GetDim(rank - 1);
    const int64_t srcCh = (rank == 2) ? 1 : ss.GetDim(rank - 1);

    OP_CHECK_IF(batch <= 0 || height <= 0 || width <= 0,
                OP_LOGE(context, "invalid shape: N=%ld H=%ld W=%ld", batch, height, width), return ge::GRAPH_FAILED);
    // 先限幅再相乘：保证后续 int64 乘法不溢出（(2^31-1)^2 仍远小于 INT64_MAX），
    // 也保证传给 DeriveGeo 的 int32 窄化不丢信息。
    OP_CHECK_IF(
        batch > kMaxDim || height > kMaxDim || width > kMaxDim,
        OP_LOGE(context, "shape out of range: N=%ld H=%ld W=%ld, must be <= %ld", batch, height, width, kMaxDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ss.GetDim(hIdx) != height || ss.GetDim(hIdx + 1) != width,
                OP_LOGE(context, "src spatial size (%ld,%ld) != guide (%ld,%ld)", ss.GetDim(hIdx), ss.GetDim(hIdx + 1),
                        height, width),
                return ge::GRAPH_FAILED);
    // rank4 的 batch 只取 guide 的 N（kernel 的 totalTasks 与 srcGm_ 尺寸都按它计算），
    // 而 dst 在 infershape 侧完全继承 src 的 shape。src.N < guide.N 会按 guide.N 越界读 src，
    // src.N > guide.N 则 dst 多出的 batch 永不写出，因此显式要求两边一致。
    OP_CHECK_IF(rank == 4 && ss.GetDim(0) != batch,
                OP_LOGE(context, "src batch N=%ld != guide batch N=%ld", ss.GetDim(0), batch), return ge::GRAPH_FAILED);
    OP_CHECK_IF(guideCh < 1 || guideCh > 3, OP_LOGE(context, "guide channels must be in [1,3], got %ld", guideCh),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(srcCh < 1 || srcCh > kMaxSrcCh,
                OP_LOGE(context, "src channels must be in [1,%ld], got %ld", kMaxSrcCh, srcCh),
                return ge::GRAPH_FAILED);
    const int64_t plane = height * width;
    OP_CHECK_IF(plane > kMaxPlaneElems, OP_LOGE(context, "H*W=%ld too large", plane), return ge::GRAPH_FAILED);

    // ---------------- 属性 ----------------
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* radiusPtr = attrs->GetInt(kAttrRadius);
    const float* epsPtr = attrs->GetFloat(kAttrEps);
    const int64_t* dDepthPtr = attrs->GetInt(kAttrDDepth);
    const float* scalePtr = attrs->GetFloat(kAttrScale);
    const int64_t radius = (radiusPtr == nullptr) ? 4 : *radiusPtr;
    const float eps = (epsPtr == nullptr) ? 0.01F : *epsPtr;
    const int64_t dDepth = (dDepthPtr == nullptr) ? kDepthFollowSrc : *dDepthPtr;
    const float scale = (scalePtr == nullptr) ? 1.0F : *scalePtr;

    OP_CHECK_IF(radius < 1 || radius > 512, OP_LOGE(context, "radius must be in [1,512], got %ld", radius),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(eps >= 0.0F) || !std::isfinite(eps), OP_LOGE(context, "eps must be >=0 and finite, got %f", eps),
                return ge::GRAPH_FAILED);
    // 首期只交付 scale == 1（快速引导滤波的下采样/上采样路径未实现），显式拒绝而非静默忽略
    OP_CHECK_IF(!(scale > 0.0F) || scale > 1.0F, OP_LOGE(context, "scale must be in (0,1], got %f", scale),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(std::fabs(scale - 1.0F) > 1e-6F, OP_LOGE(context, "scale != 1 is not supported yet, got %f", scale),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dDepth != kDepthFollowSrc && dDepth != kDepthU8 && dDepth != kDepthF32 && dDepth != kDepthF16,
                OP_LOGE(context, "dDepth must be -1/0(CV_8U)/5(CV_32F)/7(CV_16F), got %ld", dDepth),
                return ge::GRAPH_FAILED);

    // ---------------- dtype ----------------
    const auto guideDesc = context->GetInputDesc(kIdxGuide);
    const auto srcDesc = context->GetInputDesc(kIdxSrc);
    const auto dstDesc = context->GetOutputDesc(kOutDst);
    OP_CHECK_NULL_WITH_CONTEXT(context, guideDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDesc);
    const uint32_t eg = ElemeSizeOf(guideDesc->GetDataType());
    const uint32_t es = ElemeSizeOf(srcDesc->GetDataType());
    const uint32_t ed = ElemeSizeOf(dstDesc->GetDataType());
    OP_CHECK_IF(eg == 0U || es == 0U || ed == 0U,
                OP_LOGE(context, "unsupported dtype guide=%d src=%d dst=%d", static_cast<int>(guideDesc->GetDataType()),
                        static_cast<int>(srcDesc->GetDataType()), static_cast<int>(dstDesc->GetDataType())),
                return ge::GRAPH_FAILED);
    const int64_t maxCh = (guideCh > srcCh) ? guideCh : srcCh;
    const int64_t maxEs = static_cast<int64_t>(eg > es ? eg : es);
    const int64_t rowBytes = width * maxCh * maxEs;
    OP_CHECK_IF(rowBytes > kMaxRowBytes,
                OP_LOGE(context, "one row needs %ld bytes > %ld (W=%ld, C=%ld)", rowBytes, kMaxRowBytes, width, maxCh),
                return ge::GRAPH_FAILED);

    // ---------------- 平台 ----------------
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreNum = static_cast<int32_t>(platform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "no available AIV core"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(static_cast<int64_t>(ubSize) < GfGeo::kUbBudgetBytes,
                OP_LOGE(context, "platform UB %lu < required budget %d", ubSize, GfGeo::kUbBudgetBytes),
                return ge::GRAPH_FAILED);

    // ---------------- tile 几何（与 kernel 同源） ----------------
    const GfGeo::Geo geo = GfGeo::DeriveGeo(
        static_cast<int32_t>(width), static_cast<int32_t>(height), static_cast<int32_t>(batch),
        static_cast<int32_t>(radius), static_cast<int32_t>(guideCh), static_cast<int32_t>(srcCh),
        static_cast<int32_t>(eg), static_cast<int32_t>(es), static_cast<int32_t>(ed), coreNum);
    OP_CHECK_IF(!geo.ok,
                OP_LOGE(context, "cannot fit guided filter in UB: radius=%ld C=%ld Cs=%ld (budget %d bytes)", radius,
                        guideCh, srcCh, GfGeo::kUbBudgetBytes),
                return ge::GRAPH_FAILED);

    // ---------------- tiling data ----------------
    GuidedFilterTilingData* tiling = context->GetTilingData<GuidedFilterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK, OP_LOGE(context, "clear tiling failed"),
                return ge::GRAPH_FAILED);

    const uint32_t planeElems = static_cast<uint32_t>(GfGeo::Align8(static_cast<int32_t>(plane)));
    const uint64_t alphaElems = static_cast<uint64_t>(guideCh) * static_cast<uint64_t>(srcCh) *
                                static_cast<uint64_t>(batch) * planeElems;
    const uint64_t betaElems = static_cast<uint64_t>(srcCh) * static_cast<uint64_t>(batch) * planeElems;
    const uint64_t libSize = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());

    tiling->batch = static_cast<uint32_t>(batch);
    tiling->height = static_cast<uint32_t>(height);
    tiling->width = static_cast<uint32_t>(width);
    tiling->guideCh = static_cast<uint32_t>(guideCh);
    tiling->srcCh = static_cast<uint32_t>(srcCh);
    tiling->radius = static_cast<uint32_t>(radius);
    tiling->tileW = static_cast<uint32_t>(geo.tileW);
    tiling->haloW = static_cast<uint32_t>(geo.haloW);
    tiling->nTx = static_cast<uint32_t>(geo.nTx);
    tiling->nTy = static_cast<uint32_t>(geo.nTy);
    tiling->bandH = static_cast<uint32_t>(geo.bandH);
    tiling->atomNum = static_cast<uint32_t>(GfGeo::AtomNum(static_cast<int32_t>(guideCh), static_cast<int32_t>(srcCh)));
    tiling->coefNum = static_cast<uint32_t>(GfGeo::CoefNum(static_cast<int32_t>(guideCh), static_cast<int32_t>(srcCh)));
    tiling->planeElems = planeElems;
    tiling->outDtype = (ed == 4U) ? 0U : ((ed == 2U) ? 1U : 2U);
    tiling->eps = eps;
    const float win = static_cast<float>(2 * radius + 1);
    tiling->invWin = 1.0F / (win * win);
    // 偏移相对 GetUserWorkspace(workspace)，单位：字节
    tiling->alphaOff = 0ULL;
    tiling->betaOff = alphaElems * sizeof(float);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(libSize + (alphaElems + betaElems) * sizeof(float));

    context->SetTilingKey(GUIDED_FILTER_TPL_SCH_MODE_0);
    context->SetBlockDim(geo.blockDim);
    // kernel 用无参硬件 SyncAll 做阶段 A/B 的跨核同步，需 batch 调度保证所有块同时驻留
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(),
            "GuidedFilter tiling: N=%u H=%u W=%u C=%u Cs=%u r=%u tile=%u halo=%u nTx=%u nTy=%u band=%u "
            "blockDim=%u ub=%ld",
            tiling->batch, tiling->height, tiling->width, tiling->guideCh, tiling->srcCh, tiling->radius, tiling->tileW,
            tiling->haloW, tiling->nTx, tiling->nTy, tiling->bandH, geo.blockDim, static_cast<long>(geo.ubBytes));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGuidedFilter([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(GuidedFilter)
    .Tiling(GuidedFilterTilingFunc)
    .TilingParse<GuidedFilterCompileInfo>(TilingParseForGuidedFilter);
} // namespace optiling
