/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "bilateral_filter3_d_tiling.h"

namespace ops {

static constexpr size_t KERNEL_SIZE_ATTR_INDEX = 0;
static constexpr size_t SIGMA_SPACE_ATTR_INDEX = 1;
static constexpr size_t SIGMA_COLOR_ATTR_INDEX = 2;

static constexpr int64_t MAX_D = 1000;
static constexpr int64_t MAX_HW = 4096;
static constexpr int64_t MAX_LEADING_BATCH = 200 * 64; // N * C upper bound per spec
// Keep 0.5/sigma^2 finite in fp32; exp(-x) underflows to 0 for x > ~88, so any
// value above this still behaves as "weight exactly zero" while avoiding 0*inf
// NaNs for the zero-distance offsets.
static constexpr float MAX_INV2_SIGMA = 1e30F;

static float InvTwoSigmaSq(double sigma)
{
    double inv2 = 0.5 / (sigma * sigma);
    if (inv2 > static_cast<double>(MAX_INV2_SIGMA)) {
        return MAX_INV2_SIGMA;
    }
    return static_cast<float>(inv2);
}

static uint64_t TilingKeyFor(ge::DataType dt)
{
    return dt == ge::DT_FLOAT ? optiling::BF3D_TILING_KEY_FP32 : optiling::BF3D_TILING_KEY_FP16;
}

static ge::graphStatus BilateralFilter3DTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0U, OP_LOGE(context, "invalid Ascend910B AIV platform information."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto shape = inputShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 3U || rank > 5U, OP_LOGE(context, "BilateralFilter3D only supports rank 3/4/5."),
                return ge::GRAPH_FAILED);
    const int64_t dDim = shape.GetDim(rank - 3U);
    const int64_t hDim = shape.GetDim(rank - 2U);
    const int64_t wDim = shape.GetDim(rank - 1U);
    OP_CHECK_IF(dDim <= 0 || hDim <= 0 || wDim <= 0,
                OP_LOGE(context, "BilateralFilter3D does not support empty tensors."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dDim > MAX_D, OP_LOGE(context, "D=%ld exceeds the limit %ld.", dDim, MAX_D), return ge::GRAPH_FAILED);
    OP_CHECK_IF(hDim > MAX_HW || wDim > MAX_HW,
                OP_LOGE(context, "H/W=%ld/%ld exceeds the limit %ld.", hDim, wDim, MAX_HW), return ge::GRAPH_FAILED);
    uint64_t nVolume = 1U;
    for (size_t i = 0; i + 3U < rank; ++i) {
        nVolume *= static_cast<uint64_t>(shape.GetDim(i));
    }
    OP_CHECK_IF(nVolume > static_cast<uint64_t>(MAX_LEADING_BATCH),
                OP_LOGE(context, "product of leading dims %lu exceeds the limit %ld.", nVolume, MAX_LEADING_BATCH),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kernelSize = attrs->GetInt(KERNEL_SIZE_ATTR_INDEX);
    const float* sigmaSpace = attrs->GetFloat(SIGMA_SPACE_ATTR_INDEX);
    const float* sigmaColor = attrs->GetFloat(SIGMA_COLOR_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kernelSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaSpace);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaColor);
    OP_CHECK_IF(!(*kernelSize == 3 || *kernelSize == 5 || *kernelSize == 7 || *kernelSize == 9),
                OP_LOGE(context, "kernel_size must be one of 3/5/7/9, got %ld.", *kernelSize), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(std::isfinite(static_cast<double>(*sigmaSpace)) && *sigmaSpace > 0.0F),
                OP_LOGE(context, "sigma_space must be positive and finite."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(std::isfinite(static_cast<double>(*sigmaColor)) && *sigmaColor > 0.0F),
                OP_LOGE(context, "sigma_color must be positive and finite."), return ge::GRAPH_FAILED);

    auto desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const ge::DataType inDtype = desc->GetDataType();
    const int64_t dtypeBytes = inDtype == ge::DT_FLOAT ? 4 : 2;

    // 行内分块宽度推导。设备侧每个 chunk 元素占用:
    //   13 份 fp32 工作缓冲 (中心/邻域/两个累加器/临时/除法结果/ramp/掩码构造 A,B/
    //                        负偏移修复的 head,fix/常量全 1/全 0)
    // + (2r+1) 份 fp32 掩码缓冲(每个 dx 一份独立 TBuf —— 掩码必须写入完整张量,
    //   不能写偏移视图)
    // + 4 份输入 dtype (输入队列 2 槽 + 输出队列 2 槽)
    // 另加与 chunk 无关的空间权重表 k^3 个 fp32。
    const int64_t radius = *kernelSize / 2;
    const int64_t maskRows = 2 * radius + 1;
    const int64_t perChunkElem = (13 + maskRows) * 4 + 4 * dtypeBytes;
    const int64_t swBytes = *kernelSize * *kernelSize * *kernelSize * 4;
    constexpr int64_t UB_SLACK = 2048; // 预留给队列头信息等框架开销
    const int64_t budget = static_cast<int64_t>(ubSize) - swBytes - UB_SLACK;
    OP_CHECK_IF(budget <= 0,
                OP_LOGE(context, "UB too small for the spatial weight table (%ld bytes needed, %lu available).",
                        swBytes, ubSize),
                return ge::GRAPH_FAILED);
    const int64_t wPad = (wDim + 7) / 8 * 8;
    int64_t chunkLen = budget / perChunkElem / 8 * 8; // 向下对齐到 8 个元素(32 字节)
    if (chunkLen > optiling::BF3D_CHUNK_MAX) {
        chunkLen = optiling::BF3D_CHUNK_MAX;
    }
    if (chunkLen > wPad) {
        chunkLen = wPad;
    }
    OP_CHECK_IF(chunkLen < 8,
                OP_LOGE(context, "UB budget too tight for kernel_size=%ld (need >= %ld bytes per 8 elements).",
                        *kernelSize, perChunkElem * 8),
                return ge::GRAPH_FAILED);

    const uint64_t totalRows = nVolume * static_cast<uint64_t>(dDim) * static_cast<uint64_t>(hDim);
    const uint32_t usedCore = totalRows < static_cast<uint64_t>(coreNum) ? static_cast<uint32_t>(totalRows) : coreNum;

    // ops-cv 框架写法: 栈上构造 TilingData 后 SaveToBuffer(GetRawTilingData()),
    // 不用 GetTilingData<T>()(nnpbase 执行器下返回 nullptr)。
    optiling::BilateralFilter3DTilingData tiling;
    tiling.set_totalRows(totalRows);
    tiling.set_d(static_cast<uint32_t>(dDim));
    tiling.set_h(static_cast<uint32_t>(hDim));
    tiling.set_w(static_cast<uint32_t>(wDim));
    tiling.set_kernelSize(static_cast<uint32_t>(*kernelSize));
    tiling.set_usedCoreNum(usedCore);
    tiling.set_chunkLen(static_cast<uint32_t>(chunkLen));
    tiling.set_inv2ss(InvTwoSigmaSq(static_cast<double>(*sigmaSpace)));
    tiling.set_inv2sc(InvTwoSigmaSq(static_cast<double>(*sigmaColor)));

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());

    context->SetBlockDim(usedCore);
    context->SetTilingKey(TilingKeyFor(inDtype));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseBilateralFilter3D([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BilateralFilter3D)
    .Tiling(BilateralFilter3DTilingFunc)
    .TilingParse<optiling::BilateralFilter3DCompileInfo>(TilingParseBilateralFilter3D);

} // namespace ops
