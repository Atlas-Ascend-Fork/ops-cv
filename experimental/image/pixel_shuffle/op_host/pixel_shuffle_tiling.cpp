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
 * \file pixel_shuffle_tiling.cpp
 * \brief PixelShuffle tiling：按 UB 预算推导行块高度 blockH 与 W 方向分段，并给出多核切分
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "experimental/image/pixel_shuffle/op_kernel/pixel_shuffle_tiling_data.h"
#include "experimental/image/pixel_shuffle/op_kernel/pixel_shuffle_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {

// scale_factor 上界：空间维数 S<=3，故 r^S <= r^3 <= 2^60 < INT64_MAX，
// 保证下方 chanPerGroup 累乘不会溢出；否则溢出为 0 时
// "C_in % chanPerGroup" 会整数除零（host 侧 SIGFPE）。
// 该上界同时保证下方 rSq * W * elementSize 不会溢出。
static constexpr int64_t MAX_SCALE_FACTOR = 1LL << 20;

struct PixelShuffleCompileInfo {};

// 查询 AIV 核数与 UB 容量：核数决定多核切分，UB 容量决定单块尺寸上限
static ge::graphStatus QueryPlatformCapability(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platInfo);
    auto platform = platform_ascendc::PlatformAscendC(platInfo);
    coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum == 0 || ubSize == 0) {
        OP_LOGE(context->GetNodeName(), "query platform capability failed, coreNum=%ld, ubSize=%lu", coreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndValidateAttrs(gert::TilingContext* context, int64_t& r)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGW(context->GetNodeName(), "No attributes found");
        return ge::GRAPH_FAILED;
    }

    const int64_t* scale_ptr = attrs->GetInt(0);
    if (scale_ptr == nullptr) {
        OP_LOGE(context->GetNodeName(), "r attribute is required but not set");
        return ge::GRAPH_FAILED;
    }

    r = *scale_ptr;
    if (r < 2 || r > MAX_SCALE_FACTOR) {
        OP_LOGE(context->GetNodeName(), "scale_factor must be within [2, %ld], but got %ld", MAX_SCALE_FACTOR, r);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static uint32_t GetElementSize(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT16) {
        return 2;
    }
    // 默认为 float32 (4字节)
    return 4;
}

static void ComputeTilingParameters(PixelShuffleTilingData* tilingData, uint32_t elementSize)
{
    const uint64_t kDefaultTileByteLimit = 64ull * 1024ull * 1024ull; // 64MB
    int64_t totalLen = tilingData->totalLength;
    int64_t suggestedTiles = 1;

    // G.EXP.22-CPP: 确保除法运算不会导致除零错误
    if (elementSize > 0 && totalLen > 0) {
        uint64_t elemsPerTile = kDefaultTileByteLimit / elementSize;
        if (elemsPerTile > 0) {
            suggestedTiles = static_cast<int64_t>((static_cast<uint64_t>(totalLen) + elemsPerTile - 1) / elemsPerTile);
        } else {
            // 如果单个元素就超过了 limit，则每个 tile 处理一个元素
            suggestedTiles = totalLen;
        }
    }

    tilingData->tileNum = std::max<int64_t>(1, suggestedTiles);
    tilingData->tileDataNum = static_cast<uint32_t>((totalLen + tilingData->tileNum - 1) / tilingData->tileNum);

    if (tilingData->tileDataNum == 0) {
        tilingData->tileDataNum = 1;
    }
}

static ge::graphStatus PixelShuffleTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do PixelShuffleTilingFunc");

    auto inputShape = context->GetInputShape(0);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    int64_t r = 0;
    if (GetAndValidateAttrs(context, r) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto xShape = inputShape->GetOriginShape();
    PixelShuffleTilingData* tilingData = context->GetTilingData<PixelShuffleTilingData>();

    // 支持 4D（N,C,H,W）与 5D（N,C,D,H,W），通道固定 index 1（与 PyTorch 一致）
    int64_t rank = xShape.GetDimNum();
    if (rank != 4 && rank != 5) {
        OP_LOGE(context->GetNodeName(), "Input tensor must be 4D or 5D, got %ld dims", rank);
        return ge::GRAPH_FAILED;
    }
    // 仅 -1 表示未知维；dim < -1 必须在此拒绝，否则两个负维相乘为正可穿透下面
    // totalLength > 0 与通道整除校验，kernel 会得到负块数/越界偏移。
    for (int64_t i = 0; i < rank; ++i) {
        if (xShape.GetDim(i) < -1) {
            OP_LOGE(context->GetNodeName(), "dim[%ld] = %ld is invalid, only -1 is allowed for an unknown dim", i,
                    xShape.GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    tilingData->rank = rank;
    tilingData->N = xShape.GetDim(0);
    tilingData->C_in = xShape.GetDim(1);
    tilingData->D = (rank == 5) ? xShape.GetDim(2) : 1;
    tilingData->H = (rank == 5) ? xShape.GetDim(3) : xShape.GetDim(2);
    tilingData->W = xShape.GetDim(rank - 1);
    tilingData->r = r;
    tilingData->totalLength = tilingData->N * tilingData->C_in * tilingData->D * tilingData->H * tilingData->W;

    // 空间维度数 S：4D->2, 5D->3；通道缩放因子 r^S
    int64_t s = (rank == 5) ? 3 : 2;
    int64_t chanPerGroup = 1;
    for (int64_t i = 0; i < s; ++i) {
        chanPerGroup *= r;
    }

    // 校验输入合法性，防止后续计算 totalLength 为 0 导致的逻辑异常
    if (tilingData->totalLength <= 0 || (tilingData->C_in % chanPerGroup != 0)) {
        OP_LOGE(context->GetNodeName(), "Invalid dims or C_in not divisible by r^%ld", s);
        return ge::GRAPH_FAILED;
    }

    uint32_t elementSize = 4; // 默认值
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc) {
        ge::DataType dt = inputDesc->GetDataType();
        const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT, ge::DT_FLOAT16};
        if (supportedDtypes.count(dt) == 0) {
            OP_LOGE(context->GetNodeName(), "Unsupported data type: %d", dt);
            return ge::GRAPH_FAILED;
        }
        elementSize = GetElementSize(dt);
    }

    if (elementSize == 0) {
        OP_LOGW(context->GetNodeName(), "elementSize is 0, fallback to 4");
        elementSize = 4;
    }

    ComputeTilingParameters(tilingData, elementSize);

    // 获取平台核数，按可并行块数分配多核（与 kernel 的块划分保持一致）
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (QueryPlatformCapability(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "query platform capability failed");
        return ge::GRAPH_FAILED;
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }

    // 与 kernel 完全一致的块推导（任一改动必须同步 op_kernel/pixel_shuffle.h 的 Init）
    // 单块 UB 占用 = kQueueBufCount * A * elementSize + A * sizeof(uint32_t)
    //   其中 A = r^2 * align(blockH * wPerBlock, unit)
    const int64_t kQueueBufCount = 4; // 2 队列 × 双缓冲
    // 必须与 kernel 的 UB_TOTAL_BUDGET_BYTES（op_kernel/pixel_shuffle.h）保持一致
    const int64_t kUbTotalBudget = 150 * 1024;
    const int64_t kBlockByteTarget = 2 * 1024;
    const int64_t unit = 32 / static_cast<int64_t>(elementSize);
    int64_t rSq = r * r;

    int64_t ubBudget = kUbTotalBudget;
    int64_t perElemCost = kQueueBufCount * static_cast<int64_t>(elementSize) + 4;
    int64_t alignedChanLimit = (ubBudget / perElemCost) / rSq;
    alignedChanLimit = (alignedChanLimit / unit) * unit;
    if (alignedChanLimit < unit) {
        alignedChanLimit = unit;
    }

    // (1) 行块高度 blockH：取 H 的约数，既满足 kBlockByteTarget 目标，又受 UB 预算约束
    int64_t blockHCap = kBlockByteTarget / (rSq * tilingData->W * static_cast<int64_t>(elementSize));
    if (blockHCap < 1) {
        blockHCap = 1;
    }
    if (blockHCap > alignedChanLimit) {
        blockHCap = alignedChanLimit;
    }
    int64_t blockH = 1;
    for (int64_t d = blockHCap; d >= 1; --d) {
        if (tilingData->H % d == 0) {
            blockH = d;
            break;
        }
    }
    if (blockH < 1) {
        blockH = 1;
    }

    // (2) W 方向分段：保证 blockH * wPerBlock <= alignedChanLimit，
    //     避免单块字节数随 W 线性膨胀导致 kernel 侧 InitBuffer 超出 UB
    int64_t wMaxPerBlock = alignedChanLimit / blockH;
    if (wMaxPerBlock < 1) {
        wMaxPerBlock = 1;
    }
    int64_t wPerBlock = (tilingData->W < wMaxPerBlock) ? tilingData->W : wMaxPerBlock;
    if (wPerBlock < 1) {
        wPerBlock = 1;
    }
    int64_t bPerRow = (tilingData->W + wPerBlock - 1) / wPerBlock;

    // 兜底自检 1：单块 UB 占用（含索引表）不得超过内部预算（用组内分块这一更大上界做保守校验）
    int64_t chanStride = ((blockH * wPerBlock + unit - 1) / unit) * unit;
    int64_t alignedBlockElems = rSq * chanStride;
    int64_t requiredUb = kQueueBufCount * alignedBlockElems * static_cast<int64_t>(elementSize) +
                         alignedBlockElems * static_cast<int64_t>(sizeof(uint32_t));
    OP_CHECK_IF(requiredUb > ubBudget,
                OP_LOGE(context->GetNodeName(),
                        "PixelShuffle single-block UB requirement %ld exceeds budget %ld "
                        "(r=%ld, H=%ld, W=%ld, blockH=%ld, wPerBlock=%ld)",
                        requiredUb, ubBudget, r, tilingData->H, tilingData->W, blockH, wPerBlock),
                return ge::GRAPH_FAILED);

    // 兜底自检 2：平台实际 UB 容量必须容得下该预算
    OP_CHECK_IF(static_cast<int64_t>(ubSize) < kUbTotalBudget,
                OP_LOGE(context->GetNodeName(), "PixelShuffle requires UB >= %ld bytes but platform reports only %lu",
                        kUbTotalBudget, ubSize),
                return ge::GRAPH_FAILED);

    int64_t bPerPlane = tilingData->H / blockH;
    int64_t planesPerGroup = (rank == 5) ? tilingData->D * r : 1;
    int64_t blocksPerGroup = planesPerGroup * bPerPlane * bPerRow;
    int64_t totalBlocks = tilingData->N * (tilingData->C_in / chanPerGroup) * blocksPerGroup;

    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(totalBlocks, coreNum));
    context->SetBlockDim(std::max<uint32_t>(1, blockDim));
    // 与 op_kernel/pixel_shuffle_tiling_key.h 的 TPL 声明保持一致（当前只有 MODE_0）
    context->SetTilingKey(PIXEL_SHUFFLE_TPL_SCH_MODE_0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPixelShuffle([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PixelShuffle)
    .Tiling(PixelShuffleTilingFunc)
    .TilingParse<PixelShuffleCompileInfo>(TilingParseForPixelShuffle);
} // namespace optiling
