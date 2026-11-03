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
 * \file lbp_feature_tiling.cpp
 * \brief LbpFeature Tiling: validates shape/attrs, then plans core split, tiles and UB budget.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "register/tilingdata_base.h"
#include "lbp_feature_tiling_data.h"
#include "lbp_feature_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

namespace optiling {

struct LbpFeatureCompileInfo {};

// tiling 分发入口
static ge::graphStatus LbpFeatureTilingFunc(gert::TilingContext* context)
{
    LbpFeatureTilingData* tiling = context->GetTilingData<LbpFeatureTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (memset_s(tiling, sizeof(LbpFeatureTilingData), 0, sizeof(LbpFeatureTilingData)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    auto attrs = context->GetAttrs();
    const auto* inputTensor = context->GetInputTensor(0);
    if (attrs == nullptr || inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* radiusPtr = attrs->GetInt(0);
    const int64_t* neighborsPtr = attrs->GetInt(1);
    if (radiusPtr == nullptr || neighborsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Validate attribute ranges in int64 before narrowing: values such as
    // 2^32 + 1 must not truncate into a legal int32 value.
    const int64_t radius64 = *radiusPtr;
    const int64_t neighbors64 = *neighborsPtr;
    if (radius64 < 0 || radius64 > std::numeric_limits<int32_t>::max() || neighbors64 < 1 || neighbors64 > 31) {
        return ge::GRAPH_FAILED;
    }
    const int32_t radius = static_cast<int32_t>(radius64);
    const int32_t neighbors = static_cast<int32_t>(neighbors64);

    const auto shape_x = inputTensor->GetOriginShape();
    const int32_t x_dimensional = shape_x.GetDimNum();
    if (x_dimensional < 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t H64 = 1;
    int64_t W64 = 1;
    int64_t batchCount64 = 1;
    if (x_dimensional > 2) {
        for (int32_t i = 0; i < x_dimensional - 2; ++i) {
            const int64_t dim = shape_x.GetDim(i);
            if (dim <= 0 || batchCount64 > std::numeric_limits<int32_t>::max() / dim) {
                return ge::GRAPH_FAILED;
            }
            batchCount64 *= dim;
        }
        H64 = shape_x.GetDim(x_dimensional - 2);
        W64 = shape_x.GetDim(x_dimensional - 1);
    } else {
        H64 = shape_x.GetDim(0);
        W64 = shape_x.GetDim(1);
    }
    if (H64 <= 0 || W64 <= 0 || H64 > std::numeric_limits<int32_t>::max() ||
        W64 > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    // The tensor size and every per-batch product must fit into int32, both
    // for the tiling data fields and for the kernel-side arithmetic.
    const int64_t totalLength64 = static_cast<int64_t>(inputTensor->GetShapeSize());
    if (totalLength64 <= 0 || totalLength64 > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    if (H64 > std::numeric_limits<int32_t>::max() / W64) {
        return ge::GRAPH_FAILED;
    }
    const int64_t plane64 = H64 * W64;
    if (batchCount64 > totalLength64 / plane64 || batchCount64 * plane64 != totalLength64) {
        return ge::GRAPH_FAILED;
    }
    const int32_t totalLength = static_cast<int32_t>(totalLength64);
    const int32_t batchCount = static_cast<int32_t>(batchCount64);
    const int32_t H = static_cast<int32_t>(H64);
    const int32_t W = static_cast<int32_t>(W64);
    if (static_cast<int64_t>(H) <= 2 * radius64 || static_cast<int64_t>(W) <= 2 * radius64) {
        return ge::GRAPH_FAILED;
    }

    const int32_t outH = H - 2 * radius;
    const int32_t outW = W - 2 * radius;
    tiling->outH = outH;
    tiling->outW = outW;
    tiling->H = H;
    tiling->W = W;
    tiling->batchCount = batchCount;

    auto inputType = inputTensor->GetDataType();
    const uint32_t inputTypeBytes = (inputType == ge::DT_UINT8) ? 1U : 4U;
    const uint32_t bytesPerElement = (inputType == ge::DT_UINT8) ? 37U : 45U;
    constexpr uint32_t UB_RESERVED_BYTES = 12U * 1024U;
    constexpr uint32_t VECTOR_COMPUTE_ALIGN_ELEMENTS = 64U;
    constexpr uint32_t OUTPUT_ALIGN_ELEMENTS = 32U / sizeof(int32_t);
    const uint32_t inputAlignElements = 32U / inputTypeBytes;
    const uint32_t rowAlignElements = std::max(inputAlignElements, OUTPUT_ALIGN_ELEMENTS);
    const uint64_t usableUb = (ub_size > UB_RESERVED_BYTES) ? (ub_size - UB_RESERVED_BYTES) : ub_size;

    uint32_t maxTileElements = static_cast<uint32_t>(usableUb / bytesPerElement);
    maxTileElements = (maxTileElements / VECTOR_COMPUTE_ALIGN_ELEMENTS) * VECTOR_COMPUTE_ALIGN_ELEMENTS;
    if (maxTileElements < VECTOR_COMPUTE_ALIGN_ELEMENTS) {
        maxTileElements = VECTOR_COMPUTE_ALIGN_ELEMENTS;
    }

    constexpr uint32_t MAX_COPY_BYTES = 2097151U;
    uint32_t maxCopyElements = MAX_COPY_BYTES / inputTypeBytes;
    maxCopyElements = (maxCopyElements / VECTOR_COMPUTE_ALIGN_ELEMENTS) * VECTOR_COMPUTE_ALIGN_ELEMENTS;
    if (maxCopyElements < VECTOR_COMPUTE_ALIGN_ELEMENTS) {
        maxCopyElements = VECTOR_COMPUTE_ALIGN_ELEMENTS;
    }

    uint32_t tileRows = 1U;
    uint32_t tileCols = 1U;
    uint32_t rowStride = rowAlignElements;
    const uint32_t fullRowStride = ((static_cast<uint32_t>(outW) + rowAlignElements - 1U) / rowAlignElements) *
                                   rowAlignElements;
    const uint32_t totalRows = static_cast<uint32_t>(batchCount) * static_cast<uint32_t>(outH);
    tiling->totalRows = totalRows;

    // ============================================================
    // 4.5 v4 slab 几何：每个 neighbor 的整数角点列/行偏移
    // ============================================================
    int32_t fxArr[31];
    int32_t fyArr[31];
    int32_t cxArr[31];
    int32_t cyArr[31];
    for (int32_t n = 0; n < neighbors; ++n) {
        const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(n) / static_cast<float>(neighbors);
        const float x = static_cast<float>(static_cast<double>(radius) * std::cos(angle));
        const float y = static_cast<float>(-static_cast<double>(radius) * std::sin(angle));
        fxArr[n] = static_cast<int32_t>(std::floor(x));
        fyArr[n] = static_cast<int32_t>(std::floor(y));
        cxArr[n] = static_cast<int32_t>(std::ceil(x));
        cyArr[n] = static_cast<int32_t>(std::ceil(y));
    }
    int32_t kArr[31];
    for (int32_t k = 0; k < 31; ++k) {
        kArr[k] = 0;
    }
    uint32_t kCount = 0U;
    for (int32_t n = 0; n < neighbors; ++n) {
        const int32_t vals[2] = {fxArr[n], cxArr[n]};
        for (int32_t vi = 0; vi < 2; ++vi) {
            bool found = false;
            for (uint32_t k = 0U; k < kCount; ++k) {
                if (kArr[k] == vals[vi]) {
                    found = true;
                    break;
                }
            }
            if (!found && kCount < 31U) {
                kArr[kCount] = vals[vi];
                kCount += 1U;
            }
        }
    }
    int32_t fyMin = fyArr[0];
    int32_t cyMax = cyArr[0];
    for (int32_t n = 1; n < neighbors; ++n) {
        if (fyArr[n] < fyMin) {
            fyMin = fyArr[n];
        }
        if (cyArr[n] > cyMax) {
            cyMax = cyArr[n];
        }
    }
    const int32_t slabRowSpan = cyMax - fyMin;

    const uint32_t outputWidth = static_cast<uint32_t>(outW);
    const uint32_t rowGroup = (outputWidth % 8U == 0U) ?
                                  1U :
                                  ((outputWidth % 4U == 0U) ? 2U : ((outputWidth % 2U == 0U) ? 4U : 8U));
    const uint32_t rowTaskCount = (totalRows + rowGroup - 1U) / rowGroup;
    const uint32_t rowParallelCores = std::max(1U, std::min(coreNum, rowTaskCount));
    const uint32_t smallCoreRowGroupNum = rowTaskCount / rowParallelCores;
    const uint32_t bigRowCoreCount = rowTaskCount % rowParallelCores;
    const uint32_t bigCoreRowGroupNum = smallCoreRowGroupNum + ((bigRowCoreCount == 0U) ? 0U : 1U);
    const uint32_t smallCoreRows = smallCoreRowGroupNum * rowGroup;
    const uint32_t bigCoreRows = bigCoreRowGroupNum * rowGroup;
    const uint32_t maxRowsPerCore = bigCoreRows;
    const bool fullRowFits = fullRowStride <= maxTileElements && static_cast<uint32_t>(outW) <= maxCopyElements;
    const bool needColumnParallelism = fullRowFits && totalRows < coreNum &&
                                       static_cast<uint32_t>(outW) >= 2U * VECTOR_COMPUTE_ALIGN_ELEMENTS &&
                                       static_cast<uint64_t>(coreNum - totalRows) * 4U >= coreNum;
    bool useRowSplit = fullRowFits && !needColumnParallelism;

    if (useRowSplit) {
        tileCols = static_cast<uint32_t>(outW);
        rowStride = fullRowStride;
        tileRows = std::min(maxTileElements / rowStride, static_cast<uint32_t>(outH));
        tileRows = std::min(tileRows, 4095U);
        tileRows = std::min(tileRows, maxRowsPerCore);
    } else if (!fullRowFits) {
        tileRows = 1U;
        uint32_t availableCols = std::min(maxTileElements, maxCopyElements);
        availableCols = (availableCols / VECTOR_COMPUTE_ALIGN_ELEMENTS) * VECTOR_COMPUTE_ALIGN_ELEMENTS;
        availableCols = std::max(availableCols, VECTOR_COMPUTE_ALIGN_ELEMENTS);
        const uint32_t desiredColTiles = std::min(
            (coreNum + totalRows - 1U) / totalRows,
            (static_cast<uint32_t>(outW) + VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) / VECTOR_COMPUTE_ALIGN_ELEMENTS);
        const uint32_t targetCols = (static_cast<uint32_t>(outW) + desiredColTiles - 1U) / desiredColTiles;
        const uint32_t alignedTargetCols = ((targetCols + VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) /
                                            VECTOR_COMPUTE_ALIGN_ELEMENTS) *
                                           VECTOR_COMPUTE_ALIGN_ELEMENTS;
        tileCols = std::min(availableCols, alignedTargetCols);
        tileCols = std::min(tileCols, static_cast<uint32_t>(outW));
        rowStride = ((tileCols + rowAlignElements - 1U) / rowAlignElements) * rowAlignElements;
    } else {
        const uint32_t desiredColTiles = std::min(
            (coreNum + totalRows - 1U) / totalRows,
            (static_cast<uint32_t>(outW) + VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) / VECTOR_COMPUTE_ALIGN_ELEMENTS);
        const uint32_t targetCols = (static_cast<uint32_t>(outW) + desiredColTiles - 1U) / desiredColTiles;
        tileRows = 1U;
        tileCols = ((targetCols + VECTOR_COMPUTE_ALIGN_ELEMENTS - 1U) / VECTOR_COMPUTE_ALIGN_ELEMENTS) *
                   VECTOR_COMPUTE_ALIGN_ELEMENTS;
        tileCols = std::min(tileCols, static_cast<uint32_t>(outW));
        rowStride = ((tileCols + rowAlignElements - 1U) / rowAlignElements) * rowAlignElements;
    }

    if (tileRows == 0U) {
        tileRows = 1U;
    }
    if (tileCols == 0U) {
        tileCols = 1U;
    }
    if (rowStride == 0U) {
        rowStride = rowAlignElements;
    }

    // ============================================================
    // 5.5 v4 模式选择与行数上限
    // v4：每 tile 只搬 |K| 块整板（列偏移 k 各一块，覆盖全部行窗口），
    // 每个 neighbor 零搬运。UB 模型：整板队列（2 x |K| x (R+span) 存储行）
    // 替换 v2 的两队邻居缓冲。UB 装不下同尺寸整板但能容纳 ~70% 行数时，
    // 也启用并降到 maxTileRowsV4（碎片化每-tile 开销被流量收益覆盖）。
    // ============================================================
    const uint32_t planeBpe = 4U * inputTypeBytes;
    const uint32_t nonPlaneBpe = (bytesPerElement > planeBpe) ? (bytesPerElement - planeBpe) : bytesPerElement;
    const uint32_t slabSetBpe = 2U * kCount * inputTypeBytes;
    uint32_t maxTileRowsV4 = 0U;
    {
        const uint64_t perRowBytes = static_cast<uint64_t>(rowStride) * (nonPlaneBpe + slabSetBpe);
        const uint64_t spanBytes = static_cast<uint64_t>(slabSetBpe) * static_cast<uint64_t>(rowStride) *
                                   static_cast<uint64_t>(slabRowSpan);
        if (usableUb > spanBytes && perRowBytes > 0ULL) {
            maxTileRowsV4 = static_cast<uint32_t>((usableUb - spanBytes) / perRowBytes);
        }
    }
    // v4.4：流量必须赢——kCount×(R+span) ≤ 0.9×2n×(R+1)，否则退回 v2 路径
    const uint32_t candidateRows = (tileRows < maxTileRowsV4) ? tileRows : maxTileRowsV4;
    const uint64_t v4Rows = static_cast<uint64_t>(kCount) * (candidateRows + static_cast<uint64_t>(slabRowSpan));
    const uint64_t v2Rows = 2ULL * static_cast<uint64_t>(neighbors) * (candidateRows + 1ULL);
    uint32_t groupSlabMode = 0U;
    if (kCount >= 1U && kCount < 2U * static_cast<uint32_t>(neighbors) && tileRows >= 2U &&
        maxTileRowsV4 >= std::max(2U, (tileRows * 7U) / 10U) && v4Rows * 10ULL <= v2Rows * 9ULL) {
        groupSlabMode = 1U;
        if (tileRows > maxTileRowsV4) {
            tileRows = maxTileRowsV4;
        }
    }

    // ============================================================
    // 5.6 UB 真实账单校验（防 VEC UB 越界，507035）
    //
    // “45B/元素”平均模型没有计入 neighbor slab 的 halo 开销：
    // 每个邻居缓冲多 rowStride+128 个元素，v2 的 4 个缓冲合计约
    // 16×rowStride 字节；宽行 fp32（如 1832~2016 元素行宽）会低估
    // 实际 InitBuffer 总量，导致 UB 越界异常。
    // 这里按与 kernel InitBuffer 完全一致的公式核算总字节数，
    // 超出平台 UB 上限时收缩 tileRows；单行仍超则（必要时）从
    // 整行切分退化为列切分并收缩 tileCols。
    // ============================================================
    {
        auto alignUp64 = [](uint64_t value) -> uint64_t { return ((value + 63ULL) / 64ULL) * 64ULL; };
        auto trueUbBytes = [&](uint32_t rows, uint32_t cols) -> uint64_t {
            const uint64_t stride = (static_cast<uint64_t>(cols) + rowAlignElements - 1ULL) / rowAlignElements *
                                    rowAlignElements;
            const uint64_t raw = static_cast<uint64_t>(rows) * stride;
            const uint64_t cap = alignUp64(raw);
            uint64_t total = 2ULL * cap * inputTypeBytes +  // center 队列（2 缓冲）
                             2ULL * cap * sizeof(int32_t) + // 输出队列（int32，2 缓冲）
                             cap * 4ULL +                   // interp
                             2ULL * cap * 2ULL +            // condition + oneHalf
                             cap * 1ULL +                   // compareBits
                             cap * 4ULL;                    // bitValue
            if (inputTypeBytes == 1U) {
                total += 2ULL * cap * 4ULL + cap * 2ULL; // u8 专用：centerFloat + work + castHalf
            }
            if (groupSlabMode == 1U) {
                const uint64_t slabSet = alignUp64(static_cast<uint64_t>(kCount) *
                                                   (static_cast<uint64_t>(rows) + static_cast<uint64_t>(slabRowSpan)) *
                                                   stride) +
                                         64ULL;
                total += 2ULL * slabSet * inputTypeBytes;
            } else {
                const uint64_t slabElems = ((raw + stride + 2ULL * 64ULL - 1ULL) / 64ULL) * 64ULL;
                total += 4ULL * slabElems * inputTypeBytes;
            }
            return total;
        };
        const uint64_t ubBudget = ub_size;
        if (useRowSplit) {
            while (tileRows > 1U && trueUbBytes(tileRows, tileCols) > ubBudget) {
                tileRows -= 1U;
            }
            if (trueUbBytes(tileRows, tileCols) > ubBudget) {
                // 整行单行仍超预算（超宽 fp32 行）→ 退化为列切分
                useRowSplit = false;
                tileRows = 1U;
                uint32_t availableCols = std::min(maxTileElements, maxCopyElements);
                availableCols = (availableCols / VECTOR_COMPUTE_ALIGN_ELEMENTS) * VECTOR_COMPUTE_ALIGN_ELEMENTS;
                availableCols = std::max(availableCols, VECTOR_COMPUTE_ALIGN_ELEMENTS);
                tileCols = std::min(availableCols, static_cast<uint32_t>(outW));
                rowStride = ((tileCols + rowAlignElements - 1U) / rowAlignElements) * rowAlignElements;
            }
        }
        while (tileCols > VECTOR_COMPUTE_ALIGN_ELEMENTS && trueUbBytes(tileRows, tileCols) > ubBudget) {
            tileCols -= VECTOR_COMPUTE_ALIGN_ELEMENTS;
            rowStride = ((tileCols + rowAlignElements - 1U) / rowAlignElements) * rowAlignElements;
        }
    }

    tiling->tileRows = tileRows;
    tiling->tileCols = tileCols;
    tiling->rowStride = rowStride;

    const int64_t outputLength64 = static_cast<int64_t>(batchCount) * outH * outW;
    if (outputLength64 <= 0 || outputLength64 > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const int32_t output_length = static_cast<int32_t>(outputLength64);

    const double CV_PI = 3.14159265358979323846;
    // 【修复点】全部零初始化：只填前 neighbors 个，剩余元素不再携带栈上脏数据
    // 序列化进 tilingData 的内容从此完全确定，UT 基线才可固化
    int32_t offset1_ndarray[31] = {};
    int32_t offset2_ndarray[31] = {};
    int32_t offset3_ndarray[31] = {};
    int32_t offset4_ndarray[31] = {};
    float w1_ndarray[31] = {};
    float w2_ndarray[31] = {};
    float w3_ndarray[31] = {};
    float w4_ndarray[31] = {};

    for (int32_t n = 0; n < neighbors; ++n) {
        const double angle = 2.0 * CV_PI * static_cast<double>(n) / static_cast<float>(neighbors);
        const float x = static_cast<float>(static_cast<double>(radius) * std::cos(angle));
        const float y = static_cast<float>(-static_cast<double>(radius) * std::sin(angle));
        const int32_t fx = static_cast<int32_t>(std::floor(x));
        const int32_t fy = static_cast<int32_t>(std::floor(y));
        const int32_t cx = static_cast<int32_t>(std::ceil(x));
        const int32_t cy = static_cast<int32_t>(std::ceil(y));
        offset1_ndarray[n] = fy * W + fx;
        offset2_ndarray[n] = fy * W + cx;
        offset3_ndarray[n] = cy * W + fx;
        offset4_ndarray[n] = cy * W + cx;
        const float ty = y - static_cast<float>(fy);
        const float tx = x - static_cast<float>(fx);
        w1_ndarray[n] = (1.0f - tx) * (1.0f - ty);
        w2_ndarray[n] = tx * (1.0f - ty);
        w3_ndarray[n] = (1.0f - tx) * ty;
        w4_ndarray[n] = tx * ty;
    }

    tiling->radius = radius;
    tiling->neighbors = neighbors;
    tiling->totalLength = totalLength;
    tiling->outputLength = output_length;

    // 恢复使用宏自动生成的 set_ 方法，不再需要 memcpy
    memcpy_s(tiling->offset1_ndarray, sizeof(tiling->offset1_ndarray), offset1_ndarray, sizeof(offset1_ndarray));
    memcpy_s(tiling->offset2_ndarray, sizeof(tiling->offset2_ndarray), offset2_ndarray, sizeof(offset2_ndarray));
    memcpy_s(tiling->offset3_ndarray, sizeof(tiling->offset3_ndarray), offset3_ndarray, sizeof(offset3_ndarray));
    memcpy_s(tiling->offset4_ndarray, sizeof(tiling->offset4_ndarray), offset4_ndarray, sizeof(offset4_ndarray));
    memcpy_s(tiling->w1_ndarray, sizeof(tiling->w1_ndarray), w1_ndarray, sizeof(w1_ndarray));
    memcpy_s(tiling->w2_ndarray, sizeof(tiling->w2_ndarray), w2_ndarray, sizeof(w2_ndarray));
    memcpy_s(tiling->w3_ndarray, sizeof(tiling->w3_ndarray), w3_ndarray, sizeof(w3_ndarray));
    memcpy_s(tiling->w4_ndarray, sizeof(tiling->w4_ndarray), w4_ndarray, sizeof(w4_ndarray));

    // ============================================================
    // 8.5 v4 slab 视图参数
    // ============================================================
    int32_t fxIdxArr[31];
    int32_t cxIdxArr[31];
    int32_t fyOffArr[31];
    int32_t cyOffArr[31];
    for (int32_t n = 0; n < 31; ++n) {
        fxIdxArr[n] = 0;
        cxIdxArr[n] = 0;
        fyOffArr[n] = 0;
        cyOffArr[n] = 0;
    }
    for (int32_t n = 0; n < neighbors; ++n) {
        for (uint32_t k = 0U; k < kCount; ++k) {
            if (kArr[k] == fxArr[n]) {
                fxIdxArr[n] = static_cast<int32_t>(k);
            }
            if (kArr[k] == cxArr[n]) {
                cxIdxArr[n] = static_cast<int32_t>(k);
            }
        }
        fyOffArr[n] = fyArr[n] - fyMin;
        cyOffArr[n] = cyArr[n] - fyMin;
    }
    tiling->groupSlabMode = groupSlabMode;
    tiling->colOffsetCount = kCount;
    tiling->slabRowSpan = slabRowSpan;
    tiling->slabRowMin = fyMin;
    memcpy_s(tiling->colOffsets, sizeof(tiling->colOffsets), kArr, sizeof(kArr));
    memcpy_s(tiling->fxSlabIdx, sizeof(tiling->fxSlabIdx), fxIdxArr, sizeof(fxIdxArr));
    memcpy_s(tiling->cxSlabIdx, sizeof(tiling->cxSlabIdx), cxIdxArr, sizeof(cxIdxArr));
    memcpy_s(tiling->fyRowOff, sizeof(tiling->fyRowOff), fyOffArr, sizeof(fyOffArr));
    memcpy_s(tiling->cyRowOff, sizeof(tiling->cyRowOff), cyOffArr, sizeof(cyOffArr));

    const uint32_t tileRowCount = (static_cast<uint32_t>(outH) + tileRows - 1U) / tileRows;
    const uint32_t tileColCount = (static_cast<uint32_t>(outW) + tileCols - 1U) / tileCols;
    const uint32_t imageTileCount = tileRowCount * tileColCount;
    const uint32_t totalTiles = static_cast<uint32_t>(batchCount) * imageTileCount;
    const uint32_t tileUsedCore = std::max(1U, std::min(coreNum, totalTiles));
    const uint32_t smallCoreTileNum = totalTiles / tileUsedCore;
    const uint32_t bigCoreCount = totalTiles % tileUsedCore;
    const uint32_t bigCoreTileNum = smallCoreTileNum + ((bigCoreCount == 0U) ? 0U : 1U);
    tiling->tileColCount = tileColCount;
    tiling->imageTileCount = imageTileCount;
    tiling->smallCoreTileNum = smallCoreTileNum;
    tiling->bigCoreTileNum = bigCoreTileNum;
    tiling->bigCoreCount = bigCoreCount;
    const uint32_t rowUsedCore = rowParallelCores;
    tiling->smallCoreRows = smallCoreRows;
    tiling->bigCoreRows = bigCoreRows;
    tiling->bigRowCoreCount = bigRowCoreCount;

    // 直接设置 TilingKey 为 1 (RowSplit) 或 2 (Tile2D)
    context->SetTilingKey(GET_TPL_TILING_KEY(
        static_cast<uint32_t>(useRowSplit ? LBP_FEATURE_SCH_MODE_ROW_SPLIT : LBP_FEATURE_SCH_MODE_TILE_2D)));
    context->SetBlockDim(useRowSplit ? rowUsedCore : tileUsedCore);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLbpFeature([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(LbpFeature).Tiling(LbpFeatureTilingFunc).TilingParse<LbpFeatureCompileInfo>(TilingParseForLbpFeature);
} // namespace optiling
