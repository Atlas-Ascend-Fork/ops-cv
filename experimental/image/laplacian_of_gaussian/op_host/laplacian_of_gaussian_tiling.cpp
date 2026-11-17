/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include <cmath>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <limits>
#include <vector>
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/laplacian_of_gaussian_tiling_data.h"
#include "../op_kernel/laplacian_of_gaussian_tiling_key.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* c)
{
    auto fail = []() { return ge::GRAPH_FAILED; };
    if (!c || !c->GetInputShape(0) || !c->GetAttrs() || !c->GetPlatformInfo())
        return fail();
    auto& s = c->GetInputShape(0)->GetStorageShape();
    auto rank = s.GetDimNum();
    if (rank < 2 || rank > 4 || (rank == 4 && s.GetDim(1) != 1))
        return fail();
    uint64_t p = 1;
    for (size_t i = 0; i < rank; ++i) {
        auto v = s.GetDim(i);
        if (v <= 0 || p > INT64_MAX / uint64_t(v))
            return fail();
        p *= v;
    }
    const auto* a = c->GetAttrs();
    auto sp = a->GetAttrPointer<float>(0);
    auto rp = a->GetAttrPointer<int64_t>(1);
    auto tp = a->GetAttrPointer<float>(2);
    auto bp = a->GetAttrPointer<char>(3);
    if (!sp || !rp || !tp || !bp || !std::isfinite(*sp) || *sp <= 0 || !std::isfinite(*tp) || *tp <= 0 || *rp < -1)
        return fail();
    double rd = *rp >= 0 ? double(*rp) : std::floor(double(*sp) * double(*tp) + 0.5);
    if (!std::isfinite(rd) || rd > (UINT32_MAX - 1) / 2)
        return fail();
    uint32_t r = rd, k = 2 * r + 1;
    uint64_t stride = (uint64_t(k) * 4 + 31) / 32 * 32;
    uint64_t bytes = 64 + 2 * stride;
    auto raw = c->GetRawTilingData();
    if (!raw || !raw->GetData() || bytes > raw->GetCapacity())
        return fail();
    uint32_t border = 4;
    const char* names[] = {"REFLECT", "REFLECT_101", "REPLICATE", "CONSTANT"};
    for (uint32_t i = 0; i < 4; ++i)
        if (std::strcmp(bp, names[i]) == 0)
            border = i;
    if (border == 4)
        return fail();
    uint64_t h = s.GetDim(rank - 2), w = s.GetDim(rank - 1), n = rank == 2 ? 1 : s.GetDim(0);
    if (h > (INT64_MAX - r) / 2 || w > (INT64_MAX - r) / 2 || p > INT64_MAX / 4)
        return fail();
    platform_ascendc::PlatformAscendC plat(c->GetPlatformInfo());
    uint64_t ub = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    uint32_t cores = plat.GetCoreNumAiv();
    if (!cores || bytes > ub || ub - bytes < 8192 + 32 * 40)
        return fail();
    uint32_t elem = c->GetInputDesc(0)->GetDataType() == ge::DT_UINT8   ? 1 :
                    c->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16 ? 2 :
                                                                          4;
    uint32_t tile = 32, optimized = 0;
    uint64_t limit = (w + 31) / 32 * 32;
    if (limit > 4096)
        limit = 4096;
    // Per-width cost: the 16-float scratch block plus a 2k-float row ring, i.e. (16 + 2k)
    // floats per output column. Per padded column the kernel allocates in Init:
    // raw(T)=elem, half=2, halo=4, convert=4 bytes, so (10 + elem) bytes per padded column.
    // kUbSlack covers 32B alignment rounding of both terms plus framework bookkeeping.
    constexpr uint64_t kUbSlack = 16384;
    auto fits = [&](uint64_t width) {
        uint64_t padded = (width + 2 * r + 31) / 32 * 32;
        uint64_t cache = (uint64_t(16) + 2ull * k) * 4 * width;
        return cache + (10 + elem) * padded + 2 * stride + kUbSlack <= ub;
    };
    for (uint64_t width = limit; width >= 32; width -= 32) {
        uint64_t candidateTiles = n * h * (w / width + (w % width != 0));
        uint64_t oldTiles = n * h * (w / 32 + (w % 32 != 0));
        if (candidateTiles < cores && candidateTiles < oldTiles)
            continue;
        if (fits(width)) {
            tile = width;
            optimized = 1;
            break;
        }
    }
    if (!optimized && fits(32))
        optimized = 1;
    // Pick the row-chunk height that minimises per-block work:
    //   chunks = ceil(h/R), work per chunk ~ (k + R - 1) horizontal rows,
    //   blocks = min(cores, groups*chunks).
    // Short images therefore fall back to R=1 (full parallelism, no reuse),
    // tall images get a large R where the k-ring actually pays off.
    uint64_t cols = w / tile + (w % tile != 0);
    uint64_t rowChunk = 1, chunks = 1, tiles = 0;
    if (optimized) {
        uint64_t groups = n * cols;
        if (!groups)
            groups = 1;
        double bestCost = 1e30;
        for (uint64_t R = 1; R <= h; ++R) {
            uint64_t nch = (h + R - 1) / R;
            uint64_t need = groups * nch;
            uint64_t blk = need < cores ? need : cores;
            if (!blk)
                continue;
            double cost = double(nch) * (double(k) + double(R) - 1.0) / double(blk);
            if (cost < bestCost - 1e-9) {
                bestCost = cost;
                rowChunk = R;
            }
        }
        if (rowChunk < 1)
            rowChunk = 1;
        if (rowChunk > h)
            rowChunk = h;
        chunks = (h + rowChunk - 1) / rowChunk;
        tiles = n * cols * chunks;
    } else {
        // Not enough UB for the row cache: fall back to the legacy 32-wide per-row path.
        tile = 32;
        cols = w / 32 + (w % 32 != 0);
        rowChunk = 1;
        chunks = h;
        tiles = n * h * cols;
    }
    uint32_t blocks = tiles < cores ? uint32_t(tiles) : cores;
    if (blocks == 0)
        return fail();
    std::vector<double> g(k), d(k);
    double sum = 0, ss = double(*sp) * double(*sp);
    for (uint32_t i = 0; i < k; ++i) {
        double z = (double(i) - r) / double(*sp);
        g[i] = std::exp(-0.5 * z * z);
        sum += g[i];
    }
    std::vector<float> gf(k), df(k);
    bool anyD = false;
    for (uint32_t i = 0; i < k; ++i) {
        double z = (double(i) - r) / double(*sp);
        g[i] /= sum;
        d[i] = g[i] == 0 ? 0 : ((z * z - 1) / ss) * g[i];
        gf[i] = g[i];
        df[i] = d[i];
        if (!std::isfinite(g[i]) || !std::isfinite(d[i]) || !std::isfinite(gf[i]) || !std::isfinite(df[i]))
            return fail();
        anyD |= df[i] != 0;
    }
    if (!anyD)
        return fail();
    LaplacianOfGaussianTilingData t{};
    t.n = n;
    t.h = h;
    t.w = w;
    t.tiles = tiles;
    t.radius = r;
    t.k = k;
    t.border = border;
    t.blocks = blocks;
    t.weightOffset = 64;
    t.weightStride = uint32_t(stride);
    t.tileWidth = tile;
    t.reserved = optimized ? ((uint32_t(rowChunk) << 1) | 1u) : 0u;
    auto dst = static_cast<char*>(raw->GetData());
    std::memset(dst, 0, bytes);
    std::memcpy(dst, &t, sizeof(t));
    std::memcpy(dst + t.weightOffset, gf.data(), k * 4);
    std::memcpy(dst + t.weightOffset + stride, df.data(), k * 4);
    raw->SetDataSize(bytes);
    c->SetBlockDim(blocks);
    c->SetTilingKey(0);
    auto ws = c->GetWorkspaceSizes(1);
    if (!ws)
        return fail();
    ws[0] = plat.GetLibApiWorkSpaceSize();
    OP_LOGI(c->GetNodeName(),
            "LOG_RESOURCE ub=%lu cores=%u capacity=%lu tiling_bytes=%lu blocks=%u tiles=%lu T=%u halo=%u rowChunk=%lu "
            "chunks=%lu B=1 weight_stride=%lu workspace=%lu r=%u K=%u",
            ub, cores, raw->GetCapacity(), bytes, blocks, tiles, tile, optimized, rowChunk, chunks, stride, ws[0], r,
            k);
    return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(LaplacianOfGaussian).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
} // namespace optiling
