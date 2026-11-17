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

#include "kernel_operator.h"
#include "laplacian_of_gaussian_tiling_data.h"
#include "laplacian_of_gaussian_tiling_key.h"
using namespace AscendC;
namespace {
// General TwoSum: retain FP32 addition residual without magnitude ordering.
// Scratch is disjoint from both accumulator components and the product.
__aicore__ inline void Accumulate(LocalTensor<float> hi, LocalTensor<float> lo, LocalTensor<float> product,
                                  LocalTensor<float> scratch, uint32_t stride, uint32_t count)
{
    auto s = scratch;
    auto z = scratch[stride];
    auto e = scratch[2 * stride];
    auto q = scratch[3 * stride];
    auto v = scratch[4 * stride];
    Add(s, hi, product, count);
    PipeBarrier<PIPE_V>();
    Sub(z, s, hi, count);
    PipeBarrier<PIPE_V>();
    Sub(q, s, z, count);
    PipeBarrier<PIPE_V>();
    Sub(e, hi, q, count);
    PipeBarrier<PIPE_V>();
    Sub(v, product, z, count);
    PipeBarrier<PIPE_V>();
    Add(e, e, v, count);
    PipeBarrier<PIPE_V>();
    Add(q, lo, e, count);
    PipeBarrier<PIPE_V>();
    Add(hi, s, q, count);
    PipeBarrier<PIPE_V>();
    Sub(z, hi, s, count);
    PipeBarrier<PIPE_V>();
    Sub(v, hi, z, count);
    PipeBarrier<PIPE_V>();
    Sub(e, s, v, count);
    PipeBarrier<PIPE_V>();
    Sub(v, q, z, count);
    PipeBarrier<PIPE_V>();
    Add(lo, e, v, count);
    PipeBarrier<PIPE_V>();
}
template <class T>
class LoGLegacy {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf, weights, raw, halfBuf;
    GlobalTensor<T> x;
    GlobalTensor<float> y, wg;
    LaplacianOfGaussianTilingData t;
    LocalTensor<float> g, d, values, u, v, tmp, a, b, ul, vl, scratch;
    LocalTensor<T> input;
    __aicore__ inline int64_t Map(int64_t q, int64_t l)
    {
        if (t.border == 3)
            return q < 0 || q >= l ? -1 : q;
        if (t.border == 2)
            return q < 0 ? 0 : (q >= l ? l - 1 : q);
        if (l == 1)
            return 0;
        int64_t p = t.border == 0 ? 2 * l : 2 * (l - 1);
        int64_t z = q % p;
        if (z < 0)
            z += p;
        return z < l ? z : p - z - (t.border == 0 ? 1 : 0);
    }
    __aicore__ inline void Sync() { PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline void Gather(uint64_t base, int64_t col, int64_t tap, uint32_t count)
    {
        uint32_t j = 0;
        while (j < count) {
            int64_t start = Map(col + j + tap, t.w);
            if (start < 0) {
                values.SetValue(j, 0.f);
                ++j;
                continue;
            }
            uint32_t run = 1;
            while (j + run < count && Map(col + j + run + tap, t.w) == start + run)
                ++run;
            DataCopyExtParams cp{1, uint32_t(run * sizeof(T)), 0, 0, 0};
            for (uint32_t z = 0; z < 32; ++z)
                input.SetValue(z, T(0));
            Sync();
            DataCopyPad(input, x[base + start], cp, DataCopyPadExtParams<T>{true, 0, 0, T(0)});
            Sync();
            if constexpr (sizeof(T) == 1) {
                auto h = halfBuf.Get<half>();
                Cast(h, input, RoundMode::CAST_NONE, 32);
                Sync();
                auto f = buf.Get<float>()[128];
                Cast(f, h, RoundMode::CAST_NONE, 32);
                Sync();
                for (uint32_t z = 0; z < run; ++z)
                    values.SetValue(j + z, f.GetValue(z));
            } else if constexpr (sizeof(T) == 2) {
                auto f = buf.Get<float>()[128];
                Cast(f, input, RoundMode::CAST_NONE, 32);
                Sync();
                for (uint32_t z = 0; z < run; ++z)
                    values.SetValue(j + z, f.GetValue(z));
            } else {
                for (uint32_t z = 0; z < run; ++z)
                    values.SetValue(j + z, input.GetValue(z));
            }
            Sync();
            j += run;
        }
        Sync();
    }

public:
    __aicore__ inline void Init(GM_ADDR xx, GM_ADDR yy, GM_ADDR tt, const LaplacianOfGaussianTilingData& data)
    {
        t = data;
        x.SetGlobalBuffer((__gm__ T*)xx);
        y.SetGlobalBuffer((__gm__ float*)yy);
        wg.SetGlobalBuffer((__gm__ float*)(tt + t.weightOffset));
        pipe.InitBuffer(buf, 32 * 8 * 4);
        pipe.InitBuffer(raw, 32 * sizeof(T));
        pipe.InitBuffer(halfBuf, 64);
        pipe.InitBuffer(weights, 2 * t.weightStride);
        auto f = buf.Get<float>();
        values = f;
        u = f[8];
        v = f[16];
        tmp = f[24];
        a = f[32];
        b = f[40];
        ul = f[48];
        vl = f[56];
        scratch = f[64];
        input = raw.Get<T>();
        g = weights.Get<float>();
        d = g[t.weightStride / 4];
        DataCopyExtParams cp{1, t.k * 4, 0, 0, 0};
        DataCopyPad(g, wg, cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(d, wg[t.weightStride / 4], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Sync();
    }
    __aicore__ inline void Process()
    {
        uint64_t cols = t.w / 32 + (t.w % 32 != 0);
        for (uint64_t id = GetBlockIdx(); id < t.tiles; id += t.blocks) {
            uint64_t col = (id % cols) * 32, row = (id / cols) % t.h, n = id / (cols * t.h);
            uint32_t whole = t.w - col < 32 ? t.w - col : 32;
            for (uint32_t part = 0; part < whole; part += 8) {
                uint64_t subcol = col + part;
                uint32_t count = whole - part < 8 ? whole - part : 8;
                Duplicate(a, 0.f, 8);
                Duplicate(b, 0.f, 8);
                Sync();
                for (uint32_t iy = 0; iy < t.k; ++iy) {
                    int64_t sy = Map(int64_t(row) + int64_t(iy) - t.radius, t.h);
                    if (sy < 0)
                        continue;
                    Duplicate(u, 0.f, 8);
                    Duplicate(v, 0.f, 8);
                    Duplicate(ul, 0.f, 8);
                    Duplicate(vl, 0.f, 8);
                    Sync();
                    for (uint32_t ix = 0; ix < t.k; ++ix) {
                        Gather((n * t.h + sy) * t.w, subcol, int64_t(ix) - t.radius, count);
                        Muls(tmp, values, g.GetValue(ix), count);
                        Sync();
                        Accumulate(u, ul, tmp, scratch, 8, count);
                        Sync();
                        Muls(tmp, values, d.GetValue(ix), count);
                        Sync();
                        Accumulate(v, vl, tmp, scratch, 8, count);
                        Sync();
                    }
                    Muls(tmp, u, d.GetValue(iy), count);
                    Sync();
                    Accumulate(a, b, tmp, scratch, 8, count);
                    Sync();
                    Muls(tmp, ul, d.GetValue(iy), count);
                    Sync();
                    Accumulate(a, b, tmp, scratch, 8, count);
                    Sync();
                    Muls(tmp, v, g.GetValue(iy), count);
                    Sync();
                    Accumulate(a, b, tmp, scratch, 8, count);
                    Sync();
                    Muls(tmp, vl, g.GetValue(iy), count);
                    Sync();
                    Accumulate(a, b, tmp, scratch, 8, count);
                    Sync();
                }
                Add(a, a, b, count);
                Sync();
                DataCopyExtParams cp{1, count * 4, 0, 0, 0};
                DataCopyPad(y[(n * t.h + row) * t.w + subcol], a, cp);
                Sync();
            }
        }
    }
};
// Per-row halo path (original): one tile per output row, halo rebuilt for each tap row.
template <class T>
class LoGRow {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf, weights, raw, halfBuf, haloBuf, convertBuf;
    GlobalTensor<T> x;
    GlobalTensor<float> y, wg;
    LaplacianOfGaussianTilingData t;
    LocalTensor<float> g, d, values, u, v, tmp, a, b, ul, vl, scratch;
    LocalTensor<T> input;
    LocalTensor<float> halo;
    uint32_t padded;
    __aicore__ inline int64_t Map(int64_t q, int64_t l)
    {
        if (t.border == 3)
            return q < 0 || q >= l ? -1 : q;
        if (t.border == 2)
            return q < 0 ? 0 : (q >= l ? l - 1 : q);
        if (l == 1)
            return 0;
        int64_t p = t.border == 0 ? 2 * l : 2 * (l - 1);
        int64_t z = q % p;
        if (z < 0)
            z += p;
        return z < l ? z : p - z - (t.border == 0 ? 1 : 0);
    }
    __aicore__ inline void Sync() { PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline void LoadHalo(uint64_t base, int64_t col, uint32_t count)
    {
        uint32_t length = count + 2 * t.radius, j = 0;
        while (j < length) {
            int64_t start = Map(col + int64_t(j) - t.radius, t.w);
            if (start < 0) {
                halo.SetValue(j, 0.f);
                ++j;
                continue;
            }
            uint32_t run = 1;
            while (j + run < length && Map(col + int64_t(j + run) - t.radius, t.w) == start + run)
                ++run;
            Duplicate(raw.Get<int16_t>(), int16_t(0), padded * sizeof(T) / sizeof(int16_t));
            Sync();
            DataCopyExtParams cp{1, uint32_t(run * sizeof(T)), 0, 0, 0};
            DataCopyPad(input, x[base + start], cp, DataCopyPadExtParams<T>{true, 0, 0, T(0)});
            Sync();
            auto f = convertBuf.Get<float>();
            if constexpr (sizeof(T) == 1) {
                auto h = halfBuf.Get<half>();
                Cast(h, input, RoundMode::CAST_NONE, padded);
                Sync();
                Cast(f, h, RoundMode::CAST_NONE, padded);
                Sync();
                for (uint32_t z = 0; z < run; ++z)
                    halo.SetValue(j + z, f.GetValue(z));
            } else if constexpr (sizeof(T) == 2) {
                Cast(f, input, RoundMode::CAST_NONE, padded);
                Sync();
                for (uint32_t z = 0; z < run; ++z)
                    halo.SetValue(j + z, f.GetValue(z));
            } else {
                for (uint32_t z = 0; z < run; ++z)
                    halo.SetValue(j + z, input.GetValue(z));
            }
            Sync();
            j += run;
        }
        Sync();
    }

public:
    __aicore__ inline void Init(GM_ADDR xx, GM_ADDR yy, GM_ADDR tt, const LaplacianOfGaussianTilingData& data)
    {
        t = data;
        x.SetGlobalBuffer((__gm__ T*)xx);
        y.SetGlobalBuffer((__gm__ float*)yy);
        wg.SetGlobalBuffer((__gm__ float*)(tt + t.weightOffset));
        padded = (t.tileWidth + 2 * t.radius + 31) / 32 * 32;
        pipe.InitBuffer(buf, t.tileWidth * 16 * 4);
        pipe.InitBuffer(raw, padded * sizeof(T));
        pipe.InitBuffer(halfBuf, padded * 2);
        pipe.InitBuffer(haloBuf, padded * 4);
        pipe.InitBuffer(convertBuf, padded * 4);
        halo = haloBuf.Get<float>();
        pipe.InitBuffer(weights, 2 * t.weightStride);
        auto f = buf.Get<float>();
        values = f;
        u = f[t.tileWidth];
        v = f[2 * t.tileWidth];
        tmp = f[3 * t.tileWidth];
        a = f[4 * t.tileWidth];
        b = f[5 * t.tileWidth];
        ul = f[6 * t.tileWidth];
        vl = f[7 * t.tileWidth];
        scratch = f[8 * t.tileWidth];
        input = raw.Get<T>();
        g = weights.Get<float>();
        d = g[t.weightStride / 4];
        DataCopyExtParams cp{1, t.k * 4, 0, 0, 0};
        DataCopyPad(g, wg, cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(d, wg[t.weightStride / 4], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Sync();
    }
    __aicore__ inline void Process()
    {
        uint64_t cols = t.w / t.tileWidth + (t.w % t.tileWidth != 0);
        for (uint64_t id = GetBlockIdx(); id < t.tiles; id += t.blocks) {
            uint64_t col = (id % cols) * t.tileWidth, row = (id / cols) % t.h, n = id / (cols * t.h);
            uint32_t count = t.w - col < t.tileWidth ? t.w - col : t.tileWidth;
            Duplicate(a, 0.f, t.tileWidth);
            Duplicate(b, 0.f, t.tileWidth);
            Sync();
            for (uint32_t iy = 0; iy < t.k; ++iy) {
                int64_t sy = Map(int64_t(row) + int64_t(iy) - t.radius, t.h);
                if (sy < 0)
                    continue;
                Duplicate(u, 0.f, t.tileWidth);
                Duplicate(v, 0.f, t.tileWidth);
                Duplicate(ul, 0.f, t.tileWidth);
                Duplicate(vl, 0.f, t.tileWidth);
                Sync();
                LoadHalo((n * t.h + sy) * t.w, col, count);
                for (uint32_t ix = 0; ix < t.k; ++ix) {
                    for (uint32_t j = 0; j < count; ++j)
                        values.SetValue(j, halo.GetValue(ix + j));
                    Sync();
                    Muls(tmp, values, g.GetValue(ix), count);
                    Sync();
                    Accumulate(u, ul, tmp, scratch, t.tileWidth, count);
                    Sync();
                    Muls(tmp, values, d.GetValue(ix), count);
                    Sync();
                    Accumulate(v, vl, tmp, scratch, t.tileWidth, count);
                    Sync();
                }
                Muls(tmp, u, d.GetValue(iy), count);
                Sync();
                Accumulate(a, b, tmp, scratch, t.tileWidth, count);
                Sync();
                Muls(tmp, ul, d.GetValue(iy), count);
                Sync();
                Accumulate(a, b, tmp, scratch, t.tileWidth, count);
                Sync();
                Muls(tmp, v, g.GetValue(iy), count);
                Sync();
                Accumulate(a, b, tmp, scratch, t.tileWidth, count);
                Sync();
                Muls(tmp, vl, g.GetValue(iy), count);
                Sync();
                Accumulate(a, b, tmp, scratch, t.tileWidth, count);
                Sync();
            }
            Add(a, a, b, count);
            Sync();
            DataCopyExtParams cp{1, count * 4, 0, 0, 0};
            DataCopyPad(y[(n * t.h + row) * t.w + col], a, cp);
            Sync();
        }
    }
};
// Row-chunked halo path: one tile owns rowChunk output rows of a column strip and
// reuses the k horizontally filtered rows kept in a ring cache.
template <class T>
class LoGChunk {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf, weights, raw, halfBuf, haloBuf, convertBuf, ringBuf;
    GlobalTensor<T> x;
    GlobalTensor<float> y, wg;
    LaplacianOfGaussianTilingData t;
    LocalTensor<float> g, d, values, tmp, a, b, scratch, halo, ring, p0, p2;
    LocalTensor<T> input;
    uint32_t padded, w;
    __aicore__ inline int64_t Map(int64_t q, int64_t l)
    {
        if (t.border == 3)
            return q < 0 || q >= l ? -1 : q;
        if (t.border == 2)
            return q < 0 ? 0 : (q >= l ? l - 1 : q);
        if (l == 1)
            return 0;
        int64_t p = t.border == 0 ? 2 * l : 2 * (l - 1);
        int64_t z = q % p;
        if (z < 0)
            z += p;
        return z < l ? z : p - z - (t.border == 0 ? 1 : 0);
    }
    __aicore__ inline void Sync() { PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline void SyncV() { PipeBarrier<PIPE_V>(); }
    __aicore__ inline LocalTensor<float> Slot(uint32_t s, uint32_t part) { return ring[(s * 2 + part) * w]; }
    __aicore__ inline void LoadHalo(uint64_t base, int64_t col, uint32_t lo, uint32_t hi)
    {
        uint32_t j = lo;
        while (j < hi) {
            int64_t start = Map(col + int64_t(j) - t.radius, t.w);
            if (start < 0) {
                halo.SetValue(j, 0.f);
                ++j;
                continue;
            }
            uint32_t run = 1;
            while (j + run < hi && Map(col + int64_t(j + run) - t.radius, t.w) == start + run)
                ++run;
            Sync();
            DataCopyExtParams cp{1, uint32_t(run * sizeof(T)), 0, 0, 0};
            DataCopyPad(input, x[base + start], cp, DataCopyPadExtParams<T>{true, 0, 0, T(0)});
            Sync();
            auto f = convertBuf.Get<float>();
            if constexpr (sizeof(T) == 1) {
                auto h = halfBuf.Get<half>();
                Cast(h, input, RoundMode::CAST_NONE, run);
                Sync();
                Cast(f, h, RoundMode::CAST_NONE, run);
                Sync();
                for (uint32_t z = 0; z < run; ++z)
                    halo.SetValue(j + z, f.GetValue(z));
            } else if constexpr (sizeof(T) == 2) {
                Cast(f, input, RoundMode::CAST_NONE, run);
                Sync();
                for (uint32_t z = 0; z < run; ++z)
                    halo.SetValue(j + z, f.GetValue(z));
            } else {
                for (uint32_t z = 0; z < run; ++z)
                    halo.SetValue(j + z, input.GetValue(z));
            }
            Sync();
            j += run;
        }
        Sync();
    }
    __aicore__ inline void ZeroRow(uint32_t s)
    {
        Duplicate(Slot(s, 0), 0.f, w);
        Duplicate(Slot(s, 1), 0.f, w);
    }
    __aicore__ inline void HorizontalRow(uint64_t base, int64_t col, uint32_t count, uint32_t s)
    {
        auto hg = Slot(s, 0), hd = Slot(s, 1);
        Duplicate(hg, 0.f, w);
        Duplicate(hd, 0.f, w);
        Sync();

        // Columns whose whole tap window lies inside the image can be read straight from GM:
        // DataCopyPad tolerates arbitrary GM offsets, so no UB shift is needed.
        int32_t lo = (col >= int64_t(t.radius)) ? 0 : int32_t(t.radius - col);
        int64_t maxj = int64_t(t.w) - 1 - int64_t(t.radius) - col;
        int32_t hi = maxj < 0 ? -1 : int32_t(maxj < (int64_t(count) - 1) ? maxj : (int64_t(count) - 1));
        uint32_t a = uint32_t(lo + 7) / 8u * 8u;
        if (a > count)
            a = count;
        uint32_t needL = a + t.k;
        if (needL > count + 2u * t.radius)
            needL = count + 2u * t.radius;
        uint32_t needR0 = uint32_t(hi + 1);
        uint32_t needR1 = count + t.k - 1u;
        if (needR1 > count + 2u * t.radius)
            needR1 = count + 2u * t.radius;
        LoadHalo(base, col, 0, needL);
        if (needR1 > needR0)
            LoadHalo(base, col, needR0, needR1);
        for (uint32_t ix = 0; ix < t.k; ++ix) {
            if (hi >= int32_t(a)) {
                uint32_t n = uint32_t(hi - int32_t(a) + 1);
                uint64_t src = base + uint64_t(col) + uint64_t(a) + uint64_t(ix) - uint64_t(t.radius);
                if constexpr (sizeof(T) == 4) {
                    DataCopyExtParams cp4{1, uint32_t(n * 4), 0, 0, 0};
                    DataCopyPad(values[a], x[src], cp4, DataCopyPadExtParams<T>{false, 0, 0, T(0)});
                    Sync();
                } else {
                    DataCopyExtParams cp{1, uint32_t(n * sizeof(T)), 0, 0, 0};
                    DataCopyPad(input, x[src], cp, DataCopyPadExtParams<T>{true, 0, 0, T(0)});
                    Sync();
                    if constexpr (sizeof(T) == 1) {
                        auto h = halfBuf.Get<half>();
                        Cast(h, input, RoundMode::CAST_NONE, n);
                        Sync();
                        Cast(values[a], h, RoundMode::CAST_NONE, n);
                        Sync();
                    } else {
                        Cast(values[a], input, RoundMode::CAST_NONE, n);
                        Sync();
                    }
                }
            }
            Sync();
            for (uint32_t jj = 0; jj < a; ++jj)
                values.SetValue(jj, halo.GetValue(ix + jj));
            for (int32_t jj = hi + 1; jj < int32_t(count); ++jj)
                values.SetValue(uint32_t(jj), halo.GetValue(ix + jj));
            Sync();
            Muls(tmp, values, g.GetValue(ix), count);
            SyncV();
            Add(hg, hg, tmp, count);
            SyncV();
            Muls(tmp, values, d.GetValue(ix), count);
            SyncV();
            Add(hd, hd, tmp, count);
            SyncV();
        }
    }

public:
    __aicore__ inline void Init(GM_ADDR xx, GM_ADDR yy, GM_ADDR tt, const LaplacianOfGaussianTilingData& data)
    {
        t = data;
        w = t.tileWidth;
        x.SetGlobalBuffer((__gm__ T*)xx);
        y.SetGlobalBuffer((__gm__ float*)yy);
        wg.SetGlobalBuffer((__gm__ float*)(tt + t.weightOffset));
        padded = (w + 2 * t.radius + 31) / 32 * 32;
        pipe.InitBuffer(buf, w * 16 * 4);
        pipe.InitBuffer(raw, padded * sizeof(T));
        pipe.InitBuffer(halfBuf, padded * 2);
        pipe.InitBuffer(haloBuf, padded * 4);
        pipe.InitBuffer(convertBuf, padded * 4);
        pipe.InitBuffer(ringBuf, t.k * 2 * w * 4);
        pipe.InitBuffer(weights, 2 * t.weightStride);
        auto f = buf.Get<float>();
        a = f;
        b = f[w];
        tmp = f[2 * w];
        values = f[3 * w];
        scratch = f[4 * w];
        p0 = f[9 * w];
        p2 = f[10 * w];
        halo = haloBuf.Get<float>();
        ring = ringBuf.Get<float>();
        input = raw.Get<T>();
        g = weights.Get<float>();
        d = g[t.weightStride / 4];
        DataCopyExtParams cp{1, t.k * 4, 0, 0, 0};
        DataCopyPad(g, wg, cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(d, wg[t.weightStride / 4], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Sync();
    }
    __aicore__ inline void Process()
    {
        uint64_t cols = (t.w + w - 1) / w;
        uint32_t rc = t.reserved >> 1;
        if (rc == 0)
            return;
        uint64_t chunks = (t.h + rc - 1) / rc;
        uint64_t perN = cols * chunks;
        if (perN == 0 || rc == 0)
            return;
        for (uint64_t id = GetBlockIdx(); id < t.tiles; id += t.blocks) {
            uint64_t n = id / perN, rem = id % perN;
            uint64_t colIdx = rem % cols, chunk = rem / cols;
            uint64_t col = colIdx * w;
            uint32_t count = (t.w - col < w) ? uint32_t(t.w - col) : w;
            uint64_t r0 = chunk * rc, r1 = r0 + rc;
            if (r1 > t.h)
                r1 = t.h;
            if (r0 >= r1)
                continue;
            uint64_t xbase = n * t.h * t.w;
            for (uint32_t iy = 0; iy < t.k; ++iy) {
                int64_t sy = Map(int64_t(r0) + int64_t(iy) - t.radius, t.h);
                if (sy < 0)
                    ZeroRow(iy % t.k);
                else
                    HorizontalRow(xbase + uint64_t(sy) * t.w, col, count, iy % t.k);
            }
            for (uint64_t r = r0; r < r1; ++r) {
                uint32_t s0 = uint32_t((r - r0) % t.k);
                Duplicate(a, 0.f, count);
                Duplicate(b, 0.f, count);
                Sync();
                // g and d are even, so taps R+p and R-p share one weight: accumulate their
                // pre-multiplied rows together and halve the number of compensated adds.
                for (uint32_t pp = 0; pp <= t.radius; ++pp) {
                    uint32_t iy = t.radius + pp;
                    uint32_t s1 = (s0 + iy) % t.k;
                    if (pp == 0) {
                        Muls(tmp, Slot(s1, 0), d.GetValue(iy), count);
                        SyncV();
                        Accumulate(a, b, tmp, scratch, w, count);
                        SyncV();
                        Muls(tmp, Slot(s1, 1), g.GetValue(iy), count);
                        SyncV();
                        Accumulate(a, b, tmp, scratch, w, count);
                        SyncV();
                    } else {
                        uint32_t iy2 = t.radius - pp;
                        uint32_t s2 = (s0 + iy2) % t.k;
                        Add(p0, Slot(s1, 0), Slot(s2, 0), count);
                        Sync();
                        Add(p2, Slot(s1, 1), Slot(s2, 1), count);
                        Sync();
                        Muls(tmp, p0, d.GetValue(iy), count);
                        SyncV();
                        Accumulate(a, b, tmp, scratch, w, count);
                        SyncV();
                        Muls(tmp, p2, g.GetValue(iy), count);
                        SyncV();
                        Accumulate(a, b, tmp, scratch, w, count);
                        SyncV();
                    }
                }
                Add(a, a, b, count);
                Sync();
                DataCopyExtParams cp{1, count * 4, 0, 0, 0};
                DataCopyPad(y[n * t.h * t.w + r * t.w + col], a, cp);
                Sync();
                int64_t ny = Map(int64_t(r) + 1 + t.radius, t.h);
                if (ny < 0)
                    ZeroRow(s0);
                else
                    HorizontalRow(xbase + uint64_t(ny) * t.w, col, count, s0);
            }
        }
    }
};
} // namespace
template <uint32_t mode>
__global__ __aicore__ void laplacian_of_gaussian(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(LaplacianOfGaussianTilingStorage);
    GET_TILING_DATA_WITH_STRUCT(LaplacianOfGaussianTilingData, t, tiling);
    if ((t.reserved & 1) == 0) {
        LoGLegacy<DTYPE_X> op;
        op.Init(x, y, tiling, t);
        op.Process();
    } else if ((t.reserved >> 1) > 1) {
        LoGChunk<DTYPE_X> op;
        op.Init(x, y, tiling, t);
        op.Process();
    } else {
        LoGRow<DTYPE_X> op;
        op.Init(x, y, tiling, t);
        op.Process();
    }
}
