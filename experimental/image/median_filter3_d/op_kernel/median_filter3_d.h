/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEDIAN_FILTER3_D_H
#define MEDIAN_FILTER3_D_H
#include "kernel_operator.h"
#include "median_filter3_d_tiling_data.h"
#include "median_filter3_d_sort32.h"
namespace NsMedianFilter3D {
using namespace AscendC;
template <typename T>
class KernelMedianFilter3D {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MedianFilter3DTilingData* data, TPipe* pipe)
    {
        t = *data;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), t.rows * t.w);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), t.rows * t.w);
        pipe->InitBuffer(planes, t.padded * t.tile * 4);
        pipe->InitBuffer(temp, t.tile * 4);
        pipe->InitBuffer(halfTemp, t.tile * 2);
        pipe->InitBuffer(inQueue, 1, t.tile * sizeof(T));
        pipe->InitBuffer(outQueue, 1, t.tile * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        for (uint64_t task = GetBlockIdx(); task < t.tasks; task += t.cores) {
            ProcessTile(task);
        }
    }

private:
    __aicore__ inline int64_t Reflect(int64_t p, int64_t n)
    {
        p %= 2 * n;
        if (p < 0) {
            p += 2 * n;
        }
        return p < n ? p : 2 * n - 1 - p;
    }
    __aicore__ inline void Load(LocalTensor<float> dst, uint64_t base, int64_t z, int64_t y, int64_t x, uint32_t len)
    {
        bool validZY = z >= 0 && z < t.d && y >= 0 && y < t.h;
        if (validZY && x >= 0 && x + len <= t.w) {
            auto in = inQueue.template AllocTensor<T>();
            DataCopyExtParams read{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(in, xGm[base + (z * t.h + y) * t.w + x], read, pad);
            inQueue.EnQue(in);
            in = inQueue.template DeQue<T>();
            if constexpr (IsSameType<T, float>::value) {
                Adds(dst, in, 0.0f, len);
            } else if constexpr (IsSameType<T, uint8_t>::value) {
                auto h = halfTemp.Get<half>();
                Cast(h, in, RoundMode::CAST_NONE, len);
                PipeBarrier<PIPE_V>();
                Cast(dst, h, RoundMode::CAST_NONE, len);
            } else {
                Cast(dst, in, RoundMode::CAST_NONE, len);
            }
            PipeBarrier<PIPE_V>();
            inQueue.FreeTensor(in);
        } else {
            PipeBarrier<PIPE_ALL>();
            for (uint32_t j = 0; j < len; ++j) {
                int64_t xx = x + j, zz = z, yy = y;
                float value = t.cval;
                if (t.mode == 0) {
                    zz = Reflect(zz, t.d);
                    yy = Reflect(yy, t.h);
                    xx = Reflect(xx, t.w);
                }
                if (t.mode == 0 || (validZY && xx >= 0 && xx < t.w)) {
                    if constexpr (IsSameType<T, uint8_t>::value) {
                        value = static_cast<float>(
                            static_cast<int32_t>(xGm.GetValue(base + (zz * t.h + yy) * t.w + xx)));
                    } else {
                        value = static_cast<float>(xGm.GetValue(base + (zz * t.h + yy) * t.w + xx));
                    }
                }
                dst.SetValue(j, value);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ProcessTile(uint64_t task)
    {
        uint64_t row = task / t.tilesPerRow;
        uint32_t x = (task % t.tilesPerRow) * t.tile;
        uint32_t len = t.w - x < t.tile ? t.w - x : t.tile;
        uint64_t volume = row / (static_cast<uint64_t>(t.d) * t.h);
        int64_t z = (row / t.h) % t.d, y = row % t.h;
        uint64_t base = volume * static_cast<uint64_t>(t.d) * t.h * t.w;
        auto buf = planes.Get<float>();
        uint32_t idx = 0;
        for (uint32_t dz = 0; dz < t.kd; ++dz) {
            for (uint32_t dy = 0; dy < t.kh; ++dy) {
                for (uint32_t dx = 0; dx < t.kw; ++dx) {
                    Load(buf[idx++ * t.tile], base, z + dz - static_cast<int64_t>(t.kd / 2),
                         y + dy - static_cast<int64_t>(t.kh / 2), static_cast<int64_t>(x) + dx - t.kw / 2, len);
                }
            }
        }
        // +Inf is a valid neutral value even for all-infinite input windows.
        union {
            uint32_t u;
            float f;
        } infinity;
        infinity.u = 0x7f800000;
        for (; idx < t.padded; ++idx) {
            Duplicate(buf[idx * t.tile], infinity.f, len);
        }
        PipeBarrier<PIPE_V>();
        auto tmp = temp.Get<float>();
        if (t.padded == 32) {
            Sort32(buf, tmp, t.tile, len);
        } else {
            for (uint32_t size = 2; size <= t.padded; size *= 2) {
                for (uint32_t step = size / 2; step > 0; step /= 2) {
                    for (uint32_t i = 0; i < t.padded; ++i) {
                        uint32_t j = i ^ step;
                        if (j > i) {
                            auto a = buf[i * t.tile], b = buf[j * t.tile];
                            Min(tmp, a, b, len);
                            PipeBarrier<PIPE_V>();
                            if ((i & size) == 0) {
                                Max(b, a, b, len);
                                PipeBarrier<PIPE_V>();
                                Adds(a, tmp, 0.0f, len);
                            } else {
                                Max(a, a, b, len);
                                PipeBarrier<PIPE_V>();
                                Adds(b, tmp, 0.0f, len);
                            }
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
        }
        auto out = outQueue.template AllocTensor<T>();
        auto median = buf[(t.count / 2) * t.tile];
        PipeBarrier<PIPE_ALL>();
        for (uint32_t j = 0; j < len; ++j) {
            if constexpr (IsSameType<T, uint8_t>::value) {
                out.SetValue(j, static_cast<uint8_t>(static_cast<int32_t>(median.GetValue(j))));
            } else {
                out.SetValue(j, static_cast<T>(median.GetValue(j)));
            }
        }
        PipeBarrier<PIPE_ALL>();
        outQueue.EnQue(out);
        out = outQueue.template DeQue<T>();
        DataCopyExtParams write{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[row * t.w + x], out, write);
        outQueue.FreeTensor(out);
    }
    MedianFilter3DTilingData t;
    GlobalTensor<T> xGm, yGm;
    TBuf<TPosition::VECCALC> planes, temp, halfTemp;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
};
} // namespace NsMedianFilter3D
#endif
