/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CLAHE_KERNEL_H
#define CLAHE_KERNEL_H
#include "kernel_operator.h"
#include "lib/math/cumsum.h"
#include "clahe_tiling_data.h"

namespace NsClahe {
using namespace AscendC;
constexpr uint32_t ARENA_BYTES = 184320;
constexpr uint32_t MAX_PIXELS = 1024;
constexpr uint32_t INPUT_OFFSET = 131072;
constexpr uint32_t SEGMENT_BINS = 8192;
constexpr uint32_t PREFIX_BINS = 256;
constexpr CumSumConfig PREFIX_CONFIG = {true, false, false};

template <typename L, typename R>
__aicore__ inline L Small(L lhs, R rhs)
{
    return lhs < rhs ? lhs : static_cast<L>(rhs);
}
template <typename L, typename R>
__aicore__ inline L Large(L lhs, R rhs)
{
    return lhs > rhs ? lhs : static_cast<L>(rhs);
}

template <typename T>
class KernelClahe {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ClaheTilingData* data, TPipe* pipe)
    {
        d_ = data;
        pipe_ = pipe;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        lut_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(workspace));
        histGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + d_->lutElements * sizeof(uint16_t)));
        pipe_->InitBuffer(arena_, ARENA_BYTES);
    }

    __aicore__ inline void Process()
    {
        const uint64_t tileCount = static_cast<uint64_t>(d_->batch) * d_->tilesX * d_->tilesY;
        for (uint64_t tile = GetBlockIdx(); tile < tileCount; tile += d_->coreNum) {
            BuildLut(tile);
        }
        // 所有参与核都到达屏障，LUT 完整写回后才允许其他核读取。
        SyncAll();
        for (uint64_t region = GetBlockIdx(); region < d_->regionCount; region += d_->coreNum) {
            InterpolateRegion(region);
        }
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> View(uint32_t offset)
    {
        return arena_.Get<uint8_t>()[offset].template ReinterpretCast<U>();
    }
    template <HardEvent E>
    __aicore__ inline void Fence()
    {
        const auto event = static_cast<event_t>(pipe_->FetchEventID(E));
        SetFlag<E>(event);
        WaitFlag<E>(event);
    }
    __aicore__ inline int32_t FloorIndex(float value)
    {
        const int32_t truncated = static_cast<int32_t>(value);
        return value < static_cast<float>(static_cast<int32_t>(truncated)) ? truncated - 1 : truncated;
    }
    __aicore__ inline uint32_t Reflect(uint64_t index, uint32_t length)
    {
        if (length == 1) {
            return 0;
        }
        const uint64_t period = 2ULL * (length - 1);
        const uint32_t reduced = index % period;
        return reduced < length ? reduced : period - reduced;
    }
    __aicore__ inline uint16_t RoundLut(uint32_t count)
    {
        const float scaled = static_cast<float>(static_cast<int32_t>(count)) * d_->lutScale;
        uint32_t rounded = static_cast<uint32_t>(static_cast<int32_t>(scaled));
        const float fraction = scaled - static_cast<float>(static_cast<int32_t>(rounded));
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1U) != 0)) {
            ++rounded;
        }
        return static_cast<uint16_t>(Small(rounded, d_->bins - 1));
    }

    template <typename Count>
    __aicore__ inline uint32_t CountTile(uint64_t tile, LocalTensor<Count> histogram, uint32_t firstBin,
                                         uint32_t binCount)
    {
        const uint32_t tx = tile % d_->tilesX;
        const uint32_t ty = tile / d_->tilesX % d_->tilesY;
        const uint64_t batchBase = tile / (static_cast<uint64_t>(d_->tilesX) * d_->tilesY) * d_->imageElements;
        const uint64_t xStart = static_cast<uint64_t>(tx) * d_->tileWidth;
        const uint64_t yStart = static_cast<uint64_t>(ty) * d_->tileHeight;
        auto input = View<T>(INPUT_OFFSET);
        const uint32_t cap = d_->clipCount == 0 ? d_->tileWidth * d_->tileHeight : d_->clipCount;
        uint32_t clipped = 0;
        Duplicate(histogram, static_cast<Count>(0), binCount);
        Fence<HardEvent::V_S>();
        for (uint32_t row = 0; row < d_->tileHeight; ++row) {
            const uint64_t rowBase = batchBase + static_cast<uint64_t>(Reflect(yStart + row, d_->height)) * d_->width;
            for (uint32_t begin = 0; begin < d_->tileWidth; begin += 4096) {
                const uint32_t count = Small(4096U, d_->tileWidth - begin);
                const bool direct = xStart + begin + count <= d_->width;
                if (direct) {
                    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
                    DataCopyPad(input, x_[rowBase + xStart + begin], params, padding);
                    Fence<HardEvent::MTE2_S>();
                }
                for (uint32_t col = 0; col < count; ++col) {
                    const uint32_t value = direct ? input.GetValue(col) :
                                                    x_.GetValue(rowBase + Reflect(xStart + begin + col, d_->width));
                    if (value >= firstBin && value - firstBin < binCount) {
                        const auto old = histogram.GetValue(value - firstBin);
                        if (static_cast<uint32_t>(old) < cap) {
                            histogram.SetValue(value - firstBin, static_cast<Count>(old + 1));
                        } else {
                            ++clipped;
                        }
                    }
                }
                Fence<HardEvent::S_MTE2>();
            }
        }
        return clipped;
    }

    __aicore__ inline bool ResidualBin(uint32_t bin, uint32_t residual, uint32_t step)
    {
        return residual != 0 && bin % step == 0 && bin / step < residual;
    }

    union FloatBits {
        float value;
        uint32_t bits;
    };

    __aicore__ inline uint64_t ShiftJam(uint64_t value, int32_t shift)
    {
        if (shift >= 0) {
            return value << shift;
        }
        const uint32_t right = -shift;
        if (right >= 64) {
            return value != 0;
        }
        return (value >> right) | ((value & ((1ULL << right) - 1)) != 0);
    }

    // 非负有限 fp32 的单次舍入乘加；仅用于输出半整数附近的少量像素。
    __aicore__ inline float ExactMultiplyAdd(float a, float b, float c)
    {
        const FloatBits av{a};
        const FloatBits bv{b};
        const FloatBits cv{c};
        const uint32_t ae = av.bits >> 23;
        const uint32_t be = bv.bits >> 23;
        const uint32_t ce = cv.bits >> 23;
        const uint64_t am = (av.bits & 0x7fffffU) | (ae == 0 ? 0 : 0x800000U);
        const uint64_t bm = (bv.bits & 0x7fffffU) | (be == 0 ? 0 : 0x800000U);
        const uint64_t cm = (cv.bits & 0x7fffffU) | (ce == 0 ? 0 : 0x800000U);
        const int32_t pe = static_cast<int32_t>(Large(ae, 1U) + Large(be, 1U)) - 300;
        const int32_t se = static_cast<int32_t>(Large(ce, 1U)) - 150;
        const int32_t base = Large(pe + 48, se + 24) - 61;
        const uint64_t sum = ShiftJam(am * bm, pe - base) + ShiftJam(cm, se - base);
        if (sum == 0) {
            return 0.0f;
        }
        uint32_t top = 63 - __builtin_clzll(sum);
        const uint32_t shift = top - 23;
        uint64_t rounded = sum >> shift;
        const uint64_t remainder = sum & ((1ULL << shift) - 1);
        const uint64_t halfway = 1ULL << (shift - 1);
        rounded += remainder > halfway || (remainder == halfway && (rounded & 1) != 0);
        if (rounded == 0x1000000ULL) {
            rounded >>= 1;
            ++top;
        }
        FloatBits result{};
        result.bits = (static_cast<uint32_t>(base + top + 127) << 23) | (rounded & 0x7fffffU);
        return result.value;
    }

    // 有符号 int16 直接转 fp32 后补偿符号，精确保留 uint16 的全部有效位。
    __aicore__ inline void UnsignedToFloat(LocalTensor<float> dst, LocalTensor<uint16_t> src,
                                           LocalTensor<float> temporary, uint32_t maskOffset, uint32_t count)
    {
        auto mask = View<uint8_t>(maskOffset);
        Cast(dst, src.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, dst, 0.0f, CMPMODE::LT, count);
        Adds(temporary, dst, 65536.0f, count);
        PipeBarrier<PIPE_V>();
        Select(dst, mask, temporary, dst, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void EncodeUnsigned(LocalTensor<uint16_t> dst, LocalTensor<float> values,
                                          LocalTensor<float> temporary, LocalTensor<int32_t> integers,
                                          uint32_t maskOffset, uint32_t count)
    {
        auto mask = View<uint8_t>(maskOffset);
        Maxs(values, values, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Mins(values, values, static_cast<float>(static_cast<int32_t>(d_->bins - 1)), count);
        PipeBarrier<PIPE_V>();
        Cast(integers, values, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(values, integers, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, values, 32768.0f, CMPMODE::GE, count);
        Adds(temporary, values, -65536.0f, count);
        PipeBarrier<PIPE_V>();
        Select(values, mask, temporary, values, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        Cast(dst.template ReinterpretCast<int16_t>(), values, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildLut16(uint64_t tile)
    {
        auto histogram = View<uint16_t>(0);
        const uint32_t clipped = CountTile(tile, histogram, 0, 65536);
        const uint32_t batch = clipped / 65536;
        const uint32_t residual = clipped % 65536;
        const uint32_t step = residual == 0 ? 1 : Large(65536U / residual, 1U);
        for (uint32_t i = 0; i < residual; ++i) {
            const uint32_t bin = i * step;
            histogram.SetValue(bin, histogram.GetValue(bin) + 1);
        }
        const uint32_t area = d_->tileWidth * d_->tileHeight;
        if (area <= (1U << 24)) {
            auto values = View<float>(139264);
            auto prefix = View<float>(140288);
            auto temporary = View<float>(141312);
            auto integers = View<int32_t>(142336);
            auto scratch = View<uint8_t>(147456);
            auto last = prefix;
            uint32_t carry = 0;
            for (uint32_t first = 0; first < 65536; first += PREFIX_BINS) {
                Fence<HardEvent::S_V>();
                UnsignedToFloat(values, histogram[first], temporary, 143360, PREFIX_BINS);
                Adds(values, values, static_cast<float>(static_cast<int32_t>(batch)), PREFIX_BINS);
                PipeBarrier<PIPE_V>();
                CumSum<float, PREFIX_CONFIG>(prefix, last, values, scratch, CumSumInfo{1, PREFIX_BINS});
                PipeBarrier<PIPE_V>();
                Adds(prefix, prefix, static_cast<float>(static_cast<int32_t>(carry)), PREFIX_BINS);
                Fence<HardEvent::V_S>();
                carry = static_cast<uint32_t>(static_cast<int32_t>(prefix.GetValue(PREFIX_BINS - 1)));
                Fence<HardEvent::S_V>();
                Muls(prefix, prefix, d_->lutScale, PREFIX_BINS);
                PipeBarrier<PIPE_V>();
                EncodeUnsigned(histogram[first], prefix, temporary, integers, 143360, PREFIX_BINS);
                Fence<HardEvent::V_S>();
            }
            Fence<HardEvent::V_MTE3>();
        } else {
            uint32_t sum = 0;
            for (uint32_t i = 0; i < 65536; ++i) {
                sum += histogram.GetValue(i) + batch;
                histogram.SetValue(i, RoundLut(sum));
            }
            Fence<HardEvent::S_MTE3>();
        }
        DataCopy(lut_[tile * 65536], histogram, 65536);
        Fence<HardEvent::MTE3_S>();
        Fence<HardEvent::MTE3_V>();
    }

    __aicore__ inline void BuildLut(uint64_t tile)
    {
        if constexpr (sizeof(T) == 2) {
            if (d_->wideHistogram == 0) {
                BuildLut16(tile);
                return;
            }
        }
        const uint32_t segment = Small(SEGMENT_BINS, d_->bins);
        auto histogram = View<int32_t>(0);
        auto result = View<uint16_t>(32768);
        const uint64_t scratchBase = static_cast<uint64_t>(GetBlockIdx()) * 65536;
        uint32_t clipped = 0;
        for (uint32_t first = 0; first < d_->bins; first += segment) {
            clipped += CountTile(tile, histogram, first, segment);
            if (d_->wideHistogram != 0) {
                Fence<HardEvent::S_MTE3>();
                DataCopy(histGm_[scratchBase + first], histogram, segment);
                Fence<HardEvent::MTE3_S>();
                Fence<HardEvent::MTE3_V>();
            }
        }
        const uint32_t batch = clipped / d_->bins;
        const uint32_t residual = clipped % d_->bins;
        const uint32_t step = residual == 0 ? 1 : Large(d_->bins / residual, 1U);
        uint32_t sum = 0;
        for (uint32_t first = 0; first < d_->bins; first += segment) {
            if (d_->wideHistogram != 0) {
                DataCopy(histogram, histGm_[scratchBase + first], segment);
                Fence<HardEvent::MTE2_S>();
            }
            for (uint32_t i = 0; i < segment; ++i) {
                sum += histogram.GetValue(i) + batch + (ResidualBin(first + i, residual, step) ? 1 : 0);
                result.SetValue(i, RoundLut(sum));
            }
            Fence<HardEvent::S_MTE3>();
            DataCopy(lut_[tile * d_->bins + first], result, segment);
            Fence<HardEvent::MTE3_S>();
            Fence<HardEvent::MTE3_V>();
        }
    }

    // 按实际 fp32 坐标调整插值分界，覆盖中心落点附近的舍入情况。
    __aicore__ inline uint32_t Boundary(uint32_t index, uint32_t tileSize, uint32_t length, float inverse)
    {
        if (index == 0) {
            return 0;
        }
        uint32_t cut = Small(static_cast<uint64_t>(length),
                             (static_cast<uint64_t>(tileSize) * (2ULL * index - 1) + 1) / 2);
        const int32_t threshold = index - 1;
        while (cut < length && FloorIndex(static_cast<float>(static_cast<int32_t>(cut)) * inverse - 0.5f) < threshold) {
            ++cut;
        }
        while (cut > 0 && FloorIndex(static_cast<float>(static_cast<int32_t>(cut - 1)) * inverse - 0.5f) >= threshold) {
            --cut;
        }
        return cut;
    }

    __aicore__ inline void InterpolateRegion(uint64_t region)
    {
        const uint32_t ix = region % (d_->tilesX + 1);
        const uint32_t iy = region / (d_->tilesX + 1) % (d_->tilesY + 1);
        const uint64_t batch = region / ((static_cast<uint64_t>(d_->tilesX) + 1) * (d_->tilesY + 1));
        const uint32_t left = Boundary(ix, d_->tileWidth, d_->width, d_->invTileWidth);
        const uint32_t right = Boundary(ix + 1, d_->tileWidth, d_->width, d_->invTileWidth);
        const uint32_t top = Boundary(iy, d_->tileHeight, d_->height, d_->invTileHeight);
        const uint32_t bottom = Boundary(iy + 1, d_->tileHeight, d_->height, d_->invTileHeight);
        if (left >= right || top >= bottom) {
            return;
        }
        const uint32_t tx1 = ix == 0 ? 0 : ix - 1;
        const uint32_t tx2 = Small(ix, d_->tilesX - 1);
        const uint32_t ty1 = iy == 0 ? 0 : iy - 1;
        const uint32_t ty2 = Small(iy, d_->tilesY - 1);
        const uint64_t tileBase = batch * d_->tilesX * d_->tilesY;
        const uint64_t ids[4] = {tileBase + static_cast<uint64_t>(ty1) * d_->tilesX + tx1,
                                 tileBase + static_cast<uint64_t>(ty1) * d_->tilesX + tx2,
                                 tileBase + static_cast<uint64_t>(ty2) * d_->tilesX + tx1,
                                 tileBase + static_cast<uint64_t>(ty2) * d_->tilesX + tx2};
        for (uint32_t col = left; col < right; col += MAX_PIXELS) {
            const uint32_t width = Small(MAX_PIXELS, right - col);
            const uint32_t align = 32 / sizeof(T);
            const uint32_t stride = (width + align - 1) / align * align;
            const uint32_t rowsPerChunk = MAX_PIXELS / stride;
            for (uint32_t row = top; row < bottom; row += rowsPerChunk) {
                const uint32_t rows = Small(rowsPerChunk, bottom - row);
                InterpolateChunk(batch, row, col, rows, width, stride, ids);
            }
        }
    }

    __aicore__ inline void InterpolateChunk(uint64_t batch, uint32_t row, uint32_t col, uint32_t rows, uint32_t width,
                                            uint32_t stride, const uint64_t* ids)
    {
        // CompareScalar 的 count 接口要求 fp32 元素数为 64 的倍数。
        const uint32_t count = (rows * stride + 63) / 64 * 64;
        auto input = View<T>(INPUT_OFFSET);
        auto table = View<uint16_t>(0);
        auto f0 = View<float>(135168);
        auto f1 = View<float>(139264);
        auto f2 = View<float>(143360);
        auto f3 = View<float>(147456);
        auto xa = View<float>(151552);
        auto ya = View<float>(155648);
        auto temp = View<float>(159744);
        auto offsets = View<uint32_t>(167936);
        auto integers = View<int32_t>(172032);
        auto gathered = View<uint16_t>(177152);
        const uint64_t position = batch * d_->imageElements + static_cast<uint64_t>(row) * d_->width + col;
        // 向量尾部也参与 LUT gather，因此先将所有填充 lane 初始化为有效灰度 0。
        Duplicate(input.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), count * sizeof(T) / 2);
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams read{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(T)),
                               static_cast<uint32_t>((d_->width - width) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(input, x_[position], read, padding);
        Fence<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) == 1) {
            auto halfValues = f1.template ReinterpretCast<half>();
            Cast(halfValues, input, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(f0, halfValues, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        } else {
            UnsignedToFloat(f0, input, f1, 176128, count);
        }
        Muls(f0, f0, 2.0f, count);
        PipeBarrier<PIPE_V>();
        Cast(offsets.template ReinterpretCast<int32_t>(), f0, RoundMode::CAST_RINT, count);
        Duplicate(xa, 0.0f, count);
        Duplicate(ya, 0.0f, count);
        Fence<HardEvent::V_S>();
        for (uint32_t r = 0; r < rows; ++r) {
            const float y = static_cast<float>(static_cast<int32_t>(row + r)) * d_->invTileHeight - 0.5f;
            const float weightY = y - static_cast<float>(static_cast<int32_t>(FloorIndex(y)));
            for (uint32_t c = 0; c < width; ++c) {
                const float x = static_cast<float>(static_cast<int32_t>(col + c)) * d_->invTileWidth - 0.5f;
                xa.SetValue(r * stride + c, x - static_cast<float>(static_cast<int32_t>(FloorIndex(x))));
                ya.SetValue(r * stride + c, weightY);
            }
        }
        Fence<HardEvent::S_V>();
        for (uint32_t k = 0; k < 4; ++k) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(table, lut_[ids[k] * d_->bins], d_->bins);
            Fence<HardEvent::MTE2_V>();
            Gather(gathered, table, offsets, 0, count);
            PipeBarrier<PIPE_V>();
            auto target = View<float>(135168 + k * 4096);
            if constexpr (sizeof(T) == 1) {
                Cast(target, gathered.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
            } else {
                UnsignedToFloat(target, gathered, temp, 176128, count);
            }
        }
        Muls(temp, xa, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(temp, temp, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Mul(f1, f1, xa, count);
        Mul(f3, f3, xa, count);
        PipeBarrier<PIPE_V>();
        // 对齐 aarch64 OpenCV 的乘加收缩顺序，保留半整数附近的舍入结果。
        FusedMulAdd(f0, temp, f1, count);
        FusedMulAdd(f2, temp, f3, count);
        Muls(temp, ya, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(temp, temp, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Mul(f2, f2, ya, count);
        PipeBarrier<PIPE_V>();
        FusedMulAdd(f0, temp, f2, count);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        const float threshold = static_cast<float>(static_cast<int32_t>(d_->bins)) * (1.0f / 2097152.0f);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < width; ++c) {
                const uint32_t index = r * stride + c;
                const float value = f0.GetValue(index);
                const float fraction = value - static_cast<float>(static_cast<int32_t>(value));
                if (fraction >= 0.5f - threshold && fraction <= 0.5f + threshold) {
                    const uint32_t gray = input.GetValue(index);
                    const float wx = xa.GetValue(index);
                    const float wy = ya.GetValue(index);
                    const float a = static_cast<float>(static_cast<int32_t>(lut_.GetValue(ids[0] * d_->bins + gray)));
                    const float b = static_cast<float>(static_cast<int32_t>(lut_.GetValue(ids[1] * d_->bins + gray)));
                    const float c0 = static_cast<float>(static_cast<int32_t>(lut_.GetValue(ids[2] * d_->bins + gray)));
                    const float d0 = static_cast<float>(static_cast<int32_t>(lut_.GetValue(ids[3] * d_->bins + gray)));
                    const float upper = ExactMultiplyAdd(a, 1.0f - wx, b * wx);
                    const float lower = ExactMultiplyAdd(c0, 1.0f - wx, d0 * wx);
                    f0.SetValue(index, ExactMultiplyAdd(upper, 1.0f - wy, lower * wy));
                }
            }
        }
        Fence<HardEvent::S_V>();
        auto output = View<T>(177152);
        if constexpr (sizeof(T) == 2) {
            EncodeUnsigned(output, f0, temp, integers, 176128, count);
        } else {
            Cast(integers, f0, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(f0, integers, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            auto halfValues = f1.template ReinterpretCast<half>();
            Cast(halfValues, f0, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(output, halfValues, RoundMode::CAST_RINT, count);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams write{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(T)), 0,
                                static_cast<uint32_t>((d_->width - width) * sizeof(T)), 0};
        DataCopyPad(y_[position], output, write);
        Fence<HardEvent::MTE3_MTE2>();
        Fence<HardEvent::MTE3_V>();
    }

    const ClaheTilingData* d_ = nullptr;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> arena_;
    GlobalTensor<T> x_;
    GlobalTensor<T> y_;
    GlobalTensor<uint16_t> lut_;
    GlobalTensor<int32_t> histGm_;
};
} // namespace NsClahe
#endif
