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
 * \file hog_descriptor_impl.hpp
 * \brief HOGDescriptor vector kernel for Ascend 910B (AIV).
 *
 * Semantics (v1; see README.md):
 *   pad-first: image replicated right/bottom to nCells*p, all stages run on the padded grid;
 *   gradient: [-1,0,1] center difference with replicate clamp borders;
 *   C=3: per-channel gradients, first-max mag^2 channel wins (cv-compatible);
 *   angle: unsigned [0,pi) = atan(gy/gx) + floored-mod; signed [0,2pi) adds quadrant fix;
 *   binning: u = theta*nBins/range - 0.5, bilinear split to 2 adjacent bins, circular wrap (180 -> bin 0);
 *   spatial: bilinear to 4 cells on cell-center lattice ((c+0.5)/p - 0.5), edge votes folded back;
 *   block norm: L1 / L1-sqrt / L2 / L2-Hys with skimage eps forms (L2: x/sqrt(sum+eps^2));
 *   output: block row-major -> cell row-major -> bin ascending.
 *
 * Hardware rules inherited from the BilateralFilter3D 910B experience (all device-proven):
 *   - every vector-op UB operand (incl. offset views) must be 32B aligned; non-aligned offset
 *     VIEW WRITES are forbidden (silent data loss) - full-tensor writes only;
 *   - GM source addresses of DataCopyPad are arbitrary -> shifted neighborhood reads;
 *   - scalar SetValue/GetValue on UB tensors is always safe;
 *   - fp16->fp32 Cast must use CAST_NONE; queue slots must be zeroed before first use.
 *
 * No Compare/Select/Atan2 APIs are used anywhere:
 *   - equality indicators: I(x==k) = 1 - min((x-k)^2, 1);
 *   - signed angles:       hard-zero tiny gx before scaled gx indicators; compare gy exactly;
 *   - floored modulo:      x - range*floor(x/range) via the vector Floor API;
 *   - NaN-free atan:       denominator gets a +1e-30 guard, 0/0 becomes 0.
 */

#ifndef HOG_DESCRIPTOR_IMPL_H_
#define HOG_DESCRIPTOR_IMPL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hog_descriptor_tiling_def.h"
#include <type_traits>

namespace NsHogDescriptor {
using namespace AscendC;

constexpr float HOG_PI = 3.14159274f;
constexpr float HOG_2PI = 6.28318531f;
constexpr float HOG_SIGNED_GX_MIN = 1e-30f;

__aicore__ inline int32_t HogFloorToI(float v)
{
    int32_t i = static_cast<int32_t>(v);
    if (static_cast<float>(i) > v) {
        --i;
    }
    return i;
}

__aicore__ inline int32_t HogClampI(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// The aicore compiler forbids direct unsigned<->float casts; route through int32.
__aicore__ inline float HogU2F(uint32_t v) { return static_cast<float>(static_cast<int32_t>(v)); }

template <typename xType>
class HogDescriptorVectorKernel {
public:
    __aicore__ inline HogDescriptorVectorKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const optiling::HogDescriptorTilingData* t)
    {
        t_ = *t;
        nImg_ = t_.n;
        h_ = t_.h;
        w_ = t_.w;
        c_ = t_.c;
        pH_ = t_.pH;
        pW_ = t_.pW;
        bH_ = t_.bH;
        bW_ = t_.bW;
        sH_ = t_.sH;
        sW_ = t_.sW;
        nBins_ = t_.nBins;
        nCellsH_ = t_.nCellsH;
        nCellsW_ = t_.nCellsW;
        nBlocksH_ = t_.nBlocksH;
        nBlocksW_ = t_.nBlocksW;
        padH_ = t_.padH;
        padW_ = t_.padW;
        nBinsPad_ = t_.nBinsPad;
        taskCnt_ = t_.taskCnt;
        usedCoreNum_ = t_.usedCoreNum;
        nSeg_ = nCellsW_; // v1: one full-width strip, segments = cell columns

        xGm_.SetGlobalBuffer((__gm__ xType*)x, static_cast<uint64_t>(nImg_) * h_ * w_ * c_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(nImg_) * t_.dimD);

        const uint32_t stripBytes = padW_ * sizeof(float);
        const uint32_t nSeg8 = (nSeg_ + 7U) / 8U * 8U;
        stageCnt_ = bH_ * bW_ * nBinsPad_;
        const uint32_t stageBytes = stageCnt_ * sizeof(float);

        // Full-row loads (slot must hold one interleaved HWC row for C=3).
        pipe_.InitBuffer(rowQue_, 2, padW_ * c_ * sizeof(xType));
        // Block copy-out (VECOUT queue: MTE3 must not read VECCALC TBufs directly).
        pipe_.InitBuffer(outQue_, 4, nBinsPad_ * sizeof(float));
        // Cell histogram plane: bandRows * nSeg * nBinsPad fp32 (scalar-accessed).
        bandRows_ = (t_.bandLen - 1U) * sH_ + bH_;
        pipe_.InitBuffer(planeBuf_, bandRows_ * nSeg_ * nBinsPad_ * sizeof(float));
        // fp32 strip working set.
        pipe_.InitBuffer(rowMBuf_, stripBytes);
        pipe_.InitBuffer(rowPBuf_, stripBytes);
        pipe_.InitBuffer(rowYBuf_, stripBytes);
        pipe_.InitBuffer(gxABuf_, stripBytes);
        pipe_.InitBuffer(gyABuf_, stripBytes);
        pipe_.InitBuffer(gxBBuf_, stripBytes);
        pipe_.InitBuffer(gyBBuf_, stripBytes);
        pipe_.InitBuffer(m2aBuf_, stripBytes);
        pipe_.InitBuffer(m2bBuf_, stripBytes);
        pipe_.InitBuffer(uBuf_, stripBytes);
        pipe_.InitBuffer(fuBuf_, stripBytes);
        pipe_.InitBuffer(aBuf_, stripBytes);
        pipe_.InitBuffer(omABuf_, stripBytes);
        pipe_.InitBuffer(b0Buf_, stripBytes);
        pipe_.InitBuffer(b1Buf_, stripBytes);
        pipe_.InitBuffer(t0Buf_, stripBytes);
        pipe_.InitBuffer(t1Buf_, stripBytes);
        pipe_.InitBuffer(t2Buf_, stripBytes);
        pipe_.InitBuffer(t3Buf_, stripBytes);
        pipe_.InitBuffer(oneBuf_, stripBytes);
        pipe_.InitBuffer(zeroBuf_, stripBytes);
        pipe_.InitBuffer(tinyBuf_, stripBytes);
        // Segment sums / block staging.
        pipe_.InitBuffer(s0Buf_, nSeg8 * sizeof(float));
        pipe_.InitBuffer(s1Buf_, nSeg8 * sizeof(float));
        pipe_.InitBuffer(stageBuf_, stageBytes);
        pipe_.InitBuffer(normTmpBuf_, stageBytes);
        pipe_.InitBuffer(threshBuf_, stageBytes);
        pipe_.InitBuffer(sumBuf_, 8 * sizeof(float));
        pipe_.InitBuffer(invBuf_, 8 * sizeof(float));

        Duplicate(oneBuf_.Get<float>(), 1.0f, padW_);
        Duplicate(zeroBuf_.Get<float>(), 0.0f, padW_);
        Duplicate(tinyBuf_.Get<float>(), 1e-30f, padW_);
        Duplicate(invBuf_.Get<float>(), 0.0f, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        // Grid-stride over tasks; usedCoreNum comes from tiling (GetBlockNum() is unreliable here).
        for (uint32_t task = GetBlockIdx(); task < taskCnt_; task += usedCoreNum_) {
            ProcessTask(task);
        }
    }

private:
    // Load one channel row (clamped to real rows/cols) into an fp32 strip buffer, applying
    // replicate padding for virtual columns [w, padW) and optional gamma sqrt.
    __aicore__ inline void LoadRowF32(LocalTensor<float>& dst, int32_t rowY, int32_t ch)
    {
        const int32_t row = HogClampI(rowY, 0, static_cast<int32_t>(h_) - 1);
        const uint64_t rowBase = imgBase_ + static_cast<uint64_t>(row) * w_ * c_ + ch;
        LocalTensor<xType> raw = rowQue_.AllocTensor<xType>();
        if (c_ == 1U) {
            DataCopyExtParams cp{1, static_cast<uint32_t>(w_ * sizeof(xType)), 0, 0, 0};
            DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
            DataCopyPad(raw, xGm_[rowBase], cp, pad);
            rowQue_.EnQue(raw);
            raw = rowQue_.DeQue<xType>();
            if constexpr (std::is_same<xType, float>::value) {
                Adds(dst, raw, 0.0f, padW_);
            } else {
                Cast(dst, raw, RoundMode::CAST_NONE, padW_);
            }
            rowQue_.FreeTensor(raw);
            PipeBarrier<PIPE_V>();
        } else {
            // HWC: load the whole interleaved row, then de-interleave the channel scalarly
            // (byte-granular strided 2D copies proved unreliable on this chip).
            DataCopyExtParams cp{1, static_cast<uint32_t>(w_ * c_ * sizeof(xType)), 0, 0, 0};
            DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
            DataCopyPad(raw, xGm_[imgBase_ + static_cast<uint64_t>(row) * w_ * c_], cp, pad);
            rowQue_.EnQue(raw);
            raw = rowQue_.DeQue<xType>();
            for (uint32_t i = 0; i < w_; ++i) {
                dst.SetValue(i, static_cast<float>(raw.GetValue(i * c_ + static_cast<uint32_t>(ch))));
            }
            rowQue_.FreeTensor(raw);
            PipeBarrier<PIPE_V>();
        }
        if (t_.gammaFlag != 0U) {
            Sqrt(dst, dst, w_); // vector gamma on real columns only (scalar tail comes after)
            PipeBarrier<PIPE_V>();
        }
        // Replicate the last real column into virtual padding lanes (scalar -> barrier before V reads).
        const float lastCol = dst.GetValue(w_ - 1);
        for (uint32_t i = w_; i < padW_; ++i) {
            dst.SetValue(i, lastCol);
        }
        PipeBarrier<PIPE_V>();
    }

    // gx = I[clamp(c+1)] - I[clamp(c-1)], scalar per column (v1; vector tile loads are a
    // backlog item). Reads the already-loaded center row, writes the gradient strip.
    __aicore__ inline void ScalarGx(LocalTensor<float>& gx, LocalTensor<float>& rowY)
    {
        const int32_t last = static_cast<int32_t>(w_) - 1;
        for (uint32_t c = 0; c < padW_; ++c) {
            const int32_t cm = HogClampI(static_cast<int32_t>(c) - 1, 0, last);
            const int32_t cp = HogClampI(static_cast<int32_t>(c) + 1, 0, last);
            gx.SetValue(c, rowY.GetValue(cp) - rowY.GetValue(cm));
        }
        PipeBarrier<PIPE_V>();
    }

    // Per-channel gradient squared magnitude into m2, gradients into gx/gy.
    __aicore__ inline void ChannelGrad(int32_t y, int32_t ch, LocalTensor<float>& gx, LocalTensor<float>& gy,
                                       LocalTensor<float>& m2)
    {
        LocalTensor<float> rowM = rowMBuf_.Get<float>();
        LocalTensor<float> rowP = rowPBuf_.Get<float>();
        LocalTensor<float> rowY = rowYBuf_.Get<float>();
        LoadRowF32(rowM, y - 1, ch);
        LoadRowF32(rowP, y + 1, ch);
        LoadRowF32(rowY, y, ch);
        Sub(gy, rowP, rowM, padW_);
        PipeBarrier<PIPE_V>();
        ScalarGx(gx, rowY);
        Mul(m2, gx, gx, padW_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> t0 = t0Buf_.Get<float>();
        Mul(t0, gy, gy, padW_);
        PipeBarrier<PIPE_V>();
        Add(m2, m2, t0, padW_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessTask(uint32_t task)
    {
        const uint32_t img = task / t_.bandsPerImg;
        const uint32_t band = task % t_.bandsPerImg;
        imgBase_ = static_cast<uint64_t>(img) * h_ * w_ * c_;
        const uint32_t bStart = band * t_.bandLen;
        const uint32_t bLen = (bStart + t_.bandLen <= nBlocksH_) ? t_.bandLen : (nBlocksH_ - bStart);
        const int32_t cellStart = static_cast<int32_t>(bStart * sH_);
        const int32_t cellEnd = static_cast<int32_t>((bStart + bLen - 1U) * sH_ + bH_);

        LocalTensor<float> plane = planeBuf_.Get<float>();
        const uint32_t planeFloats = bandRows_ * nSeg_ * nBinsPad_;
        Duplicate(plane, 0.0f, planeFloats);
        PipeBarrier<PIPE_V>();

        HistogramPass(img, cellStart, cellEnd, plane);
        BlockPass(img, bStart, bLen, cellStart, plane);
    }

    __aicore__ inline void HistogramPass(uint32_t img, int32_t cellStart, int32_t cellEnd, LocalTensor<float>& plane)
    {
        LocalTensor<float> gxA = gxABuf_.Get<float>();
        LocalTensor<float> gyA = gyABuf_.Get<float>();
        LocalTensor<float> gxB = gxBBuf_.Get<float>();
        LocalTensor<float> gyB = gyBBuf_.Get<float>();
        LocalTensor<float> m2a = m2aBuf_.Get<float>();
        LocalTensor<float> m2b = m2bBuf_.Get<float>();
        LocalTensor<float> u = uBuf_.Get<float>();
        LocalTensor<float> fu = fuBuf_.Get<float>();
        LocalTensor<float> aV = aBuf_.Get<float>();
        LocalTensor<float> omA = omABuf_.Get<float>();
        LocalTensor<float> b0w = b0Buf_.Get<float>();
        LocalTensor<float> b1w = b1Buf_.Get<float>();
        LocalTensor<float> t0 = t0Buf_.Get<float>();
        LocalTensor<float> t1 = t1Buf_.Get<float>();
        LocalTensor<float> t2 = t2Buf_.Get<float>();
        LocalTensor<float> t3 = t3Buf_.Get<float>();
        LocalTensor<float> oneV = oneBuf_.Get<float>();
        LocalTensor<float> zeroV = zeroBuf_.Get<float>();
        LocalTensor<float> tinyV = tinyBuf_.Get<float>();
        LocalTensor<float> rowM = rowMBuf_.Get<float>();
        LocalTensor<float> rowP = rowPBuf_.Get<float>();

        const float nBinsF = HogU2F(nBins_);
        const float range = (t_.signedFlag != 0U) ? HOG_2PI : HOG_PI;
        const float invRange = 1.0f / range;
        const int32_t yLo = HogClampI(cellStart * static_cast<int32_t>(pH_) - static_cast<int32_t>(pH_), 0,
                                      static_cast<int32_t>(padH_));
        const int32_t yHi = HogClampI(cellEnd * static_cast<int32_t>(pH_) + static_cast<int32_t>(pH_), 0,
                                      static_cast<int32_t>(padH_));

        for (int32_t y = yLo; y < yHi; ++y) {
            // ---- gradients + winning channel ----
            ChannelGrad(y, 0, gxA, gyA, m2a);
            for (int32_t ch = 1; ch < static_cast<int32_t>(c_); ++ch) {
                ChannelGrad(y, ch, gxB, gyB, m2b);
                // strict-greater indicator (first channel wins ties, cv-style)
                Sub(t0, m2b, m2a, padW_);
                PipeBarrier<PIPE_V>();
                Muls(t0, t0, 1e38f, padW_);
                PipeBarrier<PIPE_V>();
                Min(t1, t0, oneV, padW_);
                PipeBarrier<PIPE_V>();
                Max(t2, t1, zeroV, padW_);
                PipeBarrier<PIPE_V>();
                // blend winner
                Mul(t0, gxB, t2, padW_);
                PipeBarrier<PIPE_V>();
                Sub(t1, oneV, t2, padW_);
                PipeBarrier<PIPE_V>();
                Mul(t1, t1, gxA, padW_);
                PipeBarrier<PIPE_V>();
                Add(gxA, t0, t1, padW_);
                PipeBarrier<PIPE_V>();
                Mul(t0, gyB, t2, padW_);
                PipeBarrier<PIPE_V>();
                Sub(t1, oneV, t2, padW_);
                PipeBarrier<PIPE_V>();
                Mul(t1, t1, gyA, padW_);
                PipeBarrier<PIPE_V>();
                Add(gyA, t0, t1, padW_);
                PipeBarrier<PIPE_V>();
                Max(m2a, m2a, m2b, padW_);
                PipeBarrier<PIPE_V>();
            }
            Sqrt(m2a, m2a, padW_); // m2a now holds mag
            PipeBarrier<PIPE_V>();

            // Signed angles: retain the original magnitude, but hard-zero tiny gx.
            // gx is then zero or |gx| >= 1e-30, so its scaled indicators are exactly 0/1.
            if (t_.signedFlag != 0U) {
                PipeBarrier<PIPE_ALL>();
                for (uint32_t column = 0; column < padW_; ++column) {
                    const float gx = gxA.GetValue(column);
                    if (gx > -HOG_SIGNED_GX_MIN && gx < HOG_SIGNED_GX_MIN) {
                        gxA.SetValue(column, 0.0f);
                    }
                    t1.SetValue(column, gyA.GetValue(column) > 0.0f ? 1.0f : 0.0f);
                }
                PipeBarrier<PIPE_ALL>();
            }

            // ---- angle: NaN-free atan + floored modulo ----
            Abs(rowM, gxA, padW_);
            PipeBarrier<PIPE_V>();
            Muls(rowM, rowM, 1e38f, padW_);
            PipeBarrier<PIPE_V>();
            Min(rowM, rowM, oneV, padW_);
            PipeBarrier<PIPE_V>();
            Sub(rowP, oneV, rowM, padW_); // ind0: |gx| == 0
            Mul(rowP, rowP, tinyV, padW_);
            PipeBarrier<PIPE_V>();
            Add(rowP, gxA, rowP, padW_); // denominator, never exactly zero in practice
            Div(rowM, gyA, rowP, padW_);
            PipeBarrier<PIPE_V>();
            Atan(rowM, rowM, padW_); // theta raw in (-pi/2, pi/2]
            PipeBarrier<PIPE_V>();
            if (t_.signedFlag != 0U) {
                Muls(t0, gxA, -1e38f, padW_);
                PipeBarrier<PIPE_V>();
                Min(t0, t0, oneV, padW_);
                PipeBarrier<PIPE_V>();
                Max(t0, t0, zeroV, padW_); // negGx
                // t1 already contains the exact I(gy > 0), including subnormal gy.
                Muls(t1, t1, 2.0f, padW_);
                PipeBarrier<PIPE_V>();
                Adds(t1, t1, -1.0f, padW_);
                PipeBarrier<PIPE_V>();
                Mul(t1, t1, t0, padW_);
                PipeBarrier<PIPE_V>();
                Muls(t1, t1, HOG_PI, padW_);
                PipeBarrier<PIPE_V>();
                Add(rowM, rowM, t1, padW_);
                PipeBarrier<PIPE_V>();
            }
            Adds(u, rowM, 0.0f, padW_);
            PipeBarrier<PIPE_V>();
            Muls(rowM, u, invRange, padW_);
            PipeBarrier<PIPE_V>();
            Floor(rowM, rowM, padW_);
            PipeBarrier<PIPE_V>();
            Muls(rowM, rowM, -range, padW_);
            PipeBarrier<PIPE_V>();
            Add(u, u, rowM, padW_); // theta in [0, range)
            PipeBarrier<PIPE_V>();

            // ---- bin split: u = theta*scale - 0.5, b0/b1 circular wrap ----
            Muls(u, u, t_.binScale, padW_);
            PipeBarrier<PIPE_V>();
            Adds(u, u, -0.5f, padW_);
            PipeBarrier<PIPE_V>();
            Floor(fu, u, padW_);
            PipeBarrier<PIPE_V>();
            Sub(aV, u, fu, padW_); // a in [0,1)
            PipeBarrier<PIPE_V>();
            Sub(omA, oneV, aV, padW_);
            PipeBarrier<PIPE_V>();
            Muls(t0, fu, -1e38f, padW_);
            PipeBarrier<PIPE_V>();
            Min(t0, t0, oneV, padW_);
            PipeBarrier<PIPE_V>();
            Max(t0, t0, zeroV, padW_); // I(fu < 0), only -1 possible
            Muls(t0, t0, nBinsF, padW_);
            PipeBarrier<PIPE_V>();
            Add(b0w, fu, t0, padW_); // wrapped b0
            Adds(t1, b0w, 1.0f, padW_);
            PipeBarrier<PIPE_V>();
            Adds(t2, t1, -nBinsF, padW_);
            PipeBarrier<PIPE_V>();
            Mul(t2, t2, t2, padW_);
            PipeBarrier<PIPE_V>();
            Min(t2, t2, oneV, padW_);
            PipeBarrier<PIPE_V>();
            Sub(t3, oneV, t2, padW_); // I(b1raw == nBins)
            Muls(t3, t3, nBinsF, padW_);
            PipeBarrier<PIPE_V>();
            Sub(b1w, t1, t3, padW_); // wrapped b1
            PipeBarrier<PIPE_V>();

            // ---- row spatial weights (scalars, with edge fold-back) ----
            float rA = 0.0f;
            float rB = 0.0f;
            int32_t i0 = 0;
            int32_t i1 = 0;
            {
                const float v = (static_cast<float>(y) + 0.5f) / HogU2F(pH_) - 0.5f;
                i0 = HogFloorToI(v);
                const float fy = v - static_cast<float>(i0);
                i1 = i0 + 1;
                rA = 1.0f - fy;
                rB = fy;
                if (i0 < 0) {
                    rB = 1.0f;
                    rA = 0.0f;
                }
                if (i1 >= static_cast<int32_t>(nCellsH_)) {
                    rA = 1.0f;
                    rB = 0.0f;
                }
            }

            // ---- scalar vote scatter (deterministic; the vector chain above is read back
            //      exactly once per row after a barrier, the plane is scalar-only) ----
            const bool hasI0 = (i0 >= cellStart) && (i0 < cellEnd);
            const bool hasI1 = (i1 >= cellStart) && (i1 < cellEnd);
            const uint32_t rowF0 = hasI0 ? static_cast<uint32_t>(i0 - cellStart) * nSeg_ * nBinsPad_ : 0U;
            const uint32_t rowF1 = hasI1 ? static_cast<uint32_t>(i1 - cellStart) * nSeg_ * nBinsPad_ : 0U;
            const float invPW = 1.0f / HogU2F(pW_);
            const int32_t lastSeg = static_cast<int32_t>(nSeg_) - 1;
            for (uint32_t c = 0; c < padW_; ++c) {
                const float magV = m2a.GetValue(c);
                if (magV == 0.0f) {
                    continue;
                }
                const float aVal = aV.GetValue(c);
                const float omAv = omA.GetValue(c);
                const int32_t b0 = static_cast<int32_t>(b0w.GetValue(c));
                const int32_t b1 = static_cast<int32_t>(b1w.GetValue(c));
                // column bilinear with edge fold-back (same math as the row weights)
                const float vc = (HogU2F(c) + 0.5f) * invPW - 0.5f;
                const int32_t j0f = HogFloorToI(vc);
                const float fx = vc - static_cast<float>(j0f);
                int32_t j1 = j0f + 1;
                float ca = 1.0f - fx;
                float cb = fx;
                if (j0f < 0) {
                    ca = 0.0f;
                    cb = 1.0f;
                }
                if (j1 > lastSeg) {
                    ca += cb;
                    cb = 0.0f;
                    j1 = lastSeg;
                }
                const int32_t j0 = j0f < 0 ? 0 : j0f;
                const uint32_t colA = static_cast<uint32_t>(j0) * nBinsPad_;
                const uint32_t colB = static_cast<uint32_t>(j1) * nBinsPad_;
                const float mBase = magV;
                for (uint32_t k = 0; k < nBins_; ++k) {
                    const int32_t kf = static_cast<int32_t>(k);
                    float wk = 0.0f;
                    if (b0 == kf) {
                        wk += omAv;
                    }
                    if (b1 == kf) {
                        wk += aVal;
                    }
                    if (wk == 0.0f) {
                        continue;
                    }
                    const float m = mBase * wk;
                    if (hasI0 && rA != 0.0f) {
                        if (ca != 0.0f) {
                            const uint32_t off = rowF0 + colA + k;
                            plane.SetValue(off, plane.GetValue(off) + m * rA * ca);
                        }
                        if (cb != 0.0f) {
                            const uint32_t off = rowF0 + colB + k;
                            plane.SetValue(off, plane.GetValue(off) + m * rA * cb);
                        }
                    }
                    if (hasI1 && rB != 0.0f) {
                        if (ca != 0.0f) {
                            const uint32_t off = rowF1 + colA + k;
                            plane.SetValue(off, plane.GetValue(off) + m * rB * ca);
                        }
                        if (cb != 0.0f) {
                            const uint32_t off = rowF1 + colB + k;
                            plane.SetValue(off, plane.GetValue(off) + m * rB * cb);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void BlockPass(uint32_t img, uint32_t bStart, uint32_t bLen, int32_t cellStart,
                                     LocalTensor<float>& plane)
    {
        LocalTensor<float> staging = stageBuf_.Get<float>();
        LocalTensor<float> normTmp = normTmpBuf_.Get<float>();
        LocalTensor<float> thresh = threshBuf_.Get<float>();
        LocalTensor<float> sumT = sumBuf_.Get<float>();
        LocalTensor<float> invT = invBuf_.Get<float>();

        const uint32_t stageCnt = stageCnt_;
        const float eps2 = t_.eps * t_.eps;
        const uint64_t imgBase = static_cast<uint64_t>(img) * t_.dimD;

        for (uint32_t bi = bStart; bi < bStart + bLen; ++bi) {
            for (uint32_t bj = 0; bj < nBlocksW_; ++bj) {
                Duplicate(staging, 0.0f, stageCnt);
                PipeBarrier<PIPE_V>();
                for (uint32_t ci = 0; ci < bH_; ++ci) {
                    const uint32_t cellRow = (bi * sH_ + ci) - static_cast<uint32_t>(cellStart);
                    const uint32_t rowOff = cellRow * nSeg_ * nBinsPad_;
                    for (uint32_t cj = 0; cj < bW_; ++cj) {
                        const uint32_t srcOff = rowOff + (bj * sW_ + cj) * nBinsPad_;
                        const uint32_t dstOff = (ci * bW_ + cj) * nBinsPad_;
                        for (uint32_t k = 0; k < nBins_; ++k) {
                            staging.SetValue(dstOff + k, plane.GetValue(srcOff + k));
                        }
                    }
                }
                PipeBarrier<PIPE_V>(); // scalar gather -> vector norm visibility
                {
                    // normalization (skimage eps forms)
                    if (t_.normMode == 0U || t_.normMode == 1U) { // L1 / L1-sqrt
                        Abs(normTmp, staging, stageCnt);
                        PipeBarrier<PIPE_V>();
                        ReduceSum<float, true>(sumT, normTmp, thresh, stageCnt);
                        const float denom = sumT.GetValue(0) + t_.eps;
                        Muls(staging, staging, 1.0f / denom, stageCnt);
                        PipeBarrier<PIPE_V>();
                        if (t_.normMode == 1U) {
                            Sqrt(staging, staging, stageCnt);
                            PipeBarrier<PIPE_V>();
                        }
                    } else {                                      // L2 / L2-Hys
                        Mul(normTmp, staging, staging, stageCnt); // v^2
                        PipeBarrier<PIPE_V>();
                        ReduceSum<float, true>(sumT, normTmp, thresh, stageCnt);
                        invT.SetValue(0, sumT.GetValue(0) + eps2);
                        PipeBarrier<PIPE_V>(); // scalar write -> vector Sqrt visibility
                        Sqrt(invT, invT, 8);
                        PipeBarrier<PIPE_V>();
                        Muls(staging, staging, 1.0f / invT.GetValue(0), stageCnt);
                        PipeBarrier<PIPE_V>();
                        if (t_.normMode == 3U) {
                            Duplicate(thresh, t_.l2Hys, stageCnt);
                            PipeBarrier<PIPE_V>();
                            Min(staging, staging, thresh, stageCnt);
                            PipeBarrier<PIPE_V>();
                            Mul(normTmp, staging, staging, stageCnt); // v^2
                            PipeBarrier<PIPE_V>();
                            ReduceSum<float, true>(sumT, normTmp, thresh, stageCnt);
                            invT.SetValue(0, sumT.GetValue(0) + eps2);
                            PipeBarrier<PIPE_V>(); // scalar write -> vector Sqrt visibility
                            Sqrt(invT, invT, 8);
                            PipeBarrier<PIPE_V>();
                            Muls(staging, staging, 1.0f / invT.GetValue(0), stageCnt);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }

                // copy out: block row-major, cell row-major, bin ascending.
                // Route through a VECOUT queue: MTE3 reads only queue tensors (bf3d-proven).
                const uint64_t blockOff = imgBase + (static_cast<uint64_t>(bi) * nBlocksW_ + bj) * bH_ * bW_ * nBins_;
                for (uint32_t ci = 0; ci < bH_; ++ci) {
                    for (uint32_t cj = 0; cj < bW_; ++cj) {
                        const uint32_t off = (ci * bW_ + cj) * nBinsPad_;
                        LocalTensor<float> outT = outQue_.AllocTensor<float>();
                        Adds(outT, staging[off], 0.0f, nBins_);
                        outQue_.EnQue(outT);
                        outT = outQue_.DeQue<float>();
                        DataCopyExtParams cp{1, static_cast<uint32_t>(nBins_ * sizeof(float)), 0, 0, 0};
                        DataCopyPad(yGm_[blockOff + (ci * bW_ + cj) * nBins_], outT, cp);
                        outQue_.FreeTensor(outT);
                    }
                }
            }
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> rowQue_;
    TQue<QuePosition::VECOUT, 4> outQue_;
    TBuf<TPosition::VECCALC> planeBuf_;
    TBuf<TPosition::VECCALC> rowMBuf_;
    TBuf<TPosition::VECCALC> rowPBuf_;
    TBuf<TPosition::VECCALC> rowYBuf_;
    TBuf<TPosition::VECCALC> gxABuf_;
    TBuf<TPosition::VECCALC> gyABuf_;
    TBuf<TPosition::VECCALC> gxBBuf_;
    TBuf<TPosition::VECCALC> gyBBuf_;
    TBuf<TPosition::VECCALC> m2aBuf_;
    TBuf<TPosition::VECCALC> m2bBuf_;
    TBuf<TPosition::VECCALC> uBuf_;
    TBuf<TPosition::VECCALC> fuBuf_;
    TBuf<TPosition::VECCALC> aBuf_;
    TBuf<TPosition::VECCALC> omABuf_;
    TBuf<TPosition::VECCALC> b0Buf_;
    TBuf<TPosition::VECCALC> b1Buf_;
    TBuf<TPosition::VECCALC> t0Buf_;
    TBuf<TPosition::VECCALC> t1Buf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> t3Buf_;
    TBuf<TPosition::VECCALC> oneBuf_;
    TBuf<TPosition::VECCALC> zeroBuf_;
    TBuf<TPosition::VECCALC> tinyBuf_;
    TBuf<TPosition::VECCALC> s0Buf_;
    TBuf<TPosition::VECCALC> s1Buf_;
    TBuf<TPosition::VECCALC> stageBuf_;
    TBuf<TPosition::VECCALC> normTmpBuf_;
    TBuf<TPosition::VECCALC> threshBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> invBuf_;

    GlobalTensor<xType> xGm_;
    GlobalTensor<float> yGm_;

    optiling::HogDescriptorTilingData t_;
    uint32_t nImg_ = 1, h_ = 1, w_ = 1, c_ = 1;
    uint32_t pH_ = 8, pW_ = 8, bH_ = 2, bW_ = 2, sH_ = 1, sW_ = 1;
    uint32_t nBins_ = 9, nCellsH_ = 1, nCellsW_ = 1, nBlocksH_ = 1, nBlocksW_ = 1;
    uint32_t padH_ = 1, padW_ = 1, nBinsPad_ = 16, nSeg_ = 1, bandRows_ = 2, stageCnt_ = 64;
    uint32_t taskCnt_ = 1, usedCoreNum_ = 1;
    uint64_t imgBase_ = 0;
};

} // namespace NsHogDescriptor

#endif // HOG_DESCRIPTOR_IMPL_H_
