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
 * \file bilateral_filter3_d_impl.hpp
 * \brief BilateralFilter3D vector kernel for Ascend 910B (AIV, 行内分块向量化).
 *
 * 语义: y = Σ(w*v)/Σ(w) over the valid kernel_size^3 neighborhood,
 * w = exp(-(|off|^2/(2*σs^2) + (v-c)^2/(2*σc^2))); fp32 累加; 边界 = 有效邻域截断 + 重归一化。
 *
 * 并行划分: 一行 = W 个连续体素, 核间按 grid-stride 领行(步长必须用 tiling 下发的
 * usedCoreNum —— 本启动方式下 GetBlockNum() 返回 1)。
 *
 * 向量化策略: 每行再按 chunkLen(8 的倍数, 由 tiling 按 UB 预算推导)分块, 块内所有算术
 * 都是整块全宽向量指令。关键手法是**邻域移位在 GM 地址里完成**: 对每个 (dz,dy,dx) 直接从
 * GM 偏移 (nbRowBase + chunkBase + dx) 处 DataCopyPad 一整块到缓冲的 lane 0, 于是
 * buf[j] 天然就是输出 lane j 所需的邻域值, UB 侧永远是零偏移的完整张量。
 *
 * 必须遵守的 910B 约束(真机实测, 失败多为静默 —— 详见 docs/design.md 第 3 节):
 *   - 向量指令的 UB 操作数必须 32 字节对齐;
 *   - 经非零偏移视图做向量**写**会静默失效 ⇒ 本文件所有向量写的 dst 都是完整张量,
 *     掩码表因此拆成 2r+1 个独立 TBuf 而不是一个按行索引的大 buffer;
 *   - 整块搬运不得复用为别的用途申请的队列;
 *   - 输出走 VECOUT 队列, 不可复用 VECIN;
 *   - 不手工 FetchEventID + SetFlag/WaitFlag(与 TQue 内部事件冲突)。
 */
#ifndef BILATERAL_FILTER_3D_IMPL_H_
#define BILATERAL_FILTER_3D_IMPL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bilateral_filter3_d_tiling_def.h"

namespace NsBilateralFilter3D {
using namespace AscendC;

constexpr uint32_t BF3D_ALIGN = 8;         // 32 字节 / sizeof(float)
constexpr uint32_t BF3D_MAX_MASK_ROWS = 9; // k <= 9 ⇒ 2r+1 <= 9

template <typename xType>
class BilateralFilter3DVectorKernel {
public:
    __aicore__ inline BilateralFilter3DVectorKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const optiling::BilateralFilter3DTilingData* tilingData)
    {
        d_ = tilingData->d;
        h_ = tilingData->h;
        w_ = tilingData->w;
        kernelSize_ = tilingData->kernelSize;
        radius_ = static_cast<int32_t>(kernelSize_ >> 1U);
        totalRows_ = tilingData->totalRows;
        inv2ss_ = tilingData->inv2ss;
        inv2sc_ = tilingData->inv2sc;
        usedCoreNum_ = tilingData->usedCoreNum;
        chunkLen_ = static_cast<int32_t>(tilingData->chunkLen);
        swCount_ = kernelSize_ * kernelSize_ * kernelSize_;
        maskRows_ = 2U * static_cast<uint32_t>(radius_) + 1U;
        totalElems_ = totalRows_ * w_;

        xGm_.SetGlobalBuffer((__gm__ xType*)x, totalElems_);
        yGm_.SetGlobalBuffer((__gm__ xType*)y, totalElems_);

        const uint32_t cBytes = static_cast<uint32_t>(chunkLen_) * sizeof(float);
        pipe_.InitBuffer(cChBuf_, cBytes);
        pipe_.InitBuffer(nbChBuf_, cBytes);
        pipe_.InitBuffer(accDBuf_, cBytes);
        pipe_.InitBuffer(accNBuf_, cBytes);
        pipe_.InitBuffer(tmpBuf_, cBytes);
        pipe_.InitBuffer(divBuf_, cBytes);
        pipe_.InitBuffer(rampBuf_, cBytes);
        pipe_.InitBuffer(mskABuf_, cBytes);
        pipe_.InitBuffer(mskBBuf_, cBytes);
        pipe_.InitBuffer(headChBuf_, cBytes);
        pipe_.InitBuffer(nbFixBuf_, cBytes);
        pipe_.InitBuffer(onesBuf_, cBytes);
        pipe_.InitBuffer(zerosBuf_, cBytes);
        // 每个 dx 一份独立掩码缓冲: 构造掩码时 dst 必须是完整张量。
        for (uint32_t i = 0; i < maskRows_; ++i) {
            pipe_.InitBuffer(mskBufs_[i], cBytes);
        }
        pipe_.InitBuffer(swBuf_, swCount_ * sizeof(float));
        pipe_.InitBuffer(inQue_, 2, static_cast<uint32_t>(chunkLen_) * sizeof(xType));
        pipe_.InitBuffer(outQue_, 2, static_cast<uint32_t>(chunkLen_) * sizeof(xType));

        // 启动时 UB 位型任意(可能是 NaN)。本实现对超出有效长度的 lane 只在权重上做掩码、
        // 不清数据, 所以必须保证这些 lane 恒为有限值, 否则 NaN*0=NaN 会穿透掩码。
        LocalTensor<xType> slot0 = inQue_.template AllocTensor<xType>();
        LocalTensor<xType> slot1 = inQue_.template AllocTensor<xType>();
        Duplicate(slot0, static_cast<xType>(0.0f), chunkLen_);
        Duplicate(slot1, static_cast<xType>(0.0f), chunkLen_);
        PipeBarrier<PIPE_V>();
        inQue_.FreeTensor(slot0);
        inQue_.FreeTensor(slot1);

        LocalTensor<float> ones = onesBuf_.Get<float>();
        LocalTensor<float> zeros = zerosBuf_.Get<float>();
        Duplicate(ones, 1.0f, chunkLen_);
        Duplicate(zeros, 0.0f, chunkLen_);
        Duplicate(cChBuf_.Get<float>(), 0.0f, chunkLen_);
        Duplicate(nbChBuf_.Get<float>(), 0.0f, chunkLen_);
        PipeBarrier<PIPE_V>();

        BuildRamp();
        CalcSpatialWeights();
        // ramp 由标量建表、随后被 BuildMasks 的向量指令读取 (S -> V);
        // sw 由 Exp 写入、随后在热循环里被 GetValue 读取 (V -> S)。
        // 两者都是跨管道依赖, 一次性用全管道屏障隔开(每核只付一次)。
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        // grid-stride 按行分配。步长必须是 tiling 下发的 usedCoreNum: 本启动方式下
        // GetBlockNum() 返回 1, 会让所有核以步长 1 重叠遍历全部行、互相覆盖输出。
        const uint64_t coreNum = usedCoreNum_;
        for (uint64_t row = GetBlockIdx(); row < totalRows_; row += coreNum) {
            ProcessRow(row);
        }
    }

private:
    // ramp[j] = (float)j, 供掩码构造使用。标量建表只在 Init 做一次。
    __aicore__ inline void BuildRamp()
    {
        LocalTensor<float> ramp = rampBuf_.Get<float>();
        for (int32_t j = 0; j < chunkLen_; ++j) {
            ramp.SetValue(j, static_cast<float>(j));
        }
        PipeBarrier<PIPE_V>();
    }

    // sw[(dz+r)*k*k + (dy+r)*k + (dx+r)] = exp(-(dz²+dy²+dx²) * inv2ss)。
    __aicore__ inline void CalcSpatialWeights()
    {
        LocalTensor<float> sw = swBuf_.Get<float>();
        int32_t idx = 0;
        for (int32_t dz = -radius_; dz <= radius_; ++dz) {
            for (int32_t dy = -radius_; dy <= radius_; ++dy) {
                for (int32_t dx = -radius_; dx <= radius_; ++dx) {
                    sw.SetValue(idx++, static_cast<float>(dz * dz + dy * dy + dx * dx));
                }
            }
        }
        Muls(sw, sw, -inv2ss_, static_cast<int32_t>(swCount_));
        PipeBarrier<PIPE_V>();
        Exp(sw, sw, static_cast<int32_t>(swCount_));
        PipeBarrier<PIPE_V>();
    }

    // 从 GM 加载 cnt 个元素到 fp32 块缓冲的 lane 0(GM 源地址可任意偏移)。
    // [cnt, chunkLen) 的 lane 保留上一次的内容(恒为有限值), 由调用方的掩码在权重上清零。
    __aicore__ inline void LoadChunk(const LocalTensor<float>& dst, uint64_t gmElemOff, int32_t cnt)
    {
        if (cnt <= 0) {
            return;
        }
        if constexpr (std::is_same<xType, float>::value) {
            LocalTensor<float> raw = inQue_.template AllocTensor<float>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(raw, xGm_[gmElemOff], cp, pad);
            inQue_.EnQue(raw);
            raw = inQue_.template DeQue<float>();
            Adds(dst, raw, 0.0f, chunkLen_);
            inQue_.FreeTensor(raw);
        } else {
            LocalTensor<xType> raw = inQue_.template AllocTensor<xType>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * sizeof(xType)), 0, 0, 0};
            DataCopyPadExtParams<xType> pad{false, 0, 0, 0};
            DataCopyPad(raw, xGm_[gmElemOff], cp, pad);
            inQue_.EnQue(raw);
            raw = inQue_.template DeQue<xType>();
            Cast(dst, raw, RoundMode::CAST_NONE, chunkLen_);
            inQue_.FreeTensor(raw);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreChunk(const LocalTensor<float>& src, uint64_t gmElemOff, int32_t cnt)
    {
        LocalTensor<xType> yUb = outQue_.template AllocTensor<xType>();
        if constexpr (std::is_same<xType, float>::value) {
            Adds(yUb, src, 0.0f, chunkLen_);
        } else {
            Cast(yUb, src, RoundMode::CAST_RINT, chunkLen_);
        }
        outQue_.EnQue(yUb);
        yUb = outQue_.template DeQue<xType>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * sizeof(xType)), 0, 0, 0};
        DataCopyPad(yGm_[gmElemOff], yUb, cp);
        outQue_.FreeTensor(yUb);
    }

    // 为当前 chunk 的每个 dx 构造掩码: lane j 有效 ⇔ lo <= j < hi。
    // 纯向量算术构造, 不用标量逐元素写(那样每行每块要 (2r+1)*chunkLen 次 SetValue):
    //   a = clamp(ramp - lo + 1, 0, 1)  -> j >= lo 时为 1
    //   b = clamp(hi - ramp,     0, 1)  -> j <  hi 时为 1
    //   mask = a * b
    // clamp 用 Min/Max 配预置的全 1/全 0 张量(标量变体 Mins/Maxs 这里不依赖)。
    __aicore__ inline void BuildMasks(int32_t chunkBase, int32_t chunkCnt)
    {
        LocalTensor<float> a = mskABuf_.Get<float>();
        LocalTensor<float> b = mskBBuf_.Get<float>();
        LocalTensor<float> ramp = rampBuf_.Get<float>();
        LocalTensor<float> ones = onesBuf_.Get<float>();
        LocalTensor<float> zeros = zerosBuf_.Get<float>();
        for (int32_t dx = -radius_; dx <= radius_; ++dx) {
            int32_t lo = -(chunkBase + dx);
            if (lo < 0) {
                lo = 0;
            }
            int32_t hi = static_cast<int32_t>(w_) - chunkBase - dx;
            if (hi > chunkCnt) {
                hi = chunkCnt;
            }
            if (hi < 0) {
                hi = 0;
            }
            Adds(a, ramp, static_cast<float>(1 - lo), chunkLen_);
            PipeBarrier<PIPE_V>();
            Min(a, a, ones, chunkLen_);
            PipeBarrier<PIPE_V>();
            Max(a, a, zeros, chunkLen_);
            PipeBarrier<PIPE_V>();
            Muls(b, ramp, -1.0f, chunkLen_);
            PipeBarrier<PIPE_V>();
            Adds(b, b, static_cast<float>(hi), chunkLen_);
            PipeBarrier<PIPE_V>();
            Min(b, b, ones, chunkLen_);
            PipeBarrier<PIPE_V>();
            Max(b, b, zeros, chunkLen_);
            PipeBarrier<PIPE_V>();
            Mul(mskBufs_[dx + radius_].Get<float>(), a, b, chunkLen_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessRow(uint64_t row)
    {
        const int32_t wI = static_cast<int32_t>(w_);
        const int64_t w64 = static_cast<int64_t>(w_);
        const uint64_t rowBase = row * w_;
        const uint64_t rem = row % (static_cast<uint64_t>(d_) * h_);
        const int32_t z = static_cast<int32_t>(rem / h_);
        const int32_t yv = static_cast<int32_t>(rem % h_);
        const int32_t zLow = z < radius_ ? -z : -radius_;
        const int32_t zHigh = z + radius_ >= static_cast<int32_t>(d_) ? static_cast<int32_t>(d_) - 1 - z : radius_;
        const int32_t yLow = yv < radius_ ? -yv : -radius_;
        const int32_t yHigh = yv + radius_ >= static_cast<int32_t>(h_) ? static_cast<int32_t>(h_) - 1 - yv : radius_;

        LocalTensor<float> cCh = cChBuf_.Get<float>();
        LocalTensor<float> nbCh = nbChBuf_.Get<float>();
        LocalTensor<float> accD = accDBuf_.Get<float>();
        LocalTensor<float> accN = accNBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> div = divBuf_.Get<float>();
        LocalTensor<float> headCh = headChBuf_.Get<float>();
        LocalTensor<float> nbFix = nbFixBuf_.Get<float>();
        LocalTensor<float> sw = swBuf_.Get<float>();
        LocalTensor<float> mk[BF3D_MAX_MASK_ROWS];
        for (uint32_t i = 0; i < maskRows_; ++i) {
            mk[i] = mskBufs_[i].Get<float>();
        }
        const int64_t totalI64 = static_cast<int64_t>(totalElems_);

        for (int32_t cb = 0; cb < wI; cb += chunkLen_) {
            const int32_t cl = (wI - cb) < chunkLen_ ? (wI - cb) : chunkLen_;
            // 无需掩码 ⇔ 对所有 dx ∈ [-r, r] 本块的每个 lane 都落在 [0, W) 内。
            // 注意不能简化成 cb > 0 && cb + cl < W: 若 cb + cl 距 W 不足 r,
            // 末尾 lane 对 dx > 0 已越界, 漏掩码会静默算错。
            const bool needMask = !(cb >= radius_ && cb + cl + radius_ <= wI);

            LoadChunk(cCh, rowBase + static_cast<uint64_t>(cb), cl);
            if (needMask) {
                BuildMasks(cb, cl);
            }
            Duplicate(accD, 0.0f, chunkLen_);
            Duplicate(accN, 0.0f, chunkLen_);
            PipeBarrier<PIPE_V>();

            for (int32_t dz = zLow; dz <= zHigh; ++dz) {
                for (int32_t dy = yLow; dy <= yHigh; ++dy) {
                    const int64_t nbRowBase = static_cast<int64_t>(rowBase) +
                                              (static_cast<int64_t>(dz) * h_ + dy) * w64;
                    const int32_t swBase = ((dz + radius_) * static_cast<int32_t>(kernelSize_) + (dy + radius_)) *
                                           static_cast<int32_t>(kernelSize_);
                    for (int32_t dx = -radius_; dx <= radius_; ++dx) {
                        int32_t lo = -(cb + dx);
                        if (lo < 0) {
                            lo = 0;
                        }
                        int32_t hi = wI - cb - dx;
                        if (hi > cl) {
                            hi = cl;
                        }
                        if (hi <= lo) {
                            continue; // 本块内该 dx 没有任何有效输出 lane
                        }
                        const float spatialW = sw.GetValue(swBase + (dx + radius_));
                        const int64_t srcStart = nbRowBase + cb + dx;

                        LocalTensor<float> nbUse;
                        if (srcStart < 0) {
                            // GM 源地址为负: 仅张量起始附近可达(nbRowBase < r, 即全局首行且
                            // dz=dy=0, 或 W 极小时紧邻起始的几行)。此时连续加载无法落位,
                            // 退化为标量组装 —— 极罕见, 不影响整体吞吐。
                            LoadChunk(headCh, static_cast<uint64_t>(nbRowBase), cl);
                            Duplicate(nbFix, 0.0f, chunkLen_);
                            // 跨管道依赖, PipeBarrier<PIPE_V> 覆盖不到:
                            //   1) headCh.GetValue 是标量读 V 管道刚写入的数据 (V -> S);
                            //   2) nbFix.SetValue 写完立即被下面的 Sub 读取     (S -> V)。
                            // 真机实测: 只用 PIPE_V 屏障时, 标量循环最后写入的 lane
                            // (即每行最后一列)会偶发读到旧值 —— 同一形状重复运行时而
                            // 通过时而失败, 出错行恰好是全部能取到"邻域行 = 第 0 行"的行。
                            // 用全管道屏障而不自行分配 event: FetchEventID 会与 TQue
                            // 内部事件冲突(见 docs/design.md 第 3 节)。
                            PipeBarrier<PIPE_ALL>();
                            for (int32_t j = lo; j < cl; ++j) {
                                const int64_t rel = static_cast<int64_t>(cb) + dx + j;
                                if (rel >= 0 && rel < static_cast<int64_t>(cl)) {
                                    nbFix.SetValue(j, headCh.GetValue(static_cast<int32_t>(rel)));
                                }
                            }
                            PipeBarrier<PIPE_ALL>();
                            nbUse = nbFix;
                        } else {
                            int32_t loadLen = cl;
                            const int64_t avail = totalI64 - srcStart;
                            if (static_cast<int64_t>(loadLen) > avail) {
                                loadLen = static_cast<int32_t>(avail);
                            }
                            if (loadLen <= 0) {
                                continue;
                            }
                            LoadChunk(nbCh, static_cast<uint64_t>(srcStart), loadLen);
                            nbUse = nbCh;
                        }

                        Sub(tmp, nbUse, cCh, chunkLen_);
                        PipeBarrier<PIPE_V>();
                        Mul(tmp, tmp, tmp, chunkLen_);
                        PipeBarrier<PIPE_V>();
                        Muls(tmp, tmp, -inv2sc_, chunkLen_);
                        PipeBarrier<PIPE_V>();
                        Exp(tmp, tmp, chunkLen_);
                        PipeBarrier<PIPE_V>();
                        Muls(tmp, tmp, spatialW, chunkLen_);
                        PipeBarrier<PIPE_V>();
                        if (needMask) {
                            Mul(tmp, tmp, mk[dx + radius_], chunkLen_);
                            PipeBarrier<PIPE_V>();
                        }
                        Add(accD, accD, tmp, chunkLen_);
                        PipeBarrier<PIPE_V>();
                        Mul(tmp, tmp, nbUse, chunkLen_); // w·v(无效 lane 的 w 已为 0)
                        PipeBarrier<PIPE_V>();
                        Add(accN, accN, tmp, chunkLen_);
                        PipeBarrier<PIPE_V>();
                    }
                }
            }

            // [cl, chunkLen) 的 lane 两个累加器都是 0 ⇒ 0/0 = NaN, 但只写回 cl 个元素,
            // 不会外泄。有效 lane 的分母恒 >= 1(中心体素权重为 1)。
            Div(div, accN, accD, chunkLen_);
            PipeBarrier<PIPE_V>();
            StoreChunk(div, rowBase + static_cast<uint64_t>(cb), cl);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQue_;
    TQue<QuePosition::VECOUT, 2> outQue_;
    TBuf<TPosition::VECCALC> cChBuf_;
    TBuf<TPosition::VECCALC> nbChBuf_;
    TBuf<TPosition::VECCALC> accDBuf_;
    TBuf<TPosition::VECCALC> accNBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> divBuf_;
    TBuf<TPosition::VECCALC> rampBuf_;
    TBuf<TPosition::VECCALC> mskABuf_;
    TBuf<TPosition::VECCALC> mskBBuf_;
    TBuf<TPosition::VECCALC> headChBuf_;
    TBuf<TPosition::VECCALC> nbFixBuf_;
    TBuf<TPosition::VECCALC> onesBuf_;
    TBuf<TPosition::VECCALC> zerosBuf_;
    TBuf<TPosition::VECCALC> mskBufs_[BF3D_MAX_MASK_ROWS];
    TBuf<TPosition::VECCALC> swBuf_;

    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> yGm_;

    uint64_t totalRows_ = 0;
    uint64_t totalElems_ = 0;
    uint32_t d_ = 0, h_ = 0, w_ = 0;
    uint32_t kernelSize_ = 0;
    uint32_t swCount_ = 0;
    uint32_t maskRows_ = 1;
    uint32_t usedCoreNum_ = 1;
    int32_t chunkLen_ = 8;
    int32_t radius_ = 0;
    float inv2ss_ = 0.0f;
    float inv2sc_ = 0.0f;
};

} // namespace NsBilateralFilter3D

#endif // BILATERAL_FILTER_3D_IMPL_H_
