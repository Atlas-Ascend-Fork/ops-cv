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
 * \file gaussian_filter_3_d.cpp
 * \brief GaussianFilter3D AscendC kernel（Ascend 910B，AIV）
 *
 * tap 搬运改用 TQue(VECIN, depth=2)，MTE2/V 事件由 TQue 自动管理（Set/Wait 严格交替）：
 *  - H/D pass：center 单独 Axpy；±i 对称两行各 Alloc/EnQue，DeQue 后先 Add 再 Axpy，
 *    乘加次数 2r+1 → r+1；
 *  - W pass：每个 tap 只搬运对齐中段（DataCopyPad，GM→UB）走向量 Axpy，越界段与少量未对齐元素走标量。
 */
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

using namespace AscendC;

namespace {
constexpr uint32_t GAUSSIAN_MODE_REFLECT = 0;
constexpr uint32_t GAUSSIAN_MODE_CONSTANT = 1;
constexpr uint32_t GAUSSIAN_MODE_NEAREST = 2;
constexpr uint32_t GAUSSIAN_MODE_MIRROR = 3;
constexpr uint32_t GAUSSIAN_MODE_WRAP = 4;
constexpr int64_t VEC_ALIGN_NUM = 8;

__aicore__ inline int64_t RoundUp8(int64_t x) { return ((x + VEC_ALIGN_NUM - 1) / VEC_ALIGN_NUM) * VEC_ALIGN_NUM; }
__aicore__ inline int64_t RoundDown8(int64_t x) { return (x / VEC_ALIGN_NUM) * VEC_ALIGN_NUM; }
__aicore__ inline float BitsToFloat(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = bits;
    return conv.f;
}
} // namespace

template <typename T>
class KernelGaussianFilter3D {
public:
    __aicore__ inline KernelGaussianFilter3D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GaussianFilter3DTilingData* tilingData)
    {
        tiling_ = tilingData;
        batch_ = tiling_->batch;
        depth_ = tiling_->depth;
        height_ = tiling_->height;
        width_ = tiling_->width;
        radius_ = tiling_->radius;
        mode_ = tiling_->mode;
        tileLen_ = tiling_->tileLen;
        chunkNum_ = tiling_->chunkNum;
        lineNum_ = tiling_->lineNum;
        tasksPerCore_ = tiling_->tasksPerCore;
        pongOffsetElems_ = static_cast<int64_t>(tiling_->pongOffset) / static_cast<int64_t>(sizeof(float));

        totalElem_ = static_cast<int64_t>(batch_) * depth_ * height_ * width_;
        totalTasks_ = static_cast<int64_t>(lineNum_) * chunkNum_;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalElem_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalElem_);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), pongOffsetElems_ + totalElem_ + 8L);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            xU16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
        }

        uint32_t blockIdx = GetBlockIdx();
        int64_t beginTask = static_cast<int64_t>(blockIdx) * tasksPerCore_;
        int64_t endTask = beginTask + tasksPerCore_;
        if (beginTask > totalTasks_) {
            beginTask = totalTasks_;
        }
        if (endTask > totalTasks_) {
            endTask = totalTasks_;
        }
        taskStart_ = beginTask;
        taskEnd_ = endTask;

        pipe_ = &pipe;
        pipe_->InitBuffer(accBuf_, tileLen_ * sizeof(float));
        pipe_->InitBuffer(tapQue_, 2, tileLen_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_->InitBuffer(rawBuf_[0], tileLen_ * sizeof(T));
            pipe_->InitBuffer(rawBuf_[1], tileLen_ * sizeof(T));
            pipe_->InitBuffer(cvtBuf_, tileLen_ * sizeof(float));
            raw_[0] = rawBuf_[0].Get<T>();
            raw_[1] = rawBuf_[1].Get<T>();
            cvt_ = cvtBuf_.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        DoPass(0);
        SyncAll();
        DoPass(1);
        SyncAll();
        DoPass(2);
    }

private:
    // ---------- 边界索引映射：返回 -1 表示 CONSTANT 越界补 0 ----------
    __aicore__ inline int64_t MapIndex(int64_t i, int64_t n) const
    {
        if (i >= 0 && i < n) {
            return i;
        }
        if (mode_ == GAUSSIAN_MODE_CONSTANT) {
            return -1;
        }
        if (n == 1) {
            return 0;
        }
        if (mode_ == GAUSSIAN_MODE_NEAREST) {
            return i < 0 ? 0 : n - 1;
        }
        if (mode_ == GAUSSIAN_MODE_REFLECT) {
            int64_t p = 2 * n;
            int64_t j = i % p;
            if (j < 0) {
                j += p;
            }
            return j >= n ? p - 1 - j : j;
        }
        if (mode_ == GAUSSIAN_MODE_MIRROR) {
            int64_t p = 2 * n - 2;
            int64_t j = i % p;
            if (j < 0) {
                j += p;
            }
            return j >= n ? p - j : j;
        }
        // WRAP
        int64_t j = i % n;
        if (j < 0) {
            j += n;
        }
        return j;
    }

    // ---------- 标量读输入（bf16 位模式还原） ----------
    __aicore__ inline float EdgeValue(int64_t elem)
    {
        if constexpr (std::is_same_v<T, float>) {
            return xGm_.GetValue(elem);
        } else if constexpr (std::is_same_v<T, half>) {
            return static_cast<float>(xGm_.GetValue(elem));
        } else {
            uint16_t bits = xU16Gm_.GetValue(elem);
            union {
                uint32_t u;
                float f;
            } conv;
            conv.u = static_cast<uint32_t>(bits) << 16;
            return conv.f;
        }
    }

    __aicore__ inline void AccScalarAdd(LocalTensor<float>& acc, int64_t i, float wk, float v)
    {
        float oldValue = acc.GetValue(i);
        acc.SetValue(i, oldValue + wk * v);
    }

    // ---------- 流水级同步（acc 写回 / DoW 标量用） ----------
    __aicore__ inline void PipeSyncVMte3()
    {
        TEventID e = pipe_->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(e);
    }
    __aicore__ inline void PipeSyncMte3V()
    {
        TEventID e = pipe_->AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(e);
        WaitFlag<HardEvent::MTE3_V>(e);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(e);
    }
    __aicore__ inline void PipeSyncVS()
    {
        TEventID e = pipe_->AllocEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(e);
        WaitFlag<HardEvent::V_S>(e);
        pipe_->ReleaseEventID<HardEvent::V_S>(e);
    }
    __aicore__ inline void PipeSyncSV()
    {
        TEventID e = pipe_->AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
        pipe_->ReleaseEventID<HardEvent::S_V>(e);
    }

    // ---------- 从 workspace 搬一行到 TQue（GM→UB），返回入队 tensor ----------
    __aicore__ inline LocalTensor<float> IssueRow(int64_t srcElem, int64_t len)
    {
        LocalTensor<float> t = tapQue_.AllocTensor<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(t, wsGm_[srcElem], cp, pad);
        tapQue_.EnQue(t);
        return t;
    }

    // ============================================================
    // W pass：每 tap 只搬"对齐中段"（TQue，dst 对齐），Axpy 处理对齐段；
    // 越界段（reflect 等）与有效未对齐段（每 tap <16 个）走标量
    // ============================================================
    __aicore__ inline void DoW(LocalTensor<float>& acc, int64_t rowBase, int64_t c0, int64_t len)
    {
        int64_t taps = 2 * static_cast<int64_t>(radius_) + 1;
        for (int64_t k = 0; k < taps; k++) {
            float wk = BitsToFloat(tiling_->weightBits[k]);
            int64_t start = c0 + k - static_cast<int64_t>(radius_);
            int64_t iLow = start < 0 ? -start : 0;
            if (iLow > len) {
                iLow = len;
            }
            int64_t iHigh = start + len > width_ ? width_ - start : len;
            if (iHigh < 0) {
                iHigh = 0;
            }
            if (iHigh > len) {
                iHigh = len;
            }
            int64_t iA = RoundUp8(iLow);
            int64_t iB = RoundDown8(iHigh);

            // 对齐中段：DataCopyPad 到对齐 t（源任意），Axpy 目标 acc[iA] 对齐
            if (iA < iB) {
                LocalTensor<float> t = tapQue_.AllocTensor<float>();
                if constexpr (std::is_same_v<T, float>) {
                    DataCopyExtParams cp{1, static_cast<uint32_t>((iB - iA) * sizeof(float)), 0, 0, 0};
                    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                    DataCopyPad(t, xGm_[rowBase + start + iA], cp, pad);
                } else {
                    LocalTensor<T> tT = t.ReinterpretCast<T>();
                    DataCopyExtParams cp{1, static_cast<uint32_t>((iB - iA) * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                    DataCopyPad(tT, xGm_[rowBase + start + iA], cp, pad);
                }
                tapQue_.EnQue(t);
                LocalTensor<float> d = tapQue_.DeQue<float>();
                if constexpr (std::is_same_v<T, float>) {
                    Axpy(acc[iA], d, wk, static_cast<uint32_t>(iB - iA));
                } else {
                    LocalTensor<T> dT = d.ReinterpretCast<T>();
                    Cast(cvt_, dT, RoundMode::CAST_NONE, static_cast<uint32_t>(iB - iA));
                    PipeBarrier<PIPE_V>();
                    Axpy(acc[iA], cvt_, wk, static_cast<uint32_t>(iB - iA));
                }
                PipeBarrier<PIPE_V>();
                tapQue_.FreeTensor(d);
            }

            // 边缘：越界段（非 CONSTANT 映射）+ 有效但未走中段的部分
            bool hasMid = iA < iB;
            bool hasOut = mode_ != GAUSSIAN_MODE_CONSTANT && (iLow > 0 || iHigh < len);
            bool hasInEdge = hasMid ? (iA > iLow || iB < iHigh) : (iHigh > iLow);
            if (hasOut || hasInEdge) {
                PipeSyncVS();
                if (hasOut) {
                    for (int64_t i = 0; i < iLow; i++) {
                        int64_t j = MapIndex(start + i, width_);
                        if (j >= 0) {
                            AccScalarAdd(acc, i, wk, EdgeValue(rowBase + j));
                        }
                    }
                    for (int64_t i = iHigh; i < len; i++) {
                        int64_t j = MapIndex(start + i, width_);
                        if (j >= 0) {
                            AccScalarAdd(acc, i, wk, EdgeValue(rowBase + j));
                        }
                    }
                }
                // 有效段中未被中段覆盖的部分：有中段拆两端，无中段整体标量
                int64_t mL = hasMid ? iA : iHigh;
                int64_t mR = hasMid ? iB : iHigh;
                for (int64_t i = iLow; i < mL; i++) {
                    AccScalarAdd(acc, i, wk, EdgeValue(rowBase + start + i));
                }
                for (int64_t i = mR; i < iHigh; i++) {
                    AccScalarAdd(acc, i, wk, EdgeValue(rowBase + start + i));
                }
                PipeSyncSV();
            }
        }
    }

    // ============================================================
    // H/D pass：TQue + 对称核合并（pair 串行，事件由 TQue 管理）
    // ============================================================
    __aicore__ inline int64_t HDSrcRow(uint32_t pass, int64_t j, int64_t nc, int64_t di, int64_t hi)
    {
        if (pass == 1) {
            return ((nc * depth_ + di) * height_ + j) * width_;
        }
        return ((nc * depth_ + j) * height_ + hi) * width_;
    }

    __aicore__ inline void DoHD(uint32_t pass, LocalTensor<float>& acc, int64_t srcOffset, int64_t nc, int64_t di,
                                int64_t hi, int64_t c0, int64_t len)
    {
        int64_t axisLen = pass == 1 ? height_ : depth_;
        int64_t base = pass == 1 ? hi : di;
        uint32_t cnt = static_cast<uint32_t>(len);

        // center
        int64_t jc = MapIndex(base, axisLen);
        if (jc >= 0) {
            float w0 = BitsToFloat(tiling_->weightBits[radius_]);
            int64_t row = HDSrcRow(pass, jc, nc, di, hi);
            (void)IssueRow(srcOffset + row + c0, len);
            LocalTensor<float> c = tapQue_.DeQue<float>();
            Axpy(acc, c, w0, cnt);
            PipeBarrier<PIPE_V>();
            tapQue_.FreeTensor(c);
        }

        // 对称 pair
        for (int64_t i = 1; i <= static_cast<int64_t>(radius_); i++) {
            float wi = BitsToFloat(tiling_->weightBits[radius_ - static_cast<uint32_t>(i)]);
            int64_t jm = MapIndex(base - i, axisLen);
            int64_t jp = MapIndex(base + i, axisLen);

            if (jm == jp) {
                if (jm >= 0) {
                    int64_t row = HDSrcRow(pass, jm, nc, di, hi);
                    (void)IssueRow(srcOffset + row + c0, len);
                    LocalTensor<float> c = tapQue_.DeQue<float>();
                    Axpy(acc, c, 2.0f * wi, cnt);
                    PipeBarrier<PIPE_V>();
                    tapQue_.FreeTensor(c);
                }
                continue;
            }

            bool hm = jm >= 0;
            bool hp = jp >= 0;
            if (hm) {
                int64_t rowm = HDSrcRow(pass, jm, nc, di, hi);
                (void)IssueRow(srcOffset + rowm + c0, len);
            }
            if (hp) {
                int64_t rowp = HDSrcRow(pass, jp, nc, di, hi);
                (void)IssueRow(srcOffset + rowp + c0, len);
            }
            LocalTensor<float> dm;
            LocalTensor<float> dp;
            if (hm) {
                dm = tapQue_.DeQue<float>();
            }
            if (hp) {
                dp = tapQue_.DeQue<float>();
            }
            if (hm && hp) {
                Add(dm, dm, dp, cnt);
                Axpy(acc, dm, wi, cnt);
            } else if (hm) {
                Axpy(acc, dm, wi, cnt);
            } else if (hp) {
                Axpy(acc, dp, wi, cnt);
            }
            PipeBarrier<PIPE_V>();
            if (hm) {
                tapQue_.FreeTensor(dm);
            }
            if (hp) {
                tapQue_.FreeTensor(dp);
            }
        }
    }

    // ============================================================
    // pass 主循环
    // ============================================================
    __aicore__ inline void DoPass(uint32_t pass)
    {
        const int64_t srcOffset = pass == 2 ? pongOffsetElems_ : 0;
        const int64_t dstOffset = pass == 1 ? pongOffsetElems_ : 0;
        const bool dstFp32 = pass != 2;
        LocalTensor<float> acc = accBuf_.Get<float>();

        // 上一 pass 用 MTE3 写 workspace，本 pass 用 MTE2 读，需 MTE3→MTE2 内存序
        if (pass == 1 || pass == 2) {
            WaitFlag<HardEvent::MTE3_MTE2>(passWriteEvt_);
            pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(passWriteEvt_);
        }

        for (int64_t task = taskStart_; task < taskEnd_; task++) {
            int64_t line = task / chunkNum_;
            int64_t chunk = task - line * chunkNum_;
            int64_t c0 = chunk * tileLen_;
            int64_t len = width_ - c0 < tileLen_ ? width_ - c0 : tileLen_;

            int64_t dh = height_ * depth_;
            int64_t nc = line / dh;
            int64_t rem = line - nc * dh;
            int64_t di = rem / height_;
            int64_t hi = rem - di * height_;
            int64_t rowBase = line * width_;

            Duplicate(acc, 0.0f, static_cast<uint32_t>(len));
            PipeBarrier<PIPE_V>();

            if (pass == 0) {
                DoW(acc, rowBase, c0, len);
            } else {
                DoHD(pass, acc, srcOffset, nc, di, hi, c0, len);
            }

            if (dstFp32) {
                PipeSyncVMte3();
                DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
                DataCopyPad(wsGm_[dstOffset + rowBase + c0], acc, cp);
                PipeSyncMte3V();
            } else {
                if constexpr (!std::is_same_v<T, float>) {
                    // 饱和转换：先 clamp 到目标 dtype 范围再 Cast
                    float maxVal = std::is_same_v<T, half> ? 65504.0f : 3.3895314e38f;
                    Mins(acc, acc, maxVal, static_cast<uint32_t>(len));
                    Maxs(acc, acc, -maxVal, static_cast<uint32_t>(len));
                    PipeBarrier<PIPE_V>();
                    // f32->bf16 仅支持 CAST_RINT；f32->fp16 用 CAST_NONE
                    RoundMode outMode = std::is_same_v<T, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
                    Cast(raw_[0], acc, outMode, static_cast<uint32_t>(len));
                    PipeBarrier<PIPE_V>();
                }
                PipeSyncVMte3();
                DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
                if constexpr (std::is_same_v<T, float>) {
                    DataCopyPad(yGm_[rowBase + c0], acc, cp);
                } else {
                    DataCopyPad(yGm_[rowBase + c0], raw_[0], cp);
                }
                PipeSyncMte3V();
            }
        }
        // 本 pass（pass0/1）用 MTE3 写了 workspace，建立到下一 pass MTE2 的依赖
        if (pass == 0 || pass == 1) {
            passWriteEvt_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(passWriteEvt_);
        }
    }

private:
    TPipe pipe;
    const GaussianFilter3DTilingData* tiling_ = nullptr;
    TPipe* pipe_ = nullptr;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> wsGm_;
    GlobalTensor<uint16_t> xU16Gm_;

    TBuf<TPosition::VECCALC> accBuf_;
    TQue<TPosition::VECIN, 2> tapQue_;
    TBuf<TPosition::VECCALC> rawBuf_[2];
    LocalTensor<T> raw_[2];
    TBuf<TPosition::VECCALC> cvtBuf_;
    LocalTensor<float> cvt_;

    TEventID passWriteEvt_;

    int64_t batch_;
    int64_t depth_;
    int64_t height_;
    int64_t width_;
    uint32_t radius_;
    uint32_t mode_;
    int64_t tileLen_;
    int64_t chunkNum_;
    int64_t lineNum_;
    int64_t tasksPerCore_;
    int64_t pongOffsetElems_;
    int64_t totalElem_;
    int64_t totalTasks_;
    int64_t taskStart_;
    int64_t taskEnd_;
};

extern "C" __global__ __aicore__ void gaussian_filter_3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        KernelGaussianFilter3D<float> op;
        op.Init(x, y, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelGaussianFilter3D<half> op;
        op.Init(x, y, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGaussianFilter3D<bfloat16_t> op;
        op.Init(x, y, workspace, &tilingData);
        op.Process();
    }
}
