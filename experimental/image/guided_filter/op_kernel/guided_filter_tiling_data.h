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
 * \file guided_filter_tiling_data.h
 * \brief GuidedFilter tiling 数据结构（host/kernel 共用的 ABI）
 */

#ifndef __GUIDED_FILTER_TILLING_DATA_H__
#define __GUIDED_FILTER_TILLING_DATA_H__

#include <cstdint>

// ---------------------------------------------------------------------------
// ABI 冻结清单：本结构体的字段顺序即 host/kernel 之间的二进制契约，改字段必须同时改
//   - op_host/guided_filter_tiling.cpp（写入端）
//   - op_kernel/guided_filter.h（读取端）
// 中间量布局：每张图都是 H*W 的平面，平面步长统一为 planeElems（32B 对齐）。
// ---------------------------------------------------------------------------
struct GuidedFilterTilingData {
    uint32_t batch;      // N
    uint32_t height;     // H
    uint32_t width;      // W
    uint32_t guideCh;    // C，引导图通道数（1~3）
    uint32_t srcCh;      // Cs，待滤波图通道数
    uint32_t radius;     // r，窗口半径（>=1）
    uint32_t tileW;      // T，列方向有效列数（= ceil(W / nTx)）
    uint32_t haloW;      // haloW = align8(T + 2r)，UB 中每行实际跨度（含左右 halo）
    uint32_t nTx;        // 列方向任务数
    uint32_t nTy;        // 行方向任务数
    uint32_t bandH;      // 每个行带的行数
    uint32_t atomNum;    // M，第一阶段原子乘积图数量（见 GfGeo::AtomNum）
    uint32_t coefNum;    // K = C*Cs + Cs，alpha/beta 图数量
    uint32_t planeElems; // 每张中间平面元素数 = align8(H*W)
    uint32_t outDtype;   // 0=float32, 1=float16, 2=uint8（调试用，实际由模板宏决定）
    float eps;           // 正则项（不随 dtype 缩放，与 OpenCV 一致）
    float invWin;        // 1/(2r+1)^2
    uint64_t alphaOff;   // workspace 中 alpha 段的元素偏移（float 单位）
    uint64_t betaOff;    // workspace 中 beta 段的元素偏移
};

// ---------------------------------------------------------------------------
// 几何推导：**host tiling 与 kernel 共用同一份实现**（单一真值，避免两侧漂移）
// 仅使用标准 C++，不得引入 AscendC 依赖（会同时被 host 侧包含）。
// ---------------------------------------------------------------------------
namespace GfGeo {
constexpr int32_t kUbBudgetBytes = 184 * 1024; // 单核 UB 预算（实际 192KB，另留 8KB 给栈/同步）
constexpr int32_t kMaxTileW = 2048;
constexpr int32_t kFixedBytes = 8192; // 索引表/常量/对齐损耗的固定预留

inline int32_t Align8(int32_t v) { return (v + 7) & ~7; }
// 与 kernel 侧 GuidedFilter::kIdxAlign 一致：所有 Gather 偏移表按 64 项对齐分配
inline int32_t IdxPad(int32_t n) { return (n + 63) / 64 * 64; }
inline int32_t CeilDivI(int32_t a, int32_t b) { return (a + b - 1) / b; }

// 原子乘积图数量：μ_c(C) + Σ_cd(C(C+1)/2) + ν_s(Cs) + ω_cs(C*Cs)
inline int32_t AtomNum(int32_t gCh, int32_t sCh) { return gCh + gCh * (gCh + 1) / 2 + sCh + gCh * sCh; }
inline int32_t CoefNum(int32_t gCh, int32_t sCh) { return gCh * sCh + sCh; }
inline int32_t PadTo(int32_t n, int32_t g) { return (g <= 1) ? n : (n + g - 1) / g * g; }

// 窗口长度 win 的素因子分解：单级滑动和需要 win-1 份移位副本，而按因子 p 逐级做
// 「步长子滑动和」只需 Σ(p-1) 份。win=9=3*3 时 8 份降到 4 份，累加指令同样减半。
// 必须与 kernel 侧 GuidedFilter::Init 里的分解循环一致。
inline int32_t ShiftSlots(int32_t win)
{
    int32_t w = win;
    int32_t sum = 0;
    for (int32_t p = 2; p <= w; ++p) {
        while (w % p == 0) {
            sum += p - 1;
            w /= p;
        }
    }
    return sum;
}

// 水平滑窗合批张数。一次 Add 覆盖 mapBatch 张图，把 rows 次树形归约从「每图各做一遍」
// 摊成「每批做一遍」—— 实测水平滑窗的树形归约占 3->3 总耗时约 33%，而它几乎全是
// 单 repeat(cnt = 1 个 tile 宽)的短指令，瓶颈在指令条数而非元素数，所以合批收益接近线性。
// 每多带一张图要多占 移位副本 rows*strideT*4 + 偏移表 rows*strideT*4 字节，
// 用 kMapBatchBytes 控制这笔额外开销的上限（避免挤掉更大 tileW 的候选几何）。
constexpr int32_t kMapBatchBytes = 32 * 1024;

// 选合批粒度：末批不满时多带出的空图也要参与 Gather/Add，是纯浪费，所以先让
// 「补齐后的图数」最小，同样补齐量下再让批数最少。例：
//   21 张图、上限 7  -> g=7 (21 图, 3 批) 远好于 g=4 (24 图, 6 批)
//   21 张图、上限 4  -> g=3 (21 图, 7 批) 好于   g=4 (24 图, 6 批)  <- 实测 +3%
inline int32_t MapBatch(int32_t strideT, int32_t rows, int32_t atomNum)
{
    const int64_t perMap = static_cast<int64_t>(2) * rows * strideT * 4; // 移位副本 + 偏移表
    int32_t gMax = (perMap > 0) ? static_cast<int32_t>(kMapBatchBytes / perMap) : 1;
    if (gMax < 1) {
        gMax = 1;
    }
    if (gMax > atomNum) {
        gMax = atomNum;
    }
    int32_t best = 1;
    int64_t bestPad = atomNum;
    int64_t bestBatches = atomNum;
    for (int32_t g = 1; g <= gMax; ++g) {
        const int64_t batches = (atomNum + g - 1) / g;
        const int64_t pad = batches * g;
        if (pad < bestPad || (pad == bestPad && batches < bestBatches)) {
            bestPad = pad;
            bestBatches = batches;
            best = g;
        }
    }
    return best;
}

struct Geo {
    int32_t tileW = 0;
    int32_t haloW = 0;
    int32_t nTx = 0;
    int32_t nTy = 0;
    int32_t bandH = 0;
    uint32_t blockDim = 0;
    int64_t ubBytes = 0;
    bool ok = false;
};

// UB 需求（字节）。必须与 op_kernel/guided_filter.h 的 InitBuffer 列表逐项一致。
inline int64_t UbNeed(int32_t tileW, int32_t haloW, int32_t radius, int32_t gCh, int32_t sCh, int32_t atomNum,
                      int32_t coefNum, int32_t eg, int32_t es, int32_t ed)
{
    const int32_t strideT = Align8(tileW);
    const int64_t rows = 2 * static_cast<int64_t>(radius) + 1;
    const int64_t ring = static_cast<int64_t>(atomNum) * (rows + 1) * haloW * 4; // 原子图环形缓冲（深度 2r+2）
    // 水平滑窗按 mapBatch 张图合批：vAll/box 要按合批粒度向上取整（末批可能有空图占位），
    // sh/idx 表则从 1 张图扩到 mapBatch 张图。必须与 kernel 的 MapBatchK() 完全一致。
    const int32_t mb = MapBatch(strideT, static_cast<int32_t>(rows), atomNum);
    const int32_t padAtom = PadTo(atomNum, mb);
    const int64_t vAll = static_cast<int64_t>(padAtom) * haloW * 4;       // 每图的垂直窗口和
    const int64_t box = static_cast<int64_t>(padAtom) * strideT * 4;      // 每图的盒式滤波结果
    const int64_t aux = static_cast<int64_t>(10 + coefNum) * strideT * 4; // 余子式 9 + det 1 + alpha/beta K
    const int64_t gRow = static_cast<int64_t>(gCh) * haloW * 4;           // guide 行（float 平面）
    const int64_t pRow = static_cast<int64_t>(sCh) * haloW * 4;           // src 行（float 平面）
    const int64_t rawG = static_cast<int64_t>(gCh) * haloW * eg;          // guide 原始行（输入 dtype）
    const int64_t rawS = static_cast<int64_t>(sCh) * haloW * es;          // src 原始行（输入 dtype）
    // 内核至少分配 32B（HlfBytes/ConvBytes 的下限），预算取两种口径的较大者
    const int64_t hlf = static_cast<int64_t>(
        (eg == 1 || es == 1) ? (static_cast<int64_t>(gCh > sCh ? gCh : sCh) * haloW * 2) : 32); // uint8->half 中转
    // 水平滑窗：移位副本 shBuf_（rows*strideT）+ 偏移索引表 idxBuf_（按 64 项对齐）
    const int64_t shSlots = rows * mb * strideT;
    const int64_t shf = (shSlots + IdxPad(static_cast<int32_t>(shSlots))) * 4;
    // 去交织索引表三段（guide / src / 系数）+ 输出阶段 guide 段，每段都按 64 项对齐，
    // 且每段尾部再留 kIdxAlign 项；另有输出交织表 IlIdxSlots（按 64 项对齐）。
    // 必须与 kernel 的 SegSrc()/SegCoef()/SegGuideOut()/DeIdxSlots()/IlIdxSlots() 逐一对应。
    const int64_t segGuide = IdxPad(gCh * haloW);
    const int64_t segSrc = IdxPad(sCh * haloW);
    // 系数行重排表：平面数够多时一次 Gather 整批重排（K 个平面），否则只占 1 个平面。
    // 阈值必须与 kernel 的 kCoefGatherMin 一致。
    const int64_t segCoef = IdxPad((coefNum >= 4) ? coefNum * haloW : haloW);
    const int64_t deIdx = (segGuide + segSrc + segCoef + segGuide + 4 * 64 + IdxPad(strideT * sCh)) * 4;
    // det 退化快判的归约暂存（ReduceMin/ReduceMax 的 dst 8+8 + sharedTmpBuffer strideT）
    const int64_t red = static_cast<int64_t>(16 + strideT) * 4;
    // StageB 系数行合批暂存（K 行 × haloW）
    const int64_t coefRow = static_cast<int64_t>(coefNum) * haloW * 4;
    // 非 float 输入的交错行 float 中转（内核最少分配 32B）
    const int64_t conv = (eg != 4 || es != 4) ? static_cast<int64_t>(gCh > sCh ? gCh : sCh) * haloW * 4 : 32;
    const int64_t qf = static_cast<int64_t>(sCh) * strideT * 4;   // 输出 q 平面
    const int64_t il = static_cast<int64_t>(strideT) * sCh * 4;   // 交织后的输出行（float）
    const int64_t out = static_cast<int64_t>(strideT) * sCh * ed; // 交织后的输出行（输出 dtype）
    return ring + vAll + box + aux + gRow + pRow + rawG + rawS + hlf + shf + deIdx + conv + red + coefRow + qf + il +
           out + kFixedBytes;
}

// 由形状/属性推导 tile 几何：先按 UB 预算取最大列宽，再按核数做二维任务划分。
inline Geo DeriveGeo(int32_t width, int32_t height, int32_t batch, int32_t radius, int32_t gCh, int32_t sCh, int32_t eg,
                     int32_t es, int32_t ed, int32_t coreNum)
{
    Geo g;
    const int32_t atomNum = AtomNum(gCh, sCh);
    const int32_t coefNum = CoefNum(gCh, sCh);

    int32_t tStart = (width < kMaxTileW) ? width : kMaxTileW;
    if (tStart > 8) {
        tStart = (tStart / 8) * 8;
    }
    int32_t bestT = 0;
    for (int32_t t = tStart; t >= 1;) {
        const int32_t haloW = Align8(t + 2 * radius);
        if (UbNeed(t, haloW, radius, gCh, sCh, atomNum, coefNum, eg, es, ed) <= kUbBudgetBytes) {
            bestT = t;
            break;
        }
        if (t <= 8) {
            break;
        }
        t = (t - 8 < 8) ? 8 : (t - 8);
    }
    if (bestT <= 0) {
        return g; // UB 放不下哪怕 1 列 -> host 侧报错
    }

    const int32_t nTxMin = CeilDivI(width, bestT);
    const int32_t cores = (coreNum > 0) ? coreNum : 1;
    int32_t maxTy = height / (2 * radius + 1);
    if (maxTy < 1) {
        maxTy = 1;
    }

    // 在 UB 可行的前提下挑 nTx：tileW 越大，向量指令的单次吞吐越高（repeat 数更多），
    // 但每个 tile 两侧各带 r 列 halo；同时 nTx*nTy 越贴近 coreNum 的整数倍，
    // wave 利用率越高（否则会出现"部分核跑 2 轮、部分核空等"）。
    // 这里按 util * (有效列占比) 打分，同分取更大的 tileW。
    int32_t bestT2 = 0;
    int32_t bestNTx = 0;
    int64_t bestScore = -1;
    const int32_t nTxTryEnd = (nTxMin * 3 < width) ? (nTxMin * 3) : width;
    for (int32_t nTx = nTxMin; nTx <= nTxTryEnd; ++nTx) {
        const int32_t tw = CeilDivI(width, nTx);
        if (tw < 8) {
            break;
        }
        // 水平滑窗的移位表会读 v[i+k]（i < strideT，k <= 2r），且 halo 起点额外让出最多 7 列（对齐用），
        // 必须保证不越过行尾
        int32_t hw = Align8(tw + 2 * radius + 8);
        const int32_t strideT = Align8(tw);
        if (hw < strideT + 2 * radius + 8) {
            hw = Align8(strideT + 2 * radius + 8);
        }
        if (UbNeed(tw, hw, radius, gCh, sCh, atomNum, coefNum, eg, es, ed) > kUbBudgetBytes) {
            // 注意必须 continue 而不是 break：haloW = Align8(tw + 2r + 8) 是阶跃函数，
            // UB 需求对 nTx 并不单调 —— 某个 nTx 恰好越过 8 列台阶就会超预算，
            // 但它后面更窄的 tileW 完全可能重新落在预算内。用 break 会漏掉这些候选，
            // 实测曾因此把 720p 3->3 从 nTx=20(util 1000) 误选成 nTx=11(util 550)，慢 36%。
            continue;
        }
        int32_t nTy = CeilDivI(cores, nTx * batch);
        if (nTy < 1) {
            nTy = 1;
        }
        if (nTy > maxTy) {
            nTy = maxTy;
        }
        const int32_t bandHTry = CeilDivI(height, nTy);
        nTy = CeilDivI(height, bandHTry);
        const int64_t tasks = static_cast<int64_t>(nTx) * nTy * batch;
        const int64_t waves = (tasks + cores - 1) / cores;
        const int64_t util = tasks * 1000 / (waves * cores);                     // 0..1000
        const int64_t eff = static_cast<int64_t>(tw) * 1000 / (tw + 2 * radius); // halo 效率
        const int64_t score = util * eff * 1000 / 1000 + tw;                     // 同分取大 tile
        if (score > bestScore) {
            bestScore = score;
            bestT2 = tw;
            bestNTx = nTx;
        }
    }
    if (bestNTx <= 0) {
        bestT2 = CeilDivI(width, nTxMin);
        bestNTx = nTxMin;
    }

    const int32_t nTx = bestNTx;
    const int32_t tileW = bestT2;
    int32_t haloW = Align8(tileW + 2 * radius + 8);
    if (haloW < Align8(tileW) + 2 * radius + 8) {
        haloW = Align8(Align8(tileW) + 2 * radius + 8);
    }

    // 行方向：让 nTx*nTy*N 覆盖所有核，同时限制行带不小于一个窗口高度（避免 halo 占比过高）
    int32_t nTy = CeilDivI(cores, nTx * batch);
    if (nTy < 1) {
        nTy = 1;
    }
    if (nTy > maxTy) {
        nTy = maxTy;
    }
    int32_t bandH = CeilDivI(height, nTy);
    nTy = CeilDivI(height, bandH);

    const int64_t tasks = static_cast<int64_t>(nTx) * nTy * batch;
    g.tileW = tileW;
    g.haloW = haloW;
    g.nTx = nTx;
    g.nTy = nTy;
    g.bandH = bandH;
    g.blockDim = static_cast<uint32_t>(tasks < cores ? tasks : cores);
    g.ubBytes = UbNeed(tileW, haloW, radius, gCh, sCh, atomNum, coefNum, eg, es, ed);
    g.ok = (g.blockDim > 0);
    return g;
}
} // namespace GfGeo

#endif // __GUIDED_FILTER_TILLING_DATA_H__
