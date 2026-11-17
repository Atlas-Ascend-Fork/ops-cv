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
 * \file guided_filter.h
 * \brief GuidedFilter（导向滤波）AscendC 内核实现，语义对齐 cv::ximgproc::guidedFilter
 *
 * 算法（与 OpenCV 一致的「显式推导 + 盒式滤波」实现，不使用积分图）：
 *   StageA：对每个 tile 求出 2r+1 窗口内的
 *             μ_c = box(g_c)、μ_cd = box(g_c·g_d)、ν_s = box(p_s)、ω_cs = box(g_c·p_s)
 *           （共 M = C + C(C+1)/2 + Cs + C·Cs 张「原子图」，C=Cs=3 时 M=21），
 *           再按像素做 3x3（或 1x1/2x2）协方差求逆，得到 alpha(c,s) / beta(s) 写入 workspace。
 *   StageB：对 alpha/beta 再各做一次盒式滤波（共 K = C·Cs + Cs 张，C=Cs=3 时 K=12），
 *           最后 q = mean_a · g + mean_b 并按输出 dtype 交织写回。
 *   两阶段之间用硬件 SyncAll 做跨核同步（KERNEL_TYPE_MIX_AIV_1_0 + batch 调度）。
 *
 * 本实现依赖以下平台硬约束（都是踩过的坑，改动前务必确认）：
 *   1. 向量指令的 LocalTensor 起始地址必须 32B 对齐，所以 1..2r 个元素的位移不能直接
 *      做偏移，必须先用 Gather 生成「移位副本」。
 *   2. Gather 的最后一个 64 元素 repeat 会整块读满偏移表 —— 索引表必须按 64 项对齐分配
 *      且尾项填 0，否则越界读到的相邻数据会被当成地址，表现为「偶发几十个像素出错」。
 *   3. Gather 不在 V 流水上执行，读它的结果前必须 PipeBarrier<PIPE_V>()。
 *   4. MTE2 -> 向量（Gather/Cast）只做 PipeBarrier<PIPE_MTE2>() 不够，必须 PIPE_ALL。
 *   5. MTE3 读 UB 的 WAR：本轮 DMA 还没读完时下一轮不能覆写同一块 UB，
 *      相关函数末尾必须 PipeBarrier<PIPE_MTE3>()。
 */
#ifndef __GUIDED_FILTER_H__
#define __GUIDED_FILTER_H__

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "guided_filter_tiling_data.h"
#include "guided_filter_tiling_key.h"

namespace NsGuidedFilter {

using namespace AscendC;

// aux 缓冲槽位：[0,9) = 余子式（按 c*3+d 下标，含对称冗余）、9 = det/临时、
// [10, 10+K) = alpha(c*Cs+s) 与 beta(s)，即 kAuxFixed 之后接 K 个槽
constexpr int32_t kAuxFixed = 10;
constexpr int32_t kDetSlot = 9;

// det 的退化阈值：|det| < 此值时把 det 置 1（与 OpenCV 的 guidedFilter 一致）
constexpr float kDetDegThresh = 1e-6F;

template <typename TG, typename TS, typename TD>
class GuidedFilter {
public:
    __aicore__ inline void Init(GM_ADDR guide, GM_ADDR src, GM_ADDR dst, GM_ADDR workspace,
                                const GuidedFilterTilingData* tiling);
    __aicore__ inline void Process();

private:
    struct Task {
        int32_t img;
        int32_t y0;
        int32_t y1;
        int32_t x0;
        int32_t x1;
    };

    // ---- 基础工具 ----
    __aicore__ inline int32_t MirrorIdx(int32_t i, int32_t n) const;
    __aicore__ inline LocalTensor<float> RingSlot(int32_t map, int32_t row);
    // 环形缓冲布局：ring[row%ringN_][map][col]。
    //
    // map 放在行内（步长 haloW）而不是行间，是为了让「垂直窗口和」一次覆盖全部 map：
    //   Add(vAll, vAll, ring[RingRowOff(y+r)], mapCount*haloW)
    // vAll 本身就是 [map][col] 连续布局，与之一致。若按 [map][row] 布局，每张图都要
    // 单独发一条 cnt=haloW 的短指令，2r+1 张图就是几十条 —— 垂直求和曾因此占掉相当大
    // 一块指令条数开销，改成整批一条后 3->3 又降约 25%。
    __aicore__ inline int32_t RingRowOff(int32_t row) const { return (row % ringN_) * (M_ * HW_); }
    __aicore__ inline int32_t RingMapStride() const { return HW_; }
    __aicore__ inline int32_t CovIdx(int32_t c, int32_t d) const;
    __aicore__ inline int32_t NuIdx(int32_t s) const;
    __aicore__ inline int32_t OmIdx(int32_t c, int32_t s) const;
    // 对称伴随矩阵的 aux 槽位：00->0, 01->1, 02->2, 11->3, 12->4, 22->5
    // （不能用 c*3+d：C=3 时 20/21 会与 num/tmp 的 6/7 号槽冲突）
    __aicore__ inline int32_t AdjSlot(int32_t c, int32_t d) const;
    __aicore__ inline Task DecodeTask(int32_t tid) const;

    // ---- 行装填 ----
    template <typename TX>
    __aicore__ inline void LoadRowToPlanar(const GlobalTensor<TX>& gm, LocalTensor<TX> raw, LocalTensor<uint32_t> idx,
                                           LocalTensor<float> dst, int32_t ch, int32_t img, int32_t y, int32_t xs,
                                           int32_t span);
    __aicore__ inline void BuildIndexTables();
    __aicore__ inline void BuildDeintIndex(LocalTensor<uint32_t> idx, int32_t ch, int32_t xs, int32_t span);
    __aicore__ inline void BuildPlanarIndex(LocalTensor<uint32_t> idx, int32_t ch, int32_t xs, int32_t span);
    __aicore__ inline void LoadGuideRow(int32_t img, int32_t y, int32_t xs, int32_t span);
    __aicore__ inline void LoadGuideRowOut(int32_t img, int32_t y, int32_t xs, int32_t span);
    __aicore__ inline void LoadSrcRow(int32_t img, int32_t y, int32_t xs, int32_t span);

    // ---- 盒式滤波 ----
    __aicore__ inline void FillAtomRing(int32_t img, int32_t y, int32_t xs, int32_t span);
    __aicore__ inline void FillCoefRing(int32_t img, int32_t y, int32_t xs, int32_t span);
    __aicore__ inline void ComputeBoxes(int32_t y, int32_t tw, int32_t span, int32_t mapCount, bool seed);
    __aicore__ inline void BuildShiftIndex();
    __aicore__ inline void HorizontalRaw(LocalTensor<float> v, LocalTensor<float> out, int32_t srcElems);
    __aicore__ inline void ClampDetDegen(LocalTensor<float> det, int32_t tw);

    // 非 float 输入时需要「交错行 float 中转」；uint8 输入时还需要 uint8->half 中转
    static constexpr bool kNonFloatIn = !(std::is_same_v<TG, float> && std::is_same_v<TS, float>);
    static constexpr bool kHasU8In = std::is_same_v<TG, uint8_t> || std::is_same_v<TS, uint8_t>;
    __aicore__ inline uint32_t ChMax() const { return static_cast<uint32_t>((C_ > Cs_) ? C_ : Cs_); }
    __aicore__ inline uint32_t HlfBytes() const { return kHasU8In ? (ChMax() * static_cast<uint32_t>(HW_) * 2U) : 32U; }
    __aicore__ inline uint32_t ConvBytes() const
    {
        return kNonFloatIn ? (ChMax() * static_cast<uint32_t>(HW_) * 4U) : 32U;
    }

    // ---- 系数与输出 ----
    __aicore__ inline void BuildAlphaBeta(int32_t tw);
    __aicore__ inline void EmitAlphaBeta(const Task& t, int32_t y, int32_t tw);
    __aicore__ inline void BuildOutRow(const Task& t, int32_t y, int32_t tw, int32_t span);

    // ---- 阶段 ----
    __aicore__ inline void StageA(const Task& t);
    __aicore__ inline void StageB(const Task& t);

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ringBuf_;    // M * (2r+2) * haloW：原子/系数图行环（环深 2r+2）
    TBuf<QuePosition::VECCALC> vBuf_;       // M * haloW：每图的垂直窗口和
    TBuf<QuePosition::VECCALC> boxBuf_;     // M * strideT：每图的盒式滤波结果
    TBuf<QuePosition::VECCALC> auxBuf_;     // (10+K) * strideT
    TBuf<QuePosition::VECCALC> redBuf_;     // 16 + strideT：det 退化快判的归约暂存
    TBuf<QuePosition::VECCALC> coefRowBuf_; // K * haloW：StageB 系数行合批暂存
    TBuf<QuePosition::VECCALC> gRowBuf_;    // C * haloW：guide 行（float 平面）
    TBuf<QuePosition::VECCALC> pRowBuf_;    // Cs * haloW：src 行（float 平面）
    TBuf<QuePosition::VECCALC> rawGBuf_;    // C * haloW：guide 原始行（输入 dtype）
    TBuf<QuePosition::VECCALC> rawSBuf_;    // Cs * haloW：src 原始行（输入 dtype）
    TBuf<QuePosition::VECCALC> hlfBuf_;     // uint8 -> half 中转（仅 uint8 输入；其余至少 32B）
    TBuf<QuePosition::VECCALC> convBuf_;    // 非 float 输入的交错行 float 中转（其余至少 32B）
    // 去交织索引表，共 4 段且每段都按 64 项对齐（段间再留 kIdxAlign 项）：
    //   [0, SegSrc)          guide 行（StageA，xs = x0 - R_，含左 halo）
    //   [SegSrc, SegCoef)    src 行
    //   [SegCoef, SegGuideOut)  系数行（单通道，StageB）
    //   [SegGuideOut, DeIdxSlots) guide 行（StageB 输出阶段，xs = x0，不含 halo）
    TBuf<QuePosition::VECCALC> deIdxBuf_;
    TBuf<QuePosition::VECCALC> shBuf_;    // rows * strideT：水平滑动窗口用的「移位副本」
    TBuf<QuePosition::VECCALC> idxBuf_;   // IdxSlots() = align64(rows*strideT)：移位副本的字节偏移索引表
    TBuf<QuePosition::VECCALC> ilIdxBuf_; // IlIdxSlots() = align64(strideT*Cs)：输出交织索引表
    TBuf<QuePosition::VECCALC> qBuf_;     // Cs * strideT：每通道输出 q
    TBuf<QuePosition::VECCALC> ilBuf_;    // strideT * Cs：交织后的输出行（float）
    TBuf<QuePosition::VECCALC> outBuf_;   // strideT * Cs：交织后的输出行（输出 dtype）

    // Gather 的最后一个 repeat 会读满整个 64 项偏移块，所以每张索引表都必须按 64 项对齐
    // 分配并把尾项填 0 —— 否则会越界读到相邻数据当作偏移，随机踩到 UB 其他位置
    // （表现为"偶发几十个像素出错"，极难复现）。
    static constexpr int32_t kIdxAlign = 64;
    __aicore__ inline int32_t IdxPad(int32_t n) const { return (n + kIdxAlign - 1) / kIdxAlign * kIdxAlign; }
    __aicore__ inline int32_t SegSrc() const { return IdxPad(C_ * HW_); }
    __aicore__ inline int32_t SegCoef() const { return SegSrc() + IdxPad(Cs_ * HW_); }
    // StageB 输出阶段只需 guide 的「输出列」（xs = t.x0，不含左 halo），
    // 与 StageA 的 xs = t.x0 - R_ 不同，必须单独一段索引表。
    // 系数行重排表：合批时覆盖 K_ 个平面（一次 Gather 整批搬入环槽），
    // 平面数很少（K_ < kCoefGatherMin）时只占 1 个平面 —— 逐平面 Gather 更快，
    // 也不该为用不到的表付 UB。必须与 host 侧 GfGeo::CoefGatherNum 一致。
    static constexpr int32_t kCoefGatherMin = 4;
    __aicore__ inline int32_t CoefGatherNum() const { return (K_ >= kCoefGatherMin) ? K_ : 1; }
    __aicore__ inline int32_t SegGuideOut() const { return SegCoef() + IdxPad(CoefGatherNum() * HW_) + kIdxAlign; }
    __aicore__ inline int32_t DeIdxSlots() const { return SegGuideOut() + IdxPad(C_ * HW_) + kIdxAlign; }
    // 水平滑窗合批张数：一次 Add 覆盖 mapBatch_ 张图。必须与 host 侧 GfGeo::MapBatch 一致。
    __aicore__ inline int32_t MapBatchK() const
    {
        const int32_t perMap = 2 * rows_ * strideT_ * 4; // 移位副本 + 偏移表
        int32_t g = (perMap > 0) ? (GfGeo::kMapBatchBytes / perMap) : 1;
        if (g < 1) {
            g = 1;
        }
        if (g > M_) {
            g = M_;
        }
        return g;
    }
    // vAll / box 按合批粒度向上取整（末批允许有空图占位，其值为垃圾但不会被消费）
    __aicore__ inline int32_t MapSlots() const { return (M_ + mapBatch_ - 1) / mapBatch_ * mapBatch_; }
    // 每个槽位覆盖全部合批图，所以「一份移位副本」= mapBatch_ * strideT_ 个元素
    __aicore__ inline int32_t ShSlots() const { return rows_ * mapBatch_ * strideT_; }
    __aicore__ inline int32_t IdxSlots() const { return IdxPad(rows_ * mapBatch_ * strideT_); }
    __aicore__ inline int32_t IlIdxSlots() const { return IdxPad(strideT_ * Cs_); }

    GlobalTensor<TG> guideGm_;
    GlobalTensor<TS> srcGm_;
    GlobalTensor<TD> dstGm_;
    GlobalTensor<float> alphaGm_;
    GlobalTensor<float> betaGm_;
    const GuidedFilterTilingData* t_ = nullptr;

    int32_t N_ = 1;
    int32_t H_ = 0;
    int32_t W_ = 0;
    int32_t C_ = 1;
    int32_t Cs_ = 1;
    int32_t R_ = 0;
    int32_t T_ = 0;
    int32_t HW_ = 0; // haloW
    int32_t nTx_ = 1;
    int32_t nTy_ = 1;
    int32_t bandH_ = 1;
    int32_t rows_ = 1;     // 2r+1（窗口行数）
    int32_t ringN_ = 2;    // 2r+2（环形缓冲深度：增量垂直求和需要同时存活 y-r-1 与 y+r）
    int32_t strideT_ = 0;  // align8(tileW)
    int32_t M_ = 0;        // atomNum
    int32_t mapBatch_ = 1; // 水平滑窗合批张数（见 MapBatchK）
    int32_t K_ = 0;        // coefNum
    int32_t planeElems_ = 0;
    float invWin_ = 1.0F;
    float eps_ = 0.0F;
};

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::Init(GM_ADDR guide, GM_ADDR src, GM_ADDR dst, GM_ADDR workspace,
                                                      const GuidedFilterTilingData* tiling)
{
    t_ = tiling;
    N_ = static_cast<int32_t>(tiling->batch);
    H_ = static_cast<int32_t>(tiling->height);
    W_ = static_cast<int32_t>(tiling->width);
    C_ = static_cast<int32_t>(tiling->guideCh);
    Cs_ = static_cast<int32_t>(tiling->srcCh);
    R_ = static_cast<int32_t>(tiling->radius);
    T_ = static_cast<int32_t>(tiling->tileW);
    HW_ = static_cast<int32_t>(tiling->haloW);
    nTx_ = static_cast<int32_t>(tiling->nTx);
    nTy_ = static_cast<int32_t>(tiling->nTy);
    bandH_ = static_cast<int32_t>(tiling->bandH);
    rows_ = 2 * R_ + 1;
    // 环深 = 窗口行数 + 1：垂直和改成增量式后，V(y) 需要同时拿到 y+r（新进）与 y-r-1（要减掉的），
    // 二者相差 2r+1 行，若环深仍为 2r+1 就会槽位相撞。
    ringN_ = rows_ + 1;
    // 注意：这里不能用 GfGeo::Align8 —— 该头文件同时给 host 用，不能标 __aicore__，
    // 在 device 侧会留下未解析符号（ld.lld: undefined symbol GfGeo::Align8）。
    strideT_ = (T_ + 7) & ~7;
    M_ = static_cast<int32_t>(tiling->atomNum);
    K_ = static_cast<int32_t>(tiling->coefNum);
    mapBatch_ = MapBatchK();
    planeElems_ = static_cast<int32_t>(tiling->planeElems);
    invWin_ = tiling->invWin;
    eps_ = tiling->eps;

    guideGm_.SetGlobalBuffer((__gm__ TG*)guide, static_cast<uint64_t>(N_) * H_ * W_ * C_);
    srcGm_.SetGlobalBuffer((__gm__ TS*)src, static_cast<uint64_t>(N_) * H_ * W_ * Cs_);
    dstGm_.SetGlobalBuffer((__gm__ TD*)dst, static_cast<uint64_t>(N_) * H_ * W_ * Cs_);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    alphaGm_.SetGlobalBuffer((__gm__ float*)(userWs + tiling->alphaOff),
                             static_cast<uint64_t>(N_) * C_ * Cs_ * planeElems_);
    betaGm_.SetGlobalBuffer((__gm__ float*)(userWs + tiling->betaOff), static_cast<uint64_t>(N_) * Cs_ * planeElems_);

    pipe_.InitBuffer(ringBuf_, static_cast<uint32_t>(M_) * ringN_ * HW_ * 4U);
    pipe_.InitBuffer(vBuf_, static_cast<uint32_t>(MapSlots()) * HW_ * 4U);
    pipe_.InitBuffer(boxBuf_, static_cast<uint32_t>(MapSlots()) * strideT_ * 4U);
    pipe_.InitBuffer(auxBuf_, static_cast<uint32_t>(kAuxFixed + K_) * strideT_ * 4U);
    pipe_.InitBuffer(gRowBuf_, static_cast<uint32_t>(C_) * HW_ * 4U);
    pipe_.InitBuffer(pRowBuf_, static_cast<uint32_t>(Cs_) * HW_ * 4U);
    pipe_.InitBuffer(rawGBuf_, static_cast<uint32_t>(C_) * HW_ * sizeof(TG));
    pipe_.InitBuffer(rawSBuf_, static_cast<uint32_t>(Cs_) * HW_ * sizeof(TS));
    pipe_.InitBuffer(hlfBuf_, static_cast<uint32_t>(HlfBytes()));
    pipe_.InitBuffer(convBuf_, static_cast<uint32_t>(ConvBytes()));
    pipe_.InitBuffer(deIdxBuf_, static_cast<uint32_t>(DeIdxSlots()) * 4U);
    pipe_.InitBuffer(shBuf_, static_cast<uint32_t>(ShSlots()) * 4U);
    pipe_.InitBuffer(redBuf_, static_cast<uint32_t>(16 + strideT_) * 4U);
    pipe_.InitBuffer(coefRowBuf_, static_cast<uint32_t>(K_ * HW_) * 4U);
    pipe_.InitBuffer(idxBuf_, static_cast<uint32_t>(IdxSlots()) * 4U);
    pipe_.InitBuffer(ilIdxBuf_, static_cast<uint32_t>(IlIdxSlots()) * 4U);
    pipe_.InitBuffer(qBuf_, static_cast<uint32_t>(Cs_) * strideT_ * 4U);
    pipe_.InitBuffer(ilBuf_, static_cast<uint32_t>(strideT_) * Cs_ * 4U);
    pipe_.InitBuffer(outBuf_, static_cast<uint32_t>(strideT_) * Cs_ * sizeof(TD));

    // 环槽每行只写 [0, span) 段，[span, haloW) 尾段从不写；而垂直/水平滑窗按 haloW
    // 整段求和会读到尾段。这里一次性清零，保证行为确定（这些列只污染 box 的
    // [tw, strideT) 区间，下游不读）。
    Duplicate(ringBuf_.Get<float>(), 0.0F, static_cast<int32_t>(M_ * ringN_ * HW_));
    PipeBarrier<PIPE_V>();

    // 索引表只与 tile 几何 / 通道数相关，整个 kernel 只需构建一次
    BuildIndexTables();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline int32_t GuidedFilter<TG, TS, TD>::MirrorIdx(int32_t i, int32_t n) const
{
    // cv::BORDER_REFLECT: fedcba|abcdefgh|hgfedcb，周期 2n（-1 -> 0, -2 -> 1, n -> n-1）
    // 配合 LoadRowToPlanar 的列映射：image 列 x 落在 UB 内的下标 (x - xs)，xs 是本 tile 的
    // halo 起点（可为负）。xs < 0 时 DMA 只能从第 0 列开始搬，整行在 UB 里等效右移 r 列，
    // 这个偏移被折进去交织索引表，从而保证 Gather 的 dst 永远 32B 对齐。
    const int32_t period = 2 * n;
    if (period <= 0) {
        return 0;
    }
    int32_t j = i % period;
    if (j < 0) {
        j += period;
    }
    return (j < n) ? j : (period - 1 - j);
}

template <typename TG, typename TS, typename TD>
__aicore__ inline LocalTensor<float> GuidedFilter<TG, TS, TD>::RingSlot(int32_t map, int32_t row)
{
    return ringBuf_.Get<float>()[RingRowOff(row) + map * RingMapStride()];
}

template <typename TG, typename TS, typename TD>
__aicore__ inline int32_t GuidedFilter<TG, TS, TD>::CovIdx(int32_t c, int32_t d) const
{
    int32_t k = 0;
    for (int32_t i = 0; i < c; ++i) {
        k += (C_ - i);
    }
    k += (d - c);
    return C_ + k;
}

template <typename TG, typename TS, typename TD>
__aicore__ inline int32_t GuidedFilter<TG, TS, TD>::NuIdx(int32_t s) const
{
    return C_ + C_ * (C_ + 1) / 2 + s;
}

template <typename TG, typename TS, typename TD>
__aicore__ inline int32_t GuidedFilter<TG, TS, TD>::OmIdx(int32_t c, int32_t s) const
{
    return C_ + C_ * (C_ + 1) / 2 + Cs_ + c * Cs_ + s;
}

template <typename TG, typename TS, typename TD>
__aicore__ inline int32_t GuidedFilter<TG, TS, TD>::AdjSlot(int32_t c, int32_t d) const
{
    if (c > d) {
        const int32_t sw = c;
        c = d;
        d = sw;
    }
    if (c == 0) {
        return (d == 0) ? 0 : ((d == 1) ? 1 : 2);
    }
    if (c == 1) {
        return (d == 1) ? 3 : 4;
    }
    return 5;
}

template <typename TG, typename TS, typename TD>
__aicore__ inline typename GuidedFilter<TG, TS, TD>::Task GuidedFilter<TG, TS, TD>::DecodeTask(int32_t tid) const
{
    Task t;
    const int32_t perImg = nTx_ * nTy_;
    t.img = tid / perImg;
    const int32_t rem = tid % perImg;
    const int32_t ty = rem / nTx_;
    const int32_t tx = rem % nTx_;
    t.y0 = ty * bandH_;
    t.y1 = (t.y0 + bandH_ < H_) ? (t.y0 + bandH_) : H_;
    t.x0 = tx * T_;
    t.x1 = (t.x0 + T_ < W_) ? (t.x0 + T_) : W_;
    return t;
}

template <typename TG, typename TS, typename TD>
template <typename TX>
__aicore__ inline void GuidedFilter<TG, TS, TD>::LoadRowToPlanar(const GlobalTensor<TX>& gm, LocalTensor<TX> raw,
                                                                 LocalTensor<uint32_t> idx, LocalTensor<float> dst,
                                                                 int32_t ch, int32_t img, int32_t y, int32_t xs,
                                                                 int32_t span)
{
    int32_t cs = (xs < 0) ? 0 : xs;
    int32_t ce = xs + span;
    if (ce > W_) {
        ce = W_;
    }
    const int32_t cl = (ce > cs) ? (ce - cs) : 0;
    if (cl <= 0) {
        return;
    }
    const int64_t base = ((static_cast<int64_t>(img) * H_ + y) * W_ + cs) * ch;
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint16_t>(cl * ch * static_cast<int32_t>(sizeof(TX)));
    p.srcStride = 0;
    p.dstStride = 0;
    DataCopyPad(raw, gm[base], p, {false, 0, 0, 0});
    // 实测：MTE2 -> 向量不能只做 PIPE_MTE2 屏障（结果会错），必须全流水同步。
    PipeBarrier<PIPE_ALL>();

    // 先把交错行统一转成 float（uint8/half 走一次或两次 Cast），
    // 再用一次 Gather 按 BORDER_REFLECT 索引摊平成 ch 张 float 平面。
    LocalTensor<float> inter;
    if constexpr (std::is_same_v<TX, float>) {
        inter = raw; // 已是 float 交错行，无需中转
    } else if constexpr (std::is_same_v<TX, half>) {
        inter = convBuf_.Get<float>();
        Cast(inter, raw, RoundMode::CAST_NONE, static_cast<uint32_t>(cl * ch));
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<half> tmp = hlfBuf_.Get<half>();
        Cast(tmp, raw, RoundMode::CAST_NONE, static_cast<uint32_t>(cl * ch));
        PipeBarrier<PIPE_V>();
        inter = convBuf_.Get<float>();
        Cast(inter, tmp, RoundMode::CAST_NONE, static_cast<uint32_t>(cl * ch));
        PipeBarrier<PIPE_V>();
    }
    // dst[c*HW_ + j] = inter[(MirrorIdx(xs+j) - cs)*ch + c]
    // 用预建的字节偏移表一次 Gather 摊平一个通道：每行 ch 条向量指令，
    for (int32_t c = 0; c < ch; ++c) {
        Gather(dst[c * HW_], inter, idx[c * HW_], 0U, static_cast<uint32_t>(span));
    }
    PipeBarrier<PIPE_V>(); // Gather 不在 V 流水上执行，读它的结果前必须同步
}

// 构建只与 tile 几何相关的索引表：水平滑窗的移位副本表 + 输出交织表。
template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::BuildIndexTables()
{
    BuildShiftIndex();

    // 输出交织：out[j*Cs + s] = q[s*strideT + j]
    LocalTensor<uint32_t> ilIdx = ilIdxBuf_.Get<uint32_t>();
    for (int32_t j = 0; j < strideT_; ++j) {
        for (int32_t s = 0; s < Cs_; ++s) {
            ilIdx.SetValue(j * Cs_ + s,
                           static_cast<uint32_t>((s * strideT_ + j) * static_cast<int32_t>(sizeof(float))));
        }
    }
    for (int32_t k = strideT_ * Cs_; k < IlIdxSlots(); ++k) {
        ilIdx.SetValue(k, 0U); // Gather 末轮会读满 64 项，尾项必须安全
    }
    PipeBarrier<PIPE_ALL>();
}

// 去交织索引表（每个 task 的 xs 不同，所以在 Stage 开头重建；规模仅 ch*haloW，可忽略）。
template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::BuildDeintIndex(LocalTensor<uint32_t> idx, int32_t ch, int32_t xs,
                                                                 int32_t span)
{
    const int32_t cs = (xs < 0) ? 0 : xs;
    for (int32_t c = 0; c < ch; ++c) {
        for (int32_t j = 0; j < HW_; ++j) {
            uint32_t off = 0U; // j >= span 的填充项指向 0，保证 Gather 恒在界内
            if (j < span) {
                const int32_t mcol = MirrorIdx(xs + j, W_);
                off = static_cast<uint32_t>(((mcol - cs) * ch + c) * static_cast<int32_t>(sizeof(float)));
            }
            idx.SetValue(c * HW_ + j, off);
        }
    }
    for (int32_t k = ch * HW_; k < IdxPad(ch * HW_); ++k) {
        idx.SetValue(k, 0U); // Gather 末轮会读满 64 项，尾项必须安全
    }
    PipeBarrier<PIPE_ALL>();
}

// 平面布局的行重排表：第 c 个平面独立占一行（步长 haloW），
//   idx[c*haloW + j] = (c*haloW + mirror(j) - cs) * sizeof(float)
// 与 BuildDeintIndex 的区别只在源地址是「平面」而不是「按通道交织」。
// 用它可以一次 Gather 把 K_ 个系数平面整批搬进环槽（环槽的 map 维本来就在行内连续）。
template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::BuildPlanarIndex(LocalTensor<uint32_t> idx, int32_t ch, int32_t xs,
                                                                  int32_t span)
{
    const int32_t cs = (xs < 0) ? 0 : xs;
    for (int32_t c = 0; c < ch; ++c) {
        for (int32_t j = 0; j < HW_; ++j) {
            uint32_t off = 0U; // j >= span 的填充项指向本平面首元素，保证 Gather 恒在界内
            if (j < span) {
                const int32_t mcol = MirrorIdx(xs + j, W_);
                off = static_cast<uint32_t>((c * HW_ + (mcol - cs)) * static_cast<int32_t>(sizeof(float)));
            }
            idx.SetValue(c * HW_ + j, off);
        }
    }
    for (int32_t k = ch * HW_; k < IdxPad(ch * HW_); ++k) {
        idx.SetValue(k, 0U); // Gather 末轮会读满 64 项，尾项必须安全
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::LoadGuideRow(int32_t img, int32_t y, int32_t xs, int32_t span)
{
    LocalTensor<uint32_t> idx = deIdxBuf_.Get<uint32_t>();
    LoadRowToPlanar(guideGm_, rawGBuf_.Get<TG>(), idx, gRowBuf_.Get<float>(), C_, img, y, xs, span);
}

// StageB 输出阶段：guide 只取输出列（xs = t.x0，不含左 halo），用单独一段索引表。
template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::LoadGuideRowOut(int32_t img, int32_t y, int32_t xs, int32_t span)
{
    LocalTensor<uint32_t> idx = deIdxBuf_.Get<uint32_t>()[SegGuideOut()];
    LoadRowToPlanar(guideGm_, rawGBuf_.Get<TG>(), idx, gRowBuf_.Get<float>(), C_, img, y, xs, span);
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::LoadSrcRow(int32_t img, int32_t y, int32_t xs, int32_t span)
{
    LocalTensor<uint32_t> idx = deIdxBuf_.Get<uint32_t>()[SegSrc()];
    LoadRowToPlanar(srcGm_, rawSBuf_.Get<TS>(), idx, pRowBuf_.Get<float>(), Cs_, img, y, xs, span);
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::FillAtomRing(int32_t img, int32_t y, int32_t xs, int32_t span)
{
    LoadGuideRow(img, y, xs, span);
    LoadSrcRow(img, y, xs, span);

    LocalTensor<float> g = gRowBuf_.Get<float>();
    LocalTensor<float> p = pRowBuf_.Get<float>();
    LocalTensor<float> ring = ringBuf_.Get<float>();
    const uint32_t cnt = static_cast<uint32_t>(span);
    const int32_t roff = RingRowOff(y); // 本行所在环槽，行内所有 map 共用
    const int32_t mstr = RingMapStride();
    int32_t m = 0;
    // μ_c = g_c
    for (int32_t c = 0; c < C_; ++c) {
        Adds(ring[roff + m * mstr], g[c * HW_], 0.0F, cnt);
        ++m;
    }
    // Σ_cd 的原子 = g_c * g_d (c <= d)
    for (int32_t c = 0; c < C_; ++c) {
        for (int32_t d = c; d < C_; ++d) {
            if (c == d) {
                Mul(ring[roff + m * mstr], g[c * HW_], g[c * HW_], cnt);
            } else {
                Mul(ring[roff + m * mstr], g[c * HW_], g[d * HW_], cnt);
            }
            ++m;
        }
    }
    // ν_s = p_s
    for (int32_t s = 0; s < Cs_; ++s) {
        Adds(ring[roff + m * mstr], p[s * HW_], 0.0F, cnt);
        ++m;
    }
    // ω_cs 的原子 = g_c * p_s
    for (int32_t c = 0; c < C_; ++c) {
        for (int32_t s = 0; s < Cs_; ++s) {
            Mul(ring[roff + m * mstr], g[c * HW_], p[s * HW_], cnt);
            ++m;
        }
    }
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::FillCoefRing(int32_t img, int32_t y, int32_t xs, int32_t span)
{
    // 先把 K_ 行全部搬进 K_*haloW 的暂存区（系数图是单通道 float 平面，
    //  行字节数一致），只做一次 PIPE_ALL，再统一做镜像重排 Gather。
    int32_t cs = (xs < 0) ? 0 : xs;
    int32_t ce = xs + span;
    if (ce > W_) {
        ce = W_;
    }
    const int32_t cl = (ce > cs) ? (ce - cs) : 0;
    LocalTensor<float> rows = coefRowBuf_.Get<float>();
    if (cl > 0) {
        const int32_t alphaCnt = C_ * Cs_;
        const int64_t rowOff = static_cast<int64_t>(y) * W_ + cs;
        // 同样不能合并成多块 DMA（见 EmitAlphaBeta 的说明）；只把首平面基址提到循环外。
        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint16_t>(cl * static_cast<int32_t>(sizeof(float)));
        p.srcStride = 0;
        p.dstStride = 0;
        const int64_t alphaBase = (static_cast<int64_t>(img) * alphaCnt) * planeElems_ + rowOff;
        const int64_t betaBase = (static_cast<int64_t>(img) * Cs_) * planeElems_ + rowOff;
        for (int32_t k = 0; k < alphaCnt; ++k) {
            DataCopyPad(rows[k * HW_], alphaGm_[alphaBase + static_cast<int64_t>(k) * planeElems_], p,
                        {false, 0, 0, 0});
        }
        for (int32_t k = 0; k < Cs_; ++k) {
            DataCopyPad(rows[(alphaCnt + k) * HW_], betaGm_[betaBase + static_cast<int64_t>(k) * planeElems_], p,
                        {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>(); // 一次同步覆盖 2 条 DMA（MTE2 -> 向量需全流水同步）
    }
    LocalTensor<uint32_t> coefIdx = deIdxBuf_.Get<uint32_t>()[SegCoef()];
    LocalTensor<float> ring = ringBuf_.Get<float>();
    const int32_t roff = RingRowOff(y);
    // 一次 Gather 重排全部 K_ 个系数平面：coefRowBuf_ 与环槽的 map 维都是行内连续、
    // 步长都是 haloW，所以一张平面表就能服务整批（原来要 K_ 次 Gather + K_ 个 barrier）。
    if (K_ >= kCoefGatherMin) {
        // 合批：表里带绝对平面偏移，基址取整个暂存区
        Gather(ring[roff], rows, coefIdx, 0U, static_cast<uint32_t>(K_ * HW_));
    } else {
        // 平面数很少时逐平面搬更快；表只有 1 个平面（相对偏移），基址逐平面下移
        for (int32_t k = 0; k < K_; ++k) {
            Gather(ring[roff + k * HW_], rows[k * HW_], coefIdx, 0U, static_cast<uint32_t>(HW_));
        }
    }
    PipeBarrier<PIPE_V>(); // Gather 不在 V 流水上执行，读它的结果前必须同步
}

// 先用 ReduceMin/ReduceMax 判一次：只要 min|det| 明显大于阈值就整行跳过，
// 正常图像下永远走这条快路径，省掉 tw 次标量访问和后续那个 S->V 屏障。
template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::ClampDetDegen(LocalTensor<float> det, int32_t tw)
{
    LocalTensor<float> red = redBuf_.Get<float>();
    LocalTensor<float> rmin = red;     // 8 个 float（32B）放归约结果
    LocalTensor<float> rmax = red[8];  // 8 个 float
    LocalTensor<float> work = red[16]; // strideT 个 float 作为 sharedTmpBuffer
    ReduceMin(rmin, det, work, tw);
    ReduceMax(rmax, det, work, tw);
    PipeBarrier<PIPE_ALL>(); // Vector 归约结果 -> 标量读取
    const float lo = rmin.GetValue(0);
    const float hi = rmax.GetValue(0);
    // 整行都不满足 |d| < kDetDegThresh 的两种情形：全 >= 阈值 或 全 <= -阈值
    if (lo >= kDetDegThresh || hi <= -kDetDegThresh) {
        return;
    }
    for (int32_t i = 0; i < tw; ++i) {
        float d = det.GetValue(i);
        if ((d < 0.0F ? -d : d) < kDetDegThresh) {
            det.SetValue(i, 1.0F);
        }
    }
    PipeBarrier<PIPE_ALL>(); // 标量写 -> 后续 Vector 读
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::HorizontalRaw(LocalTensor<float> v, LocalTensor<float> out,
                                                               int32_t srcElems)
{
    // 未归一化的一维窗口和：out(i) = Σ_{j=i}^{i+2r} v(j)。
    //
    // 向量化方案：向量指令的 LocalTensor 起始地址必须 32B 对齐，所以不能直接对 v 施加
    // 1..2r 个元素的偏移（实测会触发 EZ9999）。这里先用 Gather 按预建的字节偏移表一次性
    // 生成 rows_ 份「移位副本」（sh[k*strideT + i] = v[i + k]，strideT 为 8 的倍数，
    // 保证每份副本的起始地址仍然 32B 对齐），再对副本做普通的对齐 Add 累加。
    //
    // 合批：一次处理 mapBatch_ 张图（每张图占连续 strideT_ 个元素），于是 rows_ 次
    // 树形归约只做一遍而不是每张图各做一遍 —— 指令条数直接降为 1/mapBatch_。
    //
    // 注：「把 rows_ 分解成素因子、逐级做子滑动和」能再省一半份数与 Add 条数，
    // 但每一级都要多发一次 Gather，而 Gather 的固定开销远大于元素数（实测 1->1 因为
    // 多一次 Gather 调用慢了 55%，3->3 只快 2.5%），因此不采用。
    const uint32_t cnt = static_cast<uint32_t>(mapBatch_ * strideT_);
    LocalTensor<float> sh = shBuf_.Get<float>();
#ifdef ASCENDC_CPU_DEBUG
    // ICPU 模型的 Gather 只处理 min(count, 源张量元素数) 个偏移，超出的那部分目标
    // 根本不写（残留未初始化 UB，实测会表现为 1e19~1e36 与 NaN），真机没有这个限制
    // （同一份内核上板 27/27 通过）。这里按「源张量可用元素数」把一次调用切成若干次
    // 等价调用：索引表里第 k 段的偏移本来就是绝对字节偏移，所以分段结果与整段一致。
    // 该分支只在 -DASCENDC_CPU_DEBUG 编译时存在，真机生成代码完全不变。
    int32_t kStep = (cnt > 0U) ? (srcElems / static_cast<int32_t>(cnt)) : rows_;
    if (kStep < 1) {
        kStep = 1;
    }
    if (kStep > rows_) {
        kStep = rows_;
    }
    LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
    for (int32_t k = 0; k < rows_; k += kStep) {
        const int32_t n = ((k + kStep <= rows_) ? kStep : (rows_ - k));
        Gather(sh[k * static_cast<int32_t>(cnt)], v, idx[k * static_cast<int32_t>(cnt)], 0U,
               static_cast<uint32_t>(n) * cnt);
    }
#else
    Gather(sh, v, idxBuf_.Get<uint32_t>(), 0U, static_cast<uint32_t>(rows_) * cnt);
#endif
    PipeBarrier<PIPE_V>();
    // 树形归约代替串行累加：out = Σ_{k<rows_} sh[k]。
    // 最后一层直接把结果写进 out，省掉一次整段拷贝。
    for (int32_t d = 1; d < rows_; d <<= 1) {
        for (int32_t i = 0; i + d < rows_; i += (d << 1)) {
            if (i == 0 && (d << 1) >= rows_) {
                Add(out, sh, sh[d * static_cast<int32_t>(cnt)], cnt);
            } else {
                Add(sh[i * static_cast<int32_t>(cnt)], sh[i * static_cast<int32_t>(cnt)],
                    sh[(i + d) * static_cast<int32_t>(cnt)], cnt);
            }
        }
    }
    if (rows_ == 1) {
        Adds(out, sh, 0.0F, cnt);
    }
}

// 构建水平滑窗的移位索引表：idx[k*strideT + i] = (i + k) * sizeof(float)
// 每个 task 只需在 tile 几何确定后构建一次。
template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::BuildShiftIndex()
{
    // 索引表只与 tile 几何相关，整个 kernel 只构建一次（约 rows*strideT 个元素），
    // 用标量写完全可接受；避免 CreateVecIndex/Adds 对无符号类型的不支持。
    // 合批布局：[k][m][i]，槽内地址 = k*(mb*strideT) + m*strideT + i，
    // 对应源偏移 = m*HW_ + i + k。偏移以 srcBase 为基准，所以同一张表可以服务每一批
    // （调用方把 srcBase 移到该批首图）。
    LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
    const int32_t mb = mapBatch_;
    for (int32_t k = 0; k < rows_; ++k) {
        for (int32_t m = 0; m < mb; ++m) {
            for (int32_t i = 0; i < strideT_; ++i) {
                const int32_t src = m * HW_ + i + k;
                idx.SetValue(k * mb * strideT_ + m * strideT_ + i,
                             static_cast<uint32_t>(src * static_cast<int32_t>(sizeof(float))));
            }
        }
    }
    for (int32_t k = rows_ * mb * strideT_; k < IdxSlots(); ++k) {
        idx.SetValue(k, 0U); // Gather 末轮会读满 64 项，尾项必须安全
    }
    PipeBarrier<PIPE_ALL>(); // 标量写 -> Vector(Gather) 读
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::ComputeBoxes(int32_t y, int32_t tw, int32_t span, int32_t mapCount,
                                                              bool seed)
{
    LocalTensor<float> vAll = vBuf_.Get<float>();
    (void)span;
    LocalTensor<float> ring = ringBuf_.Get<float>();
    // 垂直求和一次覆盖全部 mapCount 张图：ring 的 map 维在行内连续，vAll 也是，
    // 所以整批是一条长指令而不是 mapCount 条短指令。
    // 宽度取 haloW 而非 span：环槽的 [span, haloW) 尾段由 Init 一次性置 0，只会污染
    // box 的 [tw, strideT) 列，而下游从不读这些列。
    const uint32_t mcnt = static_cast<uint32_t>(mapCount) * static_cast<uint32_t>(HW_);
    if (seed) {
        // 行带首行：全量求 2r+1 行和（镜像索引每 dy 只算一次）
        Adds(vAll, ring[RingRowOff(MirrorIdx(y - R_, H_))], 0.0F, mcnt);
        for (int32_t dy = -R_ + 1; dy <= R_; ++dy) {
            Add(vAll, vAll, ring[RingRowOff(MirrorIdx(y + dy, H_))], mcnt);
        }
    } else {
        // 其余行：V(y) = V(y-1) + ring[MirrorIdx(y+r)] - ring[MirrorIdx(y-r-1)]
        // 减少 2r-1 次向量加，且与镜像边界无关（望远镜求和恒成立）
        const int32_t roffAdd = RingRowOff(MirrorIdx(y + R_, H_));
        const int32_t roffSub = RingRowOff(MirrorIdx(y - R_ - 1, H_));
        Add(vAll, vAll, ring[roffAdd], mcnt);
        Sub(vAll, vAll, ring[roffSub], mcnt);
    }
    // 消费者是 HorizontalRaw 里的 Gather（不在 V 流水上），只需等 V 队列排空，
    // 不需要全流水刷新。
    PipeBarrier<PIPE_V>();

    LocalTensor<float> box = boxBuf_.Get<float>();
    // 按 mapBatch_ 张图一批做水平滑窗；末批不足时多带出的空图落在 padding 槽位，
    // 结果不会被下游消费（下游只读前 mapCount 张）。
    (void)tw;
    for (int32_t m = 0; m < mapCount; m += mapBatch_) {
        HorizontalRaw(vAll[m * HW_], box[m * strideT_], MapSlots() * HW_ - m * HW_);
    }
    // 这里原来是 PipeBarrier<PIPE_ALL>()：前后都是向量操作（HorizontalRaw 末尾的 Adds
    // 与下面的 Muls），同流水按序执行，屏障纯属多余，直接去掉。

    // 21 个 map 的 box 结果在 boxBuf_ 里是连续的，直接一次 Muls 处理全部 map，
    Muls(box, box, invWin_, static_cast<uint32_t>(mapCount * strideT_));
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::BuildAlphaBeta(int32_t tw)
{
    LocalTensor<float> box = boxBuf_.Get<float>();
    LocalTensor<float> aux = auxBuf_.Get<float>();
    const uint32_t cnt = static_cast<uint32_t>(tw);
    LocalTensor<float> det = aux[kDetSlot * strideT_];
    LocalTensor<float> num = aux[6 * strideT_];
    LocalTensor<float> tmp = aux[7 * strideT_];

    // ---- 第一步：把盒式均值变换成「中心化协方差 + 互协方差」（原地写回 box）----
    //   Σ_cc = box(g_c^2) - μ_c^2 + eps ;  Σ_cd = box(g_c g_d) - μ_c μ_d
    //   ω_cs = box(g_c p_s) - μ_c ν_s
    for (int32_t c = 0; c < C_; ++c) {
        for (int32_t d = c; d < C_; ++d) {
            LocalTensor<float> cov = box[CovIdx(c, d) * strideT_];
            Mul(tmp, box[c * strideT_], box[d * strideT_], cnt);
            Sub(cov, cov, tmp, cnt);
            if (c == d) {
                Adds(cov, cov, eps_, cnt);
            }
        }
    }
    for (int32_t c = 0; c < C_; ++c) {
        for (int32_t s = 0; s < Cs_; ++s) {
            LocalTensor<float> om = box[OmIdx(c, s) * strideT_];
            Mul(tmp, box[c * strideT_], box[NuIdx(s) * strideT_], cnt);
            Sub(om, om, tmp, cnt);
        }
    }

    if (C_ == 1) {
        Adds(det, box[CovIdx(0, 0) * strideT_], 0.0F, cnt);
        ClampDetDegen(det, tw);
        for (int32_t s = 0; s < Cs_; ++s) {
            Div(aux[(kAuxFixed + s) * strideT_], box[OmIdx(0, s) * strideT_], det, cnt);
        }
    } else if (C_ == 2) {
        LocalTensor<float> s00 = box[CovIdx(0, 0) * strideT_];
        LocalTensor<float> s01 = box[CovIdx(0, 1) * strideT_];
        LocalTensor<float> s11 = box[CovIdx(1, 1) * strideT_];
        LocalTensor<float> ns01 = aux[0];
        Mul(det, s00, s11, cnt);
        Mul(tmp, s01, s01, cnt);
        Sub(det, det, tmp, cnt);
        Muls(ns01, s01, -1.0F, cnt);
        ClampDetDegen(det, tw);
        for (int32_t s = 0; s < Cs_; ++s) {
            // alpha(0,s) = (s11*w0 - s01*w1)/det
            Mul(num, s11, box[OmIdx(0, s) * strideT_], cnt);
            Mul(tmp, ns01, box[OmIdx(1, s) * strideT_], cnt);
            Add(num, num, tmp, cnt);
            Div(aux[(kAuxFixed + 0 * Cs_ + s) * strideT_], num, det, cnt);
            // alpha(1,s) = (-s01*w0 + s00*w1)/det
            Mul(num, ns01, box[OmIdx(0, s) * strideT_], cnt);
            Mul(tmp, s00, box[OmIdx(1, s) * strideT_], cnt);
            Add(num, num, tmp, cnt);
            Div(aux[(kAuxFixed + 1 * Cs_ + s) * strideT_], num, det, cnt);
        }
    } else {
        LocalTensor<float> s00 = box[CovIdx(0, 0) * strideT_];
        LocalTensor<float> s01 = box[CovIdx(0, 1) * strideT_];
        LocalTensor<float> s02 = box[CovIdx(0, 2) * strideT_];
        LocalTensor<float> s11 = box[CovIdx(1, 1) * strideT_];
        LocalTensor<float> s12 = box[CovIdx(1, 2) * strideT_];
        LocalTensor<float> s22 = box[CovIdx(2, 2) * strideT_];
        LocalTensor<float> c00 = aux[0];
        LocalTensor<float> c01 = aux[1 * strideT_];
        LocalTensor<float> c02 = aux[2 * strideT_];
        LocalTensor<float> c11 = aux[3 * strideT_];
        LocalTensor<float> c12 = aux[4 * strideT_];
        LocalTensor<float> c22 = aux[5 * strideT_];
        // 伴随矩阵（对称）
        Mul(c00, s11, s22, cnt);
        Mul(tmp, s12, s12, cnt);
        Sub(c00, c00, tmp, cnt);
        Mul(c01, s02, s12, cnt);
        Mul(tmp, s01, s22, cnt);
        Sub(c01, c01, tmp, cnt);
        Mul(c02, s01, s12, cnt);
        Mul(tmp, s02, s11, cnt);
        Sub(c02, c02, tmp, cnt);
        Mul(c11, s00, s22, cnt);
        Mul(tmp, s02, s02, cnt);
        Sub(c11, c11, tmp, cnt);
        Mul(c12, s01, s02, cnt);
        Mul(tmp, s00, s12, cnt);
        Sub(c12, c12, tmp, cnt);
        Mul(c22, s00, s11, cnt);
        Mul(tmp, s01, s01, cnt);
        Sub(c22, c22, tmp, cnt);
        // det = s00*c00 + s01*c01 + s02*c02
        Mul(det, s00, c00, cnt);
        Mul(tmp, s01, c01, cnt);
        Add(det, det, tmp, cnt);
        Mul(tmp, s02, c02, cnt);
        Add(det, det, tmp, cnt);
        ClampDetDegen(det, tw);
        for (int32_t s = 0; s < Cs_; ++s) {
            LocalTensor<float> w0 = box[OmIdx(0, s) * strideT_];
            LocalTensor<float> w1 = box[OmIdx(1, s) * strideT_];
            LocalTensor<float> w2 = box[OmIdx(2, s) * strideT_];
            for (int32_t c = 0; c < 3; ++c) {
                LocalTensor<float> a0 = aux[AdjSlot(c, 0) * strideT_];
                LocalTensor<float> a1 = aux[AdjSlot(c, 1) * strideT_];
                LocalTensor<float> a2 = aux[AdjSlot(c, 2) * strideT_];
                Mul(num, a0, w0, cnt);
                Mul(tmp, a1, w1, cnt);
                Add(num, num, tmp, cnt);
                Mul(tmp, a2, w2, cnt);
                Add(num, num, tmp, cnt);
                Div(aux[(kAuxFixed + c * Cs_ + s) * strideT_], num, det, cnt);
            }
        }
    }

    // beta_s = ν_s - Σ_c alpha(c,s) * μ_c
    for (int32_t s = 0; s < Cs_; ++s) {
        LocalTensor<float> b = aux[(kAuxFixed + C_ * Cs_ + s) * strideT_];
        Adds(b, box[NuIdx(s) * strideT_], 0.0F, cnt);
        for (int32_t c = 0; c < C_; ++c) {
            Mul(tmp, aux[(kAuxFixed + c * Cs_ + s) * strideT_], box[c * strideT_], cnt);
            Sub(b, b, tmp, cnt);
        }
    }
    // 向量写 -> EmitAlphaBeta 的 MTE3 读取，必须全流水屏障
    PipeBarrier<PIPE_ALL>();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::EmitAlphaBeta(const Task& t, int32_t y, int32_t tw)
{
    LocalTensor<float> aux = auxBuf_.Get<float>();
    const int64_t rowOff = static_cast<int64_t>(y) * W_ + t.x0;
    const int32_t alphaCnt = C_ * Cs_;
    // 注：不能把 K_ 个平面合并成一条多块 DataCopyPad —— 本平台的 DataCopyPad
    // （带 pad 语义的那条接口）只支持 blockCount == 1，blockCount > 1 直接 EZ9999。
    // 这里退而把「首平面基址」与「平面步长」提到循环外，循环内只做加法。
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint16_t>(tw * static_cast<int32_t>(sizeof(float)));
    p.srcStride = 0;
    p.dstStride = 0;
    const int64_t alphaBase = (static_cast<int64_t>(t.img) * alphaCnt) * planeElems_ + rowOff;
    const int64_t betaBase = (static_cast<int64_t>(t.img) * Cs_) * planeElems_ + rowOff;
    for (int32_t k = 0; k < alphaCnt; ++k) {
        DataCopyPad(alphaGm_[alphaBase + static_cast<int64_t>(k) * planeElems_], aux[(kAuxFixed + k) * strideT_], p);
    }
    for (int32_t k = 0; k < Cs_; ++k) {
        DataCopyPad(betaGm_[betaBase + static_cast<int64_t>(k) * planeElems_],
                    aux[(kAuxFixed + alphaCnt + k) * strideT_], p);
    }
    // WAR：行循环下一轮的 BuildAlphaBeta 会覆写 auxBuf_，必须等本轮 MTE3 读完。
    // 缺这一条会偶发把新行数据写进尚未落盘的 alpha/beta（表现为随机几十个像素出错）。
    PipeBarrier<PIPE_MTE3>();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::BuildOutRow(const Task& t, int32_t y, int32_t tw, int32_t span)
{
    ComputeBoxes(y, tw, span, K_, y == t.y0);

    // 输出只用图像列 [x0, x1)，不需要 halo
    LoadGuideRowOut(t.img, y, t.x0, tw);

    LocalTensor<float> box = boxBuf_.Get<float>();
    LocalTensor<float> g = gRowBuf_.Get<float>();
    LocalTensor<float> tmp = auxBuf_.Get<float>();
    const uint32_t cnt = static_cast<uint32_t>(tw);
    const int32_t alphaCnt = C_ * Cs_;

    for (int32_t s = 0; s < Cs_; ++s) {
        LocalTensor<float> q = qBuf_.Get<float>()[s * strideT_];
        Adds(q, box[(alphaCnt + s) * strideT_], 0.0F, cnt);
        for (int32_t c = 0; c < C_; ++c) {
            Mul(tmp, box[(c * Cs_ + s) * strideT_], g[c * HW_], cnt);
            Add(q, q, tmp, cnt);
        }
    }
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> finalRow;
    int32_t nOut = tw;
    if (Cs_ == 1) {
        finalRow = qBuf_.Get<float>();
    } else {
        // 交织：out[j*Cs+s] = q_s[j]（Gather 一次完成，原本是 tw*Cs 次标量搬移）
        finalRow = ilBuf_.Get<float>();
        Gather(finalRow, qBuf_.Get<float>(), ilIdxBuf_.Get<uint32_t>(), 0U, static_cast<uint32_t>(tw * Cs_));
        PipeBarrier<PIPE_V>();
        nOut = tw * Cs_;
    }

    const int64_t base = (static_cast<int64_t>(t.img) * H_ + y) * W_ * Cs_ + static_cast<int64_t>(t.x0) * Cs_;
    DataCopyParams p;
    p.blockCount = 1;
    p.srcStride = 0;
    p.dstStride = 0;
    if constexpr (std::is_same_v<TD, float>) {
        p.blockLen = static_cast<uint16_t>(nOut * static_cast<int32_t>(sizeof(float)));
        DataCopyPad(dstGm_[base], finalRow, p);
    } else if constexpr (std::is_same_v<TD, half>) {
        LocalTensor<half> out = outBuf_.Get<half>();
        Cast(out, finalRow, RoundMode::CAST_RINT, static_cast<uint32_t>(nOut));
        PipeBarrier<PIPE_ALL>();
        p.blockLen = static_cast<uint16_t>(nOut * 2);
        DataCopyPad(dstGm_[base], out, p);
    } else {
        // uint8：round-half-away + 饱和到 [0,255]（与 saturate_cast<uchar>(cvRound(x)) 在非 .5 边界一致）
        LocalTensor<uint8_t> out = outBuf_.Get<uint8_t>();
        for (int32_t i = 0; i < nOut; ++i) {
            float v = finalRow.GetValue(i);
            v = (v >= 0.0F) ? (v + 0.5F) : (v - 0.5F);
            int32_t iv = static_cast<int32_t>(v);
            if (iv < 0) {
                iv = 0;
            } else if (iv > 255) {
                iv = 255;
            }
            out.SetValue(i, static_cast<uint8_t>(iv));
        }
        PipeBarrier<PIPE_ALL>();
        p.blockLen = static_cast<uint16_t>(nOut);
        DataCopyPad(dstGm_[base], out, p);
    }
    // WAR：下一轮的 BuildOutRow 会覆写 qBuf_/ilBuf_/outBuf_，必须等本轮输出 DMA 读完。
    PipeBarrier<PIPE_MTE3>();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::StageA(const Task& t)
{
    const int32_t xs = t.x0 - R_;
    const int32_t tw = t.x1 - t.x0;
    const int32_t span = tw + 2 * R_;

    // 去交织索引表依赖本 task 的 xs，先建好（规模仅 (C+Cs)*haloW，可忽略）
    LocalTensor<uint32_t> deIdx = deIdxBuf_.Get<uint32_t>();
    BuildDeintIndex(deIdx, C_, xs, span);
    BuildDeintIndex(deIdx[SegSrc()], Cs_, xs, span);

    int32_t lo = t.y0 - R_;
    if (lo < 0) {
        lo = 0;
    }
    int32_t hi = t.y0 + R_;
    if (hi > H_ - 1) {
        hi = H_ - 1;
    }
    for (int32_t yy = lo; yy <= hi; ++yy) {
        FillAtomRing(t.img, yy, xs, span);
    }
    for (int32_t y = t.y0; y < t.y1; ++y) {
        if (y > t.y0) {
            const int32_t nr = y + R_;
            if (nr <= H_ - 1) {
                FillAtomRing(t.img, nr, xs, span);
            }
        }
        ComputeBoxes(y, tw, span, M_, y == t.y0);
        BuildAlphaBeta(tw);
        EmitAlphaBeta(t, y, tw);
    }
    PipeBarrier<PIPE_MTE3>();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::StageB(const Task& t)
{
    const int32_t xs = t.x0 - R_;
    const int32_t tw = t.x1 - t.x0;
    const int32_t span = tw + 2 * R_;

    // 系数行是单通道平面，只需一段索引（放在 guide/src 之后）
    LocalTensor<uint32_t> deIdx = deIdxBuf_.Get<uint32_t>();
    BuildPlanarIndex(deIdx[SegCoef()], CoefGatherNum(), xs, span);
    // 输出阶段 guide 行：xs = t.x0、span = tw，与 StageA 的 (x0-R_, tw+2R) 不同
    BuildDeintIndex(deIdx[SegGuideOut()], C_, t.x0, tw);

    int32_t lo = t.y0 - R_;
    if (lo < 0) {
        lo = 0;
    }
    int32_t hi = t.y0 + R_;
    if (hi > H_ - 1) {
        hi = H_ - 1;
    }
    for (int32_t yy = lo; yy <= hi; ++yy) {
        FillCoefRing(t.img, yy, xs, span);
    }
    for (int32_t y = t.y0; y < t.y1; ++y) {
        if (y > t.y0) {
            const int32_t nr = y + R_;
            if (nr <= H_ - 1) {
                FillCoefRing(t.img, nr, xs, span);
            }
        }
        BuildOutRow(t, y, tw, span);
    }
    PipeBarrier<PIPE_MTE3>();
}

template <typename TG, typename TS, typename TD>
__aicore__ inline void GuidedFilter<TG, TS, TD>::Process()
{
    const int32_t totalTasks = N_ * nTx_ * nTy_;
    if (totalTasks <= 0) {
        return;
    }
    const int32_t blk = static_cast<int32_t>(GetBlockIdx());
    const int32_t dim = static_cast<int32_t>(GetBlockNum());

    for (int32_t i = blk; i < totalTasks; i += dim) {
        StageA(DecodeTask(i));
    }
    // 阶段 A 的 alpha/beta 全部落盘后，阶段 B 才能读；跨核屏障 + batch 调度保证可见性/驻留
    PipeBarrier<PIPE_MTE3>();
    SyncAll();
    for (int32_t i = blk; i < totalTasks; i += dim) {
        StageB(DecodeTask(i));
    }
}

} // namespace NsGuidedFilter
#endif // GUIDED_FILTER_H
