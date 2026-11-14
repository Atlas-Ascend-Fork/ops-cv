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
 * \file crop_and_resize_simt.h
 * \brief SIMT kernel implementation for crop_and_resize operator
 *
 * 并行策略：以 warp 为粒度对 (b, cy, cx) 输出位置做 Grid-Stride 遍历，每个 warp 负责一个输出位置；
 * warp 内 32 个 lane 分摊 C 个通道（lane 依次处理 d = laneId, laneId+32, ...），相邻 lane 访问相邻
 * 通道，形成合并访存（coalesced）。位置级计算（box 读取 / 坐标映射 / 非有限与越界判定）由 warp
 * 内所有 lane 在 SIMT 锁步下以相同输入冗余执行，因此结果位级一致。每个 (pos, d) 的浮点计算序列与
 * 线程逐位置（thread-per-position）版本完全相同（仅任务重划分）。数据排布（NHWC/NCHW）是编译期
 * 模板参数，经 TilingKey schMode 编码。
 *
 * 索引位宽模板化 IDX_T(uint32_t/uint64_t)：由 host/kernel 根据所有输入/输出 tensor 的元素总数选择。
 * 32 位快路径要求 x、y、boxes 的寻址范围均不超过 INT32_MAX；否则走 64 位路径。
 * __launch_bounds__ 按 IDX_T 位宽模板化，uint32 路径开 1024 线程，uint64 路径开 512 线程。
 *
 * 标量参数直接作为 VF 函数入参传递，不再经 UB 中转。
 *
 * 精度对齐方案（与 TF image.crop_and_resize bit-exact，中间计算统一使用 float32）：
 *   1. volatile — 阻止 NPU 编译器将 a*b+c 收缩为 fma(a,b,c)（单舍入），
 *      匹配 TF Eigen tensor 的 separate mul+add（双舍入）。
 *      -ffp-contract=off 和 #pragma clang fp contract(off) 在 NPU 编译器上均不可靠，
 *      volatile 是经实证验证的唯一有效 FMA 防护手段。
 *   2. DsDiv — fmaf 修正除法残差，消除 NPU 与 numpy 除法的 1-ULP 坐标差异，
 *      使 OOB 判定完全一致。
 *   3. IEEE 754 自然传播 — 不使用 t==0 guard，0*Inf=NaN 自然传播匹配 TF。
 *   4. FTZ 对齐 — x86 golden（TF）编译启用 Flush-To-Zero/DAZ：次正规运算结果/输入被冲刷为 +0，
 *      NPU VF 硬件按 IEEE 渐进下溢保留次正规，满量程输入（|x|≈3.4e38）下与 golden 分叉，
 *      经 Ftz() 冲刷对齐；完整语义、全部应用位置与例外详见 Ftz 定义处注释。
 */

#ifndef CROP_AND_RESIZE_SIMT_H_
#define CROP_AND_RESIZE_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/device_sync_functions.h"
#include "crop_and_resize_tiling_data.h"
#include "crop_and_resize_tiling_key.h"

namespace NsCropAndResize {
using namespace AscendC;

// 线程数模板：uint32 路径开 1024 线程，uint64 路径开 512 线程
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == sizeof(uint32_t)) ? 1024 : 512;

// warp 内 lane 数：warp 处理一个输出位置，lane 分摊通道维（d ≡ laneId mod 32），相邻 lane 读相邻通道
static constexpr uint32_t WARP_LANES = 32;

// 每个 box 的坐标数 [y1, x1, y2, x2]
static constexpr int32_t BOX_COORDS_PER_BOX = 4;

// 数据排布（编译期模板参数，经 TilingKey schMode 编码由 apt.cpp 分流传入，不经 TilingData/UB 传递）：
// LAYOUT_NHWC=0 与原单值模式 schMode=0 对应（ND/NHWC 路径二进制兼容）；LAYOUT_NCHW=1 对应 schMode=1
enum Layout : uint32_t {
    LAYOUT_NHWC = 0, // ND/NHWC 输入语义合并：x=(N,H,W,C)，y=(b,cy,cx,C)
    LAYOUT_NCHW = 1, // NCHW 输入：x=(N,C,H,W)，y=(b,C,cy,cx)，数据不转置
};

// MapCoordinate cropSize==1 时的中点权重
static constexpr float MIDPOINT_WEIGHT = 0.5f;

// 以下 union 位转换依赖编译器 union 双关扩展（Clang/GCC 支持），非 ISO C++ 标准行为。
// int32_t/uint32_t/float 均为 32 位同尺寸类型，无数据损坏风险。
static_assert(sizeof(int32_t) == sizeof(float), "int32_t and float must have same size");
static_assert(sizeof(uint32_t) == sizeof(float), "uint32_t and float must have same size");

// VF 标量参数集合
template <typename IDX_T>
struct ScalarParams {
    IDX_T batch;
    IDX_T imageHeight;
    IDX_T imageWidth;
    IDX_T depth;
    IDX_T cropHeight;
    IDX_T cropWidth;
    float extrapolationValue;
};

// GM 指针集合
template <typename T_X, typename T_BOXES, typename T_Y>
struct GmPointers {
    __gm__ T_X* xGm;
    __gm__ T_BOXES* boxesGm;
    __gm__ int32_t* boxIndexGm;
    __gm__ T_Y* yGm;
};

// 预计算的几何常量
template <typename IDX_T>
struct Geometry {
    float imgHMinus1;
    float imgWMinus1;
    IDX_T imgHWC; // imageHeight*imageWidth*depth：NHWC 即 H*W*C，NCHW 为 C*H*W（数值相同，两 layout 复用）
    IDX_T cropHWC; // cropHeight*cropWidth*depth：输出 batch 步长，两 layout 数值相同
};

// 插值几何参数（邻居索引 + 权重 + 偏移）
template <typename IDX_T>
struct InterpGeometry {
    IDX_T topY;
    IDX_T bottomY;
    IDX_T leftX;
    IDX_T rightX;
    float wy1;
    float wx1;
    IDX_T imgBase;
    IDX_T topRowOff;
    IDX_T bottomRowOff;
    IDX_T leftColOff;
    IDX_T rightColOff;
    IDX_T channelInStride; // 输入图读侧通道步长：NCHW = H*W（d 迭代跨步）；NHWC = 1
};

// 位置级计算结果：填充路径（box_index 越界 / boxes 或映射坐标非有限 / OOB）时 isFill 为 true
// 且 fillVal 有效；正常路径时 outBase/channelOutStride/ig 有效，由通道级计算消费。
template <typename IDX_T>
struct PosLevelResult {
    bool isFill;
    float fillVal;
    IDX_T outBase;
    IDX_T channelOutStride;
    InterpGeometry<IDX_T> ig;
};

// dtype 转换
template <typename T>
__simt_callee__ inline float CastToFloat(T val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return static_cast<float>(val);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return __int2float_rn(static_cast<int>(val));
    } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
        return __int2float_rn(static_cast<int32_t>(val));
    } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>) {
        return __int2float_rn(static_cast<int32_t>(val));
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
__simt_callee__ inline T CastFromFloat(float val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else {
        return static_cast<T>(val);
    }
}

// DsDiv: 精确除法，消除 NPU 与 numpy 的 1-ULP 差异
// 普通除法 a/b 在 NPU 和 numpy 上可能因舍入方向不同产生 1-ULP 差异，
// 导致坐标映射的 OOB 判定不一致。DsDiv 用 fmaf 计算残差并修正：
//   q = a/b, residual = fmaf(q, b, -a), result = q - residual/b
// 调用方保证 b != 0（cropSize > 1 时除数 cropSize-1 >= 1）。
__simt_callee__ inline float DsDiv(float a, float b)
{
    float q = a / b;
    float residual = fmaf(q, b, -a);
    return q - residual / b;
}

// 将整数索引 clamp 到 [0, hi]；调用前已保证 val 非负
template <typename IDX_T>
__simt_callee__ inline IDX_T ClampIdx(IDX_T val, IDX_T hi)
{
    return (val > hi) ? hi : val;
}

constexpr float kF32MinNormal = 1.17549435e-38f; // 2^-126，float32 最小正规数（FLT_MIN）

// Ftz: 次正规数冲刷（Flush-To-Zero / Denormals-Are-Zero），对齐 x86 golden 的浮点语义。
// TF golden 运行在 x86 上（-ffast-math 类编译同时启用 FTZ 与 DAZ）：运算结果为次正规
// （|v| < 2^-126 且非 0）时被冲刷为 +0（FTZ），次正规输入参与运算前同样被冲刷为 +0（DAZ）；
// NPU VF 硬件按 IEEE 754 渐进下溢保留次正规，满量程输入（|x|≈3.4e38）下插值乘积/加和
// 下溢进次正规域，导致与 golden 分叉。经 dump 实测验证（215 用例 bit 全对齐）：
//   - FTZ 输出冲刷：覆盖插值链路全部乘/除/加结果（BilinearSampleChannel 的
//     prod0/prod1/prod2/row0/row1/返回值、MapCoordinate 的 boxDiff/num/scale/prod/base/coord）；
//   - DAZ 输入冲刷：外部输入（x 像素、boxes 坐标）加载后冲刷（内部中间量已被输出 FTZ
//     冲刷，无需重复 DAZ）；
//   - 减法输出在权重 ∈ [0,1) 的 lerp 段不显式冲刷（乘积 <= |diff|，下游乘法 Ftz 兜底）；
//     MapCoordinate 的 boxDiff 因下游 imgSizeMinus1 可放大（>1）而显式冲刷；
//   - DsDiv 内部不冲刷（其 fmaf 修正依赖完整精度，除法输出在调用处冲刷）；
//   - NaN/Inf/±0 恒等返回（比较条件不命中），NaN/Inf 传播路径零触碰；
//   - 正规数恒等返回：真实业务像素值域（±1e4 级）下插值结果远大于 2^-126，Ftz 为 no-op。
__simt_callee__ inline float Ftz(float v) { return (v != 0.0f && fabsf(v) < kF32MinNormal) ? 0.0f : v; }

// 计算插值几何参数：邻居索引 + 权重 + 地址偏移（layout 编译期特化寻址步长）。
template <typename IDX_T, Layout LAYOUT, bool IS_NEAREST>
__simt_callee__ inline InterpGeometry<IDX_T> ComputeInterpGeometry(float inY, float inX, IDX_T boxIdx,
                                                                   const ScalarParams<IDX_T>& params,
                                                                   const Geometry<IDX_T>& geom)
{
    InterpGeometry<IDX_T> ig;
    if constexpr (IS_NEAREST) {
        ig.topY = ClampIdx(static_cast<IDX_T>(roundf(inY)), params.imageHeight - 1);
        ig.bottomY = ig.topY;
        ig.leftX = ClampIdx(static_cast<IDX_T>(roundf(inX)), params.imageWidth - 1);
        ig.rightX = ig.leftX;
        ig.wy1 = 0.0f;
        ig.wx1 = 0.0f;
    } else {
        ig.topY = ClampIdx(static_cast<IDX_T>(floorf(inY)), params.imageHeight - 1);
        ig.bottomY = ClampIdx(static_cast<IDX_T>(ceilf(inY)), params.imageHeight - 1);
        ig.leftX = ClampIdx(static_cast<IDX_T>(floorf(inX)), params.imageWidth - 1);
        ig.rightX = ClampIdx(static_cast<IDX_T>(ceilf(inX)), params.imageWidth - 1);
        ig.wy1 = inY - static_cast<float>(ig.topY);
        ig.wx1 = inX - static_cast<float>(ig.leftX);
    }
    // batch 基址：NHWC 的 H*W*C 与 NCHW 的 C*H*W 数值相同，两 layout 复用
    ig.imgBase = boxIdx * geom.imgHWC;
    if constexpr (LAYOUT == LAYOUT_NCHW) {
        // NCHW: x=(N,C,H,W)，行偏移=row*W、列偏移=col、通道步长=H*W
        ig.topRowOff = ig.topY * params.imageWidth;
        ig.bottomRowOff = ig.bottomY * params.imageWidth;
        ig.leftColOff = ig.leftX;
        ig.rightColOff = ig.rightX;
        ig.channelInStride = params.imageHeight * params.imageWidth;
    } else {
        // NHWC: x=(N,H,W,C)，行偏移=row*W*C、列偏移=col*C、通道步长=1
        IDX_T depth = params.depth;
        ig.topRowOff = ig.topY * params.imageWidth * depth;
        ig.bottomRowOff = ig.bottomY * params.imageWidth * depth;
        ig.leftColOff = ig.leftX * depth;
        ig.rightColOff = ig.rightX * depth;
        ig.channelInStride = 1;
    }
    return ig;
}

// 单通道双线性插值：像素布局 v00---v01 / v10---v11，lerp 形式 a+t*(b-a)，volatile 阻止 FMA contraction 匹配 TF
template <typename T_X, typename IDX_T, Layout LAYOUT>
__simt_callee__ inline float BilinearSampleChannel(const __gm__ T_X* xGm, const InterpGeometry<IDX_T>& ig, IDX_T d)
{
    IDX_T chOff = (LAYOUT == LAYOUT_NCHW) ? d * ig.channelInStride : d;
    // DAZ：外部输入像素参与运算前冲刷次正规（x86 -ffast-math 的 DAZ 语义，dump 实测 bit 对齐验证）；
    // fp16 次正规输入转 fp32 后落入正规域，Ftz 恒等
    float v00 = Ftz(CastToFloat(xGm[ig.imgBase + ig.topRowOff + ig.leftColOff + chOff]));
    float v01 = Ftz(CastToFloat(xGm[ig.imgBase + ig.topRowOff + ig.rightColOff + chOff]));
    float v10 = Ftz(CastToFloat(xGm[ig.imgBase + ig.bottomRowOff + ig.leftColOff + chOff]));
    float v11 = Ftz(CastToFloat(xGm[ig.imgBase + ig.bottomRowOff + ig.rightColOff + chOff]));

    // volatile 防护段逐行保留（乘法结果物化经 Ftz 冲刷，volatile 写/读链不变，FMA 仍被阻止）。
    // x86 FTZ 为全引擎语义：乘/除/加运算结果为次正规时均被冲刷为 +0（实测 14 用例：
    // golden=0 vs NPU=5.588e-39，即加法结果次正规未被冲刷导致的分叉）。
    // 因此除 prod0/prod1/prod2（乘法）外，row0/row1/返回值（加法）结果同样经 Ftz 冲刷；
    // 减法（diff0/diff1/diff2）不显式冲刷：wx1/wy1 ∈ [0,1)，乘积 <= |diff|，
    // 减法次正规残差由下游乘法 Ftz 兜底冲刷，结果等价。
    volatile float diff0 = v01 - v00;
    volatile float prod0 = Ftz(ig.wx1 * diff0);
    float row0 = Ftz(v00 + prod0);

    volatile float diff1 = v11 - v10;
    volatile float prod1 = Ftz(ig.wx1 * diff1);
    float row1 = Ftz(v10 + prod1);

    volatile float diff2 = row1 - row0;
    volatile float prod2 = Ftz(ig.wy1 * diff2);
    return Ftz(row0 + prod2);
}

template <typename T_X, typename IDX_T, Layout LAYOUT>
__simt_callee__ inline float NearestSampleChannel(const __gm__ T_X* xGm, const InterpGeometry<IDX_T>& ig, IDX_T d)
{
    IDX_T chOff = (LAYOUT == LAYOUT_NCHW) ? d * ig.channelInStride : d;
    return CastToFloat(xGm[ig.imgBase + ig.topRowOff + ig.leftColOff + chOff]);
}

// 坐标映射：将 crop 坐标映射到原图坐标
// TF 公式：
//   if cropSize > 1: scale = (boxEnd - boxStart) * (imgSize-1) / (cropSize-1)
//                    coord = boxStart * (imgSize-1) + cropIdx * scale
//   else:             coord = 0.5 * (boxStart + boxEnd) * (imgSize-1)
// DsDiv 修正 NPU plain division 与 x86 的 1-ULP 差异，匹配 TF 除法结果；
// DsDiv 结果在调用处经 Ftz 冲刷（除法结果为次正规时对齐 x86 FTZ），DsDiv 内部不冲刷。
// x86 FTZ/DAZ 全引擎语义：boxStart/boxEnd 入口 DAZ（外部输入冲刷）；
// boxDiff/num/scale/prod/base/coord 各步运算输出为次正规时冲刷 +0
// （boxDiff/num/base 显式冲刷：下游 imgSizeMinus1 > 1 可放大次正规，无法由下游兜底）。
// volatile 阻止 FMA contraction。
template <typename IDX_T>
__simt_callee__ inline float MapCoordinate(float boxStart, float boxEnd, IDX_T cropIdx, IDX_T cropSize,
                                           float imgSizeMinus1)
{
    // DAZ：box 坐标为外部输入，参与运算前冲刷次正规（NaN/Inf 恒等，入口检测不受影响）
    float bs = Ftz(boxStart);
    float be = Ftz(boxEnd);
    if (cropSize > static_cast<IDX_T>(1)) {
        volatile float boxDiff = Ftz(be - bs);
        volatile float num = Ftz(boxDiff * imgSizeMinus1);
        volatile float scale = Ftz(DsDiv(num, static_cast<float>(cropSize - 1)));
        volatile float prod = Ftz(static_cast<float>(cropIdx) * scale);
        volatile float base = Ftz(bs * imgSizeMinus1);
        volatile float coord = Ftz(base + prod);
        return coord;
    } else {
        // midpoint 分支：bs/be 已 DAZ（各自 ∈ {0} ∪ [FLT_MIN, +∞)，任意符号），|sum| >= FLT_MIN 或 = 0，
        // 结果域不会次正规，无需 FTZ（x86 对应运算在该输入域下同样不触发冲刷）
        float sum = bs + be;
        float coord = MIDPOINT_WEIGHT * sum * imgSizeMinus1;
        return coord;
    }
}

// 预计算几何常量
template <typename IDX_T>
__simt_callee__ inline Geometry<IDX_T> ComputeGeometry(const ScalarParams<IDX_T>& params)
{
    Geometry<IDX_T> geom;
    // 防御性处理：tiling 不应下发 0 维图像；这里避免无符号 IDX_T 下溢。
    geom.imgHMinus1 = params.imageHeight > 0 ? static_cast<float>(params.imageHeight - 1) : 0.0f;
    geom.imgWMinus1 = params.imageWidth > 0 ? static_cast<float>(params.imageWidth - 1) : 0.0f;
    geom.imgHWC = params.imageHeight * params.imageWidth * params.depth;
    geom.cropHWC = params.cropHeight * params.cropWidth * params.depth;
    return geom;
}

// 统一填充（warp lane 分摊）：所有通道写同一个值，覆盖 4 条整通道填充路径
template <typename T_Y, typename IDX_T, Layout LAYOUT>
__simt_callee__ inline void FillAllChannelsStrided(IDX_T depth, IDX_T laneId, IDX_T outBase, IDX_T channelOutStride,
                                                   float val, __gm__ T_Y* yGm)
{
    for (IDX_T d = laneId; d < depth; d += static_cast<IDX_T>(WARP_LANES)) {
        IDX_T off = (LAYOUT == LAYOUT_NCHW) ? d * channelOutStride : d;
        yGm[outBase + off] = CastFromFloat<T_Y>(val);
    }
}

// 输出布局（基址 outBase + 写侧通道步长 channelOutStride），编译期特化。
// NHWC = b*cropHWC+cy*cropW*C+cx*C（步长 d）；NCHW = b*cropHWC+cy*cropW+cx（步长 d*cropHW）
template <typename IDX_T, Layout LAYOUT>
__simt_callee__ inline void ComputeOutLayout(IDX_T b, IDX_T cy, IDX_T cx, const ScalarParams<IDX_T>& params,
                                             const Geometry<IDX_T>& geom, IDX_T& outBase, IDX_T& channelOutStride)
{
    if constexpr (LAYOUT == LAYOUT_NCHW) {
        // NCHW: y=(b,C,cy,cx)，outBase = b*cropHWC + cy*cropW + cx，d 循环写 outBase + d*cropH*cropW
        channelOutStride = params.cropHeight * params.cropWidth;
        outBase = b * geom.cropHWC + cy * params.cropWidth + cx;
    } else {
        // NHWC: y=(b,cy,cx,C)，outBase = b*cropHWC + cy*cropW*C + cx*C，d 循环写 outBase + d
        outBase = b * geom.cropHWC + cy * params.cropWidth * params.depth + cx * params.depth;
        channelOutStride = 1;
    }
}

// 位置级计算：读 box -> 坐标映射 -> 非有限/OOB 检查 -> 插值几何。
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T, Layout LAYOUT, bool IS_NEAREST>
__simt_callee__ inline PosLevelResult<IDX_T> ComputePosLevel(IDX_T b, IDX_T cy, IDX_T cx,
                                                             const ScalarParams<IDX_T>& params,
                                                             const Geometry<IDX_T>& geom,
                                                             GmPointers<T_X, T_BOXES, T_Y> gms)
{
    PosLevelResult<IDX_T> posResult;
    posResult.isFill = false;
    posResult.fillVal = 0.0f;
    ComputeOutLayout<IDX_T, LAYOUT>(b, cy, cx, params, geom, posResult.outBase, posResult.channelOutStride);

    // box_index 越界：防御性填充 0（TF 会抛 InvalidArgumentError，测试用例应避免越界）
    int32_t boxIdx = gms.boxIndexGm[b];
    if (boxIdx < 0 || static_cast<IDX_T>(boxIdx) >= params.batch) {
        posResult.isFill = true;
        posResult.fillVal = 0.0f;
        return posResult;
    }

    // 防御性处理：零维图像视为整块越界，统一填 extrapolationValue，不读取 x。
    if (params.imageHeight == 0 || params.imageWidth == 0) {
        posResult.isFill = true;
        posResult.fillVal = params.extrapolationValue;
        return posResult;
    }

    // 读取 box 归一化坐标 [y1, x1, y2, x2]（范围 0~1）
    IDX_T boxOffset = b * static_cast<IDX_T>(BOX_COORDS_PER_BOX);
    float y1 = CastToFloat(gms.boxesGm[boxOffset + 0]);
    float x1 = CastToFloat(gms.boxesGm[boxOffset + 1]);
    float y2 = CastToFloat(gms.boxesGm[boxOffset + 2]);
    float x2 = CastToFloat(gms.boxesGm[boxOffset + 3]);

    // 坐标映射：crop 坐标 -> 原图坐标
    float inY = MapCoordinate<IDX_T>(y1, y2, cy, params.cropHeight, geom.imgHMinus1);
    float inX = MapCoordinate<IDX_T>(x1, x2, cx, params.cropWidth, geom.imgWMinus1);

    // master GPU 语义：NaN/Inf 均走取反区间判断，整 box 填 extrapolationValue。
    // !(x >= 0 && x <= max) 对 NaN 为 true，避免 floorf/ceilf(NaN) 产生未定义索引。
    if (!(inY >= 0.0f && inY <= geom.imgHMinus1) || !(inX >= 0.0f && inX <= geom.imgWMinus1)) {
        posResult.isFill = true;
        posResult.fillVal = params.extrapolationValue;
        return posResult;
    }

    // 邻居索引 + 权重 + 偏移计算（layout 编译期特化）
    posResult.ig = ComputeInterpGeometry<IDX_T, LAYOUT, IS_NEAREST>(inY, inX, static_cast<IDX_T>(boxIdx), params, geom);
    return posResult;
}

// 处理单个输出位置的 warp 级实现：位置级计算由 warp 内 32 lane 以相同输入锁步冗余执行；
// 通道级计算 lane 分摊（lane 处理 d ≡ laneId mod 32），相邻 lane 读相邻通道实现 coalesced 访存。
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T, Layout LAYOUT, bool IS_NEAREST>
__simt_callee__ inline void ProcessOnePositionWarp(IDX_T b, IDX_T cy, IDX_T cx, IDX_T laneId,
                                                   const ScalarParams<IDX_T>& params, const Geometry<IDX_T>& geom,
                                                   GmPointers<T_X, T_BOXES, T_Y> gms)
{
    PosLevelResult<IDX_T> pr = ComputePosLevel<T_X, T_BOXES, T_Y, IDX_T, LAYOUT, IS_NEAREST>(b, cy, cx, params, geom,
                                                                                             gms);
    if (pr.isFill) {
        // 填充路径：lane 分摊写整通道
        FillAllChannelsStrided<T_Y, IDX_T, LAYOUT>(params.depth, laneId, pr.outBase, pr.channelOutStride, pr.fillVal,
                                                   gms.yGm);
        return;
    }

// 逐通道采样：lane 分摊 d ≡ laneId (mod 32)（展开 4 次提升 ILP）
#pragma unroll 4
    for (IDX_T d = laneId; d < params.depth; d += static_cast<IDX_T>(WARP_LANES)) {
        float result;
        if constexpr (IS_NEAREST) {
            result = NearestSampleChannel<T_X, IDX_T, LAYOUT>(gms.xGm, pr.ig, d);
        } else {
            result = BilinearSampleChannel<T_X, IDX_T, LAYOUT>(gms.xGm, pr.ig, d);
        }
        // 输出通道偏移：NCHW = d*cropH*cropW；NHWC = d（编译期折叠）
        IDX_T outOff = (LAYOUT == LAYOUT_NCHW) ? d * pr.channelOutStride : d;
        gms.yGm[pr.outBase + outOff] = CastFromFloat<T_Y>(result);
    }
}

// VF 核心计算：Grid-Stride 循环以 warp 粒度遍历输出位置
// 标量参数直接作为 VF 入参传递，避免 UB 中转。
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T, Layout LAYOUT, bool IS_NEAREST>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpCropAndResizeSimtKernel(
    IDX_T totalPositions, IDX_T batch, IDX_T imageHeight, IDX_T imageWidth, IDX_T depth, IDX_T cropHeight,
    IDX_T cropWidth, float extrapolationValue, __gm__ T_X* xGm, __gm__ T_BOXES* boxesGm, __gm__ int32_t* boxIndexGm,
    __gm__ T_Y* yGm)
{
    ScalarParams<IDX_T> params = {batch, imageHeight, imageWidth, depth, cropHeight, cropWidth, extrapolationValue};
    Geometry<IDX_T> geom = ComputeGeometry(params);
    GmPointers<T_X, T_BOXES, T_Y> gms = {xGm, boxesGm, boxIndexGm, yGm};

    IDX_T cropHW = params.cropHeight * params.cropWidth;
    IDX_T cropWidthT = params.cropWidth;

    // 除零防御：host tiling 不一致（cropHeight=0 或 cropWidth=0）时避免整数除零异常
    if (cropHW == 0 || cropWidthT == 0) {
        return;
    }

    // warp 分解：全局线程 id -> (warpId, laneId)。线程总数 = THREADS × 核数，均为 32 的倍数，
    // warpStride 为全局 warp 槽位数，grid-stride 兜底保证任意 totalPositions 正确
    IDX_T tid = static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
    IDX_T laneId = tid % static_cast<IDX_T>(WARP_LANES);
    IDX_T warpId = tid / static_cast<IDX_T>(WARP_LANES);
    IDX_T warpStride = static_cast<IDX_T>(blockDim.x) * static_cast<IDX_T>(gridDim.x) / static_cast<IDX_T>(WARP_LANES);

    for (IDX_T pos = warpId; pos < totalPositions; pos += warpStride) {
        IDX_T b = pos / cropHW;
        IDX_T rem = pos - b * cropHW;
        IDX_T cy = rem / cropWidthT;
        IDX_T cx = rem - cy * cropWidthT;
        ProcessOnePositionWarp<T_X, T_BOXES, T_Y, IDX_T, LAYOUT, IS_NEAREST>(b, cy, cx, laneId, params, geom, gms);
    }
}

// VF 启动函数：直接把标量参数传给 VF kernel
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T, Layout LAYOUT, bool IS_NEAREST>
__aicore__ inline void LaunchVf(const CropAndResizeTilingData* tilingData, __gm__ T_X* xGm, __gm__ T_BOXES* boxesGm,
                                __gm__ int32_t* boxIndexGm, __gm__ T_Y* yGm)
{
    constexpr uint32_t TIDX = THREADS<IDX_T>;
    asc_vf_call<OpCropAndResizeSimtKernel<T_X, T_BOXES, T_Y, IDX_T, LAYOUT, IS_NEAREST>>(
        dim3(TIDX), static_cast<IDX_T>(tilingData->totalPositions), static_cast<IDX_T>(tilingData->batch),
        static_cast<IDX_T>(tilingData->imageHeight), static_cast<IDX_T>(tilingData->imageWidth),
        static_cast<IDX_T>(tilingData->depth), static_cast<IDX_T>(tilingData->cropHeight),
        static_cast<IDX_T>(tilingData->cropWidth), tilingData->extrapolationValue, xGm, boxesGm, boxIndexGm, yGm);
}

// 根据所有输入/输出 tensor 的元素总数选择 32/64 位索引路径。
// 32 位路径要求 x、y、boxes 的寻址范围均不超过 INT32_MAX。
__aicore__ inline bool NeedInt64Index(const CropAndResizeTilingData* tilingData)
{
    constexpr int64_t kInt32Max = 2147483647LL;
    int64_t xTotal = tilingData->batch * tilingData->imageHeight * tilingData->imageWidth * tilingData->depth;
    int64_t yTotal = tilingData->numBoxes * tilingData->cropHeight * tilingData->cropWidth * tilingData->depth;
    int64_t boxesTotal = tilingData->numBoxes * 4;
    return xTotal > kInt32Max || yTotal > kInt32Max || boxesTotal > kInt32Max;
}

// Process 入口函数：按所有输入/输出元素总数选择 uint32/uint64 路径
template <typename T_X, typename T_BOXES, typename T_Y, Layout LAYOUT, bool IS_NEAREST>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR boxes, GM_ADDR boxIndex, GM_ADDR cropSize, GM_ADDR y,
                               const CropAndResizeTilingData* tilingData)
{
    (void)cropSize;

    if (tilingData == nullptr) {
        return;
    }

    if (tilingData->totalPositions == 0) {
        return;
    }

    __gm__ T_X* xGm = (__gm__ T_X*)x;
    __gm__ T_BOXES* boxesGm = (__gm__ T_BOXES*)boxes;
    __gm__ int32_t* boxIndexGm = (__gm__ int32_t*)boxIndex;
    __gm__ T_Y* yGm = (__gm__ T_Y*)y;

    bool needInt64 = NeedInt64Index(tilingData);
    if (!needInt64) {
        LaunchVf<T_X, T_BOXES, T_Y, uint32_t, LAYOUT, IS_NEAREST>(tilingData, xGm, boxesGm, boxIndexGm, yGm);
    } else {
        LaunchVf<T_X, T_BOXES, T_Y, uint64_t, LAYOUT, IS_NEAREST>(tilingData, xGm, boxesGm, boxIndexGm, yGm);
    }
}
} // namespace NsCropAndResize
#endif // CROP_AND_RESIZE_SIMT_H_
