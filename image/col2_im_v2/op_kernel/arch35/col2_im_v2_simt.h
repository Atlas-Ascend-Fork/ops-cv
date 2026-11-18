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
 * \file col2_im_v2_simt.h
 * \brief col2_im_v2 SIMT kernel（输出 centric，fp32 提升累加，写回时单次舍入 —— GPU col2im_device 语义）
 *
 * 累加语义：
 *   1) 累加顺序：对单个输出元素，贡献按 kernel offset (h_k, w_k) 字典序升序到达
 *      （遍历结构与 CPU col2im scatter 循环的贡献到达顺序一致）；
 *   2) 累加精度：fp32 提升累加（ACC_T = float，全程无中间舍入），写回输出时单次舍入回 D_T，
 *      与 GPU/CUDA col2im_device 的 float 累加器语义一致。
 * 已知偏差：与 CPU golden（torch F.fold，输出 dtype 逐步舍入）在 fp16 高重叠场景存在
 * ~1 ULP 级偏差（TTK 第 6 轮实测 mere 最高 36.2，与 GPU 语义同量级）。
 */

#ifndef __COL2_IM_V2_SIMT_H__
#define __COL2_IM_V2_SIMT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "col2_im_v2_tiling_data.h"
#include "col2_im_v2_tiling_key.h"

namespace NsCol2ImV2 {

using namespace AscendC;

constexpr uint32_t VF_MAX_THREAD_NUM = 1024; // VF 线程数，编译期常量，禁止从 tiling 获取

// 贡献累加子函数：遍历结构不变——(hK, wK) 字典序升序 + stride 整除过滤 + 边界过滤 + uint64 偏移；
// 累加精度为 fp32 提升累加（val 为 float，全程无中间舍入），写回时单次舍入 —— GPU col2im_device 语义
template <typename D_T, typename StrideT>
__simt_callee__ inline void AccumulateWindows(__gm__ D_T* xGmAddr, float& val, const uint32_t cIm, const uint32_t hIm,
                                              const uint32_t wIm, const uint32_t colH, const uint32_t colW,
                                              const uint32_t kernelSizeH, const uint32_t kernelSizeW,
                                              const StrideT strideH, const StrideT strideW, const uint32_t dilationH,
                                              const uint32_t dilationW, const StrideT shiftSH, const StrideT mSH,
                                              const StrideT shiftSW, const StrideT mSW)
{
    for (uint32_t hK = 0; hK < kernelSizeH; hK += 1) {
        // 贡献存在条件：存在 hCol >= 0 使 hIm = hCol*strideH + hK*dilationH
        // （hIm 为含 pad 坐标，等价于 CPU col2im 的 0 <= h_im < outH 越界检查）
        if (hIm < hK * dilationH) {
            continue; // remH < 0：无合法 hCol（防无符号下溢）
        }
        const StrideT remH = hIm - hK * dilationH;
        const StrideT hCol = Simt::UintDiv(remH, mSH, shiftSH);
        if (remH - hCol * strideH != 0 || hCol >= colH) {
            continue; // stride 整除过滤 + hCol 越界过滤
        }
        for (uint32_t wK = 0; wK < kernelSizeW; wK += 1) {
            if (wIm < wK * dilationW) {
                continue;
            }
            const StrideT remW = wIm - wK * dilationW;
            const StrideT wCol = Simt::UintDiv(remW, mSW, shiftSW);
            if (remW - wCol * strideW != 0 || wCol >= colW) {
                continue;
            }
            // 3-D x 线性偏移：((cIm*kH + hK)*kW + wK)*ho*wo + hCol*wo + wCol
            // 数学等价于 x[n, c*kH*kW + hK*kW + wK, hCol*wo + wCol]（cIm = n*C + c 合并）
            // uint64_t 合成，防大 shape 偏移溢出
            uint64_t xIdx = (((static_cast<uint64_t>(cIm) * kernelSizeH + hK) * kernelSizeW + wK) *
                                 static_cast<uint64_t>(colH) +
                             hCol) *
                                colW +
                            wCol;
            // fp32 提升累加（GPU col2im_device 语义）：float 累加器全程无中间舍入，
            // 写回输出时单次舍入回 D_T
            val += static_cast<float>(xGmAddr[xIdx]);
        }
    }
}

// VF 内核：输出 centric（每线程独立一个输出元素，无 atomicAdd），贡献按 (hK, wK) 升序累加
template <typename D_T, typename StrideT>
__simt_vf__ __aicore__ __launch_bounds__(VF_MAX_THREAD_NUM) inline void Col2ImV2SimtCompute(
    __gm__ D_T* xGmAddr, __gm__ D_T* yGmAddr, const uint32_t cnt, const uint32_t outH, const uint32_t outW,
    const uint32_t kernelSizeH, const uint32_t kernelSizeW, const uint32_t padH, const uint32_t padW,
    const StrideT strideH, const StrideT strideW, const uint32_t dilationH, const uint32_t dilationW,
    const uint32_t shiftGW, const uint32_t mGW, const uint32_t shiftGWH, const uint32_t mGWH, const StrideT shiftSW,
    const StrideT mSW, const StrideT shiftSH, const StrideT mSH)
{
    // colH/colW（ho/wo）VF 内自算（循环外一次）：复用 stride 快除，与 tiling 侧公式一致（tiling 已校验 >= 1）
    const StrideT numeratorH = outH + 2 * padH - dilationH * (kernelSizeH - 1) - 1;
    const StrideT numeratorW = outW + 2 * padW - dilationW * (kernelSizeW - 1) - 1;
    const uint32_t colH = static_cast<uint32_t>(Simt::UintDiv(numeratorH, mSH, shiftSH) + 1);
    const uint32_t colW = static_cast<uint32_t>(Simt::UintDiv(numeratorW, mSW, shiftSW) + 1);

    for (uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < cnt; idx += gridDim.x * blockDim.x) {
        float val = 0.0f; // fp32 提升累加器（ACC_T = float）；隐式补零：不被任何滑窗覆盖的位置输出 0
        const uint32_t wIm = idx % outW + padW;
        const uint32_t hIm = Simt::UintDiv(idx, mGW, shiftGW) % outH + padH;
        const uint32_t cIm = Simt::UintDiv(idx, mGWH, shiftGWH); // n*c 合并通道（3-D x 直接消费的关键）

        AccumulateWindows<D_T>(xGmAddr, val, cIm, hIm, wIm, colH, colW, kernelSizeH, kernelSizeW, strideH, strideW,
                               dilationH, dilationW, shiftSH, mSH, shiftSW, mSW);
        // fp32 累加结果写回时单次舍入回 D_T（GPU col2im_device 语义）
        yGmAddr[idx] = static_cast<D_T>(val);
    }
}

// 在 scalar 侧预计算 4 组 UintDiv magic/shift，再调用相应位宽的 VF。
template <typename T, typename StrideT>
__aicore__ inline void LaunchSimt(GM_ADDR x, GM_ADDR y, const Col2ImV2TilingData* tilingData)
{
    uint32_t shiftGW, mGW, shiftGWH, mGWH;
    StrideT shiftSW, mSW, shiftSH, mSH;
    uint32_t outW = static_cast<uint32_t>(tilingData->outputSizeW);
    uint32_t outWH = static_cast<uint32_t>(tilingData->outputSizeH * tilingData->outputSizeW);
    const StrideT strideH = static_cast<StrideT>(tilingData->strideH);
    const StrideT strideW = static_cast<StrideT>(tilingData->strideW);
    GetUintDivMagicAndShift(mGW, shiftGW, outW);    // / outW
    GetUintDivMagicAndShift(mGWH, shiftGWH, outWH); // / (outH*outW)
    GetUintDivMagicAndShift(mSW, shiftSW, strideW);
    GetUintDivMagicAndShift(mSH, shiftSH, strideH);

    // fp32 提升累加 + 写回单次舍入（GPU col2im_device 语义），遍历顺序仍为 (hK, wK) 升序
    asc_vf_call<Col2ImV2SimtCompute<T, StrideT>>(
        dim3(VF_MAX_THREAD_NUM), (__gm__ T*)x, (__gm__ T*)y, static_cast<uint32_t>(tilingData->totalLength),
        static_cast<uint32_t>(tilingData->outputSizeH), outW, static_cast<uint32_t>(tilingData->kernelSizeH),
        static_cast<uint32_t>(tilingData->kernelSizeW), static_cast<uint32_t>(tilingData->paddingH),
        static_cast<uint32_t>(tilingData->paddingW), strideH, strideW, static_cast<uint32_t>(tilingData->dilationH),
        static_cast<uint32_t>(tilingData->dilationW), shiftGW, mGW, shiftGWH, mGWH, shiftSW, mSW, shiftSH, mSH);
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const Col2ImV2TilingData* tilingData)
{
    // uint32 快除要求除数不超过 INT32_MAX，被除数过大时加法也可能回绕。
    // out + 2 * pad - 1 是窗口分子上界，tiling 保证 out + pad <= INT32_MAX。
    if (tilingData->strideH > INT32_MAX || tilingData->strideW > INT32_MAX ||
        tilingData->outputSizeH + 2 * tilingData->paddingH - 1 > INT32_MAX ||
        tilingData->outputSizeW + 2 * tilingData->paddingW - 1 > INT32_MAX) {
        LaunchSimt<T, uint64_t>(x, y, tilingData);
    } else {
        LaunchSimt<T, uint32_t>(x, y, tilingData);
    }
}

} // namespace NsCol2ImV2

#endif
