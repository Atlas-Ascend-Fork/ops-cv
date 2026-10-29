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
 * \file pixel_shuffle_infershape.cpp
 * \brief PixelShuffle（亚像素卷积）shape 推导：把通道维按 r^S 拆分到 S 个空间维
 *
 * 4D(N,C,H,W)   : C 可拆为 C_out * r^2，输出 (N, C/r^2, H*r, W*r)
 * 5D(N,C,D,H,W) : C 可拆为 C_out * r^3，输出 (N, C/r^3, D*r, H*r, W*r)
 * 某维为 -1 表示动态 shape 未知：跳过整除性校验，并把该维原样传递给输出。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
// PixelShuffle 单输入单输出，输入固定为 x、输出固定为 y
constexpr size_t kInputIdx = 0;
constexpr size_t kOutputIdx = 0;
constexpr int64_t kDimUnknown = -1;

// scale_factor 上界：4D 需算 r^2、5D 需算 r^3，限 r <= 2^20 可保证 r^3 <= 2^60 < INT64_MAX，
// 下方 rPow 累乘不会溢出；否则溢出为 0 时 "C_in % rPow" 会整数除零（host 侧 SIGFPE）。
static constexpr int64_t MAX_SCALE_FACTOR = 1LL << 20;

static ge::graphStatus InferShapePixelShuffle(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(kInputIdx);
    gert::Shape* outShape = context->GetOutputShape(kOutputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    // 仅支持 4D(N,C,H,W) 与 5D(N,C,D,H,W)，通道固定在 index 1
    const size_t rank = inShape->GetDimNum();
    if (rank != 4 && rank != 5) {
        OP_LOGE(context->GetNodeName(), "PixelShuffle expects a 4D or 5D input, but got %zu dims", rank);
        return GRAPH_FAILED;
    }
    // 仅 -1 表示未知维；dim < -1 均为畸形 shape，必须在此拒绝：两个负维相乘为正，
    // 仅靠“长度 > 0 + 通道可整除”无法拦下（例 (1,-4,-3,2) r=2 全通过），
    // kernel 会按负的 H/C_out 推导出错误块数与越界 GM 偏移。
    for (size_t i = 0; i < rank; ++i) {
        if (inShape->GetDim(i) < kDimUnknown) {
            OP_LOGE(context->GetNodeName(), "dim[%zu] = %ld is invalid, only -1 is allowed for an unknown dim", i,
                    inShape->GetDim(i));
            return GRAPH_FAILED;
        }
    }
    OP_LOGD(context->GetNodeName(), "PixelShuffle infershape begin, rank=%zu", rank);

    const bool is5d = (rank == 5);
    const int64_t n = inShape->GetDim(0);
    const int64_t cIn = inShape->GetDim(1);
    const int64_t dIn = is5d ? inShape->GetDim(2) : 1;
    const int64_t hIn = is5d ? inShape->GetDim(3) : inShape->GetDim(2);
    const int64_t wIn = inShape->GetDim(rank - 1);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGW(context->GetNodeName(), "PixelShuffle has no attributes, scale_factor falls back to default");
        return GRAPH_FAILED;
    }
    const int64_t* scalePtr = attrs->GetInt(0);
    if (scalePtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "attribute scale_factor is required but not set");
        return GRAPH_FAILED;
    }
    const int64_t r = *scalePtr;
    if (r < 2 || r > MAX_SCALE_FACTOR) {
        OP_LOGE(context->GetNodeName(), "scale_factor must be within [2, %ld], but got %ld", MAX_SCALE_FACTOR, r);
        return GRAPH_FAILED;
    }

    // 通道被拆分的因子：4D 为 r^2，5D 为 r^3
    const int64_t spatialNum = is5d ? 3 : 2;
    int64_t chanFactor = 1;
    for (int64_t i = 0; i < spatialNum; ++i) {
        chanFactor *= r;
    }

    // 动态 shape：C=-1 未知时无法判断整除性，跳过校验；否则 C 必须能被 r^S 整除
    const bool cUnknown = (cIn == kDimUnknown);
    if (!cUnknown && cIn % chanFactor != 0) {
        OP_LOGE(context->GetNodeName(), "input channels %ld is not divisible by r^%ld = %ld", cIn, spatialNum,
                chanFactor);
        return GRAPH_FAILED;
    }
    const int64_t cOut = cUnknown ? kDimUnknown : cIn / chanFactor;
    // 空间维放大 r 倍；未知维(-1)保持 -1，避免写出非法的负维度
    auto upscale = [r](int64_t dim) -> int64_t { return (dim == kDimUnknown) ? kDimUnknown : dim * r; };

    if (is5d) {
        outShape->SetDimNum(5);
        outShape->SetDim(0, n);
        outShape->SetDim(1, cOut);
        outShape->SetDim(2, upscale(dIn));
        outShape->SetDim(3, upscale(hIn));
        outShape->SetDim(4, upscale(wIn));
    } else {
        outShape->SetDimNum(4);
        outShape->SetDim(0, n);
        outShape->SetDim(1, cOut);
        outShape->SetDim(2, upscale(hIn));
        outShape->SetDim(3, upscale(wIn));
    }

    OP_LOGD(context->GetNodeName(), "PixelShuffle infershape end, c_out=%ld", cOut);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PixelShuffle).InferShape(InferShapePixelShuffle);
} // namespace ops
