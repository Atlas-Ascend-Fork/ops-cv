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
 * \file guided_filter_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t IDX_GUIDE = 0;
static constexpr size_t IDX_SRC = 1;
static constexpr size_t OUT_DST = 0;

// 布局：rank2=[H,W]，rank3=[H,W,C]，rank4=[N,H,W,C]（通道后置）。
// guide 与 src 必须同 rank、同空间尺寸；通道数可以不同。dst 与 src 完全同 shape。
static ge::graphStatus InferShapeGuidedFilter(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeGuidedFilter");

    const gert::Shape* guideShape = context->GetInputShape(IDX_GUIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, guideShape);
    const gert::Shape* srcShape = context->GetInputShape(IDX_SRC);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcShape);
    gert::Shape* dstShape = context->GetOutputShape(OUT_DST);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstShape);

    const size_t guideRank = guideShape->GetDimNum();
    const size_t srcRank = srcShape->GetDimNum();
    if (guideRank != srcRank) {
        OP_LOGE(context->GetNodeName(), "guide rank %zu != src rank %zu", guideRank, srcRank);
        return GRAPH_FAILED;
    }

    // dst 完全拷贝 src 的 shape（含通道数）
    dstShape->SetDimNum(srcRank);
    for (size_t i = 0; i < srcRank; i++) {
        dstShape->SetDim(i, srcShape->GetDim(i));
    }

    // rank4 的 batch 维一致性：dst 完全继承 src 的 shape，而 kernel 的 block 数
    // 按 guide 的 N 计算，N 不一致会导致越界读 src（src.N 更小）或 dst 尾部 batch
    // 永不写出（src.N 更大）；未知维(-1)跳过
    if (srcRank == 4) {
        const int64_t g = guideShape->GetDim(0);
        const int64_t s = srcShape->GetDim(0);
        if (g != -1 && s != -1 && g != s) {
            OP_LOGE(context->GetNodeName(), "guide batch[0]=%ld 与 src=%ld 不一致", g, s);
            return GRAPH_FAILED;
        }
    }

    // 空间维一致性校验（rank2/3 的 H,W 在 [0,1]，rank4 的 H,W 在 [1,2]）；未知维(-1)跳过
    if (srcRank >= 2) {
        const size_t hIdx = (srcRank == 4) ? 1 : 0;
        const size_t wIdx = hIdx + 1;
        for (size_t i = 0; i < 2; i++) {
            const int64_t g = guideShape->GetDim(hIdx + i);
            const int64_t s = srcShape->GetDim(hIdx + i);
            if (g != -1 && s != -1 && g != s) {
                OP_LOGE(context->GetNodeName(), "guide 空间维[%zu]=%ld 与 src=%ld 不一致", hIdx + i, g, s);
                return GRAPH_FAILED;
            }
        }
        (void)wIdx;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeGuidedFilter");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GuidedFilter).InferShape(InferShapeGuidedFilter);
} // namespace ops
