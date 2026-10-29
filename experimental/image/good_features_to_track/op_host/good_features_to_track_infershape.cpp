/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kImageRank = 3;   // [N, H, W]
constexpr size_t kCornersRank = 3; // [N, max_corners, 2]
constexpr size_t kCountRank = 1;   // [N]
constexpr int64_t kCoordSize = 2;
constexpr int64_t kDefaultMaxCorners = 1000;
constexpr int64_t kUnknown = -1;
constexpr uint32_t kMaxCornersAttrIdx = 0;
} // namespace

static ge::graphStatus InferShapeGoodFeaturesToTrack(gert::InferShapeContext* context)
{
    const gert::Shape* imageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageShape);
    OP_CHECK_IF(imageShape->GetDimNum() != kImageRank, OP_LOGE(context, "image rank must be %zu", kImageRank),
                return GRAPH_FAILED);

    // 可选掩码：存在时须与 image 同 rank 且各维兼容
    const gert::Shape* maskShape = context->GetInputShape(1);
    if (maskShape != nullptr) {
        OP_CHECK_IF(maskShape->GetDimNum() != kImageRank, OP_LOGE(context, "mask rank must be %zu", kImageRank),
                    return GRAPH_FAILED);
        for (size_t d = 0; d < kImageRank; ++d) {
            const int64_t lhs = imageShape->GetDim(d);
            const int64_t rhs = maskShape->GetDim(d);
            OP_CHECK_IF(lhs != kUnknown && rhs != kUnknown && lhs != rhs,
                        OP_LOGE(context, "mask dim %zu must match image", d), return GRAPH_FAILED);
        }
    }

    // 输出上界：max_corners > 0 时为其值，否则退化为 H*W
    int64_t maxCorners = kDefaultMaxCorners;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetInt(kMaxCornersAttrIdx) != nullptr) {
        maxCorners = *attrs->GetInt(kMaxCornersAttrIdx);
    }
    const int64_t height = imageShape->GetDim(1);
    const int64_t width = imageShape->GetDim(2);
    int64_t capacity = maxCorners;
    if (maxCorners <= 0) {
        capacity = (height == kUnknown || width == kUnknown) ? kUnknown : height * width;
    }

    gert::Shape* cornersShape = context->GetOutputShape(0);
    gert::Shape* countShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, cornersShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, countShape);

    cornersShape->SetDimNum(kCornersRank);
    cornersShape->SetDim(0, imageShape->GetDim(0));
    cornersShape->SetDim(1, capacity);
    cornersShape->SetDim(2, kCoordSize);

    countShape->SetDimNum(kCountRank);
    countShape->SetDim(0, imageShape->GetDim(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GoodFeaturesToTrack).InferShape(InferShapeGoodFeaturesToTrack);
} // namespace ops
