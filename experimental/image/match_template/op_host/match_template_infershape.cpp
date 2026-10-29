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
 * \file match_template_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMatchTemplate(gert::InferShapeContext* context)
{
    constexpr size_t kImageIndex = 0;
    constexpr size_t kTemplateIndex = 1;
    constexpr size_t kScoreIndex = 0;
    constexpr size_t kBestScoreIndex = 1;
    constexpr size_t kBestLocationIndex = 2;
    constexpr size_t kExpectedRank = 4;

    const gert::Shape* imageShape = context->GetInputShape(kImageIndex);
    const gert::Shape* templateShape = context->GetInputShape(kTemplateIndex);
    const gert::CompileTimeTensorDesc* imageDesc = context->GetInputDesc(kImageIndex);
    const gert::CompileTimeTensorDesc* templateDesc = context->GetInputDesc(kTemplateIndex);
    gert::Shape* scoreShape = context->GetOutputShape(kScoreIndex);
    gert::Shape* bestScoreShape = context->GetOutputShape(kBestScoreIndex);
    gert::Shape* bestLocationShape = context->GetOutputShape(kBestLocationIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, templateShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, templateDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bestScoreShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bestLocationShape);

    OP_CHECK_IF(imageShape->GetDimNum() != kExpectedRank || templateShape->GetDimNum() != kExpectedRank,
                OP_LOGE(context->GetNodeName(), "image and template must both be rank-4 NHWC tensors"),
                return GRAPH_FAILED);

    const ge::DataType imageType = imageDesc->GetDataType();
    const ge::DataType templateType = templateDesc->GetDataType();
    OP_CHECK_IF(imageType != templateType, OP_LOGE(context->GetNodeName(), "image and template dtypes must match"),
                return GRAPH_FAILED);
    OP_CHECK_IF(imageType != ge::DT_UINT8 && imageType != ge::DT_FLOAT16 && imageType != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "supported dtypes are uint8, float16, and float32"),
                return GRAPH_FAILED);

    const int64_t n = imageShape->GetDim(0);
    const int64_t imageH = imageShape->GetDim(1);
    const int64_t imageW = imageShape->GetDim(2);
    const int64_t imageC = imageShape->GetDim(3);
    const int64_t templateN = templateShape->GetDim(0);
    const int64_t templateH = templateShape->GetDim(1);
    const int64_t templateW = templateShape->GetDim(2);
    const int64_t templateC = templateShape->GetDim(3);

    OP_CHECK_IF(n == 0 || imageH == 0 || imageW == 0 || imageC == 0 || templateN == 0 || templateH == 0 ||
                    templateW == 0 || templateC == 0,
                OP_LOGE(context->GetNodeName(), "MatchTemplate does not accept zero-sized dimensions"),
                return GRAPH_FAILED);
    OP_CHECK_IF(templateN > 0 && templateN != 1, OP_LOGE(context->GetNodeName(), "template batch must be 1"),
                return GRAPH_FAILED);
    OP_CHECK_IF(imageC > 0 && templateC > 0 && imageC != templateC,
                OP_LOGE(context->GetNodeName(), "image and template channel counts must match"), return GRAPH_FAILED);
    OP_CHECK_IF(imageH > 0 && templateH > 0 && templateH > imageH,
                OP_LOGE(context->GetNodeName(), "template height must not exceed image height"), return GRAPH_FAILED);
    OP_CHECK_IF(imageW > 0 && templateW > 0 && templateW > imageW,
                OP_LOGE(context->GetNodeName(), "template width must not exceed image width"), return GRAPH_FAILED);

    const int64_t outputH = (imageH > 0 && templateH > 0) ? imageH - templateH + 1 : -1;
    const int64_t outputW = (imageW > 0 && templateW > 0) ? imageW - templateW + 1 : -1;
    scoreShape->SetDimNum(3);
    scoreShape->SetDim(0, n);
    scoreShape->SetDim(1, outputH);
    scoreShape->SetDim(2, outputW);

    bestScoreShape->SetDimNum(1);
    bestScoreShape->SetDim(0, n);

    bestLocationShape->SetDimNum(2);
    bestLocationShape->SetDim(0, n);
    bestLocationShape->SetDim(1, 2);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatchTemplate).InferShape(InferShapeMatchTemplate);
} // namespace ops
