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

namespace {
constexpr int64_t kScoresIndex = 1;
constexpr int64_t kNmsedBoxesIndex = 0;
constexpr int64_t kNmsedScoresIndex = 1;
constexpr int64_t kNmsedClassesIndex = 2;
constexpr int64_t kNmsedNumIndex = 3;
constexpr int64_t kMaxTotalSizeAttrIndex = 3;
constexpr int64_t kTransposeBoxAttrIndex = 5;
constexpr int64_t kMaxTotalSize = 1000;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForBatchMultiClassNonMaxSuppression(gert::InferShapeContext* context)
{
    const gert::Shape* scoresShape = context->GetInputShape(kScoresIndex);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(scoresShape->GetDimNum() == 0, OP_LOGE(context, "scores shape is empty."), return ge::GRAPH_FAILED);
    const int64_t* maxTotalSize = attrs->GetAttrPointer<int64_t>(kMaxTotalSizeAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxTotalSize);
    OP_CHECK_IF(*maxTotalSize < 1 || *maxTotalSize > kMaxTotalSize,
                OP_LOGE(context, "max_total_size must be in [1, %ld].", kMaxTotalSize), return ge::GRAPH_FAILED);
    const bool* transposeBoxAttr = attrs->GetAttrPointer<bool>(kTransposeBoxAttrIndex);
    const bool transposeBox = transposeBoxAttr != nullptr && *transposeBoxAttr;

    // Match the legacy GE inference, including its conservative unknown-batch
    // rule. The fusion pass uses transpose_box for both input and output layouts.
    int64_t batch = scoresShape->GetDim(0);
    for (size_t dim = 0; dim < scoresShape->GetDimNum(); ++dim) {
        if (scoresShape->GetDim(dim) < 0) {
            batch = ge::UNKNOWN_DIM;
            break;
        }
    }

    gert::Shape* nmsedBoxesShape = context->GetOutputShape(kNmsedBoxesIndex);
    gert::Shape* nmsedScoresShape = context->GetOutputShape(kNmsedScoresIndex);
    gert::Shape* nmsedClassesShape = context->GetOutputShape(kNmsedClassesIndex);
    gert::Shape* nmsedNumShape = context->GetOutputShape(kNmsedNumIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedBoxesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedScoresShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedClassesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, nmsedNumShape);

    constexpr int64_t kBoxCoordinateCount = 4;
    constexpr int64_t kPaddedCountStride = 8;
    constexpr size_t kBoxesOutputRank = 3;
    constexpr size_t kMatrixOutputRank = 2;
    constexpr size_t kVectorOutputRank = 1;
    constexpr size_t kBatchAxis = 0;
    constexpr size_t kFirstInnerAxis = 1;
    constexpr size_t kSecondInnerAxis = 2;
    nmsedBoxesShape->SetDimNum(kBoxesOutputRank);
    nmsedBoxesShape->SetDim(kBatchAxis, batch);
    nmsedBoxesShape->SetDim(kFirstInnerAxis, transposeBox ? kBoxCoordinateCount : *maxTotalSize);
    nmsedBoxesShape->SetDim(kSecondInnerAxis, transposeBox ? *maxTotalSize : kBoxCoordinateCount);
    for (gert::Shape* output : {nmsedScoresShape, nmsedClassesShape}) {
        output->SetDimNum(kMatrixOutputRank);
        output->SetDim(kBatchAxis, batch);
        output->SetDim(kFirstInnerAxis, *maxTotalSize);
    }
    nmsedNumShape->SetDimNum(transposeBox ? kMatrixOutputRank : kVectorOutputRank);
    nmsedNumShape->SetDim(kBatchAxis, batch);
    if (transposeBox) {
        nmsedNumShape->SetDim(kFirstInnerAxis, kPaddedCountStride);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForBatchMultiClassNonMaxSuppression(gert::InferDataTypeContext* context)
{
    const ge::DataType scoresType = context->GetInputDataType(kScoresIndex);
    context->SetOutputDataType(kNmsedBoxesIndex, scoresType);
    context->SetOutputDataType(kNmsedScoresIndex, scoresType);
    context->SetOutputDataType(kNmsedClassesIndex, scoresType);
    context->SetOutputDataType(kNmsedNumIndex, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchMultiClassNonMaxSuppression)
    .InferShape(InferShapeForBatchMultiClassNonMaxSuppression)
    .InferDataType(InferDataTypeForBatchMultiClassNonMaxSuppression);
} // namespace ops
