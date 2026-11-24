/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scale_and_translate_infershape.cpp
 * \brief ScaleAndTranslate infershape
 *
 * Output shape (NHWC): y = [images.shape[0], size[0], size[1], images.shape[3]]
 */

#include <string>

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kImagesIndex = 0U;
constexpr size_t kSizeIndex = 1U;
constexpr size_t kScaleIndex = 2U;
constexpr size_t kTranslationIndex = 3U;
constexpr size_t kOutputIndex = 0U;
constexpr size_t kInputRank = 4U;
constexpr size_t kOutputRank = 4U;
constexpr size_t kBatchDim = 0U;
constexpr size_t kOutputHeightDim = 1U;
constexpr size_t kOutputWidthDim = 2U;
constexpr size_t kChannelDim = 3U;
constexpr size_t kControlInputRank = 1U;
constexpr int64_t kControlInputElements = 2;
} // namespace

namespace ops {

// Read output_height/output_width from the size input tensor.
// When size is non-const or its shape is unknown, both dims remain UNKNOWN_DIM.
static ge::graphStatus ReadSizeValues(gert::InferShapeContext* context, const gert::Shape* sizeShape,
                                      int64_t& outputHeight, int64_t& outputWidth)
{
    const gert::Tensor* sizeTensor = context->GetInputTensor(kSizeIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);

    outputHeight = ge::UNKNOWN_DIM;
    outputWidth = ge::UNKNOWN_DIM;

    // Guard: size may be unknown-rank (validation loop skips it); accessing GetDim(0)
    // without a rank check reads the -2 sentinel which is not UNKNOWN_DIM(-1), leading
    // to spurious data reads. Keep output dims as UNKNOWN_DIM and return success.
    if (Ops::Base::IsUnknownRank(*sizeShape) || sizeShape->GetDimNum() != kControlInputRank) {
        OP_LOGD(context->GetNodeName(), "size is unknown-rank or wrong rank, keep output dims as UNKNOWN_DIM");
        return ge::GRAPH_SUCCESS;
    }

    if (sizeShape->GetDim(0) == ge::UNKNOWN_DIM) {
        OP_LOGD(context->GetNodeName(), "size.shape[0] is UNKNOWN_DIM, skip reading value");
        return ge::GRAPH_SUCCESS;
    }

    if (sizeTensor->GetAddr() != nullptr) {
        const int32_t* sizeData = sizeTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, sizeData);
        outputHeight = static_cast<int64_t>(sizeData[0]);
        outputWidth = static_cast<int64_t>(sizeData[1]);

        if (outputHeight <= 0 || outputWidth <= 0) {
            std::string valMsg = "[" + std::to_string(outputHeight) + ", " + std::to_string(outputWidth) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", valMsg.c_str(),
                                                  "output_height and output_width must be positive");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGD(context->GetNodeName(), "size is non-const tensor, output dims remain UNKNOWN_DIM");
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeScaleAndTranslate(gert::InferShapeContext* context)
{
    const gert::Shape* imagesShape = context->GetInputShape(kImagesIndex);
    const gert::Shape* sizeShape = context->GetInputShape(kSizeIndex);
    const gert::Shape* scaleShape = context->GetInputShape(kScaleIndex);
    const gert::Shape* translationShape = context->GetInputShape(kTranslationIndex);
    gert::Shape* yShape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, imagesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, translationShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Unknown-rank passthrough: graph-time dynamic shape, resolved at tiling
    if (Ops::Base::IsUnknownRank(*imagesShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    // Validate images rank (NHWC 4D)
    OP_CHECK_IF(imagesShape->GetDimNum() != kInputRank,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "images",
                                                         (std::to_string(imagesShape->GetDimNum()) + "D").c_str(),
                                                         "images must be rank 4"),
                return ge::GRAPH_FAILED);

    // Validate size/scale/translation: rank 1 with exactly 2 elements each
    const gert::Shape* controlInputs[] = {sizeShape, scaleShape, translationShape};
    const char* controlNames[] = {"size", "scale", "translation"};
    for (size_t i = 0; i < sizeof(controlInputs) / sizeof(controlInputs[0]); ++i) {
        if (Ops::Base::IsUnknownRank(*controlInputs[i])) {
            continue;
        }
        OP_CHECK_IF(controlInputs[i]->GetDimNum() != kControlInputRank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), controlNames[i],
                        (std::to_string(controlInputs[i]->GetDimNum()) + "D").c_str(), "must be rank 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            controlInputs[i]->GetDim(0) != kControlInputElements && controlInputs[i]->GetDim(0) != ge::UNKNOWN_DIM,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), controlNames[i],
                                                      std::to_string(controlInputs[i]->GetDim(0)).c_str(),
                                                      "must have exactly 2 elements"),
            return ge::GRAPH_FAILED);
    }

    // Read output height/width from size tensor (const) or keep UNKNOWN_DIM
    int64_t outputHeight = ge::UNKNOWN_DIM;
    int64_t outputWidth = ge::UNKNOWN_DIM;
    OP_CHECK_IF(ReadSizeValues(context, sizeShape, outputHeight, outputWidth) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ReadSizeValues failed"), return ge::GRAPH_FAILED);

    // Construct output: y = [batch, output_height, output_width, channels]
    yShape->SetDimNum(kOutputRank);
    yShape->SetDim(kBatchDim, imagesShape->GetDim(kBatchDim));
    yShape->SetDim(kOutputHeightDim, outputHeight);
    yShape->SetDim(kOutputWidthDim, outputWidth);
    yShape->SetDim(kChannelDim, imagesShape->GetDim(kChannelDim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeScaleAndTranslate(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Output dtype is always DT_FLOAT (aligned with def.cpp Output("y").DataType({DT_FLOAT}))
    context->SetOutputDataType(kOutputIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScaleAndTranslate)
    .InferShape(InferShapeScaleAndTranslate)
    .InferDataType(InferDataTypeScaleAndTranslate)
    .InputsDataDependency({kSizeIndex});
} // namespace ops
