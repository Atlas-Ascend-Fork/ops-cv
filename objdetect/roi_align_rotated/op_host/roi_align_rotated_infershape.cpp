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
 * \file roi_align_rotated.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t ROIS_INDEX = 1;
const uint32_t OUTPUT_INDEX = 0;
const uint32_t ROIS_NUM_INDEX = 1;

const uint32_t BS_INDEX = 0;
const uint32_t H_INDEX = 1;
const uint32_t W_INDEX = 2;
const uint32_t CHANNEL_INDEX = 3;

const uint32_t PH_INDEX = 0;
const uint32_t PW_INDEX = 1;
const uint32_t SPATIAL_INDEX = 2;
const uint32_t SAMPLING_INDEX = 3;
const uint32_t ALIGNED_INDEX = 4;
const uint32_t CLOCKWISE_INDEX = 5;

const uint32_t ALIGN_VALUE = 8;
const uint32_t TILING_KEY = 1;

const uint32_t DIM_NUM_ZERO = 0;
const uint32_t DIM_NUM_ONE = 1;
const uint32_t DIM_NUM_TWO = 2;
const uint32_t DIM_NUM_THREE = 3;
const uint32_t INPUT_RANK = 4;
const uint32_t ROIS_RANK = 2;
const uint32_t OUTPUT_DIM_NUM = INPUT_RANK;
} // namespace

namespace ops {
ge::graphStatus InferShapeRoiAlignRotatedRegbase(gert::InferShapeContext* context);

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto input_shape = context->GetInputShape(INPUT_INDEX);
    auto rois_shape = context->GetInputShape(ROIS_INDEX);
    auto output_shape = context->GetOutputShape(OUTPUT_INDEX);
    if (input_shape == nullptr || rois_shape == nullptr || output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*input_shape) || Ops::Base::IsUnknownRank(*rois_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }
    // ascend950（arch35）regbase 芯片分发到 NCHW 推导；
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    OP_CHECK_IF((fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
                 ge::GRAPH_SUCCESS),
                OP_LOGE(context, "Cannot get platform info!"), return ge::GRAPH_FAILED);
    if (platformInfo.str_info.short_soc_version == "Ascend950") {
        return InferShapeRoiAlignRotatedRegbase(context);
    }
    if (input_shape->GetDimNum() != INPUT_RANK || rois_shape->GetDimNum() != ROIS_RANK) {
        return ge::GRAPH_FAILED;
    }

    int64_t rois_num = rois_shape->GetDim(ROIS_NUM_INDEX);
    int64_t channels = input_shape->GetDim(CHANNEL_INDEX);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* pooled_height = attrsPtr->GetAttrPointer<int64_t>(PH_INDEX);
    const int64_t* pooled_width = attrsPtr->GetAttrPointer<int64_t>(PW_INDEX);
    if (pooled_height == nullptr || pooled_width == nullptr || *pooled_height <= 0 || *pooled_width <= 0) {
        return ge::GRAPH_FAILED;
    }

    auto output_shape_length = OUTPUT_DIM_NUM;
    output_shape->SetDimNum(output_shape_length);
    output_shape->SetDim(DIM_NUM_ZERO, rois_num);
    output_shape->SetDim(DIM_NUM_ONE, *pooled_height); // 设置输出形状的第二个维度为pooled_height
    output_shape->SetDim(DIM_NUM_TWO, *pooled_width);  // 设置输出形状的第三个维度为pooled_width
    output_shape->SetDim(DIM_NUM_THREE, channels);     // 设置输出形状的第四个维度为channels

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeRoiAlignRotated(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX, value_dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiAlignRotated).InferShape(InferShape);
} // namespace ops
