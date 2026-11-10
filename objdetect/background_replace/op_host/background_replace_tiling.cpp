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
 * \file background_replace.cc
 * \brief BackgroundReplace算子Tiling入口
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "background_replace_tiling.h"
#include <limits>

namespace optiling {

static constexpr uint64_t TILING_KEY_HALF_C1 = 1;
static constexpr uint64_t TILING_KEY_UINT8_C1 = 2;
static constexpr uint64_t TILING_KEY_HALF_C3 = 3;
static constexpr uint64_t TILING_KEY_UINT8_C3 = 4;

static constexpr uint32_t ASCEND_310P_BLOCK_DIM = 8;
static constexpr size_t MIN_HW_RANK = 2;
static constexpr size_t HWC_RANK = 3;
static constexpr size_t CHANNEL_DIM = 2; // HWC layout: dim0=H, dim1=W, dim2=C
static constexpr int64_t CHANNEL_C1 = 1;
static constexpr int64_t CHANNEL_C3 = 3;

static int64_t GetChannel(const gert::Shape& shape)
{
    return shape.GetDimNum() < HWC_RANK ? CHANNEL_C1 : shape.GetDim(CHANNEL_DIM);
}

static bool IsShapeContractValid(const gert::Shape& bkgShape, const gert::Shape& srcShape, const gert::Shape& maskShape,
                                 const gert::Shape& outShape)
{
    if (bkgShape.GetDimNum() < MIN_HW_RANK || srcShape.GetDimNum() < MIN_HW_RANK ||
        maskShape.GetDimNum() < MIN_HW_RANK || outShape.GetDimNum() < MIN_HW_RANK) {
        return false;
    }
    if (bkgShape.GetDim(0) != srcShape.GetDim(0) || bkgShape.GetDim(1) != srcShape.GetDim(1) ||
        bkgShape.GetDim(0) != maskShape.GetDim(0) || bkgShape.GetDim(1) != maskShape.GetDim(1) ||
        bkgShape.GetDim(0) != outShape.GetDim(0) || bkgShape.GetDim(1) != outShape.GetDim(1)) {
        return false;
    }
    const int64_t bkgChannel = GetChannel(bkgShape);
    if ((bkgChannel != CHANNEL_C1 && bkgChannel != CHANNEL_C3) || GetChannel(srcShape) != bkgChannel ||
        GetChannel(outShape) != bkgChannel) {
        return false;
    }
    return GetChannel(maskShape) == CHANNEL_C1;
}

static uint64_t ResolveBackgroundReplaceTilingKey(int64_t maskSize, int64_t bkgSize, ge::DataType bkgDataType)
{
    uint64_t tiling_key = 0;
    if (maskSize == bkgSize && bkgDataType == ge::DT_FLOAT16) {
        tiling_key = TILING_KEY_HALF_C1;
    } else if (maskSize == bkgSize && bkgDataType == ge::DT_UINT8) {
        tiling_key = TILING_KEY_UINT8_C1;
    } else if (maskSize != bkgSize && bkgDataType == ge::DT_FLOAT16) {
        tiling_key = TILING_KEY_HALF_C3;
    } else if (maskSize != bkgSize && bkgDataType == ge::DT_UINT8) {
        tiling_key = TILING_KEY_UINT8_C3;
    }
    return tiling_key;
}

static ge::graphStatus TilingBackgroundReplace(gert::TilingContext* context)
{
    TilingDataBackgroundReplace tiling;
    auto tensorBkg = context->GetInputTensor(0);
    auto tensorSrc = context->GetInputTensor(1);
    auto tensorMask = context->GetInputTensor(2);
    auto outDesc = context->GetOutputDesc(0);
    if (tensorBkg == nullptr || tensorSrc == nullptr || tensorMask == nullptr || outDesc == nullptr) {
        OP_LOGE("BackgroundReplace", "bkg, src, mask or out tensor is null.");
        return ge::GRAPH_FAILED;
    }
    auto bkgShape = context->GetInputShape(0);
    auto srcShape = context->GetInputShape(1);
    auto maskShape = context->GetInputShape(2);
    auto outShape = context->GetOutputShape(0);
    if (bkgShape == nullptr || srcShape == nullptr || maskShape == nullptr || outShape == nullptr ||
        !IsShapeContractValid(bkgShape->GetStorageShape(), srcShape->GetStorageShape(), maskShape->GetStorageShape(),
                              outShape->GetStorageShape())) {
        OP_LOGE("BackgroundReplace", "bkg, src, mask or out shape is invalid.");
        return ge::GRAPH_FAILED;
    }
    int64_t maskSize = tensorMask->GetShapeSize();
    int64_t bkgSize = tensorBkg->GetShapeSize();
    if (maskSize < 0 || bkgSize < 0 || maskSize > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE("BackgroundReplace", "mask or background shape size is invalid.");
        return ge::GRAPH_FAILED;
    }
    auto bkgDataType = tensorBkg->GetDataType();
    if (tensorSrc->GetDataType() != bkgDataType || outDesc->GetDataType() != bkgDataType) {
        OP_LOGE("BackgroundReplace", "bkg, src and out dtype must be the same.");
        return ge::GRAPH_FAILED;
    }
    const uint64_t tiling_key = ResolveBackgroundReplaceTilingKey(maskSize, bkgSize, bkgDataType);
    tiling.set_size(static_cast<uint32_t>(maskSize));
    context->SetTilingKey(tiling_key);
    context->SetBlockDim(ASCEND_310P_BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForBackgroundReplace(gert::TilingParseContext* context)
{
    OP_LOGD("BackgroundReplace", "TilingPrepareForBackgroundReplace start.");
    return ge::GRAPH_SUCCESS;
}
struct BackgroundReplaceCompileInfo {};
IMPL_OP_OPTILING(BackgroundReplace)
    .Tiling(TilingBackgroundReplace)
    .TilingParse<BackgroundReplaceCompileInfo>(TilingPrepareForBackgroundReplace);
} // namespace optiling
