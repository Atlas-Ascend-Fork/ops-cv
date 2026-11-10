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
 * \file mrgba_custom_tiling.cc
 * \brief
 */
#include "mrgba_custom_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr size_t RGB_RANK = 3;
constexpr size_t ALPHA_MIN_RANK = 2;
constexpr size_t CHANNEL_DIM = 2; // HWC: dim2 = C
constexpr size_t DIM_H = 0;
constexpr size_t DIM_W = 1;
constexpr int64_t RGB_CHANNEL = 3;

namespace {
bool IsMrgbaCustomShapeValid(const gert::Shape& rgbShape, const gert::Shape& alphaShape)
{
    if (rgbShape.GetDimNum() != RGB_RANK || rgbShape.GetDim(CHANNEL_DIM) != RGB_CHANNEL) {
        return false;
    }
    if (alphaShape.GetDimNum() < ALPHA_MIN_RANK) {
        return false;
    }
    return rgbShape.GetDim(DIM_H) == alphaShape.GetDim(DIM_H) && rgbShape.GetDim(DIM_W) == alphaShape.GetDim(DIM_W);
}
} // namespace

static ge::graphStatus TilingFuncForMrgbaCustom(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto rgbShape = context->GetInputShape(0);
    auto alphaShape = context->GetInputShape(1);
    if (rgbShape == nullptr || alphaShape == nullptr ||
        !IsMrgbaCustomShapeValid(rgbShape->GetStorageShape(), alphaShape->GetStorageShape())) {
        return ge::GRAPH_FAILED;
    }
    TilingDataMrgba tiling;
    auto tensorY = context->GetInputTensor(1);
    if (tensorY == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = tensorY->GetShapeSize();
    tiling.set_alphaLen(totalLength);

    context->SetBlockDim(BLOCK_DIM);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMrgbaCustom(gert::TilingParseContext* context)
{
    OP_LOGD("MrgbaCustom", "TilingPrepareForMrgbaCustom start.");
    return ge::GRAPH_SUCCESS;
}
struct MrgbaCustomCompileInfo {};
IMPL_OP_OPTILING(MrgbaCustom)
    .Tiling(TilingFuncForMrgbaCustom)
    .TilingParse<MrgbaCustomCompileInfo>(TilingPrepareForMrgbaCustom);
} // namespace optiling
