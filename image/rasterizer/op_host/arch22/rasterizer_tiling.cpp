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
 * \file rasterizer_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../rasterizer_tiling.h"

namespace optiling {

static constexpr int64_t DIM_NUM2 = 2;
static constexpr int64_t DIM_NUM3 = 3;

static constexpr int64_t DIM_VAL3 = 3;
static constexpr int64_t DIM_VAL4 = 4;
static constexpr size_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
static constexpr size_t MAX_PROC_ELENUM = 1920;
static constexpr size_t RSV = 64;
static constexpr size_t BUFFER_NUM = 2;

static constexpr uint32_t IDX_0 = 0;
static constexpr uint32_t IDX_1 = 1;
static constexpr uint32_t IDX_2 = 2;
static constexpr uint32_t IDX_3 = 3;

static constexpr int64_t MAX_SHAPE_VALUE = 4096;
static constexpr int64_t MIN_SHAPE_VALUE = 1;

static ge::graphStatus CheckParam(gert::TilingContext* context, const gert::StorageShape* vShape,
                                  const gert::StorageShape* fShape)
{
    const gert::StorageShape* findicesShape = context->GetOutputShape(IDX_0);
    const gert::StorageShape* baryShape = context->GetOutputShape(IDX_1);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* width = attrs->GetAttrPointer<int64_t>(IDX_0);
    const int64_t* height = attrs->GetAttrPointer<int64_t>(IDX_1);
    const int64_t* useDepthPrior = attrs->GetAttrPointer<int64_t>(IDX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, fShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, findicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, baryShape);

    auto vDimNum = vShape->GetStorageShape().GetDimNum();
    auto fDimNum = fShape->GetStorageShape().GetDimNum();
    auto findicesDimNum = findicesShape->GetStorageShape().GetDimNum();
    auto barycentricDimNum = baryShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(vDimNum != DIM_NUM2 || fDimNum != DIM_NUM2 || findicesDimNum != DIM_NUM2,
                OP_LOGE(context, "v/f/findices dim num is not 2, please check"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(barycentricDimNum != DIM_NUM3, OP_LOGE(context, "barycentric dim num is not 3, please check"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        fShape->GetStorageShape().GetDim(IDX_1) != DIM_VAL3 || baryShape->GetStorageShape().GetDim(IDX_2) != DIM_VAL3,
        OP_LOGE(context, "dim1 of f and dim2 of barycentric should be 3, please check"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(vShape->GetStorageShape().GetDim(IDX_1) != DIM_VAL4,
                OP_LOGE(context, "dim1 of v should be 4, please check"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(vShape->GetStorageShape().GetDim(IDX_0) < MIN_SHAPE_VALUE ||
                    fShape->GetStorageShape().GetDim(IDX_0) < MIN_SHAPE_VALUE,
                OP_LOGE(context, "numVertices and numFaces should be greater than 0, please check"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *height > MAX_SHAPE_VALUE || *width > MAX_SHAPE_VALUE || *height < MIN_SHAPE_VALUE || *width < MIN_SHAPE_VALUE,
        OP_LOGE(context, "height/width should be no greater than 4096 and greater than 0, please check"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        findicesShape->GetStorageShape().GetDim(IDX_0) != baryShape->GetStorageShape().GetDim(IDX_0) ||
            findicesShape->GetStorageShape().GetDim(IDX_1) != baryShape->GetStorageShape().GetDim(IDX_1) ||
            findicesShape->GetStorageShape().GetDim(IDX_0) != *height ||
            findicesShape->GetStorageShape().GetDim(IDX_1) != *width,
        OP_LOGE(context, "dim0 and dim1 of findices/barycentric should be equal to height and width, please check"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*useDepthPrior != 0, OP_LOGE(context, "useDepthPrior should be 0, please check"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FillTilingData(gert::TilingContext* context, const gert::StorageShape* vShape, const gert::StorageShape* fShape)
{
    uint32_t numFaces = fShape->GetStorageShape().GetDim(IDX_0);
    uint32_t numVertices = vShape->GetStorageShape().GetDim(IDX_0);

    auto attrs = context->GetAttrs();
    const int64_t* width = attrs->GetAttrPointer<int64_t>(IDX_0);
    const int64_t* height = attrs->GetAttrPointer<int64_t>(IDX_1);
    const float* occlusionTruncation = attrs->GetAttrPointer<float>(IDX_2);
    const int64_t* useDepthPrior = attrs->GetAttrPointer<int64_t>(IDX_3);

    RasterizerTilingData tiling;

    tiling.set_numFaces(numFaces);
    tiling.set_numVertices(numVertices);
    tiling.set_height(static_cast<uint32_t>(*height));
    tiling.set_width(static_cast<uint32_t>(*width));
    tiling.set_occlusionTruncation(*occlusionTruncation);
    tiling.set_useDepthPrior(static_cast<uint32_t>(*useDepthPrior));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aivCoreNum = ascendcPlatform.GetCoreNumAiv();

    size_t* workSpaceSize = context->GetWorkspaceSizes(1);
    workSpaceSize[0] = static_cast<size_t>(*height) * static_cast<size_t>(*width) * (sizeof(int32_t) + sizeof(float)) *
                           aivCoreNum +
                       DIM_VAL3 * MAX_PROC_ELENUM * sizeof(uint32_t) + BUFFER_NUM * RSV * sizeof(uint32_t) +
                       WORK_SPACE_SIZE;

    context->SetTilingKey(1);
    context->SetBlockDim(aivCoreNum);
}

static ge::graphStatus RasterizerTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context, "Enter in RasterizerTilingFunc");

    const gert::StorageShape* vShape = context->GetInputShape(IDX_0);
    const gert::StorageShape* fShape = context->GetInputShape(IDX_1);

    OP_CHECK_IF(CheckParam(context, vShape, fShape) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckInputShapes failed"),
                return ge::GRAPH_FAILED);

    FillTilingData(context, vShape, fShape);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRasterizer([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Rasterizer).Tiling(RasterizerTilingFunc).TilingParse<RasterizerCompileInfo>(TilingParseForRasterizer);
} // namespace optiling
