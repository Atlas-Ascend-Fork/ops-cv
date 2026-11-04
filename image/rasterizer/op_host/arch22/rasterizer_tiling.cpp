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

#include <string>
#include <vector>
#include "base/err_msg.h"
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
    const char* opName = context->GetNodeName() == nullptr ? "Rasterizer" : context->GetNodeName();
    const gert::StorageShape* findicesShape = context->GetOutputShape(IDX_0);
    const gert::StorageShape* baryShape = context->GetOutputShape(IDX_1);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName,
                "Runtime attributes are unavailable; check Tiling context construction and Rasterizer registration."}));
        return ge::GRAPH_FAILED;
    }
    const int64_t* width = attrs->GetAttrPointer<int64_t>(IDX_0);
    const int64_t* height = attrs->GetAttrPointer<int64_t>(IDX_1);
    const int64_t* useDepthPrior = attrs->GetAttrPointer<int64_t>(IDX_3);
    if (vShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "v");
        return ge::GRAPH_FAILED;
    }
    if (fShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "f");
        return ge::GRAPH_FAILED;
    }
    if (findicesShape == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName, "Output shape findices is unavailable; check Tiling output context construction."}));
        return ge::GRAPH_FAILED;
    }
    if (baryShape == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName, "Output shape barycentric is unavailable; check Tiling output context construction."}));
        return ge::GRAPH_FAILED;
    }

    auto vDimNum = vShape->GetStorageShape().GetDimNum();
    auto fDimNum = fShape->GetStorageShape().GetDimNum();
    auto findicesDimNum = findicesShape->GetStorageShape().GetDimNum();
    auto barycentricDimNum = baryShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(
        vDimNum != DIM_NUM2 || fDimNum != DIM_NUM2 || findicesDimNum != DIM_NUM2,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "v,f,findices",
            (std::to_string(vDimNum) + "," + std::to_string(fDimNum) + "," + std::to_string(findicesDimNum)).c_str(),
            "v, f and findices must each have rank 2; reshape the tensors accordingly."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(barycentricDimNum != DIM_NUM3,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "barycentric", std::to_string(barycentricDimNum).c_str(), "3"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        fShape->GetStorageShape().GetDim(IDX_1) != DIM_VAL3 || baryShape->GetStorageShape().GetDim(IDX_2) != DIM_VAL3,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName, "f,barycentric",
            (std::string("f.dim1=") + std::to_string(fShape->GetStorageShape().GetDim(IDX_1)) +
             ",barycentric.dim2=" + std::to_string(baryShape->GetStorageShape().GetDim(IDX_2)))
                .c_str(),
            "f dimension 1 and barycentric dimension 2 must both equal 3."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(vShape->GetStorageShape().GetDim(IDX_1) != DIM_VAL4,
                OP_LOGE_FOR_INVALID_SHAPESIZE(opName, "v",
                                              std::to_string(vShape->GetStorageShape().GetDim(IDX_1)).c_str(), "4"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(vShape->GetStorageShape().GetDim(IDX_0) < MIN_SHAPE_VALUE ||
                    fShape->GetStorageShape().GetDim(IDX_0) < MIN_SHAPE_VALUE,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    opName, "v,f",
                    (std::string("v.dim0=") + std::to_string(vShape->GetStorageShape().GetDim(IDX_0)) +
                     ",f.dim0=" + std::to_string(fShape->GetStorageShape().GetDim(IDX_0)))
                        .c_str(),
                    "The vertex and face counts must both be greater than zero."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *height > MAX_SHAPE_VALUE || *width > MAX_SHAPE_VALUE || *height < MIN_SHAPE_VALUE || *width < MIN_SHAPE_VALUE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "height,width",
                                               (std::to_string(*height) + "," + std::to_string(*width)).c_str(),
                                               "height and width must each be within [1,4096]."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(findicesShape->GetStorageShape().GetDim(IDX_0) != baryShape->GetStorageShape().GetDim(IDX_0) ||
                    findicesShape->GetStorageShape().GetDim(IDX_1) != baryShape->GetStorageShape().GetDim(IDX_1) ||
                    findicesShape->GetStorageShape().GetDim(IDX_0) != *height ||
                    findicesShape->GetStorageShape().GetDim(IDX_1) != *width,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    opName, "findices,barycentric,height,width",
                    (std::string("findices=[") + std::to_string(findicesShape->GetStorageShape().GetDim(IDX_0)) + "," +
                     std::to_string(findicesShape->GetStorageShape().GetDim(IDX_1)) + "],barycentric=[" +
                     std::to_string(baryShape->GetStorageShape().GetDim(IDX_0)) + "," +
                     std::to_string(baryShape->GetStorageShape().GetDim(IDX_1)) +
                     "],height=" + std::to_string(*height) + ",width=" + std::to_string(*width))
                        .c_str(),
                    "Dimensions 0 and 1 of both outputs must equal height and width, respectively."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*useDepthPrior != 0,
                OP_LOGE_FOR_INVALID_VALUE(opName, "use_depth_prior", std::to_string(*useDepthPrior).c_str(), "0"),
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

    OP_CHECK_IF(CheckParam(context, vShape, fShape) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "CheckParam failed; retain its first error."), return ge::GRAPH_FAILED);

    FillTilingData(context, vShape, fShape);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRasterizer([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Rasterizer).Tiling(RasterizerTilingFunc).TilingParse<RasterizerCompileInfo>(TilingParseForRasterizer);
} // namespace optiling
