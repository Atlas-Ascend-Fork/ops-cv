/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file yolo_tiling.cpp
 * \brief Tiling implementation for yolo operator
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "objdetect/yolo/op_kernel/arch35/yolo_tiling_data.h"
#include "objdetect/yolo/op_kernel/arch35/yolo_tiling_key.h"
#include <cstring>
#include <string>

namespace optiling {

constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

struct YoloCompileInfo {
    std::string _pattern;
};

static int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    const char* opName = context->GetNodeName();
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, "tiling", "coreNum", "0", "coreNum is 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, "tiling", "ubSize", "0", "ubSize is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Extract and validate input shape and attributes
static ge::graphStatus ExtractAndValidate(gert::TilingContext* context, int64_t& N, int64_t& HW, int64_t& boxes,
                                          int64_t& classes)
{
    const char* opName = context->GetNodeName();
    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(
        xShape->GetStorageShape().GetDimNum() != 4,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "x", std::to_string(xShape->GetStorageShape().GetDimNum()).c_str(), "input dim num must be 4"),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* boxesPtr = attrs->GetInt(0);
    const int64_t* coordsPtr = attrs->GetInt(1);
    const int64_t* classesPtr = attrs->GetInt(2);
    boxes = (boxesPtr != nullptr) ? *boxesPtr : 3;
    int64_t coords = (coordsPtr != nullptr) ? *coordsPtr : 4;
    classes = (classesPtr != nullptr) ? *classesPtr : 80;

    OP_CHECK_IF(coords != 4, OP_LOGE_FOR_INVALID_VALUE(opName, "coords", std::to_string(coords).c_str(), "4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        boxes <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "boxes", std::to_string(boxes).c_str(), "boxes must be > 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(classes <= 0 || classes > 1024,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "classes", std::to_string(classes).c_str(),
                                                      "classes must be in [1, 1024]"),
                return ge::GRAPH_FAILED);

    N = xShape->GetStorageShape().GetDim(0);
    int64_t C = xShape->GetStorageShape().GetDim(1);
    int64_t H = xShape->GetStorageShape().GetDim(2);
    int64_t W = xShape->GetStorageShape().GetDim(3);
    OP_CHECK_IF(N <= 0, OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "N", std::to_string(N).c_str(), "N must be >= 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H <= 0 || W <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "H, W", (std::to_string(H) + ", " + std::to_string(W)).c_str(), "H and W must be > 0"),
                return ge::GRAPH_FAILED);
    int64_t expectedC = boxes * (coords + 1 + classes);
    OP_CHECK_IF(C != expectedC,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "C", std::to_string(C).c_str(),
                    ("expected boxes*(coords+1+classes)=" + std::to_string(expectedC)).c_str()),
                return ge::GRAPH_FAILED);
    HW = H * W;
    return ge::GRAPH_SUCCESS;
}

// Validate input and output dtypes: all tensors must use the same float16 or float32 dtype.
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName();
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDataType = xDesc->GetDataType();
    OP_CHECK_IF(
        xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "x", std::to_string(static_cast<int32_t>(xDataType)).c_str(),
                                               "x dtype must be float16/float32"),
        return ge::GRAPH_FAILED);

    auto coordDataDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, coordDataDesc);
    const ge::DataType coordDataType = coordDataDesc->GetDataType();
    OP_CHECK_IF(coordDataType != ge::DT_FLOAT16 && coordDataType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "coord_data",
                                                       std::to_string(static_cast<int32_t>(coordDataType)).c_str(),
                                                       "coord_data dtype must be float16/float32"),
                return ge::GRAPH_FAILED);

    auto objProbDesc = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, objProbDesc);
    const ge::DataType objProbDataType = objProbDesc->GetDataType();
    OP_CHECK_IF(objProbDataType != ge::DT_FLOAT16 && objProbDataType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "obj_prob",
                                                       std::to_string(static_cast<int32_t>(objProbDataType)).c_str(),
                                                       "obj_prob dtype must be float16/float32"),
                return ge::GRAPH_FAILED);

    auto classesProbDesc = context->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, classesProbDesc);
    const ge::DataType classesProbDataType = classesProbDesc->GetDataType();
    OP_CHECK_IF(classesProbDataType != ge::DT_FLOAT16 && classesProbDataType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName, "classes_prob", std::to_string(static_cast<int32_t>(classesProbDataType)).c_str(),
                    "classes_prob dtype must be float16/float32"),
                return ge::GRAPH_FAILED);

    const std::string actualDtypes = std::to_string(static_cast<int32_t>(xDataType)) + ", " +
                                     std::to_string(static_cast<int32_t>(coordDataType)) + ", " +
                                     std::to_string(static_cast<int32_t>(objProbDataType)) + ", " +
                                     std::to_string(static_cast<int32_t>(classesProbDataType));
    OP_CHECK_IF(
        coordDataType != xDataType || objProbDataType != xDataType || classesProbDataType != xDataType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "x, coord_data, obj_prob, classes_prob", actualDtypes.c_str(),
                                               "x and all outputs must have the same dtype"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Validate input and output storage formats: must be ND (kernel reads/writes linear ND layout)
static ge::graphStatus ValidateFormat(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName();
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format xFormat = xDesc->GetStorageFormat();
    OP_CHECK_IF(xFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                    opName, "x", std::to_string(static_cast<int32_t>(xFormat)).c_str(), "x format must be ND"),
                return ge::GRAPH_FAILED);

    auto coordDataDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, coordDataDesc);
    const ge::Format coordDataFormat = coordDataDesc->GetStorageFormat();
    OP_CHECK_IF(coordDataFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, "coord_data",
                                                        std::to_string(static_cast<int32_t>(coordDataFormat)).c_str(),
                                                        "coord_data format must be ND"),
                return ge::GRAPH_FAILED);

    auto objProbDesc = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, objProbDesc);
    const ge::Format objProbFormat = objProbDesc->GetStorageFormat();
    OP_CHECK_IF(objProbFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, "obj_prob",
                                                        std::to_string(static_cast<int32_t>(objProbFormat)).c_str(),
                                                        "obj_prob format must be ND"),
                return ge::GRAPH_FAILED);

    auto classesProbDesc = context->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, classesProbDesc);
    const ge::Format classesProbFormat = classesProbDesc->GetStorageFormat();
    OP_CHECK_IF(classesProbFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, "classes_prob",
                                                        std::to_string(static_cast<int32_t>(classesProbFormat)).c_str(),
                                                        "classes_prob format must be ND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Determine yolo_mode from yolo_version, softmax, background attributes
static int32_t DetermineYoloMode(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* version = attrs->GetStr(3);
    const bool* softmaxPtr = attrs->GetBool(4);
    const bool* backgroundPtr = attrs->GetBool(5);
    bool softmax = (softmaxPtr != nullptr) ? *softmaxPtr : false;
    bool background = (backgroundPtr != nullptr) ? *backgroundPtr : false;

    if (version != nullptr && strcmp(version, "V3") == 0) {
        return YOLO_MODE_1;
    }
    if (version != nullptr && strcmp(version, "V2") == 0) {
        if (!softmax && !background)
            return YOLO_MODE_1;
        if (softmax && !background)
            return YOLO_MODE_2;
        if (!softmax && background)
            return YOLO_MODE_3;
        return YOLO_MODE_4;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "yolo_version", (version != nullptr) ? version : "null",
                                          "yolo_version must be \"V2\" or \"V3\"");
    return -1;
}

static ge::graphStatus YoloTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter TilingYolo");
    const char* opName = context->GetNodeName();
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateFormat failed"),
                return ge::GRAPH_FAILED);

    int64_t N = 0, HW = 0, boxes = 0, classes = 0;
    OP_CHECK_IF(ExtractAndValidate(context, N, HW, boxes, classes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ExtractAndValidate error"), return ge::GRAPH_FAILED);

    int32_t yoloMode = DetermineYoloMode(context);
    OP_CHECK_IF(yoloMode < 0, OP_LOGE(context, "DetermineYoloMode error"), return ge::GRAPH_FAILED);

    // Core splitting: two-step method
    int64_t totalPoints = N * boxes * HW;
    int64_t perCoreElements = CeilDiv(totalPoints, coreNum);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int64_t needCoreNum = CeilDiv(totalPoints, perCoreElements);

    // Compute CeilX-aligned output dimensions (must match yolo_infershape.cpp)
    // CeilX(size, 32) = (size + 31) / 32 * 32
    // ceilHW = CeilX(HW * 2 + 32, 32) / 2
    // ceilBoxesHw = CeilX(boxes * HW * 2 + 32, 32) / 2
    int64_t ceilHW = (HW * 2 + 32 + 31) / 32 * 32 / 2;
    int64_t ceilBoxesHw = (boxes * HW * 2 + 32 + 31) / 32 * 32 / 2;

    // Fill TilingData
    YoloTilingData* tiling = context->GetTilingData<YoloTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->N = static_cast<int32_t>(N);
    tiling->boxes = static_cast<int32_t>(boxes);
    tiling->classes = static_cast<int32_t>(classes);
    tiling->HW = HW;
    tiling->ceilHW = ceilHW;
    tiling->ceilBoxesHw = ceilBoxesHw;

    context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(yoloMode)));

    // Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    // LocalMemory (DCache + dynamic UB)
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE,
                OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, "tiling", "ubSize", std::to_string(ubSize).c_str(),
                                                       "ubSize too small"),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForYolo(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<YoloCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->_pattern = "ElemWise";
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Yolo).Tiling(YoloTilingFunc).TilingParse<YoloCompileInfo>(TilingParseForYolo);

} // namespace optiling
