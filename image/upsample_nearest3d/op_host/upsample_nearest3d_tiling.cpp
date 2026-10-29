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
 * \file upsample_nearest3d_tiling.cpp
 * \brief
 */
#include <iostream>
#include <string>
#include "register/tilingdata_base.h"
#include "upsample_nearest3d_tiling_common.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_util.h"
#include "upsample_nearest3d_tiling.h"
#include "op_api/error_report_util.h"

using namespace ge;
using namespace UpsampleNearest3d;

namespace optiling {
constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;
constexpr uint8_t SCHEDULE_MODE = 1;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
const std::string EXACT_3D_TYPE = "UpsampleNearestExact3d";

class UpsampleNearest3dTiling {
public:
    explicit UpsampleNearest3dTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init() const;
    ge::graphStatus RunBigKernelTiling(gert::TilingContext* context);

private:
    inline bool CheckMaxSizes(const gert::TilingContext* context);
    void GetTilingKey() const;
    void getWorkSpace();

private:
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    const gert::ContinuousVector* outputSize = nullptr;
    const float* scaleD = nullptr;
    const float* scaleH = nullptr;
    const float* scaleW = nullptr;

    int64_t outputShapes[3] = {0};
    int64_t inputShapes[3] = {0};
};

ge::graphStatus UpsampleNearest3dTiling::Init() const { return ge::GRAPH_SUCCESS; }

ge::graphStatus UpsampleNearest3dTiling::RunBigKernelTiling(gert::TilingContext* context)
{
    bool regBase = Ops::Cv::OpTiling::IsRegbaseSocVersion(context);
    std::string opType(tilingContext->GetNodeType());
    if (regBase) {
        OP_LOGI(tilingContext->GetNodeName(), "enter Tiling4UpsampleNearest3dRegbase");
        return Tiling4UpsampleNearest3dRegbase(tilingContext);
    }
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t idx = 0;
    outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    scaleD = attrs->GetAttrPointer<float>(idx++);
    scaleH = attrs->GetAttrPointer<float>(idx++);
    scaleW = attrs->GetAttrPointer<float>(idx++);
    OP_CHECK_IF(outputSize == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "outputSize is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);

    size_t outputSizeNum = outputSize->GetSize();
    if (outputSizeNum != DIM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(tilingContext->GetNodeName(), "output_size",
                                      std::to_string(outputSizeNum).c_str(), std::to_string(DIM).c_str());
        return ge::GRAPH_FAILED;
    }

    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();

    const int64_t* outputSizeArray = reinterpret_cast<const int64_t*>(outputSize->GetData());
    for (int8_t i = 0; i < DIM; i++) {
        inputShapes[i] = inputShape.GetDim(i + BATCH_DIM);
        outputShapes[i] = outputSizeArray[i];
    }
    if (!CheckMaxSizes(tilingContext)) {
        return ge::GRAPH_FAILED;
    }

    float scales[3] = {*scaleD, *scaleH, *scaleW};
    auto compileInfo = reinterpret_cast<const UpsampleNearest3dCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "compileInfo is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    UpsampleNearest3d::UpsampleNearest3dTilingData*
        tilingData = tilingContext->GetTilingData<UpsampleNearest3d::UpsampleNearest3dTilingData>();
    UpsampleNearest3d::UpsampleNearest3dTiling::UpsampleNearest3dCommonTiling<gert::Shape>(
        srcShape->GetStorageShape(), scales, outputShapes, *tilingData, compileInfo->coreNum);

    GetTilingKey();
    tilingContext->SetBlockDim(tilingData->needCoreNum);
    getWorkSpace();
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dTiling::GetTilingKey() const
{
    uint32_t D_T_X = UPSAMPLE_NEAREST3D_TPL_FP32;
    uint32_t D_T_Y = UPSAMPLE_NEAREST3D_TPL_FP32;
    ge::DataType dtype_x = tilingContext->GetInputDesc(0)->GetDataType();
    if (dtype_x == ge::DataType::DT_FLOAT) {
        D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP32);
        D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP32);
    } else if (dtype_x == ge::DataType::DT_FLOAT16) {
        D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP16);
        D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP16);
    } else if (dtype_x == ge::DataType::DT_BF16) {
        D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_BF16);
        D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_BF16);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y);
    tilingContext->SetTilingKey(tilingKey);
}

void UpsampleNearest3dTiling::getWorkSpace()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        workspaces[0] = 0;
    } else {
        workspaces[0] = WORK_SPACE_SIZE;
    }
}

inline bool UpsampleNearest3dTiling::CheckMaxSizes(const gert::TilingContext* context)
{
    if (inputShape.GetDim(0) > INT32_MAX) {
        std::string reasonMsg = "The N-dimension of x must be less than or equal to INT32_MAX, where N is the 0th axis";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
                                              reasonMsg.c_str());
        return false;
    }
    if (inputShape.GetDim(1) > INT32_MAX) {
        std::string reasonMsg = "The C-dimension of x must be less than or equal to INT32_MAX, where C is the 1st axis";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
                                              reasonMsg.c_str());
        return false;
    }
    if (inputShapes[0] > INT32_MAX) {
        std::string reasonMsg = "The D-dimension of x must be less than or equal to INT32_MAX, where D is the 2nd axis";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
                                              reasonMsg.c_str());
        return false;
    }
    if (inputShapes[1] > INT32_MAX) {
        std::string reasonMsg = "The H-dimension of x must be less than or equal to INT32_MAX, where H is the 3rd axis";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
                                              reasonMsg.c_str());
        return false;
    }
    if (inputShapes[2] > INT32_MAX) {
        std::string
            reasonMsg = "The W-dimension of x must be less than or equal to INT32_MAX, where W is the last axis";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
                                              reasonMsg.c_str());
        return false;
    }

    if (outputShapes[0] > INT32_MAX || outputShapes[1] > INT32_MAX || outputShapes[2] > INT32_MAX) {
        std::string valueMsg = "(" + std::to_string(outputShapes[0]) + ", " + std::to_string(outputShapes[1]) + ", " +
                               std::to_string(outputShapes[2]) + ")";
        std::string reasonMsg = "Each value of output_size must be less than or equal to INT32_MAX";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size", valueMsg.c_str(),
                                              reasonMsg.c_str());
        return false;
    }
    return true;
}

static ge::graphStatus Tiling4UpsampleNearest3dTiling(gert::TilingContext* context)
{
    UpsampleNearest3dTiling tilingObject(context);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    platform_ascendc::SocVersion nearest3dSocVersion = ascendc_platform.GetSocVersion();
    if (nearest3dSocVersion == platform_ascendc::SocVersion::ASCEND310P) {
        context->SetScheduleMode(SCHEDULE_MODE);
    }
    return tilingObject.RunBigKernelTiling(context);
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleNearest3dCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "compileInfo is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
                error_report::ReportTilingStageError(context->GetNodeName(),
                                                     "failed to get the hardware info from the platform info"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearest3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

IMPL_OP_OPTILING(UpsampleNearestExact3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

} // namespace optiling
