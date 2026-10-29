/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_nearest3d_grad_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "upsample_nearest3d_grad_tiling.h"
#include "image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_tiling_data.h"
#include "image/upsample_nearest3d_grad/op_kernel/arch35/upsample_nearest3d_grad_tiling_key.h"
#include "op_api/error_report_util.h"

namespace {
constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int64_t INPUT_DIMS = 5;
constexpr int32_t CACHE_LINE = 128;
constexpr int64_t THREAD_NUM = 2048; // simt开的线程数
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr const float EPSILON = 1e-8f;

struct BaseTilingData {
    int64_t dimN = 0;
    int64_t dimC = 0;
    int64_t inD = 0;
    int64_t outD = 0;
    int64_t inH = 0;
    int64_t outH = 0;
    int64_t inW = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t blkProcessNum = 0;
    float scaleD = 0.0;
    float scaleH = 0.0;
    float scaleW = 0.0;
    int32_t isUint32 = 1;
    int32_t ubSize = 0;
    int32_t coreNum = 0;
    int32_t schId = 0;
    int32_t realCoreNum = 0;
    int32_t tailBlockNum = 0;
    int32_t dtypeSize = 0;
    int32_t ubFactor = 0;
    int32_t cacheLineNum = 0;
    int32_t oneBlockNum = 0;
};

static const std::map<ge::DataType, int32_t> inputDtypeList = {
    {ge::DT_DOUBLE, 8}, {ge::DT_UINT8, 1}, {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};

} // namespace

namespace optiling {
using namespace Ops::Cv::OpTiling;

class UpsampleNearest3dGradRegbaseTiling {
public:
    explicit UpsampleNearest3dGradRegbaseTiling(gert::TilingContext* context) : tilingContext(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScales(float scaleD, float scaleH, float scaleW);
    void ComputeDataCopy();
    void CalTilingData();
    void FillTilingData();
    void PrintTilingData();

private:
    BaseTilingData baseTiling;
    gert::TilingContext* tilingContext = nullptr;
    UpsampleNearest3dGradRegBaseTilingData* tilingData{nullptr};
};
void UpsampleNearest3dGradRegbaseTiling::ComputeDataCopy()
{
    int64_t coreNum = static_cast<int64_t>(baseTiling.coreNum);
    if (baseTiling.outSize <= static_cast<int64_t>(baseTiling.cacheLineNum * baseTiling.coreNum)) {
        baseTiling.realCoreNum = static_cast<int32_t>(baseTiling.outSize) / baseTiling.cacheLineNum;
    }
    if (baseTiling.realCoreNum == 0) {
        baseTiling.realCoreNum = 1;
    }
    baseTiling.blkProcessNum = baseTiling.outSize / static_cast<int64_t>(baseTiling.realCoreNum);
    baseTiling.tailBlockNum = static_cast<int32_t>(baseTiling.outSize % static_cast<int64_t>(baseTiling.realCoreNum));
    baseTiling.ubFactor = (baseTiling.ubSize - baseTiling.oneBlockNum * baseTiling.dtypeSize) /
                          (CONST_2 * baseTiling.dtypeSize);
    baseTiling.ubFactor = (baseTiling.ubFactor / baseTiling.oneBlockNum) * baseTiling.oneBlockNum;
    return;
}

void UpsampleNearest3dGradRegbaseTiling::CalTilingData()
{
    int64_t outDHW = baseTiling.outD * baseTiling.outH * baseTiling.outW;
    int64_t outCDHW = baseTiling.dimC * outDHW;
    int64_t outNCDHW = baseTiling.dimN * outCDHW;
    int64_t maxNum = static_cast<int64_t>(baseTiling.coreNum) * THREAD_NUM;
    bool isDataCopy = baseTiling.outD == baseTiling.inD && baseTiling.outH == baseTiling.inH &&
                      baseTiling.outW == baseTiling.inW && std::abs(baseTiling.scaleD - 1.0f) <= EPSILON &&
                      std::abs(baseTiling.scaleH - 1.0f) <= EPSILON && std::abs(baseTiling.scaleW - 1.0f) <= EPSILON;
    baseTiling.realCoreNum = baseTiling.coreNum;
    baseTiling.cacheLineNum = CACHE_LINE / baseTiling.dtypeSize;

    if (isDataCopy) {
        OP_LOGI(tilingContext, "enter datacopy");
        baseTiling.schId = 0; // 纯copy模板
        baseTiling.isUint32 = 0;
        ComputeDataCopy();
    } else {
        int64_t allNum = 0;
        if (outDHW >= maxNum) {
            OP_LOGI(tilingContext, "enter simt dhw");
            baseTiling.schId = 1; // dhw 分线程， nc拉成一维
            baseTiling.dimN = baseTiling.dimN * baseTiling.dimC;
            allNum = outDHW;
        } else if (outCDHW >= maxNum) {
            OP_LOGI(tilingContext, "enter simt cdhw");
            baseTiling.schId = CONST_2; // cdhw 分线程, 和竞品一致
            allNum = outCDHW;
        } else {
            OP_LOGI(tilingContext, "enter simt ncdhw");
            int64_t coreNum = static_cast<int64_t>(baseTiling.coreNum);
            if (outNCDHW < coreNum) {
                baseTiling.realCoreNum = static_cast<int32_t>(outNCDHW);
            }
            allNum = outNCDHW;
            baseTiling.schId = CONST_3; // ncdhw 分线程
        }
        baseTiling.blkProcessNum = allNum / static_cast<int64_t>(baseTiling.realCoreNum);
        baseTiling.tailBlockNum = static_cast<int32_t>(allNum % static_cast<int64_t>(baseTiling.realCoreNum));
    }
    return;
}

ge::graphStatus UpsampleNearest3dGradRegbaseTiling::CheckInputParams()
{
    auto input = tilingContext->GetInputDesc(CONST_0);
    OP_CHECK_IF(
        input == nullptr,
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "input",
                                            "the input desc/shape of required input input is missing in the tiling "
                                            "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto inputDtype = input->GetDataType();
    OP_CHECK_IF(
        inputDtypeList.count(inputDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(inputDtype).c_str(),
                                              "input dtype is not supported"),
        return ge::GRAPH_FAILED);
    baseTiling.dtypeSize = inputDtypeList.find(inputDtype)->second;
    int32_t ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(tilingContext));
    baseTiling.oneBlockNum = ubBlockSize / baseTiling.dtypeSize;
    auto outDescPtr0 = tilingContext->GetOutputDesc(0);
    OP_CHECK_IF(outDescPtr0 == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "outDescPtr0 (output index 0) is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto outDtype = outDescPtr0->GetDataType();
    OP_CHECK_IF(
        outDtype != inputDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(outDtype).c_str(),
                                              "input and output dtype must be the same"),
        return ge::GRAPH_FAILED);
    auto inputX = tilingContext->GetInputShape(0);
    auto outY = tilingContext->GetOutputShape(0);
    OP_CHECK_IF(
        inputX == nullptr,
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "inputX",
                                            "the input desc/shape of required input inputX (index 0) is missing in the "
                                            "tiling context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outY == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "outY (output index 0) is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outShape.GetDimNum() != INPUT_DIMS),
                error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "dim",
                                                    "the dim of input0 or output0 should be equal to 5"),
                return ge::GRAPH_FAILED);
    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outputSize = outShape.GetShapeSize();
    baseTiling.dimN = inputShape.GetDim(CONST_0);
    baseTiling.dimC = inputShape.GetDim(CONST_1);
    baseTiling.inD = inputShape.GetDim(CONST_2);
    baseTiling.inH = inputShape.GetDim(CONST_3);
    baseTiling.inW = inputShape.GetDim(CONST_4);
    baseTiling.outD = outShape.GetDim(CONST_2);
    baseTiling.outH = outShape.GetDim(CONST_3);
    baseTiling.outW = outShape.GetDim(CONST_4);
    baseTiling.outSize = outputSize;
    OP_CHECK_IF(inputSize == 0 || outputSize == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    tilingContext->GetNodeName(), "empty",
                    (std::to_string(inputSize) + ", " + std::to_string(outputSize)).c_str(),
                    "empty input or output is not supported"),
                return ge::GRAPH_FAILED);
    int64_t uint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    int32_t isUint32 = static_cast<int32_t>((inputSize <= uint32Max) && (outputSize <= uint32Max));
    baseTiling.isUint32 = isUint32;
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dGradRegbaseTiling::ComputeScales(float scaleD, float scaleH, float scaleW)
{
    baseTiling.scaleD = static_cast<float>(baseTiling.inD) / static_cast<float>(baseTiling.outD);
    baseTiling.scaleH = static_cast<float>(baseTiling.inH) / static_cast<float>(baseTiling.outH);
    baseTiling.scaleW = static_cast<float>(baseTiling.inW) / static_cast<float>(baseTiling.outW);
    if (scaleD > 0.0f) {
        baseTiling.scaleD = scaleD;
    }
    if (scaleH > 0.0f) {
        baseTiling.scaleH = scaleH;
    }
    if (scaleW > 0.0f) {
        baseTiling.scaleW = scaleW;
    }
    return;
}

ge::graphStatus UpsampleNearest3dGradRegbaseTiling::CheckInputShapeAndAttr()
{
    auto outDescPtr0 = tilingContext->GetOutputShape(0);
    auto outShape = outDescPtr0->GetStorageShape();
    int64_t outN = outShape.GetDim(CONST_0);
    int64_t outC = outShape.GetDim(CONST_1);
    OP_CHECK_IF(
        (outN != baseTiling.dimN) || (outC != baseTiling.dimC),
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "input",
                                            "the N and C dimensions of the input and output need to be the same"),
        return ge::GRAPH_FAILED);
    auto* attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            tilingContext->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    auto gradInputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_0);
    OP_CHECK_IF(gradInputSize == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "gradInputSize is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);

    auto gradOutputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_1);
    OP_CHECK_IF(gradOutputSize == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "gradOutputSize is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);

    auto scales = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_2);
    OP_CHECK_IF(
        scales == nullptr,
        error_report::ReportTilingStageError(
            tilingContext->GetNodeName(), "scales is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const float* scalesArray = reinterpret_cast<const float*>(scales->GetData());
    float scaleD = scalesArray[CONST_0];
    float scaleH = scalesArray[CONST_1];
    float scaleW = scalesArray[CONST_2];

    OP_LOGI(tilingContext, "scaleD %f, scaleH %f, scaleW %f", scaleD, scaleH, scaleW);
    int64_t outSizeNum = gradInputSize->GetSize();
    OP_CHECK_IF(outSizeNum > 0 && (outSizeNum != INPUT_DIMS),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input",
                                                          std::to_string(outSizeNum).c_str(),
                                                          "the num of gradInputSize is , invalid, must be 3"),
                return ge::GRAPH_FAILED);
    const int64_t* outData = reinterpret_cast<const int64_t*>(gradInputSize->GetData());
    int64_t outD = baseTiling.outD;
    int64_t outH = baseTiling.outH;
    int64_t outW = baseTiling.outW;
    if (outSizeNum == INPUT_DIMS) {
        outD = outData[CONST_2];
        outH = outData[CONST_3];
        outW = outData[CONST_4];
    }
    OP_CHECK_IF((baseTiling.outD != outD) || (baseTiling.outH != outH) || (baseTiling.outW != outW),
                error_report::ReportInvalidArgument(
                    tilingContext->GetNodeName(), "output",
                    "the output D H W dimensions must be the same as the attribute gradInputSize"),
                return ge::GRAPH_FAILED);
    ComputeScales(scaleD, scaleH, scaleW);
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dGradRegbaseTiling::FillTilingData()
{
    tilingData->ubFactor = baseTiling.ubFactor;
    tilingData->tailBlockNum = baseTiling.tailBlockNum;
    tilingData->blkProcessNum = baseTiling.blkProcessNum;
    tilingData->lenN = baseTiling.dimN;
    tilingData->lenC = baseTiling.dimC;
    tilingData->inD = baseTiling.inD;
    tilingData->inH = baseTiling.inH;
    tilingData->inW = baseTiling.inW;
    tilingData->outD = baseTiling.outD;
    tilingData->outH = baseTiling.outH;
    tilingData->outW = baseTiling.outW;
    tilingData->scaleD = baseTiling.scaleD;
    tilingData->scaleH = baseTiling.scaleH;
    tilingData->scaleW = baseTiling.scaleW;
}

void UpsampleNearest3dGradRegbaseTiling::PrintTilingData()
{
    OP_LOGD(tilingContext,
            "ubFactor %d, tailBlockNum %d, blkProcessNum %ld, lenN %ld, lenC %ld, inD %ld, inH %ld, inW %ld, outD %ld, "
            "outH %ld, outW %ld, scaleD %f, scaleH %f, scaleW %f",
            tilingData->ubFactor, tilingData->tailBlockNum, tilingData->blkProcessNum, tilingData->lenN,
            tilingData->lenC, tilingData->inD, tilingData->inH, tilingData->inW, tilingData->outD, tilingData->outH,
            tilingData->outW, tilingData->scaleD, tilingData->scaleH, tilingData->scaleW);
}

ge::graphStatus UpsampleNearest3dGradRegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(coreNum).c_str(),
                                              "coreNum must be greater than 0"),
        return ge::GRAPH_FAILED);
    baseTiling.coreNum = coreNum;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize == 0UL,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(ubSize).c_str(),
                                                  "ubSize must be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_LOGI(tilingContext, "coreNum is %ld, ubSize is %lu", coreNum, ubSize);
    baseTiling.ubSize = static_cast<int32_t>(ubSize);
    if (tilingData == nullptr) {
        tilingData = tilingContext->GetTilingData<UpsampleNearest3dGradRegBaseTilingData>();
        OP_CHECK_IF(
            tilingData == nullptr,
            OP_LOGE_WITHOUT_REPORT(
                tilingContext, "get tilingdata ptr failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData, sizeof(UpsampleNearest3dGradRegBaseTilingData), 0,
                          sizeof(UpsampleNearest3dGradRegBaseTilingData)) != EOK),
                OP_LOGE_WITHOUT_REPORT(
                    tilingContext, "memset tilingdata failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearest3dGradRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    tilingContext, "CheckInputParams failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    tilingContext, "CheckInputShapes failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();
    PrintTilingData();
    uint64_t schId = static_cast<uint64_t>(baseTiling.schId);
    uint64_t isUint32 = static_cast<uint64_t>(baseTiling.isUint32);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isUint32);
    OP_LOGI(tilingContext, "tilingKey %lu, schId %lu, isUint32 %lu, realCoreNum %d", tilingKey, schId, isUint32,
            baseTiling.realCoreNum);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(baseTiling.realCoreNum);

    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "workspaces is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UpsampleNearest3dGradRegbase(gert::TilingContext* context)
{
    UpsampleNearest3dGradRegbaseTiling tilingImpl = UpsampleNearest3dGradRegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(
            context,
            "Tiling4UpsampleNearest3dGradRegbase init failed; propagate the first error reported by the failing stage");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "Tiling4UpsampleNearest3dGradRegbase do tiling failed; propagate the first "
                                        "error reported by the failing stage");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "End Tiling4UpsampleNearest3dGradRegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
