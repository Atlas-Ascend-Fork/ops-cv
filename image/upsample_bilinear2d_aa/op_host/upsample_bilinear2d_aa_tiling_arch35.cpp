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
 * \file upsample_bilinear2d_aa_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "upsample_bilinear2d_aa_tiling.h"
#include "image/upsample_bilinear2d_aa/op_kernel/arch35/upsample_bilinear2d_aa_tiling_data.h"
#include "image/upsample_bilinear2d_aa/op_kernel/arch35/upsample_bilinear2d_aa_tiling_key.h"
#include "op_api/error_report_util.h"

namespace optiling {
using namespace Ops::Cv::OpTiling;

constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int64_t INPUT_DIMS = 4;
constexpr int32_t CACHE_LINE = 128;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);

struct UpsampleBilinear2dAABaseTiling {
    int64_t dimN = 0;
    int64_t dimC = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t alignCorners = 0;
    int64_t blkProcessNum = 0;
    int64_t realCoreNum = 0;
    int64_t tailBlockNum = 0;
    float scaleH = 0.0;
    float scaleW = 0.0;
    float invScaleH = 0.0;
    float invScaleW = 0.0;
    float supportH = 0.0;
    float supportW = 0.0;
    int32_t ubSize = 0;
    int32_t ubFactor = 0;
    uint64_t isInt32 = 1;
    uint64_t schId = 0;
    int32_t coreNum = 0;
    int32_t dtypeSize = 0;
    int32_t cacheLineNum = 0;
    int32_t oneBlockNum = 0;
};

static const std::map<ge::DataType, int32_t> inputDtypeList = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
class UpsampleBilinear2dAARegbaseTiling {
public:
    explicit UpsampleBilinear2dAARegbaseTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScalesSupportValues(float originalScaleH, float originalScaleW);
    void CalTilingData();
    void DataCopyTiling();
    void FillTilingData();
    void PrintTilingData();

private:
    UpsampleBilinear2dAABaseTiling baseTiling_;
    gert::TilingContext* context_ = nullptr;
    UpsampleBilinear2dAARegBaseTilingData* tilingData_{nullptr};
};

void UpsampleBilinear2dAARegbaseTiling::DataCopyTiling()
{
    baseTiling_.cacheLineNum = CACHE_LINE / baseTiling_.dtypeSize;
    if (baseTiling_.outSize <= static_cast<int64_t>(baseTiling_.cacheLineNum * baseTiling_.coreNum)) {
        baseTiling_.realCoreNum = baseTiling_.outSize / baseTiling_.cacheLineNum;
    }
    if (baseTiling_.realCoreNum == 0) {
        baseTiling_.realCoreNum = 1;
    }
    baseTiling_.blkProcessNum = baseTiling_.outSize / baseTiling_.realCoreNum;
    baseTiling_.tailBlockNum = baseTiling_.outSize % baseTiling_.realCoreNum;
    baseTiling_.ubFactor = (baseTiling_.ubSize - baseTiling_.oneBlockNum * baseTiling_.dtypeSize) /
                           (CONST_2 * baseTiling_.dtypeSize);
    baseTiling_.ubFactor = (baseTiling_.ubFactor / baseTiling_.oneBlockNum) * baseTiling_.oneBlockNum;
}

void UpsampleBilinear2dAARegbaseTiling::CalTilingData()
{
    bool isDataCopy = baseTiling_.outH == baseTiling_.inH && baseTiling_.outW == baseTiling_.inW &&
                      std::abs(baseTiling_.scaleH - 1.0f) < 1e-6f && std::abs(baseTiling_.scaleW - 1.0f) < 1e-6f;
    if (isDataCopy) {
        OP_LOGI(context_, "enter datacopy");
        baseTiling_.schId = CONST_0; // 纯copy模板
        baseTiling_.isInt32 = 0;
        DataCopyTiling();
    } else {
        OP_LOGI(context_, "enter simt nchw");
        int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
        baseTiling_.realCoreNum = baseTiling_.outSize < coreNum ? baseTiling_.outSize : coreNum;
        baseTiling_.schId = CONST_1; // nchw 分线程
        baseTiling_.blkProcessNum = baseTiling_.outSize / baseTiling_.realCoreNum;
        baseTiling_.tailBlockNum = baseTiling_.outSize % baseTiling_.realCoreNum;
    }
}

void UpsampleBilinear2dAARegbaseTiling::ComputeScalesSupportValues(float originalScaleH, float originalScaleW)
{
    if (baseTiling_.alignCorners) {
        if (baseTiling_.outW > 1) {
            baseTiling_.scaleW = static_cast<float>(baseTiling_.inW - 1) / (baseTiling_.outW - 1);
        } else {
            baseTiling_.scaleW = static_cast<float>(0);
        }
        if (baseTiling_.outH > 1) {
            baseTiling_.scaleH = static_cast<float>(baseTiling_.inH - 1) / (baseTiling_.outH - 1);
        } else {
            baseTiling_.scaleH = static_cast<float>(0);
        }
    } else {
        if (originalScaleW > 0.0f) {
            baseTiling_.scaleW = originalScaleW;
        } else {
            baseTiling_.scaleW = static_cast<float>(baseTiling_.inW) / baseTiling_.outW;
        }
        if (originalScaleH > 0.0f) {
            baseTiling_.scaleH = originalScaleH;
        } else {
            baseTiling_.scaleH = static_cast<float>(baseTiling_.inH) / baseTiling_.outH;
        }
    }
    baseTiling_.invScaleH = (baseTiling_.scaleH >= 1.0f) ? (1.0f / baseTiling_.scaleH) : 1.0f;
    baseTiling_.invScaleW = (baseTiling_.scaleW >= 1.0f) ? (1.0f / baseTiling_.scaleW) : 1.0f;
    baseTiling_.supportH = (baseTiling_.scaleH >= 1.0f) ? baseTiling_.scaleH : 1.0f;
    baseTiling_.supportW = (baseTiling_.scaleW >= 1.0f) ? baseTiling_.scaleW : 1.0f;
}

ge::graphStatus UpsampleBilinear2dAARegbaseTiling::CheckInputParams()
{
    auto input = context_->GetInputDesc(CONST_0);
    OP_CHECK_IF(
        input == nullptr,
        error_report::ReportInvalidArgument(context_->GetNodeName(), "input",
                                            "the input desc/shape of required input input is missing in the tiling "
                                            "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto inputDtype = input->GetDataType();
    OP_CHECK_IF(
        inputDtypeList.count(inputDtype) == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(inputDtype).c_str(),
                                              "input dtype is not supported"),
        return ge::GRAPH_FAILED);
    auto inputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(input->GetStorageFormat()));
    OP_CHECK_IF((inputFormat != ge::Format::FORMAT_ND && inputFormat != ge::Format::FORMAT_NCHW),
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "input",
                                                        std::to_string(static_cast<int32_t>(inputFormat)).c_str(),
                                                        "input format is not supported"),
                return ge::GRAPH_FAILED);
    baseTiling_.dtypeSize = inputDtypeList.find(inputDtype)->second;
    int32_t ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
    baseTiling_.oneBlockNum = ubBlockSize / baseTiling_.dtypeSize;
    auto outDescPtr0 = context_->GetOutputDesc(0);
    OP_CHECK_IF(outDescPtr0 == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "outDescPtr0 (output index 0) is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto outputDtype = outDescPtr0->GetDataType();
    OP_CHECK_IF(
        outputDtype != inputDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(outputDtype).c_str(),
                                              "input and output dtype must be the same"),
        return ge::GRAPH_FAILED);
    auto outputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outDescPtr0->GetStorageFormat()));
    OP_CHECK_IF(outputFormat != inputFormat,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "input",
                                                        (std::to_string(static_cast<int32_t>(inputFormat)) + ", " +
                                                         std::to_string(static_cast<int32_t>(outputFormat)))
                                                            .c_str(),
                                                        "input and output format must be the same"),
                return ge::GRAPH_FAILED);
    auto inputX = context_->GetInputShape(0);
    auto outY = context_->GetOutputShape(0);
    OP_CHECK_IF(
        inputX == nullptr,
        error_report::ReportInvalidArgument(context_->GetNodeName(), "inputX",
                                            "the input desc/shape of required input inputX (index 0) is missing in the "
                                            "tiling context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outY == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "outY (output index 0) is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outShape.GetDimNum() != INPUT_DIMS),
                error_report::ReportInvalidArgument(context_->GetNodeName(), "dim",
                                                    "the dim of input0 or output0 should be equal to four"),
                return ge::GRAPH_FAILED);
    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outSize = outShape.GetShapeSize();
    baseTiling_.dimN = inputShape.GetDim(CONST_0);
    baseTiling_.dimC = inputShape.GetDim(CONST_1);
    baseTiling_.inH = inputShape.GetDim(CONST_2);
    baseTiling_.inW = inputShape.GetDim(CONST_3);
    baseTiling_.outH = outShape.GetDim(CONST_2);
    baseTiling_.outW = outShape.GetDim(CONST_3);
    baseTiling_.outSize = outSize;
    OP_CHECK_IF(
        inputSize == 0 || outSize == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "empty",
                                                   (std::to_string(inputSize) + ", " + std::to_string(outSize)).c_str(),
                                                   "empty input or output is not supported"),
        return ge::GRAPH_FAILED);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    baseTiling_.isInt32 = static_cast<uint64_t>((inputSize <= int32Max) && (outSize <= int32Max));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBilinear2dAARegbaseTiling::CheckInputShapeAndAttr()
{
    auto outDescPtr0 = context_->GetOutputShape(0);
    auto outShape = outDescPtr0->GetStorageShape();
    int64_t outC = outShape.GetDim(CONST_1);
    int64_t outN = outShape.GetDim(CONST_0);
    OP_CHECK_IF(
        (outN != baseTiling_.dimN) || (outC != baseTiling_.dimC),
        error_report::ReportInvalidArgument(context_->GetNodeName(), "input",
                                            "the N and C dimensions of the input and output need to be the same"),
        return ge::GRAPH_FAILED);
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto outSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_0);
    OP_CHECK_IF(
        outSize == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "outSize is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const bool* alignCornersPtr = attrs->GetAttrPointer<bool>(CONST_1);
    OP_CHECK_IF(alignCornersPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "alignCornersPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    baseTiling_.alignCorners = *alignCornersPtr ? 1 : 0;
    const float* scaleHPtr = attrs->GetAttrPointer<float>(CONST_2);
    OP_CHECK_IF(
        scaleHPtr == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "scaleHPtr is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    float originalScaleH = *scaleHPtr;
    const float* scaleWPtr = attrs->GetAttrPointer<float>(CONST_3);
    OP_CHECK_IF(
        scaleWPtr == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "scaleWPtr is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    float originalScaleW = *scaleWPtr;
    OP_LOGI(context_, "alignCorners %ld, originalScaleH %f, originalScaleW %f", baseTiling_.alignCorners,
            originalScaleH, originalScaleW);
    int64_t outSizeNum = outSize->GetSize();
    OP_CHECK_IF(
        outSizeNum != CONST_2,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(outSizeNum).c_str(),
                                                  "the num of outSize is , invalid, must be 2"),
        return ge::GRAPH_FAILED);
    const int64_t* outData = reinterpret_cast<const int64_t*>(outSize->GetData());
    OP_CHECK_IF(
        (baseTiling_.outH != outData[CONST_0]) || (baseTiling_.outW != outData[CONST_1]),
        error_report::ReportInvalidArgument(context_->GetNodeName(), "output",
                                            "the output H W dimensions must be the same as the attribute outSize"),
        return ge::GRAPH_FAILED);
    ComputeScalesSupportValues(originalScaleH, originalScaleW);
    return ge::GRAPH_SUCCESS;
}

void UpsampleBilinear2dAARegbaseTiling::FillTilingData()
{
    tilingData_->blkProcessNum = baseTiling_.blkProcessNum;
    tilingData_->tailBlockNum = baseTiling_.tailBlockNum;
    tilingData_->lenN = baseTiling_.dimN;
    tilingData_->lenC = baseTiling_.dimC;
    tilingData_->inH = baseTiling_.inH;
    tilingData_->inW = baseTiling_.inW;
    tilingData_->outH = baseTiling_.outH;
    tilingData_->outW = baseTiling_.outW;
    tilingData_->ubFactor = baseTiling_.ubFactor;
    tilingData_->scaleH = baseTiling_.scaleH;
    tilingData_->scaleW = baseTiling_.scaleW;
    tilingData_->invScaleH = baseTiling_.invScaleH;
    tilingData_->invScaleW = baseTiling_.invScaleW;
    tilingData_->supportH = baseTiling_.supportH;
    tilingData_->supportW = baseTiling_.supportW;
}

void UpsampleBilinear2dAARegbaseTiling::PrintTilingData()
{
    OP_LOGD(context_,
            "blkProcessNum %ld, tailBlockNum %d, lenN %ld, lenC %ld, inH %ld, inW %ld, outH %ld, outW %ld, "
            "ubFactor %d, scaleH %f, scaleW %f, invScaleH %f, invScaleW %f, supportH %f, supportW %f",
            tilingData_->blkProcessNum, tilingData_->tailBlockNum, tilingData_->lenN, tilingData_->lenC,
            tilingData_->inH, tilingData_->inW, tilingData_->outH, tilingData_->outW, tilingData_->ubFactor,
            tilingData_->scaleH, tilingData_->scaleW, tilingData_->invScaleH, tilingData_->invScaleW,
            tilingData_->supportH, tilingData_->supportW);
}

ge::graphStatus UpsampleBilinear2dAARegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNumAiv <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(coreNumAiv).c_str(),
                                              "coreNum must be greater than 0"),
        return ge::GRAPH_FAILED);
    baseTiling_.coreNum = coreNumAiv;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0UL,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "input", std::to_string(ubSize).c_str(), "ubSize must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_, "coreNum is %ld, ubSize is %lu", coreNumAiv, ubSize);
    baseTiling_.ubSize = static_cast<int32_t>(ubSize);
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<UpsampleBilinear2dAARegBaseTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr,
                    OP_LOGE_WITHOUT_REPORT(
                        context_, "get tilingdata ptr failed; propagate the first error reported by the failing stage"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(UpsampleBilinear2dAARegBaseTilingData), 0,
                          sizeof(UpsampleBilinear2dAARegBaseTilingData)) != EOK),
                OP_LOGE_WITHOUT_REPORT(
                    context_, "memset tilingdata failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBilinear2dAARegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputParams failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputShapes failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();
    PrintTilingData();
    const uint64_t tilingKey = GET_TPL_TILING_KEY(baseTiling_.schId, baseTiling_.isInt32);
    OP_LOGI(context_, "tilingKey %lu, schId %lu, isInt32 %lu, realCoreNum %d", tilingKey, baseTiling_.schId,
            baseTiling_.isInt32, baseTiling_.realCoreNum);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTiling_.realCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "workspaces is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UpsampleBilinear2dAARegbase(gert::TilingContext* context)
{
    UpsampleBilinear2dAARegbaseTiling tilingImpl = UpsampleBilinear2dAARegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(
            context,
            "Tiling4UpsampleBilinear2dAARegbase init failed; propagate the first error reported by the failing stage");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "Tiling4UpsampleBilinear2dAARegbase do tiling failed; propagate the first "
                                        "error reported by the failing stage");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "end Tiling4UpsampleBilinear2dAARegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
