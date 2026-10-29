/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bicubic2d_aa_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "upsample_bicubic2d_aa_tiling.h"
#include "image/upsample_bicubic2d_aa/op_kernel/arch35/upsample_bicubic2d_aa_tiling_data.h"
#include "image/upsample_bicubic2d_aa/op_kernel/arch35/upsample_bicubic2d_aa_tiling_key.h"
#include "op_api/error_report_util.h"

namespace {
constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int64_t INPUT_DIMS = 4;
constexpr int64_t THREAD_NUM = 512; // simt开的线程数
constexpr int32_t CACHE_LINE = 128;

struct BaseTilingData {
    int64_t dimN = 0;
    int64_t dimC = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t alignCorners = 0;
    int64_t blkProcessNum = 0;
    float scaleH = 0.0;
    float scaleW = 0.0;
    float invScaleH = 0.0;
    float invScaleW = 0.0;
    float supportH = 0.0;
    float supportW = 0.0;
    int32_t isInt32 = 1;
    int32_t schId = 0;
    int32_t coreNum = 0;
    int32_t ubSize = 0;
    int32_t realCoreNum = 0;
    int32_t tailBlockNum = 0;
    int32_t ubFactor = 0;
    int32_t dtypeSize = 0;
    int32_t cacheLineNum = 0;
    int32_t oneBlockNum = 0;
};

static const std::map<ge::DataType, int32_t> inputDtypeList = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};

} // namespace

namespace optiling {
using namespace Ops::Cv::OpTiling;

class UpsampleBicubic2dAARegbaseTiling {
public:
    explicit UpsampleBicubic2dAARegbaseTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScalesSupportValues(float originalScaleH, float originalScaleW);
    void CalTilingData();
    void ComputeDataCopy();
    void FillTilingData();
    void PrintTilingData();

private:
    BaseTilingData baseTiling_;
    gert::TilingContext* context_ = nullptr;
    UpsampleBicubic2dAARegBaseTilingData* tilingData_{nullptr};
};

void UpsampleBicubic2dAARegbaseTiling::ComputeDataCopy()
{
    int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
    if (baseTiling_.outSize <= static_cast<int64_t>(baseTiling_.cacheLineNum * baseTiling_.coreNum)) {
        baseTiling_.realCoreNum = static_cast<int32_t>(baseTiling_.outSize) / baseTiling_.cacheLineNum;
    }
    if (0 == baseTiling_.realCoreNum) {
        baseTiling_.realCoreNum = 1;
    }
    baseTiling_.blkProcessNum = baseTiling_.outSize / static_cast<int64_t>(baseTiling_.realCoreNum);
    baseTiling_.tailBlockNum = static_cast<int32_t>(baseTiling_.outSize %
                                                    static_cast<int64_t>(baseTiling_.realCoreNum));
    baseTiling_.ubFactor = (baseTiling_.ubSize - baseTiling_.dtypeSize * baseTiling_.oneBlockNum) /
                           (CONST_2 * baseTiling_.dtypeSize);
    baseTiling_.ubFactor = (baseTiling_.ubFactor / baseTiling_.oneBlockNum) * baseTiling_.oneBlockNum;
    return;
}

void UpsampleBicubic2dAARegbaseTiling::CalTilingData()
{
    int64_t outHW = baseTiling_.outH * baseTiling_.outW;
    int64_t outCHW = baseTiling_.dimC * outHW;
    int64_t outNCHW = baseTiling_.dimN * outCHW;
    int64_t maxNum = static_cast<int64_t>(baseTiling_.coreNum) * THREAD_NUM;
    bool isDataCopy = baseTiling_.outH == baseTiling_.inH && baseTiling_.outW == baseTiling_.inW &&
                      std::abs(baseTiling_.scaleH - 1.0f) < 1e-6f && std::abs(baseTiling_.scaleW - 1.0f) < 1e-6f;
    baseTiling_.realCoreNum = baseTiling_.coreNum;
    baseTiling_.cacheLineNum = CACHE_LINE / baseTiling_.dtypeSize;

    if (isDataCopy) {
        OP_LOGI(context_, "enter datacopy");
        baseTiling_.schId = CONST_0; // 纯copy模板
        baseTiling_.isInt32 = 0;
        ComputeDataCopy();
    } else {
        OP_LOGI(context_, "enter simt nchw");
        int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
        if (outNCHW < coreNum) {
            baseTiling_.realCoreNum = static_cast<int32_t>(outNCHW);
        }
        baseTiling_.schId = CONST_1; // nchw 分线程
        int64_t allNum = outNCHW;
        baseTiling_.blkProcessNum = allNum / static_cast<int64_t>(baseTiling_.realCoreNum);
        baseTiling_.tailBlockNum = static_cast<int32_t>(allNum % static_cast<int64_t>(baseTiling_.realCoreNum));
    }
    return;
}

void UpsampleBicubic2dAARegbaseTiling::ComputeScalesSupportValues(float originalScaleH, float originalScaleW)
{
    if (baseTiling_.alignCorners) {
        if (baseTiling_.outH > 1) {
            baseTiling_.scaleH = static_cast<float>(baseTiling_.inH - 1) / (baseTiling_.outH - 1);
        } else {
            baseTiling_.scaleH = static_cast<float>(0.0);
        }
        if (baseTiling_.outW > 1) {
            baseTiling_.scaleW = static_cast<float>(baseTiling_.inW - 1) / (baseTiling_.outW - 1);
        } else {
            baseTiling_.scaleW = static_cast<float>(0.0);
        }
    } else {
        if (originalScaleH > 0.0f) {
            baseTiling_.scaleH = 1.0f / originalScaleH;
        } else {
            baseTiling_.scaleH = static_cast<float>(baseTiling_.inH) / baseTiling_.outH;
        }
        if (originalScaleW > 0.0f) {
            baseTiling_.scaleW = 1.0f / originalScaleW;
        } else {
            baseTiling_.scaleW = static_cast<float>(baseTiling_.inW) / baseTiling_.outW;
        }
    }
    baseTiling_.invScaleH = (baseTiling_.scaleH >= 1.0f) ? (1.0f / baseTiling_.scaleH) : 1.0f;
    baseTiling_.invScaleW = (baseTiling_.scaleW >= 1.0f) ? (1.0f / baseTiling_.scaleW) : 1.0f;
    baseTiling_.supportH = (baseTiling_.scaleH >= 1.0f) ? (2.0f * baseTiling_.scaleH) : 2.0f;
    baseTiling_.supportW = (baseTiling_.scaleW >= 1.0f) ? (2.0f * baseTiling_.scaleW) : 2.0f;
}

ge::graphStatus UpsampleBicubic2dAARegbaseTiling::CheckInputParams()
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
    auto outDtype = outDescPtr0->GetDataType();
    OP_CHECK_IF(
        outDtype != inputDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(outDtype).c_str(),
                                              "input and output dtype must be the same"),
        return ge::GRAPH_FAILED);
    auto outFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outDescPtr0->GetStorageFormat()));
    OP_CHECK_IF(outFormat != inputFormat,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "input",
                                                        (std::to_string(static_cast<int32_t>(inputFormat)) + ", " +
                                                         std::to_string(static_cast<int32_t>(outFormat)))
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
                                                    "the dim of input0 or output0 should be equal to 4"),
                return ge::GRAPH_FAILED);
    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outputSize = outShape.GetShapeSize();
    baseTiling_.outSize = outputSize;
    baseTiling_.dimN = inputShape.GetDim(CONST_0);
    baseTiling_.dimC = inputShape.GetDim(CONST_1);
    baseTiling_.inH = inputShape.GetDim(CONST_2);
    baseTiling_.inW = inputShape.GetDim(CONST_3);
    baseTiling_.outH = outShape.GetDim(CONST_2);
    baseTiling_.outW = outShape.GetDim(CONST_3);
    OP_CHECK_IF(
        inputSize == 0 || outputSize == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "empty", (std::to_string(inputSize) + ", " + std::to_string(outputSize)).c_str(),
            "empty input or output is not supported"),
        return ge::GRAPH_FAILED);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    int32_t isInt32 = static_cast<int32_t>((inputSize <= int32Max) && (outputSize <= int32Max));
    baseTiling_.isInt32 = isInt32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBicubic2dAARegbaseTiling::CheckInputShapeAndAttr()
{
    auto outDescPtr0 = context_->GetOutputShape(0);
    auto outShape = outDescPtr0->GetStorageShape();
    int64_t outN = outShape.GetDim(CONST_0);
    int64_t outC = outShape.GetDim(CONST_1);
    OP_CHECK_IF((outN != baseTiling_.dimN) || (outC != baseTiling_.dimC),
                error_report::ReportInvalidArgument(context_->GetNodeName(), "input",
                                                    "the N and C dimensions of the input and output need to be same"),
                return ge::GRAPH_FAILED);
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_0);
    OP_CHECK_IF(
        outputSize == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "outputSize is null in the tiling context; check the TilingContext construction"),
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
    int64_t outSizeNum = outputSize->GetSize();
    OP_CHECK_IF(
        outSizeNum != CONST_2,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(outSizeNum).c_str(),
                                                  "the num of outputSize is , invalid, must be 2"),
        return ge::GRAPH_FAILED);
    const int64_t* outData = reinterpret_cast<const int64_t*>(outputSize->GetData());
    int64_t outH = baseTiling_.outH;
    int64_t outW = baseTiling_.outW;
    outH = outData[CONST_0];
    outW = outData[CONST_1];
    OP_CHECK_IF(
        (baseTiling_.outH != outH) || (baseTiling_.outW != outW),
        error_report::ReportInvalidArgument(context_->GetNodeName(), "output",
                                            "the output H W dimensions must be the same as the attribute outputSize"),
        return ge::GRAPH_FAILED);
    ComputeScalesSupportValues(originalScaleH, originalScaleW);
    return ge::GRAPH_SUCCESS;
}

void UpsampleBicubic2dAARegbaseTiling::FillTilingData()
{
    tilingData_->blkProcessNum = baseTiling_.blkProcessNum;
    tilingData_->lenN = baseTiling_.dimN;
    tilingData_->lenC = baseTiling_.dimC;
    tilingData_->inH = baseTiling_.inH;
    tilingData_->inW = baseTiling_.inW;
    tilingData_->outH = baseTiling_.outH;
    tilingData_->outW = baseTiling_.outW;
    tilingData_->alignCorners = baseTiling_.alignCorners;
    tilingData_->ubFactor = baseTiling_.ubFactor;
    tilingData_->tailBlockNum = baseTiling_.tailBlockNum;
    tilingData_->scaleH = baseTiling_.scaleH;
    tilingData_->scaleW = baseTiling_.scaleW;
    tilingData_->invScaleH = baseTiling_.invScaleH;
    tilingData_->invScaleW = baseTiling_.invScaleW;
    tilingData_->supportH = baseTiling_.supportH;
    tilingData_->supportW = baseTiling_.supportW;
}

void UpsampleBicubic2dAARegbaseTiling::PrintTilingData()
{
    OP_LOGD(context_,
            "blkProcessNum %ld, lenN %ld, lenC %ld, inH %ld, inW %ld, outH %ld, outW %ld, alignCorners %ld, "
            "ubFactor %d, tailBlockNum %d, scaleH %f, scaleW %f, invScaleH %f, invScaleW %f, supportH %f, supportW %f",
            tilingData_->blkProcessNum, tilingData_->lenN, tilingData_->lenC, tilingData_->inH, tilingData_->inW,
            tilingData_->outH, tilingData_->outW, tilingData_->alignCorners, tilingData_->ubFactor,
            tilingData_->tailBlockNum, tilingData_->scaleH, tilingData_->scaleW, tilingData_->invScaleH,
            tilingData_->invScaleW, tilingData_->supportH, tilingData_->supportW);
}

ge::graphStatus UpsampleBicubic2dAARegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "input", std::to_string(coreNum).c_str(),
                                                      "coreNum must be greater than 0"),
                return ge::GRAPH_FAILED);
    baseTiling_.coreNum = coreNum;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0UL,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "input", std::to_string(ubSize).c_str(), "ubSize must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_, "coreNum is %ld, ubSize is %lu", coreNum, ubSize);
    baseTiling_.ubSize = static_cast<int32_t>(ubSize);
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<UpsampleBicubic2dAARegBaseTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr,
                    OP_LOGE_WITHOUT_REPORT(
                        context_, "get tilingdata ptr failed; propagate the first error reported by the failing stage"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(UpsampleBicubic2dAARegBaseTilingData), 0,
                          sizeof(UpsampleBicubic2dAARegBaseTilingData)) != EOK),
                OP_LOGE_WITHOUT_REPORT(
                    context_, "memset tilingdata failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBicubic2dAARegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputParams failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(
                    context_, "CheckInputShape failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();
    PrintTilingData();
    uint64_t schId = static_cast<uint64_t>(baseTiling_.schId);
    uint64_t isInt32 = static_cast<uint64_t>(baseTiling_.isInt32);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isInt32);
    OP_LOGI(context_, "tilingKey: %lu, schId: %lu, isInt32: %lu, realCoreNum: %d", tilingKey, schId, isInt32,
            baseTiling_.realCoreNum);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTiling_.realCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "workspaces is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UpsampleBicubic2dAARegbase(gert::TilingContext* context)
{
    UpsampleBicubic2dAARegbaseTiling tilingImpl = UpsampleBicubic2dAARegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(
            context,
            "Tiling4UpsampleBicubic2dAARegbase init failed; propagate the first error reported by the failing stage");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "Tiling4UpsampleBicubic2dAARegbase do tiling failed; propagate the first error "
                                        "reported by the failing stage");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "end Tiling4UpsampleBicubic2dAARegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
