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
 * \file upsample_bicubic2d_aa_grad_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "upsample_bicubic2d_aa_grad_tiling.h"
#include "upsample_bicubic2d_aa_grad_error.h"
#include "image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_tiling_data.h"
#include "image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_tiling_key.h"

namespace optiling {
using namespace Ops::Cv::OpTiling;

constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr uint32_t SCHEDULE_MODE = 1U;
constexpr int64_t INPUT_DIMS = 4;
constexpr int64_t THREAD_NUM = 512; // simt开的线程数
constexpr int32_t CACHE_LINE = 128;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);

struct Bicubic2dAAGradBaseTilingData {
    int64_t dimN = 0;
    int64_t dimC = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t blkProcessNum = 0;
    int64_t alignCorners = 0;
    float scaleH = 0.0;
    float scaleW = 0.0;
    float invScaleH = 0.0;
    float invScaleW = 0.0;
    float supportH = 0.0;
    float supportW = 0.0;
    int32_t isInt32 = 1;
    int32_t schId = 0;
    int32_t isDeterministic = 0;
    int32_t coreNum = 0;
    int32_t ubSize = 0;
    int32_t realCoreNum = 0;
    int32_t tailBlockNum = 0;
    int32_t ubFactor = 0;
    int32_t dtypeSize = 0;
    int32_t cacheLineNum = 0;
    int32_t oneBlockNum = 0;
    int32_t initHeadCoreNum = 0;
    int64_t perCoreInitEle = 0;
};

static const std::map<ge::DataType, int32_t> inputDtypeList = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
class UpsampleBicubic2dAAGradRegbaseTiling {
public:
    explicit UpsampleBicubic2dAAGradRegbaseTiling(gert::TilingContext* context) : context_(context) {};

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
    float ComputeScale(int64_t inSize, int64_t outSize, float originalScale);

private:
    Bicubic2dAAGradBaseTilingData baseTiling_;
    gert::TilingContext* context_ = nullptr;
    UpsampleBicubic2dAAGradRegBaseTilingData* tilingData_{nullptr};
};

void UpsampleBicubic2dAAGradRegbaseTiling::ComputeDataCopy()
{
    int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
    if (baseTiling_.outSize <= static_cast<int64_t>(baseTiling_.coreNum * baseTiling_.cacheLineNum)) {
        baseTiling_.realCoreNum = static_cast<int32_t>(baseTiling_.outSize) / baseTiling_.cacheLineNum;
    }
    if (baseTiling_.realCoreNum == 0) {
        baseTiling_.realCoreNum = 1;
    }
    baseTiling_.blkProcessNum = baseTiling_.outSize / static_cast<int64_t>(baseTiling_.realCoreNum);
    baseTiling_.tailBlockNum = static_cast<int32_t>(baseTiling_.outSize %
                                                    static_cast<int64_t>(baseTiling_.realCoreNum));
    baseTiling_.ubFactor = (baseTiling_.ubSize - baseTiling_.oneBlockNum * baseTiling_.dtypeSize) /
                           (baseTiling_.dtypeSize * CONST_2);
    baseTiling_.ubFactor = (baseTiling_.ubFactor / baseTiling_.oneBlockNum) * baseTiling_.oneBlockNum;
    return;
}

void UpsampleBicubic2dAAGradRegbaseTiling::CalTilingData()
{
    int64_t outHW = baseTiling_.outH * baseTiling_.outW;
    int64_t outCHW = baseTiling_.dimC * outHW;
    int64_t outNCHW = baseTiling_.dimN * outCHW;
    int64_t maxNum = static_cast<int64_t>(baseTiling_.coreNum) * THREAD_NUM;
    bool isDataCopy = baseTiling_.outH == baseTiling_.inH && baseTiling_.outW == baseTiling_.inW &&
                      std::abs(baseTiling_.scaleH - 1.0f) < 1e-6f && std::abs(baseTiling_.scaleW - 1.0f) < 1e-6f;
    baseTiling_.cacheLineNum = CACHE_LINE / baseTiling_.dtypeSize;
    baseTiling_.realCoreNum = baseTiling_.coreNum;
    if (isDataCopy) {
        OP_LOGI(context_, "enter datacopy");
        baseTiling_.schId = CONST_0; // 纯copy模板
        baseTiling_.isInt32 = 0;
        baseTiling_.isDeterministic = 1;
        ComputeDataCopy();
    } else if (baseTiling_.isDeterministic) {
        OP_LOGI(context_, "enter simt nchw deterministic");
        int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
        if (outNCHW < coreNum) {
            baseTiling_.realCoreNum = static_cast<int32_t>(outNCHW);
        }
        baseTiling_.schId = CONST_1; // nchw 分线程
        int64_t allNum = outNCHW;
        baseTiling_.blkProcessNum = allNum / static_cast<int64_t>(baseTiling_.realCoreNum);
        baseTiling_.tailBlockNum = static_cast<int32_t>(allNum % static_cast<int64_t>(baseTiling_.realCoreNum));
    } else {
        OP_LOGI(context_, "enter simt nchw non-deterministic");
        int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
        int64_t inNCHW = baseTiling_.dimN * baseTiling_.dimC * baseTiling_.inH * baseTiling_.inW;
        if (inNCHW < coreNum) {
            baseTiling_.realCoreNum = static_cast<int32_t>(inNCHW);
        }
        baseTiling_.schId = CONST_1; // nchw 分线程
        int64_t allNum = inNCHW;
        baseTiling_.blkProcessNum = allNum / static_cast<int64_t>(baseTiling_.realCoreNum);
        baseTiling_.tailBlockNum = static_cast<int32_t>(allNum % static_cast<int64_t>(baseTiling_.realCoreNum));
        baseTiling_.perCoreInitEle = outNCHW / static_cast<int64_t>(baseTiling_.realCoreNum);
        baseTiling_.initHeadCoreNum = outNCHW % baseTiling_.realCoreNum;
    }
    return;
}

float UpsampleBicubic2dAAGradRegbaseTiling::ComputeScale(int64_t inSize, int64_t outSize, float originalScale)
{
    if (baseTiling_.alignCorners) {
        if (inSize > 1) {
            return static_cast<float>(outSize - 1) / (inSize - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        if (originalScale > 0.0f) {
            return originalScale;
        } else {
            return static_cast<float>(outSize) / inSize;
        }
    }
}

void UpsampleBicubic2dAAGradRegbaseTiling::ComputeScalesSupportValues(float originalScaleH, float originalScaleW)
{
    baseTiling_.scaleH = ComputeScale(baseTiling_.inH, baseTiling_.outH, originalScaleH);
    baseTiling_.scaleW = ComputeScale(baseTiling_.inW, baseTiling_.outW, originalScaleW);
    baseTiling_.invScaleH = (baseTiling_.scaleH >= 1.0f) ? (1.0f / baseTiling_.scaleH) : 1.0f;
    baseTiling_.invScaleW = (baseTiling_.scaleW >= 1.0f) ? (1.0f / baseTiling_.scaleW) : 1.0f;
    baseTiling_.supportH = (baseTiling_.scaleH >= 1.0f) ? (2.0f * baseTiling_.scaleH) : 2.0f;
    baseTiling_.supportW = (baseTiling_.scaleW >= 1.0f) ? (2.0f * baseTiling_.scaleW) : 2.0f;
}

ge::graphStatus UpsampleBicubic2dAAGradRegbaseTiling::CheckInputParams()
{
    auto input = context_->GetInputDesc(CONST_0);
    OP_CHECK_IF(input == nullptr,
                ReportBicubicGradMissingArgument(context_->GetNodeName(), "gradOutput", "input descriptor is missing"),
                return ge::GRAPH_FAILED);
    auto inputDtype = input->GetDataType();
    OP_CHECK_IF(inputDtypeList.count(inputDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gradOutput",
                                                      Ops::Base::ToString(inputDtype).c_str(),
                                                      "supported dtypes are float16, float32 and bfloat16"),
                return ge::GRAPH_FAILED);
    auto inputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(input->GetStorageFormat()));
    OP_CHECK_IF((inputFormat != ge::Format::FORMAT_NCHW && inputFormat != ge::Format::FORMAT_ND),
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "gradOutput",
                                           std::to_string(static_cast<int32_t>(inputFormat)).c_str(), "ND,NCHW"),
                return ge::GRAPH_FAILED);
    baseTiling_.dtypeSize = inputDtypeList.find(inputDtype)->second;
    int32_t ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
    baseTiling_.oneBlockNum = ubBlockSize / baseTiling_.dtypeSize;
    auto outDescPtr0 = context_->GetOutputDesc(CONST_0);
    OP_CHECK_IF(outDescPtr0 == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "output descriptor container is missing"),
                return ge::GRAPH_FAILED);
    auto outputDtype = outDescPtr0->GetDataType();
    OP_CHECK_IF(outputDtype != inputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "gradOutput,gradInput",
                    (Ops::Base::ToString(inputDtype) + "," + Ops::Base::ToString(outputDtype)).c_str(),
                    "input and output dtypes must match"),
                return ge::GRAPH_FAILED);
    auto outFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outDescPtr0->GetStorageFormat()));
    OP_CHECK_IF(outFormat != inputFormat,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "gradOutput,gradInput",
                                                        (std::to_string(static_cast<int32_t>(inputFormat)) + "," +
                                                         std::to_string(static_cast<int32_t>(outFormat)))
                                                            .c_str(),
                                                        "storage formats must match"),
                return ge::GRAPH_FAILED);
    auto inputX = context_->GetInputShape(0);
    auto outY = context_->GetOutputShape(0);
    OP_CHECK_IF(inputX == nullptr,
                ReportBicubicGradMissingArgument(context_->GetNodeName(), "gradOutput", "input shape is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outY == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "output shape container is missing"),
                return ge::GRAPH_FAILED);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outShape.GetDimNum() != INPUT_DIMS),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "gradOutput,gradInput",
                    (std::to_string(inputShape.GetDimNum()) + "," + std::to_string(outShape.GetDimNum())).c_str(),
                    "both ranks must be 4"),
                return ge::GRAPH_FAILED);
    int64_t outputSize = outShape.GetShapeSize();
    int64_t inputSize = inputShape.GetShapeSize();
    baseTiling_.dimN = inputShape.GetDim(CONST_0);
    baseTiling_.dimC = inputShape.GetDim(CONST_1);
    baseTiling_.inH = inputShape.GetDim(CONST_2);
    baseTiling_.outH = outShape.GetDim(CONST_2);
    baseTiling_.inW = inputShape.GetDim(CONST_3);
    baseTiling_.outW = outShape.GetDim(CONST_3);
    baseTiling_.outSize = outputSize;
    OP_CHECK_IF(inputSize == 0 || outputSize == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    context_->GetNodeName(), "gradOutput,gradInput",
                    (std::to_string(inputSize) + "," + std::to_string(outputSize)).c_str(),
                    "empty tensors must be handled by the API before kernel tiling"),
                return ge::GRAPH_FAILED);
    int64_t int32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    int32_t isInt32 = static_cast<int32_t>((inputSize <= int32Max) && (outputSize <= int32Max));
    baseTiling_.isInt32 = isInt32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBicubic2dAAGradRegbaseTiling::CheckInputShapeAndAttr()
{
    auto outDescPtr0 = context_->GetOutputShape(0);
    auto outShape = outDescPtr0->GetStorageShape();
    int64_t outN = outShape.GetDim(CONST_0);
    int64_t outC = outShape.GetDim(CONST_1);
    OP_CHECK_IF((outN != baseTiling_.dimN) || (outC != baseTiling_.dimC),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "gradOutput,gradInput",
                    (std::to_string(baseTiling_.dimN) + "," + std::to_string(baseTiling_.dimC) + ";" +
                     std::to_string(outN) + "," + std::to_string(outC))
                        .c_str(),
                    "N and C dimensions must match"),
                return ge::GRAPH_FAILED);
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "attribute container is missing"),
                return ge::GRAPH_FAILED);
    auto outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_0);
    OP_CHECK_IF(
        outputSize == nullptr,
        ReportBicubicGradMissingArgument(context_->GetNodeName(), "output_size", "required attribute is missing"),
        return ge::GRAPH_FAILED);
    auto inputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_1);
    OP_CHECK_IF(
        inputSize == nullptr,
        ReportBicubicGradMissingArgument(context_->GetNodeName(), "input_size", "required attribute is missing"),
        return ge::GRAPH_FAILED);
    const bool* alignCornersPtr = attrs->GetAttrPointer<bool>(CONST_2);
    OP_CHECK_IF(
        alignCornersPtr == nullptr,
        ReportBicubicGradMissingArgument(context_->GetNodeName(), "align_corners", "required attribute is missing"),
        return ge::GRAPH_FAILED);
    baseTiling_.alignCorners = *alignCornersPtr ? 1 : 0;
    const float* scaleHPtr = attrs->GetAttrPointer<float>(CONST_3);
    OP_CHECK_IF(scaleHPtr == nullptr,
                ReportBicubicGradMissingArgument(context_->GetNodeName(), "scales_h", "required attribute is missing"),
                return ge::GRAPH_FAILED);
    float originalScaleH = *scaleHPtr;
    const float* scaleWPtr = attrs->GetAttrPointer<float>(CONST_4);
    OP_CHECK_IF(scaleWPtr == nullptr,
                ReportBicubicGradMissingArgument(context_->GetNodeName(), "scales_w", "required attribute is missing"),
                return ge::GRAPH_FAILED);
    float originalScaleW = *scaleWPtr;
    OP_LOGI(context_, "alignCorners %ld, originalScaleH %f, originalScaleW %f", baseTiling_.alignCorners,
            originalScaleH, originalScaleW);
    int64_t outSizeNum = outputSize->GetSize();
    OP_CHECK_IF(
        outSizeNum != CONST_2,
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "output_size", std::to_string(outSizeNum).c_str(), "2"),
        return ge::GRAPH_FAILED);
    int64_t inputSizeNum = inputSize->GetSize();
    OP_CHECK_IF(
        inputSizeNum != CONST_4,
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "input_size", std::to_string(inputSizeNum).c_str(), "4"),
        return ge::GRAPH_FAILED);

    const int64_t* inputData = reinterpret_cast<const int64_t*>(inputSize->GetData());
    OP_CHECK_IF(inputData == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "input_size has length 4 but no data"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((baseTiling_.outH != inputData[CONST_2]) || (baseTiling_.outW != inputData[CONST_3]),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "gradInput,input_size",
                    (std::to_string(baseTiling_.outH) + "," + std::to_string(baseTiling_.outW) + ";" +
                     std::to_string(inputData[CONST_2]) + "," + std::to_string(inputData[CONST_3]))
                        .c_str(),
                    "gradInput spatial dimensions must match input_size"),
                return ge::GRAPH_FAILED);
    const int64_t* outputData = reinterpret_cast<const int64_t*>(outputSize->GetData());
    OP_CHECK_IF(outputData == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "output_size has length 2 but no data"),
                return ge::GRAPH_FAILED);
    if ((baseTiling_.inH != outputData[CONST_0]) || (baseTiling_.inW != outputData[CONST_1])) {
        if (outputData[CONST_0] <= 0 || outputData[CONST_1] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "output_size",
                (std::to_string(outputData[CONST_0]) + "," + std::to_string(outputData[CONST_1])).c_str(),
                "elements must be positive");
        } else {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context_->GetNodeName(), "gradOutput,output_size",
                (std::to_string(baseTiling_.inH) + "," + std::to_string(baseTiling_.inW) + ";" +
                 std::to_string(outputData[CONST_0]) + "," + std::to_string(outputData[CONST_1]))
                    .c_str(),
                "gradOutput spatial dimensions must match output_size");
        }
        return ge::GRAPH_FAILED;
    }
    ComputeScalesSupportValues(originalScaleH, originalScaleW);
    return ge::GRAPH_SUCCESS;
}

void UpsampleBicubic2dAAGradRegbaseTiling::FillTilingData()
{
    tilingData_->blkProcessNum = baseTiling_.blkProcessNum;
    tilingData_->lenN = baseTiling_.dimN;
    tilingData_->lenC = baseTiling_.dimC;
    tilingData_->inH = baseTiling_.inH;
    tilingData_->inW = baseTiling_.inW;
    tilingData_->outH = baseTiling_.outH;
    tilingData_->outW = baseTiling_.outW;
    tilingData_->alignCorners = baseTiling_.alignCorners;
    tilingData_->tailBlockNum = baseTiling_.tailBlockNum;
    tilingData_->scaleH = baseTiling_.scaleH;
    tilingData_->scaleW = baseTiling_.scaleW;
    tilingData_->invScaleH = baseTiling_.invScaleH;
    tilingData_->invScaleW = baseTiling_.invScaleW;
    tilingData_->supportH = baseTiling_.supportH;
    tilingData_->supportW = baseTiling_.supportW;
    tilingData_->ubFactor = baseTiling_.ubFactor;
    tilingData_->perCoreInitEle = baseTiling_.perCoreInitEle;
    tilingData_->initHeadCoreNum = baseTiling_.initHeadCoreNum;
}

void UpsampleBicubic2dAAGradRegbaseTiling::PrintTilingData()
{
    OP_LOGD(context_,
            "blkProcessNum %ld, lenN %ld, lenC %ld, inH %ld, inW %ld, outH %ld, outW %ld, alignCorners %ld, "
            "ubFactor %d, tailBlockNum %d, scaleH %f, scaleW %f, invScaleH %f, invScaleW %f, supportH %f, supportW %f"
            "perCoreInitEle %d, initHeadCoreNum %d",
            tilingData_->blkProcessNum, tilingData_->lenN, tilingData_->lenC, tilingData_->inH, tilingData_->inW,
            tilingData_->outH, tilingData_->outW, tilingData_->alignCorners, tilingData_->ubFactor,
            tilingData_->tailBlockNum, tilingData_->scaleH, tilingData_->scaleW, tilingData_->invScaleH,
            tilingData_->invScaleW, tilingData_->supportH, tilingData_->supportW, tilingData_->perCoreInitEle,
            tilingData_->initHeadCoreNum);
}

ge::graphStatus UpsampleBicubic2dAAGradRegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "platform information is missing"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0,
                ReportBicubicGradTilingError(context_->GetNodeName(),
                                             "platform core count must be positive, got " + std::to_string(coreNum)),
                return ge::GRAPH_FAILED);
    baseTiling_.coreNum = coreNum;
    uint64_t ubMemSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubMemSize);
    OP_CHECK_IF(ubMemSize <= 0UL,
                ReportBicubicGradTilingError(context_->GetNodeName(),
                                             "platform UB size must be positive, got " + std::to_string(ubMemSize)),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_, "coreNum is %ld, ubMemSize is %lu", coreNum, ubMemSize);
    baseTiling_.ubSize = static_cast<int32_t>(ubMemSize);
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<UpsampleBicubic2dAAGradRegBaseTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr,
                    ReportBicubicGradTilingError(context_->GetNodeName(), "tiling data container is missing"),
                    return ge::GRAPH_FAILED);
    }
    baseTiling_.isDeterministic = context_->GetDeterministic();
    OP_CHECK_IF((memset_s(tilingData_, sizeof(UpsampleBicubic2dAAGradRegBaseTilingData), 0,
                          sizeof(UpsampleBicubic2dAAGradRegBaseTilingData)) != EOK),
                ReportBicubicGradTilingError(context_->GetNodeName(), "memset_s could not initialize tiling data"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBicubic2dAAGradRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context_, "CheckInputParams failed!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputShapes failed!"), return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();
    PrintTilingData();
    uint64_t schId = static_cast<uint64_t>(baseTiling_.schId);
    uint64_t isInt32 = static_cast<uint64_t>(baseTiling_.isInt32);
    uint64_t isDeterministic = static_cast<uint64_t>(baseTiling_.isDeterministic);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isInt32, isDeterministic);
    OP_LOGI(context_, "tilingKey %lu, schId %lu, isInt32 %lu, isDeterministic %lu, realCoreNum %d", tilingKey, schId,
            isInt32, isDeterministic, baseTiling_.realCoreNum);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTiling_.realCoreNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                ReportBicubicGradTilingError(context_->GetNodeName(), "workspace size container is missing"),
                return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;
    if (isDeterministic == 1 && schId == 0) {
        context_->SetScheduleMode(SCHEDULE_MODE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UpsampleBicubic2dAAGradRegbase(gert::TilingContext* context)
{
    UpsampleBicubic2dAAGradRegbaseTiling tilingImpl = UpsampleBicubic2dAAGradRegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "Tiling4UpsampleBicubic2dAAGradRegbase init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "Tiling4UpsampleBicubic2dAAGradRegbase do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "end Tiling4UpsampleBicubic2dAAGradRegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
