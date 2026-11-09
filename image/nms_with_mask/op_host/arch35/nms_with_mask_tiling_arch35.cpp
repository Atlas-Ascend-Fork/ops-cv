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
 * \file nms_with_mask_tiling_arch35.cpp
 * \brief nms_with_mask_tiling_arch35 impl info
 */

#include "nms_with_mask_tiling_arch35.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include <cmath>
#include <cstdint>
#include <string>
#include "log/log.h"
#include "op_api/error_report_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int32_t BASE_BLOCK_SIZE_FOR_MULTICORE = 256;
constexpr int32_t BIT_PER_BYTE = 8;
constexpr int32_t MAX_BLOCK_NUM = 39936;
constexpr int32_t SCHEDULE_MODE = 1;
constexpr int32_t ATTR_IOU_THR_INDEX = 0;
constexpr int32_t INDEX_ZERO = 0;
constexpr int32_t INDEX_ONE = 1;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t DIM_NUM_TWO = 2;
constexpr int32_t DIM_NUM_ONE = 1;
constexpr int32_t ELEMENT_NUM = 5;
// constexpr uint32_t NMS_WITH_MASK_TILINGKEY = 10000;
constexpr uint64_t TILING_KEY_FOR_MULTICORE = 10000UL;

template <typename Context>
const char* OpNameOf(Context* context)
{
    if (context == nullptr || context->GetNodeName() == nullptr) {
        return "NMSWithMask";
    }
    return context->GetNodeName();
}

ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    error_report::ReportTilingStageError(opName, reason);
    return ge::GRAPH_FAILED;
}
} // namespace

namespace optiling {
ge::graphStatus NMSWithMaskRegbaseTiling::CheckInputShape()
{
    const char* opName = OpNameOf(tilingContext_);
    auto inputShape = tilingContext_->GetInputShape(INDEX_ZERO);
    if (inputShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "box_scores");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shapeStorageScores = Ops::Cv::OpTiling::EnsureNotScalar(inputShape->GetStorageShape());
    // check input boxesScores
    if (shapeStorageScores.GetDimNum() != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "box_scores",
                                     (std::to_string(shapeStorageScores.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    boxesNum_ = shapeStorageScores.GetDim(INDEX_ZERO);
    if (boxesNum_ <= 0 || boxesNum_ > MAX_BLOCK_NUM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "box_scores", std::to_string(boxesNum_).c_str(),
                                                  "boxes num must be greater than 0 and less than or equal to 39936");
        return ge::GRAPH_FAILED;
    }
    if (shapeStorageScores.GetDim(INDEX_ONE) != ELEMENT_NUM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "box_scores",
                                                  std::to_string(shapeStorageScores.GetDim(INDEX_ONE)).c_str(),
                                                  "box_scores.shape[1] must be 5");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::CheckOutputShape()
{
    const char* opName = OpNameOf(tilingContext_);
    // check output selectedBoxes
    auto selectedBoxesOutput = tilingContext_->GetOutputShape(INDEX_ZERO);
    if (selectedBoxesOutput == nullptr) {
        return ReportTilingFailure(opName, "Output shape selected_boxes is unavailable.");
    }
    const gert::Shape& selectedBoxesShape = Ops::Cv::OpTiling::EnsureNotScalar(selectedBoxesOutput->GetStorageShape());
    if (selectedBoxesShape.GetDimNum() != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "selected_boxes",
                                     (std::to_string(selectedBoxesShape.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (selectedBoxesShape.GetDim(INDEX_ZERO) != boxesNum_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "selected_boxes",
                                                  std::to_string(selectedBoxesShape.GetDim(INDEX_ZERO)).c_str(),
                                                  "selected_boxes.shape[0] must be equal to box_scores.shape[0]");
        return ge::GRAPH_FAILED;
    }
    if (selectedBoxesShape.GetDim(INDEX_ONE) != ELEMENT_NUM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "selected_boxes",
                                                  std::to_string(selectedBoxesShape.GetDim(INDEX_ONE)).c_str(),
                                                  "selected_boxes.shape[1] must be 5");
        return ge::GRAPH_FAILED;
    }

    // check output selectedIndices
    auto selectedIdxOutput = tilingContext_->GetOutputShape(INDEX_ONE);
    if (selectedIdxOutput == nullptr) {
        return ReportTilingFailure(opName, "Output shape selected_idx is unavailable.");
    }
    const gert::Shape& selectedIdxShape = Ops::Cv::OpTiling::EnsureNotScalar(selectedIdxOutput->GetStorageShape());
    if (selectedIdxShape.GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "selected_idx",
                                     (std::to_string(selectedIdxShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (selectedIdxShape.GetDim(INDEX_ZERO) != boxesNum_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "selected_idx",
                                                  std::to_string(selectedIdxShape.GetDim(INDEX_ZERO)).c_str(),
                                                  "selected_idx.shape[0] must be equal to box_scores.shape[0]");
        return ge::GRAPH_FAILED;
    }

    // check output selectedMask
    auto selectedMaskOutput = tilingContext_->GetOutputShape(INDEX_TWO);
    if (selectedMaskOutput == nullptr) {
        return ReportTilingFailure(opName, "Output shape selected_mask is unavailable.");
    }
    const gert::Shape& selectedMaskShape = Ops::Cv::OpTiling::EnsureNotScalar(selectedMaskOutput->GetStorageShape());
    if (selectedMaskShape.GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "selected_mask",
                                     (std::to_string(selectedMaskShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (selectedMaskShape.GetDim(INDEX_ZERO) != boxesNum_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "selected_mask",
                                                  std::to_string(selectedMaskShape.GetDim(INDEX_ZERO)).c_str(),
                                                  "selected_mask.shape[0] must be equal to box_scores.shape[0]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::CheckShape()
{
    const char* opName = OpNameOf(tilingContext_);
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckShape");
    // check input shape
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckInputShape");
    OP_CHECK_IF(CheckInputShape() == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(opName, "CheckInputShape failed; retain the first error."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckInputShape success");

    // check output shape
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckOutputShape");
    OP_CHECK_IF(CheckOutputShape() == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(opName, "CheckOutputShape failed; retain the first error."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckOutputShape success");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::CheckDtype()
{
    const char* opName = OpNameOf(tilingContext_);
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckDtype");
    auto inputDesc = tilingContext_->GetInputDesc(INDEX_ZERO);
    if (inputDesc == nullptr) {
        return ReportTilingFailure(opName, "Input desc box_scores is unavailable; check Tiling context construction.");
    }
    this->inputDtype_ = inputDesc->GetDataType();
    if (this->inputDtype_ != ge::DT_BF16 && this->inputDtype_ != ge::DT_FLOAT16 && this->inputDtype_ != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "box_scores", Ops::Base::ToString(this->inputDtype_).c_str(),
                                  "DT_FLOAT, DT_FLOAT16, DT_BF16");
        return ge::GRAPH_FAILED;
    }
    auto selectedBoxesDesc = tilingContext_->GetOutputDesc(INDEX_ZERO);
    if (selectedBoxesDesc == nullptr) {
        return ReportTilingFailure(opName, "Output desc selected_boxes is unavailable.");
    }
    auto selectedBoxesDtype = selectedBoxesDesc->GetDataType();
    if (selectedBoxesDtype != this->inputDtype_) {
        const std::string dtypes = Ops::Base::ToString(this->inputDtype_) + " and " +
                                   Ops::Base::ToString(selectedBoxesDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "box_scores and selected_boxes", dtypes.c_str(),
                                               "box_scores and selected_boxes must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    auto selectedIdxDesc = tilingContext_->GetOutputDesc(INDEX_ONE);
    if (selectedIdxDesc == nullptr) {
        return ReportTilingFailure(opName, "Output desc selected_idx is unavailable.");
    }
    auto selectedIdxDtype = selectedIdxDesc->GetDataType();
    if (selectedIdxDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "selected_idx", Ops::Base::ToString(selectedIdxDtype).c_str(), "DT_INT32");
        return ge::GRAPH_FAILED;
    }
    auto selectedMaskDesc = tilingContext_->GetOutputDesc(INDEX_TWO);
    if (selectedMaskDesc == nullptr) {
        return ReportTilingFailure(opName, "Output desc selected_mask is unavailable.");
    }
    auto selectedMaskDtype = selectedMaskDesc->GetDataType();
    if (selectedMaskDtype != ge::DT_UINT8) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "selected_mask", Ops::Base::ToString(selectedMaskDtype).c_str(), "DT_UINT8");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::SetTilingData()
{
    const char* opName = OpNameOf(tilingContext_);
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::SetTilingData");
    auto compileInfo = static_cast<const NMSWithMaskCompileInfo*>(tilingContext_->GetCompileInfo());
    if (compileInfo == nullptr) {
        return ReportTilingFailure(opName, "compileInfo is null in the tiling context.");
    }
    auto vectorCoreNum = compileInfo->coreNum;
    tilingData_ = tilingContext_->GetTilingData<NMSWithMaskTilingData>();
    if (tilingData_ == nullptr) {
        return ReportTilingFailure(opName, "tilingData is null in the tiling context.");
    }
    auto attrs = tilingContext_->GetAttrs();
    if (attrs == nullptr) {
        return ReportTilingFailure(opName, "Runtime attrs are unavailable; check the TilingContext construction.");
    }
    auto iouThrPtr = attrs->GetAttrPointer<float>(ATTR_IOU_THR_INDEX);
    if (iouThrPtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "iou_threshold", "nullptr",
                                              "iou_threshold is required but not provided");
        return ge::GRAPH_FAILED;
    }
    iouThreshold_ = *iouThrPtr;
    if (!std::isfinite(iouThreshold_) || iouThreshold_ < 0.0f || iouThreshold_ > 1.0f) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "iou_threshold", std::to_string(iouThreshold_).c_str(),
                                              "iou_threshold must be a finite value within [0, 1]");
        return ge::GRAPH_FAILED;
    }
    groupSize_ = BASE_BLOCK_SIZE_FOR_MULTICORE;
    groupNum_ = (boxesNum_ + groupSize_ - 1) / groupSize_;
    blockNum_ = groupNum_ * (groupNum_ + 1) / 2;
    usedCoreNum_ = blockNum_ > vectorCoreNum ? vectorCoreNum : blockNum_;
    headCoreNum_ = blockNum_ % usedCoreNum_;
    headCoreNum_ = headCoreNum_ == 0 ? usedCoreNum_ : headCoreNum_;
    blockPerHead_ = (blockNum_ + usedCoreNum_ - 1) / usedCoreNum_;
    OP_LOGD(tilingContext_, "set tiling data and workspace size");
    tilingContext_->SetTilingKey(TILING_KEY_FOR_MULTICORE);
    tilingData_->boxesNum = boxesNum_;
    tilingData_->iouThreshold = iouThreshold_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->groupSize = groupSize_;
    tilingData_->groupNum = groupNum_;
    tilingData_->blockNum = blockNum_;
    tilingData_->headCoreNum = headCoreNum_;
    tilingData_->blockPerHead = blockPerHead_;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaceSize = tilingContext_->GetWorkspaceSizes(1);
    if (workspaceSize == nullptr) {
        return ReportTilingFailure(opName, "workspaceSize is null in the tiling context.");
    }
    workspaceSize[0] = sysWorkspaceSize;
    uint32_t workspacePerBlock = static_cast<uint32_t>(groupSize_) * static_cast<uint32_t>(groupSize_) /
                                 static_cast<uint32_t>(BIT_PER_BYTE);
    workspaceSize[0] += blockNum_ * workspacePerBlock;
    OP_LOGD(tilingContext_, "current tiling key is:%lu", TILING_KEY_FOR_MULTICORE);
    auto rawTilingData = tilingContext_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        return ReportTilingFailure(opName, "rawTilingData is null in the tiling context.");
    }
    size_t tilingDataSize = sizeof(NMSWithMaskTilingData);
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), static_cast<void*>(tilingData_),
                           tilingDataSize);
    if (ret != EOK) {
        return ReportTilingFailure(opName, "Save tiling data to buffer failed.");
    }
    rawTilingData->SetDataSize(tilingDataSize);
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingContext_->SetScheduleMode(SCHEDULE_MODE);
    OP_LOGD(tilingContext_, "SetTilingData success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::RunTiling()
{
    const char* opName = OpNameOf(tilingContext_);
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::RunTiling");
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(opName, "CheckShape failed; retain the first error."), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckShape success");

    OP_CHECK_IF(CheckDtype() == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(opName, "CheckDtype failed; retain the first error."), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckDtype success");

    return SetTilingData();
}

static ge::graphStatus TilingPrepare4NMSWithMask(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForNMSWithMask running");
    const char* opName = OpNameOf(context);
    auto compileInfo = context->GetCompiledInfo<NMSWithMaskCompileInfo>();
    if (compileInfo == nullptr) {
        return ReportTilingFailure(
            opName, "compileInfo is null in the tiling parse context; check the TilingParseContext construction.");
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ReportTilingFailure(opName, "platformInfo is null in the tiling parse context.");
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->maxBoxesNum = MAX_BLOCK_NUM;
    if (compileInfo->coreNum <= 0 || compileInfo->ubSize <= 0) {
        const std::string reason = "vectorCoreNum=" + std::to_string(compileInfo->coreNum) +
                                   ", ubSize=" + std::to_string(compileInfo->ubSize) +
                                   "; platform resources are unavailable or invalid";
        return ReportTilingFailure(opName, reason);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4NMSWithMask(gert::TilingContext* context)
{
    OP_LOGD(context, "Entering Tiling4NMSWithMask");
    if (context == nullptr) {
        return ReportTilingFailure("NMSWithMask", "tiling context is nullptr.");
    }
    auto compileInfo = static_cast<const NMSWithMaskCompileInfo*>(context->GetCompileInfo());
    if (compileInfo == nullptr) {
        return ReportTilingFailure(OpNameOf(context), "compileInfo is null in the tiling context.");
    }
    OP_LOGD(context, "Entering Tiling4NMSWithMaskRegbase");
    NMSWithMaskRegbaseTiling opTiling(context);
    return opTiling.RunTiling();
}

IMPL_OP_OPTILING(NMSWithMask).Tiling(Tiling4NMSWithMask).TilingParse<NMSWithMaskCompileInfo>(TilingPrepare4NMSWithMask);
} // namespace optiling
