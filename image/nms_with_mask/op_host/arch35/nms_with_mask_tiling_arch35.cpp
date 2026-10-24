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
#include <graph/utils/type_utils.h>
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
} // namespace

namespace optiling {
ge::graphStatus NMSWithMaskRegbaseTiling::CheckInputShape()
{
    auto inputShape = tilingContext_->GetInputShape(INDEX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    const gert::Shape& shapeStorageScores = Ops::Cv::OpTiling::EnsureNotScalar(inputShape->GetStorageShape());
    // check input boxesScores
    OP_CHECK_IF(shapeStorageScores.GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(tilingContext_, "Input box_scores' shape only supports 2-D, got dim num:%lu.",
                        shapeStorageScores.GetDimNum()),
                return ge::GRAPH_FAILED);
    boxesNum_ = shapeStorageScores.GetDim(INDEX_ZERO);
    OP_CHECK_IF(
        boxesNum_ <= 0 || boxesNum_ > MAX_BLOCK_NUM,
        OP_LOGE(tilingContext_, "Input boxes num must be greater than 0 and less than or equal to 39936, got :%lld.",
                boxesNum_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(shapeStorageScores.GetDim(INDEX_ONE) != ELEMENT_NUM,
                OP_LOGE(tilingContext_, "Input box_scores' second dim must be 5, got :%lu.",
                        shapeStorageScores.GetDim(INDEX_ONE)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::CheckOutputShape()
{
    // check output selectedBoxes
    auto selectedBoxesOutput = tilingContext_->GetOutputShape(INDEX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, selectedBoxesOutput);
    const gert::Shape& selectedBoxesShape = Ops::Cv::OpTiling::EnsureNotScalar(selectedBoxesOutput->GetStorageShape());
    OP_CHECK_IF(selectedBoxesShape.GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(tilingContext_, "Output selected_boxes' shape only supports 2-D, got dim num:%lu.",
                        selectedBoxesShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        selectedBoxesShape.GetDim(INDEX_ZERO) != boxesNum_,
        OP_LOGE(tilingContext_, "Output selected_boxes' first dim must be equal to box_scores' first dim, got :%lu.",
                selectedBoxesShape.GetDim(INDEX_ZERO)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBoxesShape.GetDim(INDEX_ONE) != ELEMENT_NUM,
                OP_LOGE(tilingContext_, "Output selected_boxes' second dim must be 5, got :%lu.",
                        selectedBoxesShape.GetDim(INDEX_ONE)),
                return ge::GRAPH_FAILED);

    // check output selectedIndices
    auto selectedIdxOutput = tilingContext_->GetOutputShape(INDEX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, selectedIdxOutput);
    const gert::Shape& selectedIdxShape = Ops::Cv::OpTiling::EnsureNotScalar(selectedIdxOutput->GetStorageShape());
    OP_CHECK_IF(selectedIdxShape.GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(tilingContext_, "Output selected_idx' shape only supports 1-D, got dim num:%lu.",
                        selectedIdxShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        selectedIdxShape.GetDim(INDEX_ZERO) != boxesNum_,
        OP_LOGE(tilingContext_, "Output selected_idx' first dim must be equal to box_scores' first dim, got :%lu.",
                selectedIdxShape.GetDim(INDEX_ZERO)),
        return ge::GRAPH_FAILED);

    // check output selectedMask
    auto selectedMaskOutput = tilingContext_->GetOutputShape(INDEX_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, selectedMaskOutput);
    const gert::Shape& selectedMaskShape = Ops::Cv::OpTiling::EnsureNotScalar(selectedMaskOutput->GetStorageShape());
    OP_CHECK_IF(selectedMaskShape.GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(tilingContext_, "Output selected_mask' shape only supports 1-D, got dim num:%lu.",
                        selectedMaskShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        selectedMaskShape.GetDim(INDEX_ZERO) != boxesNum_,
        OP_LOGE(tilingContext_, "Output selected_mask' first dim must be equal to box_scores' first dim, got :%lu.",
                selectedMaskShape.GetDim(INDEX_ZERO)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::CheckShape()
{
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckShape");
    // check input shape
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckInputShape");
    OP_CHECK_IF(CheckInputShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext_, "CheckInputShape failed."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckInputShape success");

    // check output shape
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckOutputShape");
    OP_CHECK_IF(CheckOutputShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext_, "CheckOutputShape failed."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckOutputShape success");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::CheckDtype()
{
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::CheckDtype");
    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    this->inputDtype_ = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype_ != ge::DT_BF16 && this->inputDtype_ != ge::DT_FLOAT16 && this->inputDtype_ != ge::DT_FLOAT,
        OP_LOGE(tilingContext_,
                "Input box_scores dtype is not supported, only [DT_FLOAT, DT_FLOAT16, DT_BF16] are supported, got %s",
                ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str()),
        return ge::GRAPH_FAILED);
    auto selectedBoxesDesc = tilingContext_->GetOutputDesc(INDEX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, selectedBoxesDesc);
    auto selectedBoxesDtype = selectedBoxesDesc->GetDataType();
    OP_CHECK_IF(
        selectedBoxesDtype != this->inputDtype_,
        OP_LOGE(tilingContext_, "Input box_scores' dtype[%s] and output selected_boxes' dtype[%s] should be the same.",
                ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str(),
                ge::TypeUtils::DataTypeToSerialString(selectedBoxesDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto selectedIdxDesc = tilingContext_->GetOutputDesc(INDEX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, selectedIdxDesc);
    auto selectedIdxDtype = selectedIdxDesc->GetDataType();
    OP_CHECK_IF(
        selectedIdxDtype != ge::DT_INT32,
        OP_LOGE(tilingContext_, "Output selected_idx dtype is not supported, only [DT_INT32] is supported, got %s",
                ge::TypeUtils::DataTypeToSerialString(selectedIdxDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto selectedMaskDesc = tilingContext_->GetOutputDesc(INDEX_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, selectedMaskDesc);
    auto selectedMaskDtype = selectedMaskDesc->GetDataType();
    OP_CHECK_IF(
        selectedMaskDtype != ge::DT_UINT8,
        OP_LOGE(tilingContext_, "Output selected_mask dtype is not supported, only [DT_UINT8] is supported, got %s",
                ge::TypeUtils::DataTypeToSerialString(selectedMaskDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::SetTilingData()
{
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::SetTilingData");
    auto compileInfo = static_cast<const NMSWithMaskCompileInfo*>(tilingContext_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    auto vectorCoreNum = compileInfo->coreNum;
    tilingData_ = tilingContext_->GetTilingData<NMSWithMaskTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tilingData_);
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    auto iouThrPtr = attrs->GetAttrPointer<float>(ATTR_IOU_THR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, iouThrPtr);
    iouThreshold_ = *iouThrPtr;
    if (!std::isfinite(iouThreshold_) || iouThreshold_ < 0.0f || iouThreshold_ > 1.0f) {
        OP_LOGE(tilingContext_, "iou_threshold_ must be in [0, 1], got %f", iouThreshold_);
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
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaceSize);
    workspaceSize[0] = sysWorkspaceSize;
    uint32_t workspacePerBlock = static_cast<uint32_t>(groupSize_) * static_cast<uint32_t>(groupSize_) /
                                 static_cast<uint32_t>(BIT_PER_BYTE);
    workspaceSize[0] += blockNum_ * workspacePerBlock;
    OP_LOGD(tilingContext_, "current tiling key is:%lu", TILING_KEY_FOR_MULTICORE);
    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);
    size_t tilingDataSize = sizeof(NMSWithMaskTilingData);
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), static_cast<void*>(tilingData_),
                           tilingDataSize);
    OP_CHECK_IF(ret != EOK, OP_LOGE(tilingContext_, "Save tiling data to buffer failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataSize);
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingContext_->SetScheduleMode(SCHEDULE_MODE);
    OP_LOGD(tilingContext_, "SetTilingData success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NMSWithMaskRegbaseTiling::RunTiling()
{
    OP_LOGD(tilingContext_, "Entering NMSWithMaskRegbaseTiling::RunTiling");
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext_, "CheckShape failed."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckShape success");

    OP_CHECK_IF(CheckDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext_, "CheckDtype failed."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_, "CheckDtype success");

    return SetTilingData();
}

static ge::graphStatus TilingPrepare4NMSWithMask(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForNMSWithMask running");
    auto compileInfo = context->GetCompiledInfo<NMSWithMaskCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->maxBoxesNum = MAX_BLOCK_NUM;
    OP_CHECK_IF((compileInfo->coreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context, "NMSWithMask GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%lu.",
                        compileInfo->coreNum, ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4NMSWithMask(gert::TilingContext* context)
{
    OP_LOGD(context, "Entering Tiling4NMSWithMask");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "tiling context is nullptr"), return ge::GRAPH_FAILED);
    auto compileInfo = static_cast<const NMSWithMaskCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Entering Tiling4NMSWithMaskRegbase");
    NMSWithMaskRegbaseTiling opTiling(context);
    return opTiling.RunTiling();
}

IMPL_OP_OPTILING(NMSWithMask).Tiling(Tiling4NMSWithMask).TilingParse<NMSWithMaskCompileInfo>(TilingPrepare4NMSWithMask);
} // namespace optiling
