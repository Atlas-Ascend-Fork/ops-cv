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
 * \file roi_pooling_grad_with_arg_max_tiling_arch35.cpp
 * \brief
 */

#include <string>

#include "register/op_def_registry.h"
#include "log/log.h"
#include "op_api/error_report_util.h"
#include "roi_pooling_grad_with_arg_max_tiling_arch35.h"
#include "../../op_kernel/arch35/roi_pooling_grad_with_arg_max_tiling_key.h"
#include "util/math_util.h"

namespace optiling {
constexpr int32_t ATTR_0 = 0;
constexpr int32_t ATTR_1 = 1;
constexpr int32_t ATTR_2 = 2;
constexpr int32_t ATTR_3 = 3;
constexpr int32_t ATTR_4 = 4;
constexpr uint64_t TYPE_MODE1 = 0; // 为fp32
constexpr uint64_t TYPE_MODE2 = 1; // 为fp16
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t INPUT_GRADOUT_IDX = 0;
constexpr uint32_t INPUT_X_IDX = 1;
constexpr uint32_t DIM_NUM_4D = 4;
constexpr uint32_t N_DIM = 0;
constexpr uint32_t C_DIM = 1;
constexpr uint32_t H_DIM = 2;
constexpr uint32_t W_DIM = 3;
constexpr uint64_t NUM_ZERO = 0;
constexpr uint32_t FP32_TYPESIZE = 4;

void RoiPoolingGradWithArgMaxTiling::SetTilingData()
{
    yTotalCoreNum_ = static_cast<int64_t>(
        Ops::Base::CeilDiv(yTotalLength_, Ops::Base::CeilDiv(yTotalLength_, totalCoreNum_)));
    yDataPerCore_ = Ops::Base::CeilDiv(yTotalLength_, totalCoreNum_);
    yDataTailCore_ = yTotalLength_ - (yTotalCoreNum_ - 1) * yDataPerCore_;
    useCoreNum_ = std::max(
        static_cast<int64_t>(Ops::Base::CeilDiv(totalLength_, Ops::Base::CeilDiv(totalLength_, totalCoreNum_))),
        yTotalCoreNum_);
    context_->SetBlockDim(useCoreNum_);
    tilingData_->totalLength = totalLength_;
    tilingData_->yTotalCoreNum = yTotalCoreNum_;
    tilingData_->yDataPerCore = yDataPerCore_;
    tilingData_->yDataTailCore = yDataTailCore_;
    tilingData_->yTotalLength = yTotalLength_;
    tilingData_->pooledH = pooledH_;
    tilingData_->pooledW = pooledW_;
    tilingData_->height = height_;
    tilingData_->width = width_;
    tilingData_->poolChannel = poolChannel_;
    tilingData_->useCoreNum = useCoreNum_;
    tilingData_->batch = batch_;
}

ge::graphStatus RoiPoolingGradWithArgMaxTiling::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const auto pooledHPtr = attrs->GetAttrPointer<int64_t>(ATTR_0);
    OP_CHECK_IF(
        pooledHPtr == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "pooledHPtr is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    pooledH_ = static_cast<int64_t>(*pooledHPtr);
    const auto pooledWPtr = attrs->GetAttrPointer<int64_t>(ATTR_1);
    OP_CHECK_IF(
        pooledWPtr == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "pooledWPtr is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    pooledW_ = static_cast<int64_t>(*pooledWPtr);
    const auto spatialScaleHPtr = attrs->GetAttrPointer<float>(ATTR_2);
    OP_CHECK_IF(spatialScaleHPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "spatialScaleHPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    spatialScaleH_ = static_cast<float>(*spatialScaleHPtr);
    const auto spatialScaleWPtr = attrs->GetAttrPointer<float>(ATTR_3);
    OP_CHECK_IF(spatialScaleWPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "spatialScaleWPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    spatialScaleW_ = static_cast<float>(*spatialScaleWPtr);
    const auto poolChannelPtr = attrs->GetAttrPointer<int64_t>(ATTR_4);
    OP_CHECK_IF(poolChannelPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "poolChannelPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    poolChannel_ = static_cast<int64_t>(*poolChannelPtr);
    OP_CHECK_IF(
        pooledH_ <= 0 || pooledW_ <= 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "pooled_h,pooled_w",
                                               (std::to_string(pooledH_) + "," + std::to_string(pooledW_)).c_str(),
                                               "pooled_h and pooled_w must be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(poolChannel_ <= NUM_ZERO,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "pool_channel",
                                                      std::to_string(poolChannel_).c_str(),
                                                      "pool_channel must be greater than 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoiPoolingGradWithArgMaxTiling::GetPlatformInfo()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "platformInfo is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        totalCoreNum_ <= NUM_ZERO,
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(context_->GetNodeName(), "platform", "coreNum",
                                               std::to_string(totalCoreNum_).c_str(), "Failed to get core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= NUM_ZERO,
                OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(context_->GetNodeName(), "platform", "ubSize",
                                                       std::to_string(ubSize).c_str(), "ubSize must be greater than 0"),
                return ge::GRAPH_FAILED);
    auto localMemorySize = context_->SetLocalMemorySize(ubSize - DCACHE_SIZE);
    OP_LOGD(context_, "ubSize = %lu, localMemorySize = %d.", ubSize, localMemorySize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoiPoolingGradWithArgMaxTiling::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<RoiPoolingGradWithArgMaxRegBaseTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr,
                    error_report::ReportTilingStageError(context_->GetNodeName(), "get tilingdata ptr failed"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(RoiPoolingGradWithArgMaxRegBaseTilingData), 0,
                          sizeof(RoiPoolingGradWithArgMaxRegBaseTilingData)) != EOK),
                error_report::ReportTilingStageError(context_->GetNodeName(), "memset tilingdata failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool RoiPoolingGradWithArgMaxTiling::GetDataTypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dataTypeTilingKey_ = TYPE_MODE2;
            break;
        case ge::DT_FLOAT:
            dataTypeTilingKey_ = TYPE_MODE1;
            break;
        default:
            return false;
    }

    return true;
}

void RoiPoolingGradWithArgMaxTiling::SetTilingKey()
{
    OP_LOGI(context_, "dtype is %lu", dataTypeTilingKey_);
    tilingKey_ = GET_TPL_TILING_KEY(dataTypeTilingKey_);
}

uint64_t RoiPoolingGradWithArgMaxTiling::GetTilingKey() { return tilingKey_; }

ge::graphStatus RoiPoolingGradWithArgMaxTiling::GetInputTensorInfo()
{
    auto gradOut = context_->GetDynamicInputTensor(INPUT_GRADOUT_IDX, 0);
    OP_CHECK_IF(gradOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "grad"),
                return ge::GRAPH_FAILED);
    auto& gradOutShape = gradOut->GetOriginShape();
    auto gradOutDimNum = gradOutShape.GetDimNum();
    OP_CHECK_IF(
        gradOutDimNum != DIM_NUM_4D,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad", std::to_string(gradOutDimNum).c_str(), "4"),
        return ge::GRAPH_FAILED);

    // 获取所需处理的输入数据量totalLength、输出数据量yTotalLength以及height、width
    totalLength_ = 1;
    totalLength_ *= gradOutShape[N_DIM];
    totalLength_ *= gradOutShape[C_DIM];
    totalLength_ *= gradOutShape[H_DIM];
    totalLength_ *= gradOutShape[W_DIM];
    OP_CHECK_IF(
        totalLength_ > INT32_MAX || totalLength_ <= NUM_ZERO,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "grad", std::to_string(totalLength_).c_str(),
                                                  "grad.size must be less or equal than 2^31-1."),
        return ge::GRAPH_FAILED);

    auto inputX = context_->GetDynamicInputTensor(INPUT_X_IDX, 0);
    OP_CHECK_IF(inputX == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    auto& inputXShape = inputX->GetOriginShape();
    auto inputXDimNum = inputXShape.GetDimNum();
    OP_CHECK_IF(inputXDimNum != DIM_NUM_4D,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(inputXDimNum).c_str(), "4"),
                return ge::GRAPH_FAILED);
    height_ = inputXShape[H_DIM];
    width_ = inputXShape[W_DIM];
    batch_ = inputXShape[N_DIM];
    OP_CHECK_IF(
        batch_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x.shape[0]", std::to_string(batch_).c_str(),
                                              "x.shape[0] must be greater than 0"),
        return ge::GRAPH_FAILED);
    usrWorkspaceSize_ *= inputXShape[N_DIM];
    usrWorkspaceSize_ *= inputXShape[C_DIM];
    usrWorkspaceSize_ *= inputXShape[H_DIM];
    usrWorkspaceSize_ *= inputXShape[W_DIM];
    yTotalLength_ = usrWorkspaceSize_;
    OP_CHECK_IF(
        yTotalLength_ > INT32_MAX || yTotalLength_ <= NUM_ZERO,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(yTotalLength_).c_str(),
                                                  "x.size must be less or equal than 2^31-1."),
        return ge::GRAPH_FAILED);
    usrWorkspaceSize_ *= FP32_TYPESIZE;

    auto gradOutDesc = context_->GetInputDesc(INPUT_GRADOUT_IDX);
    OP_CHECK_IF(
        gradOutDesc == nullptr,
        error_report::ReportInvalidArgument(context_->GetNodeName(), "grad", "grad desc is null in the tiling context"),
        return ge::GRAPH_FAILED);
    gardInDType_ = gradOutDesc->GetDataType();
    OP_CHECK_IF(GetDataTypeKey(gardInDType_) == false,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad", Ops::Base::ToString(gardInDType_).c_str(),
                                          "float32 or float16"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RoiPoolingGradWithArgMaxTiling::DoTiling()
{
    OP_LOGD(context_, "Enter RoiPoolingGradWithArgMaxRegBaseTilingData DoTiling");

    OP_CHECK_IF(SetAttrParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "SetAttrParams failed; propagate the first error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputTensorInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "GetInputTensorInfo failed; propagate the first error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "GetPlatformInfo failed; propagate the first error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "InitTilingData failed; propagate the first error"),
                return ge::GRAPH_FAILED);
    SetTilingData();

    SetTilingKey();
    context_->SetTilingKey(GetTilingKey());

    OP_LOGI(context_, "tiling data:%s.", tilingData_->toString().c_str());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "workspaceSizes is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    currentWorkspace
        [0] = usrWorkspaceSize_ +
              ascendcPlatform
                  .GetLibApiWorkSpaceSize(); // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。

    OP_LOGD(context_, "End dotiling");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForRoiPoolingGradWithArgMax(gert::TilingContext* context)
{
    OP_LOGD(context, "TilingForRoiPoolingGradWithArgMax running begin.");

    RoiPoolingGradWithArgMaxTiling tilingObj(context);

    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepareForRoiPoolingGradWithArgMax(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RoiPoolingGradWithArgMax)
    .Tiling(TilingForRoiPoolingGradWithArgMax)
    .TilingParse<RoiPoolingGradWithArgMaxCompileInfo>(TilingPrepareForRoiPoolingGradWithArgMax);
} // namespace optiling
