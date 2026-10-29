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
 * \file resize_bicubic_v2_grad_base_tiling_arch35.cpp
 * \brief resize_bicubic_v2_grad_base_tiling_arch35
 */
#include "resize_bicubic_v2_grad_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "util/platform_util.h"
#include "op_api/error_report_util.h"

namespace optiling {

constexpr size_t NUM_0 = 0;
constexpr size_t NUM_1 = 1;
constexpr size_t NUM_2 = 2;
constexpr size_t NUM_3 = 3;
constexpr size_t NUM_4 = 4;
constexpr int64_t RSV_BLOCK_NUM = 8;

bool ResizeBicubicV2GradBaseTiling::IsCapable() { return true; }

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo_.coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compileInfo_.ubSize = ubSize;
    } else {
        auto compileInfo = static_cast<const ResizeBicubicV2GradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr,
                    error_report::ReportTilingStageError(
                        context_->GetNodeName(),
                        "compileInfo is null in the tiling context; check the TilingContext construction"),
                    return ge::GRAPH_FAILED);
        compileInfo_.coreNum = compileInfo->coreNum;
        compileInfo_.ubSize = compileInfo->ubSize;
    }

    compileInfo_.ubBlockSize = Ops::Base::GetUbBlockSize(context_);

    OP_CHECK_IF(compileInfo_.coreNum <= 0,
                error_report::ReportTilingStageError(context_->GetNodeName(),
                                                     "failed to get the core num from the platform info"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo_.ubSize <= 0,
                error_report::ReportTilingStageError(context_->GetNodeName(),
                                                     "failed to get the ub size from the platform info"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo_.ubBlockSize <= 0,
                error_report::ReportTilingStageError(context_->GetNodeName(),
                                                     "failed to get the ub block size from the platform info"),
                return ge::GRAPH_FAILED);

    compileInfo_.isDetermine = context_->GetDeterministic() == 1 ? 1 : 0;

    calcInfo_.ubBlockNum = Ops::Base::CeilDiv(compileInfo_.ubSize, compileInfo_.ubBlockSize) - RSV_BLOCK_NUM;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetTensorInfo()
{
    // Shape 与 format 同源：均取运行时(storage)语义，避免 GE 转排布后按 NHWC 规则解析
    // NCHW 轴序 origin shape 导致维度取错轴（详见 resize_bilinear_v2_grad 同类修复）。
    auto gradsShapePtr = context_->GetInputShape(NUM_0);
    OP_CHECK_IF(gradsShapePtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "gradsShapePtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    inputInfo_.gradsShape = Ops::Cv::OpTiling::EnsureNotScalar(gradsShapePtr->GetStorageShape());
    auto gradsDescPtr = context_->GetInputDesc(NUM_0);
    OP_CHECK_IF(gradsDescPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "gradsDescPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    inputInfo_.gradsDtype = gradsDescPtr->GetDataType();
    inputInfo_.gradsFormat = gradsDescPtr->GetFormat().GetStorageFormat();

    auto originalImageShapePtr = context_->GetInputShape(NUM_1);
    OP_CHECK_IF(originalImageShapePtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "originalImageShapePtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    inputInfo_.originalImageShape = Ops::Cv::OpTiling::EnsureNotScalar(originalImageShapePtr->GetStorageShape());
    auto originalImageDescPtr = context_->GetInputDesc(NUM_1);
    OP_CHECK_IF(originalImageDescPtr == nullptr,
                error_report::ReportTilingStageError(
                    context_->GetNodeName(),
                    "originalImageDescPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    inputInfo_.imageDtype = originalImageDescPtr->GetDataType();
    inputInfo_.imageFormat = originalImageDescPtr->GetFormat().GetStorageFormat();

    auto yShapePtr = context_->GetOutputShape(NUM_0);
    OP_CHECK_IF(
        yShapePtr == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "yShapePtr is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    inputInfo_.yShape = Ops::Cv::OpTiling::EnsureNotScalar(yShapePtr->GetStorageShape());
    auto yDescPtr = context_->GetOutputDesc(NUM_0);
    OP_CHECK_IF(
        yDescPtr == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "yDescPtr is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    inputInfo_.yDtype = yDescPtr->GetDataType();
    inputInfo_.yFormat = yDescPtr->GetFormat().GetStorageFormat();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckDtypeValid()
{
    OP_CHECK_IF(
        inputInfo_.gradsDtype != ge::DT_FLOAT && inputInfo_.gradsDtype != ge::DT_FLOAT16 &&
            inputInfo_.gradsDtype != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grads", Ops::Base::ToString(inputInfo_.gradsDtype).c_str(),
                                  "FLOAT, FLOAT16 and BFLOAT16"),
        return ge::GRAPH_FAILED);

    if (inputInfo_.imageDtype != inputInfo_.gradsDtype || inputInfo_.yDtype != inputInfo_.gradsDtype) {
        std::string dtypeMsg = Ops::Base::ToString(inputInfo_.gradsDtype) + ", " +
                               Ops::Base::ToString(inputInfo_.imageDtype) + " and " +
                               Ops::Base::ToString(inputInfo_.yDtype);
        std::string reasonMsg = "The dtypes of input grads, original_image and output y must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "grads, original_image and y", dtypeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    calcInfo_.gradsDtypeSize = GetSizeByDataType(inputInfo_.gradsDtype);
    calcInfo_.yDtypeSize = GetSizeByDataType(inputInfo_.yDtype);
    if (calcInfo_.gradsDtypeSize <= 0 || calcInfo_.yDtypeSize <= 0) {
        std::string dtypeMsg = Ops::Base::ToString(inputInfo_.gradsDtype) + " and " +
                               Ops::Base::ToString(inputInfo_.yDtype);
        std::string reasonMsg = "The dtype sizes of input grads and output y must be greater than 0";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "grads and y", dtypeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckFormatValid()
{
    OP_CHECK_IF(inputInfo_.gradsFormat != ge::FORMAT_NCHW && inputInfo_.gradsFormat != ge::FORMAT_NHWC &&
                    inputInfo_.gradsFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "grads",
                                           Ops::Base::ToString(inputInfo_.gradsFormat).c_str(), "NCHW, NHWC and ND"),
                return ge::GRAPH_FAILED);

    if (inputInfo_.imageFormat != inputInfo_.gradsFormat || inputInfo_.imageFormat != inputInfo_.yFormat) {
        std::string formatMsg = Ops::Base::ToString(inputInfo_.gradsFormat) + ", " +
                                Ops::Base::ToString(inputInfo_.imageFormat) + " and " +
                                Ops::Base::ToString(inputInfo_.yFormat);
        std::string reasonMsg = "The formats of input grads, original_image and output y must be the same";
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "grads, original_image and y",
                                                formatMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    inputInfo_.format = inputInfo_.yFormat == ge::FORMAT_NHWC ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckShapeDimValid()
{
    if (inputInfo_.gradsShape.GetDimNum() != NUM_4 || inputInfo_.yShape.GetDimNum() != NUM_4) {
        std::string dimMsg = std::to_string(inputInfo_.gradsShape.GetDimNum()) + " and " +
                             std::to_string(inputInfo_.yShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "grads and y", dimMsg.c_str(),
                                                  "The shapes of grads and y must be 4D");
        return ge::GRAPH_FAILED;
    }

    if (inputInfo_.originalImageShape != inputInfo_.yShape) {
        std::string shapeMsg = Ops::Base::ToString(inputInfo_.originalImageShape) + " and " +
                               Ops::Base::ToString(inputInfo_.yShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "original_image and y", shapeMsg.c_str(),
                                               "The shapes of original_image and y must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckAxesValid()
{
    if (inputInfo_.gradsShape.GetDim(NUM_0) != inputInfo_.yShape.GetDim(NUM_0)) {
        std::string shapeMsg = Ops::Base::ToString(inputInfo_.gradsShape) + " and " +
                               Ops::Base::ToString(inputInfo_.yShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "grads and y", shapeMsg.c_str(),
            "The N-dimension of input grads and output y must be the same, where N is the 0th axis of them");
        return ge::GRAPH_FAILED;
    }
    inputInfo_.lenN = inputInfo_.gradsShape.GetDim(NUM_0);

    bool formatIsNHWC = false;
    if (inputInfo_.gradsFormat == ge::FORMAT_NCHW || inputInfo_.gradsFormat == ge::FORMAT_ND) {
        if (inputInfo_.gradsShape.GetDim(NUM_1) != inputInfo_.yShape.GetDim(NUM_1)) {
            std::string shapeMsg = Ops::Base::ToString(inputInfo_.gradsShape) + " and " +
                                   Ops::Base::ToString(inputInfo_.yShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grads and y", shapeMsg.c_str(),
                                                   "The C-dimension of input grads and output y must be the same, "
                                                   "where C is the 1st axis when their formats are NCHW or ND");
            return ge::GRAPH_FAILED;
        }
    } else if (inputInfo_.gradsFormat == ge::FORMAT_NHWC) {
        if (inputInfo_.gradsShape.GetDim(NUM_3) != inputInfo_.yShape.GetDim(NUM_3)) {
            std::string shapeMsg = Ops::Base::ToString(inputInfo_.gradsShape) + " and " +
                                   Ops::Base::ToString(inputInfo_.yShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grads and y", shapeMsg.c_str(),
                                                   "The C-dimension of input grads and output y must be the same, "
                                                   "where C is the last axis when their formats are NHWC");
            return ge::GRAPH_FAILED;
        }
        formatIsNHWC = true;
    }
    inputInfo_.lenC = inputInfo_.gradsShape.GetDim(formatIsNHWC ? NUM_3 : NUM_1);
    inputInfo_.lenSrcH = inputInfo_.yShape.GetDim(formatIsNHWC ? NUM_1 : NUM_2);
    inputInfo_.lenSrcW = inputInfo_.yShape.GetDim(formatIsNHWC ? NUM_2 : NUM_3);
    inputInfo_.lenDstH = inputInfo_.gradsShape.GetDim(formatIsNHWC ? NUM_1 : NUM_2);
    inputInfo_.lenDstW = inputInfo_.gradsShape.GetDim(formatIsNHWC ? NUM_2 : NUM_3);

    if (inputInfo_.lenN <= 0 || inputInfo_.lenC <= 0) {
        std::string reasonMsg = "The N-dimension and C-dimension of grads and y must be greater than 0, "
                                "where C is inferred from the 4D shape of grads based on its format "
                                "(axis 1 for NCHW, axis 3 for NHWC), and N is the size of axis 0";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grads",
                                              Ops::Base::ToString(inputInfo_.gradsShape).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (inputInfo_.lenSrcH <= 0 || inputInfo_.lenSrcW <= 0 || inputInfo_.lenDstH <= 0 || inputInfo_.lenDstW <= 0) {
        std::string shapeMsg = Ops::Base::ToString(inputInfo_.gradsShape) + " and " +
                               Ops::Base::ToString(inputInfo_.yShape);
        std::string reasonMsg = "The H-dimension and W-dimension of grads and y must be greater than 0, "
                                "where H and W are inferred from the 4D shapes of input grads and output y "
                                "based on their formats: axes 2/3 for NCHW, or axes 1/2 for NHWC";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grads and y", shapeMsg.c_str(),
                                               reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::CheckShapeValid()
{
    OP_CHECK_IF(
        (CheckShapeDimValid() != ge::GRAPH_SUCCESS),
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                               "CheckShapeDimValid failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckAxesValid() != ge::GRAPH_SUCCESS),
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                               "CheckAxesValid failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    calcInfo_.gradsShapeSize = inputInfo_.gradsShape.GetShapeSize();
    calcInfo_.yShapeSize = inputInfo_.yShape.GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    size_t attrNum = attrs->GetAttrNum();
    if (attrNum > NUM_0) {
        inputInfo_.alignCorners = *(attrs->GetAttrPointer<bool>(NUM_0)) ? 1 : 0;
    }
    if (attrNum > NUM_1) {
        auto scales = attrs->GetAttrPointer<gert::ContinuousVector>(NUM_1);
        OP_CHECK_IF(
            scales->GetSize() != NUM_2,
            OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "scales", std::to_string(scales->GetSize()).c_str(),
                                         std::to_string(NUM_2).c_str()),
            return ge::GRAPH_FAILED);
        const float* scalesData = static_cast<const float*>(scales->GetData());
        OP_CHECK_IF(scalesData == nullptr,
                    error_report::ReportTilingStageError(
                        context_->GetNodeName(),
                        "scalesData is null in the tiling context; check the TilingContext construction"),
                    return ge::GRAPH_FAILED);
        inputInfo_.oriScaleH = scalesData[NUM_0];
        inputInfo_.oriScaleW = scalesData[NUM_1];
        OP_LOGI(context_->GetNodeName(), "original scales(%f, %f)", inputInfo_.oriScaleH, inputInfo_.oriScaleW);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF((GetTensorInfo() != ge::GRAPH_SUCCESS),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                                       "GetTensorInfo failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckDtypeValid() != ge::GRAPH_SUCCESS),
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                               "CheckDtypeValid failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckFormatValid() != ge::GRAPH_SUCCESS),
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                               "CheckFormatValid failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckShapeValid() != ge::GRAPH_SUCCESS),
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                               "CheckShapeValid failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetAttrInfo() != ge::GRAPH_SUCCESS),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                                       "GetAttrInfo failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ResizeBicubicV2GradBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ResizeBicubicV2GradBaseTiling::GetTilingKey() const { return 0; }

ge::graphStatus ResizeBicubicV2GradBaseTiling::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        error_report::ReportTilingStageError(
            context_->GetNodeName(), "workspaces is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    // split-K places its partial-sum buffer in the user workspace (after the system-reserved
    // lib API region); request its bytes on top of the reserve. The kernel reaches the same
    // region via GetUserWorkspace(). partials count = yShapeSize * segsPerOutput floats.
    size_t workspaceSize = static_cast<size_t>(sysWorkspaceSize);
    if (calcInfo_.splitK != 0) {
        workspaceSize += static_cast<size_t>(calcInfo_.yShapeSize) * static_cast<size_t>(calcInfo_.segsPerOutput) *
                         sizeof(float);
    }
    workspaces[0] = workspaceSize;
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBicubicV2GradBaseTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

void ResizeBicubicV2GradBaseTiling::SetScales()
{
    if (inputInfo_.alignCorners) {
        if (calcInfo_.isMatchDetermine) {
            if (inputInfo_.lenSrcH > 1 && inputInfo_.lenDstH > 1) {
                calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH - 1) / (inputInfo_.lenDstH - 1);
                calcInfo_.inverseScaleH = static_cast<float>(inputInfo_.lenDstH - 1) / (inputInfo_.lenSrcH - 1);
            } else {
                calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH) / inputInfo_.lenDstH;
                calcInfo_.inverseScaleH = static_cast<float>(inputInfo_.lenDstH) / (inputInfo_.lenSrcH);
            }
            if (inputInfo_.lenSrcW > 1 && inputInfo_.lenDstW > 1) {
                calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW - 1) / (inputInfo_.lenDstW - 1);
                calcInfo_.inverseScaleW = static_cast<float>(inputInfo_.lenDstW - 1) / (inputInfo_.lenSrcW - 1);
            } else {
                calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW) / inputInfo_.lenDstW;
                calcInfo_.inverseScaleW = static_cast<float>(inputInfo_.lenDstW) / (inputInfo_.lenSrcW);
            }
        } else {
            if (inputInfo_.lenDstH > 1) {
                calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH - 1) / (inputInfo_.lenDstH - 1);
            } else {
                calcInfo_.scaleH = 0.0f;
            }
            if (inputInfo_.lenDstW > 1) {
                calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW - 1) / (inputInfo_.lenDstW - 1);
            } else {
                calcInfo_.scaleW = 0.0f;
            }
        }
    } else {
        if (inputInfo_.oriScaleH > 0.0f) {
            calcInfo_.scaleH = 1.0f / inputInfo_.oriScaleH;
            calcInfo_.inverseScaleH = inputInfo_.oriScaleH;
        } else {
            calcInfo_.scaleH = static_cast<float>(inputInfo_.lenSrcH) / inputInfo_.lenDstH;
            calcInfo_.inverseScaleH = static_cast<float>(inputInfo_.lenDstH) / inputInfo_.lenSrcH;
        }
        if (inputInfo_.oriScaleW > 0.0f) {
            calcInfo_.scaleW = 1.0f / inputInfo_.oriScaleW;
            calcInfo_.inverseScaleW = inputInfo_.oriScaleW;
        } else {
            calcInfo_.scaleW = static_cast<float>(inputInfo_.lenSrcW) / inputInfo_.lenDstW;
            calcInfo_.inverseScaleW = static_cast<float>(inputInfo_.lenDstW) / inputInfo_.lenSrcW;
        }
    }
}

bool ResizeBicubicV2GradBaseTiling::IsUseIdx32() const
{
    return calcInfo_.gradsShapeSize <= UINT32_MAX && calcInfo_.yShapeSize <= UINT32_MAX &&
           inputInfo_.lenSrcH <= INT32_MAX && inputInfo_.lenSrcW <= INT32_MAX && inputInfo_.lenDstH <= INT32_MAX &&
           inputInfo_.lenDstW <= INT32_MAX;
}

ge::graphStatus Tiling4ResizeBicubicV2Grad(gert::TilingContext* context)
{
    return Ops::Cv::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4ResizeBicubicV2Grad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4ResizeBicubicV2Grad start");

    auto compileInfo = context->GetCompiledInfo<ResizeBicubicV2GradCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "compileInfo is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "platformInfo is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;

    OP_CHECK_IF((compileInfo->coreNum <= 0),
                error_report::ReportTilingStageError(context->GetNodeName(),
                                                     "failed to get the core num from the platform info"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                error_report::ReportTilingStageError(context->GetNodeName(),
                                                     "failed to get the ub size from the platform info"),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4ResizeBicubicV2Grad end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ResizeBicubicV2Grad)
    .Tiling(Tiling4ResizeBicubicV2Grad)
    .TilingParse<ResizeBicubicV2GradCompileInfo>(TilingPrepare4ResizeBicubicV2Grad);
} // namespace optiling
