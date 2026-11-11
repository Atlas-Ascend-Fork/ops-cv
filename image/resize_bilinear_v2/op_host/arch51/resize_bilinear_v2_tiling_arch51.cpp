/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_bilinear_v2_tiling_arch51.cpp
 * \brief Ascend516（arch51）tiling：TilingKey 50000（SIMD NHWC 专用）
 */
#include "../arch35/resize_bilinear_v2_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {
constexpr size_t ATTR_ALIGN_CORERS_IDX = 0;
constexpr size_t ATTR_HALF_PIXEL_CENTERS_IDX = 1;
constexpr size_t ATTR_SCALES_IDX = 3;
constexpr size_t SCALES_NUM = 2;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t DIM_LEN_4D = 4;
constexpr size_t C_DIM_IDX_NHWC = 3;
constexpr size_t N_DIM_IDX = 0;
constexpr size_t H_DIM_IDX_NHWC = 1;
constexpr size_t WORKSPACE_SIZE = 32;
constexpr int64_t TILING_KEY_SIMD_NHWC = 50000;
constexpr int64_t ONE_BLOCK_SIZE = 32;
constexpr int64_t RSV_BLOCK_NUM = 8;
constexpr int64_t DB_BUFF_NUM = 2;
constexpr int64_t SIMD_NHWC_X_TENSOR_NUM = 4;
constexpr int64_t SIMD_NHWC_Y_TENSOR_NUM = 1;
constexpr int64_t IDX_DST_H = 0;
constexpr int64_t IDX_DST_W = 1;

class ResizeBilinearV2Arch51TilingImpl {
public:
    explicit ResizeBilinearV2Arch51TilingImpl(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init(const ResizeBilinearV2CompileInfo* compileInfo);
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckFormatMatchDims();
    void SetDimsByFormat();
    void DoTilingSimdNhwc();
    void FillTilingData();
    void PrintTilingData();
    ge::graphStatus SetScales();
    ge::graphStatus CheckDtypeAndFormat();
    ge::graphStatus SetShape();
    inline int64_t Min(int64_t x, int64_t y);
    float ComputeScale(float scale, int64_t lenSrc, int64_t lenDes);

private:
    int32_t dtypeSizeX_ = 0;
    int32_t dtypeSizeY_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockNum_ = 0;
    int64_t tilingKey_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t lenC_ = 0;
    int64_t lenN_ = 0;
    int64_t lenSrcH_ = 0;
    int64_t lenSrcW_ = 0;
    int64_t lenDesH_ = 0;
    int64_t lenDesW_ = 0;

    int64_t nFactor_ = 0;
    int64_t hFactor_ = 0;
    int64_t wFactor_ = 0;
    int64_t cFactor_ = 0;
    int64_t hwFactor_ = 0;

    int64_t ubNFactor_ = 0;
    int64_t ubHFactor_ = 0;
    int64_t ubWFactor_ = 0;
    int64_t ubCFactor_ = 0;
    int64_t ubHWFactor_ = 0;

    int64_t splitBlockFactor_ = 0;
    int64_t splitBlockTailFactor_ = 0;
    int64_t alignCorners_ = 0;
    int64_t halfPixelCenters_ = 0;
    float scaleW_ = 0.0f;
    float scaleH_ = 0.0f;
    bool scaleValid_ = false;

    int64_t lenCAlign_ = 0;
    bool isAlign_ = false;
    bool noAvailableSchedule_ = false;

    ge::DataType dtypeX_ = ge::DT_MAX;
    ge::DataType dtypeY_ = ge::DT_MAX;
    ge::Format format_ = ge::FORMAT_MAX;
    gert::Shape xShape_;
    gert::Shape yShape_;
    gert::TilingContext* context_ = nullptr;
    ResizeBilinearV2TilingData tilingData_;
};

inline int64_t ResizeBilinearV2Arch51TilingImpl::Min(int64_t x, int64_t y) { return (x < y) ? x : y; }

ge::graphStatus ResizeBilinearV2Arch51TilingImpl::CheckFormatMatchDims()
{
    OP_CHECK_IF(alignCorners_ && halfPixelCenters_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "align_corners and half_pixel_centers", "true and true",
                    "The values of attributes align_corners and half_pixel_centers cannot be true at the same time"),
                return ge::GRAPH_FAILED);

    if (xShape_.GetDimNum() != DIM_LEN_4D || yShape_.GetDimNum() != DIM_LEN_4D) {
        std::string dimMsg = std::to_string(xShape_.GetDimNum()) + " and " + std::to_string(yShape_.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x and y", dimMsg.c_str(),
                                                  "The shape of input x and output y must be 4D");
        return ge::GRAPH_FAILED;
    }

    const int64_t inputSize = xShape_.GetShapeSize();
    const int64_t outSize = yShape_.GetShapeSize();
    if (inputSize == 0 || outSize == 0) {
        std::string shapesizeMsg = std::to_string(inputSize) + " and " + std::to_string(outSize);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "x and y", shapesizeMsg.c_str(),
                                                   "Input x and output y do not support empty tensors");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

float ResizeBilinearV2Arch51TilingImpl::ComputeScale(float scale, int64_t lenSrc, int64_t lenDes)
{
    float newScale = 0.0f;
    if (scale > 0.0f) {
        newScale = static_cast<float>(1.0f) / scale;
    } else {
        newScale = static_cast<float>(lenSrc) / static_cast<float>(lenDes);
    }
    return newScale;
}

void ResizeBilinearV2Arch51TilingImpl::SetDimsByFormat()
{
    lenN_ = xShape_.GetDim(N_DIM_IDX);
    lenC_ = xShape_.GetDim(C_DIM_IDX_NHWC);
    lenSrcH_ = xShape_.GetDim(H_DIM_IDX_NHWC);
    lenDesH_ = yShape_.GetDim(H_DIM_IDX_NHWC);
    lenSrcW_ = xShape_.GetDim(H_DIM_IDX_NHWC + 1);
    lenDesW_ = yShape_.GetDim(H_DIM_IDX_NHWC + 1);

    nFactor_ = lenN_;
    hFactor_ = lenDesH_;
    wFactor_ = lenDesW_;
    cFactor_ = lenC_;
    hwFactor_ = lenDesH_ * lenDesW_;
    ubNFactor_ = nFactor_;
    ubHFactor_ = hFactor_;
    ubWFactor_ = wFactor_;
    ubCFactor_ = cFactor_;
    ubHWFactor_ = hwFactor_;

    OP_LOGI(context_->GetNodeName(), "lenN_:%ld , lenC_: %ld, srcH:%ld, srcW:%ld, dstH:%ld, dstW:%ld", lenN_, lenC_,
            lenSrcH_, lenSrcW_, lenDesH_, lenDesW_);

    isAlign_ = false;
    if ((lenC_ * dtypeSizeX_) % ONE_BLOCK_SIZE == 0) {
        isAlign_ = true;
    }
    lenCAlign_ = Ops::Base::CeilAlign(lenC_ * dtypeSizeX_, ONE_BLOCK_SIZE) / dtypeSizeX_;

    if (alignCorners_ == 0) {
        scaleH_ = ComputeScale(scaleH_, lenSrcH_, lenDesH_);
        scaleW_ = ComputeScale(scaleW_, lenSrcW_, lenDesW_);
        OP_LOGI(context_->GetNodeName(), "attr new scaleH is %f, scaleW is %f", scaleH_, scaleW_);
    } else {
        scaleH_ = static_cast<float>(lenSrcH_) / static_cast<float>(lenDesH_);
        scaleW_ = static_cast<float>(lenSrcW_) / static_cast<float>(lenDesW_);
        if (lenDesH_ > 1) {
            scaleH_ = static_cast<float>(lenSrcH_ - 1) / static_cast<float>(lenDesH_ - 1);
        }
        if (lenDesW_ > 1) {
            scaleW_ = static_cast<float>(lenSrcW_ - 1) / static_cast<float>(lenDesW_ - 1);
        }
        OP_LOGI(context_->GetNodeName(), "compute scaleH is %f, scaleW is %f", scaleH_, scaleW_);
    }
}

void ResizeBilinearV2Arch51TilingImpl::DoTilingSimdNhwc()
{
    int64_t totalLen = lenDesH_ * lenDesW_;

    if (totalLen * lenN_ <= coreNum_) {
        int64_t tileNum = coreNum_ / (totalLen * lenN_);
        hwFactor_ = 1;
        nFactor_ = 1;
        cFactor_ = Ops::Base::CeilDiv(lenC_, tileNum);
        realCoreNum_ = totalLen * lenN_ * Ops::Base::CeilDiv(lenC_, cFactor_);
    } else if (totalLen <= coreNum_) {
        int64_t tileNum = coreNum_ / totalLen;
        hwFactor_ = 1;
        nFactor_ = Ops::Base::CeilDiv(lenN_, tileNum);
        cFactor_ = lenC_;
        realCoreNum_ = totalLen * Ops::Base::CeilDiv(lenN_, nFactor_);
    } else {
        hwFactor_ = Ops::Base::CeilDiv(totalLen, coreNum_);
        nFactor_ = lenN_;
        cFactor_ = lenC_;
        realCoreNum_ = Ops::Base::CeilDiv(totalLen, hwFactor_);
    }

    realCoreNum_ = Min(realCoreNum_, coreNum_);

    int64_t vol4AxisC = ((ubBlockNum_ / DB_BUFF_NUM) * ONE_BLOCK_SIZE) /
                        (SIMD_NHWC_X_TENSOR_NUM * dtypeSizeX_ + SIMD_NHWC_Y_TENSOR_NUM * dtypeSizeY_);

    if (vol4AxisC < lenC_) {
        ubNFactor_ = 1;
        ubCFactor_ = vol4AxisC;
    } else {
        ubNFactor_ = vol4AxisC / lenC_;
        ubNFactor_ = (ubNFactor_ < nFactor_) ? ubNFactor_ : nFactor_;
        ubCFactor_ = lenC_;
    }

    // 小 C 打包窗口（仅非对齐小 C 启用，ubWFactor=pBatch，0 → kernel 逐点路径）
    ubWFactor_ = 0;
    if ((lenC_ * dtypeSizeX_) % ONE_BLOCK_SIZE != 0 || (lenC_ * dtypeSizeY_) % ONE_BLOCK_SIZE != 0) {
        const int64_t idxSize = (dtypeSizeX_ == 2) ? 2 : 4;
        const int64_t stgPerPt = lenC_ * (4 * dtypeSizeX_ + 2 * idxSize + 16 + dtypeSizeY_);
        const int64_t budgetBytes = (ubBlockNum_ * ONE_BLOCK_SIZE * 3) / 4;
        const int64_t colBytesX = lenC_ * dtypeSizeX_;
        // span 余量 +2 = kernel 侧 NEIGHBOR_MARGIN（段首尾各 1 邻域列）；先按最小批量 8 探测可行性
        const int64_t spanMin8 = Ops::Base::CeilDiv(8 * lenSrcW_, lenDesW_) + 2;
        if (2 * spanMin8 * colBytesX + 8 * stgPerPt <= budgetBytes) {
            int64_t pBatch = (budgetBytes - 4 * colBytesX) / (2 * colBytesX + stgPerPt);
            pBatch = Min(pBatch, lenDesW_);
            pBatch = (pBatch / 8) * 8;
            while (pBatch >= 8) {
                const int64_t span = Ops::Base::CeilDiv(pBatch * lenSrcW_, lenDesW_) + 2;
                if (2 * span * colBytesX + pBatch * stgPerPt <= budgetBytes) {
                    break;
                }
                pBatch -= 8;
            }
            if (pBatch >= 8) {
                ubWFactor_ = pBatch;
            }
        }
    }
}

void ResizeBilinearV2Arch51TilingImpl::FillTilingData()
{
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_alignCorners(alignCorners_);
    tilingData_.set_halfPixelCenters(halfPixelCenters_);
    tilingData_.set_lenN(lenN_);
    tilingData_.set_lenC(lenC_);
    tilingData_.set_lenSrcH(lenSrcH_);
    tilingData_.set_lenSrcW(lenSrcW_);
    tilingData_.set_lenDesH(lenDesH_);
    tilingData_.set_lenDesW(lenDesW_);
    tilingData_.set_nFactor(nFactor_);
    tilingData_.set_hFactor(hFactor_);
    tilingData_.set_wFactor(wFactor_);
    tilingData_.set_cFactor(cFactor_);
    tilingData_.set_hwFactor(hwFactor_);
    tilingData_.set_ubNFactor(ubNFactor_);
    tilingData_.set_ubHFactor(ubHFactor_);
    tilingData_.set_ubWFactor(ubWFactor_);
    tilingData_.set_ubCFactor(ubCFactor_);
    tilingData_.set_ubHWFactor(ubHWFactor_);
    tilingData_.set_splitBlockFactor(splitBlockFactor_);
    tilingData_.set_splitBlockTailFactor(splitBlockTailFactor_);
    tilingData_.set_scaleW(scaleW_);
    tilingData_.set_scaleH(scaleH_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void ResizeBilinearV2Arch51TilingImpl::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "arch51 tilingData is tilingKey:%ld, realCoreNum:%ld, ubSize: %ld, lenN:%ld, lenC:%ld, srcH:%ld, "
            "srcW:%ld, desH:%ld, desW:%ld, nFactor:%ld, hwFactor:%ld, cFactor:%ld, ubNFactor:%ld, ubCFactor:%ld, "
            "scaleW: %f, scaleH: %f",
            tilingData_.get_tilingKey(), tilingData_.get_realCoreNum(), tilingData_.get_ubSize(),
            tilingData_.get_lenN(), tilingData_.get_lenC(), tilingData_.get_lenSrcH(), tilingData_.get_lenSrcW(),
            tilingData_.get_lenDesH(), tilingData_.get_lenDesW(), tilingData_.get_nFactor(), tilingData_.get_hwFactor(),
            tilingData_.get_cFactor(), tilingData_.get_ubNFactor(), tilingData_.get_ubCFactor(),
            tilingData_.get_scaleW(), tilingData_.get_scaleH());
}

ge::graphStatus ResizeBilinearV2Arch51TilingImpl::SetScales()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto scales = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SCALES_IDX);

    if (scales != nullptr) {
        int64_t scales_num = scales->GetSize();
        const float* scales_data = reinterpret_cast<const float*>(scales->GetData());
        OP_CHECK_NULL_WITH_CONTEXT(context_, scales_data);
        OP_CHECK_IF(scales_num != SCALES_NUM,
                    OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "scales", std::to_string(scales_num).c_str(),
                                                 std::to_string(SCALES_NUM).c_str()),
                    return ge::GRAPH_FAILED);
        OP_LOGI(context_->GetNodeName(), "arch51 init: num[%ld]scales(%f %f)", scales_num, scales_data[0],
                scales_data[1]);
        scaleH_ = scales_data[IDX_DST_H];
        scaleW_ = scales_data[IDX_DST_W];
        if (scaleH_ > 0 || scaleW_ > 0) {
            scaleValid_ = true;
        }
    } else {
        OP_LOGI(context_->GetNodeName(), "arch51 init: There is no scales attr.");
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBilinearV2Arch51TilingImpl::CheckDtypeAndFormat()
{
    auto inputXDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    dtypeX_ = inputXDesc->GetDataType();
    dtypeY_ = outputYDesc->GetDataType();
    dtypeSizeX_ = GetSizeByDataType(dtypeX_);
    dtypeSizeY_ = GetSizeByDataType(dtypeY_);
    if (dtypeSizeX_ <= 0 || dtypeSizeY_ <= 0) {
        std::string dtypeMsg = Ops::Base::ToString(dtypeX_) + " and " + Ops::Base::ToString(dtypeY_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and y", dtypeMsg.c_str(),
                                               "The dtype sizes of input x and output y must be greater than zero");
        return ge::GRAPH_FAILED;
    }

    format_ = static_cast<ge::Format>(ge::GetPrimaryFormat(inputXDesc->GetStorageFormat()));
    auto outFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outputYDesc->GetStorageFormat()));
    if (format_ != outFormat) {
        std::string formatMsg = Ops::Base::ToString(format_) + " and " + Ops::Base::ToString(outFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "x and y", formatMsg.c_str(),
                                                "The formats of input x and output y must be the same");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((format_ != ge::FORMAT_NHWC),
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(format_).c_str(), "NHWC"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBilinearV2Arch51TilingImpl::SetShape()
{
    auto xStorage = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    xShape_ = Ops::Cv::OpTiling::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorage);
    yShape_ = Ops::Cv::OpTiling::EnsureNotScalar(yStorage->GetStorageShape());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBilinearV2Arch51TilingImpl::Init(const ResizeBilinearV2CompileInfo* compileInfo)
{
    OP_LOGD(context_->GetNodeName(), "Enter ResizeBilinearV2Arch51TilingImpl init.");
    coreNum_ = compileInfo->core_num;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(coreNum_ <= 0 || ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "coreNum or ubSize is less than 0"),
                return ge::GRAPH_FAILED);
    ubBlockNum_ = Ops::Base::CeilDiv(ubSize_, ONE_BLOCK_SIZE) - RSV_BLOCK_NUM;
    OP_LOGI(context_->GetNodeName(), "coreNum_ is %ld, ubSize_ is %ld, ubBlockNum_ is %ld", coreNum_, ubSize_,
            ubBlockNum_);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const bool* alignCornersPtr = attrs->GetAttrPointer<bool>(ATTR_ALIGN_CORERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alignCornersPtr);
    const bool* halfPixelCentersPtr = attrs->GetAttrPointer<bool>(ATTR_HALF_PIXEL_CENTERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, halfPixelCentersPtr);
    alignCorners_ = *alignCornersPtr ? 1 : 0;
    halfPixelCenters_ = *halfPixelCentersPtr ? 1 : 0;

    if (SetScales() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckDtypeAndFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckFormatMatchDims() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetDimsByFormat();

    OP_LOGD(context_->GetNodeName(), "Exit ResizeBilinearV2Arch51TilingImpl init.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeBilinearV2Arch51TilingImpl::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter ResizeBilinearV2Arch51TilingImpl DoTiling.");

    if (format_ == ge::FORMAT_NHWC) {
        tilingKey_ = TILING_KEY_SIMD_NHWC;
    } else {
        noAvailableSchedule_ = true;
        OP_LOGE(context_->GetNodeName(), "arch51 tiling only supports NHWC for ascend516, unsupported format %ld.",
                static_cast<int64_t>(format_));
        return ge::GRAPH_FAILED;
    }

    DoTilingSimdNhwc();
    FillTilingData();
    PrintTilingData();

    context_->SetBlockDim(realCoreNum_);
    context_->SetTilingKey(tilingKey_);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(context_->GetNodeName(), "Exit ResizeBilinearV2Arch51TilingImpl DoTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ResizeBilinearV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start Tiling4ResizeBilinearV2 (arch51).");
    auto compileInfo = reinterpret_cast<const ResizeBilinearV2CompileInfo*>(context->GetCompileInfo());
    ResizeBilinearV2Arch51TilingImpl tilingImpl = ResizeBilinearV2Arch51TilingImpl(context);
    if (tilingImpl.Init(compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ResizeBilinearV2 arch51 init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ResizeBilinearV2 arch51 do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ResizeBilinearV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepare4ResizeBilinearV2 (arch51).");
    auto compileInfo = context->GetCompiledInfo<ResizeBilinearV2CompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->core_num <= 0), OP_LOGE(context->GetNodeName(), "core num invalid."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ResizeBilinearV2)
    .Tiling(Tiling4ResizeBilinearV2)
    .TilingParse<ResizeBilinearV2CompileInfo>(TilingPrepare4ResizeBilinearV2);
} // namespace optiling
