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
 * \file resize_bicubic_v2_grad_simt_tiling_arch35.cpp
 * \brief resize_bicubic_v2_grad_simt_tiling_arch35
 */
#include "resize_bicubic_v2_grad_tiling_arch35.h"
#include "op_api/error_report_util.h"

namespace optiling {

constexpr uint64_t TILING_KEY_SIMT = 10000;
constexpr uint64_t TILING_KEY_SIMT_IDX64 = 10001;
constexpr uint64_t TILING_PRIORITY_SIMT = 3000;

bool ResizeBicubicV2GradSimtTiling::IsCapable() { return true; }

void ResizeBicubicV2GradSimtTiling::SetTilingData()
{
    tilingData_.set_lenC(inputInfo_.lenC);
    tilingData_.set_lenSrcH(inputInfo_.lenSrcH);
    tilingData_.set_lenSrcW(inputInfo_.lenSrcW);
    tilingData_.set_lenDstH(inputInfo_.lenDstH);
    tilingData_.set_lenDstW(inputInfo_.lenDstW);
    tilingData_.set_format(inputInfo_.format);
    tilingData_.set_alignCorners(inputInfo_.alignCorners);
    tilingData_.set_initYUseCoreNum(calcInfo_.initYUseCoreNum);
    tilingData_.set_initYCoreFactor(calcInfo_.initYCoreFactor);
    tilingData_.set_initYCoreTailFactor(calcInfo_.initYCoreTailFactor);
    tilingData_.set_useCoreNum(calcInfo_.useCoreNum);
    tilingData_.set_coreFactor(calcInfo_.coreFactor);
    tilingData_.set_coreTailFactor(calcInfo_.coreTailFactor);
    tilingData_.set_scaleH(calcInfo_.scaleH);
    tilingData_.set_scaleW(calcInfo_.scaleW);
}

void ResizeBicubicV2GradSimtTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "ResizeBicubicV2Grad tilingData: lenC is %ld, lenSrcH is %ld, lenSrcW is %ld, lenDstH is %ld, lenDstW is %ld, "
        "format is %ld, alignCorners is %ld, initYUseCoreNum is %ld, initYCoreFactor is %ld, "
        "initYCoreTailFactor is %ld, useCoreNum is %ld, coreFactor is %ld, coreTailFactor is %ld, "
        "scaleH is %f, scaleW is %f",
        tilingData_.get_lenC(), tilingData_.get_lenSrcH(), tilingData_.get_lenSrcW(), tilingData_.get_lenDstH(),
        tilingData_.get_lenDstW(), tilingData_.get_format(), tilingData_.get_alignCorners(),
        tilingData_.get_initYUseCoreNum(), tilingData_.get_initYCoreFactor(), tilingData_.get_initYCoreTailFactor(),
        tilingData_.get_useCoreNum(), tilingData_.get_coreFactor(), tilingData_.get_coreTailFactor(),
        tilingData_.get_scaleH(), tilingData_.get_scaleW());
    return;
}

ge::graphStatus ResizeBicubicV2GradSimtTiling::DoOpTiling()
{
    this->SetScales();

    calcInfo_.initYUseCoreNum = calcInfo_.yShapeSize < compileInfo_.coreNum ? calcInfo_.yShapeSize :
                                                                              compileInfo_.coreNum;
    calcInfo_.initYCoreFactor = Ops::Base::FloorDiv(calcInfo_.yShapeSize, calcInfo_.initYUseCoreNum);
    calcInfo_.initYCoreTailFactor = calcInfo_.yShapeSize - calcInfo_.initYCoreFactor * calcInfo_.initYUseCoreNum;

    calcInfo_.useCoreNum = calcInfo_.gradsShapeSize < compileInfo_.coreNum ? calcInfo_.gradsShapeSize :
                                                                             compileInfo_.coreNum;
    calcInfo_.coreFactor = Ops::Base::FloorDiv(calcInfo_.gradsShapeSize, calcInfo_.useCoreNum);
    calcInfo_.coreTailFactor = calcInfo_.gradsShapeSize - calcInfo_.coreFactor * calcInfo_.useCoreNum;

    SetTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t ResizeBicubicV2GradSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_KEY_SIMT_IDX64;
    if (this->IsUseIdx32()) {
        tilingKey = TILING_KEY_SIMT;
    }
    return tilingKey;
}

ge::graphStatus ResizeBicubicV2GradSimtTiling::PostTiling()
{
    context_->SetBlockDim(calcInfo_.useCoreNum);
    if (calcInfo_.useCoreNum < calcInfo_.initYUseCoreNum) {
        context_->SetBlockDim(calcInfo_.initYUseCoreNum);
    }

    OP_CHECK_IF(tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "actual",
                                                           (std::to_string(tilingData_.GetDataSize()) + ", " +
                                                            std::to_string(context_->GetRawTilingData()->GetCapacity()))
                                                               .c_str(),
                                                           "actual simt tiling data size > context tiling data size"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ResizeBicubicV2Grad, ResizeBicubicV2GradSimtTiling, TILING_PRIORITY_SIMT);
} // namespace optiling
