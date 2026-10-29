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
 * \file resize_bicubic_v2_grad_all_copy_tiling_arch35.cpp
 * \brief resize_bicubic_v2_grad_all_copy_tiling_arch35
 */
#include "resize_bicubic_v2_grad_tiling_arch35.h"
#include "op_api/error_report_util.h"

namespace optiling {

constexpr int64_t DB_BUFFER_NUM = 2;
constexpr uint64_t TILING_KEY_ALL_COPY = 30000;
constexpr uint64_t TILING_PRIORITY_ALL_COPY = 1000;

bool ResizeBicubicV2GradAllCopyTiling::IsCapable()
{
    if (inputInfo_.lenSrcH == inputInfo_.lenDstH && inputInfo_.lenSrcW == inputInfo_.lenDstW) {
        return true;
    }

    return false;
}

void ResizeBicubicV2GradAllCopyTiling::SetTilingData()
{
    tilingData_.set_useCoreNum(calcInfo_.useCoreNum);
    tilingData_.set_coreFactor(calcInfo_.coreFactor);
    tilingData_.set_coreTailFactor(calcInfo_.coreTailFactor);
    tilingData_.set_ubFactor(calcInfo_.ubFactor);
}

void ResizeBicubicV2GradAllCopyTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "ResizeBicubicV2Grad tilingData: useCoreNum is %ld, coreFactor is %ld, coreTailFactor is %ld, ubFactor is %ld",
        tilingData_.get_useCoreNum(), tilingData_.get_coreFactor(), tilingData_.get_coreTailFactor(),
        tilingData_.get_ubFactor());
    return;
}

ge::graphStatus ResizeBicubicV2GradAllCopyTiling::DoOpTiling()
{
    calcInfo_.useCoreNum = calcInfo_.yShapeSize < compileInfo_.coreNum ? calcInfo_.yShapeSize : compileInfo_.coreNum;
    calcInfo_.coreFactor = Ops::Base::FloorDiv(calcInfo_.yShapeSize, calcInfo_.useCoreNum);
    calcInfo_.coreTailFactor = calcInfo_.yShapeSize - calcInfo_.coreFactor * calcInfo_.useCoreNum;
    calcInfo_.ubFactor = calcInfo_.ubBlockNum / DB_BUFFER_NUM * compileInfo_.ubBlockSize / calcInfo_.yDtypeSize;

    SetTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t ResizeBicubicV2GradAllCopyTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_KEY_ALL_COPY;
    return tilingKey;
}

ge::graphStatus ResizeBicubicV2GradAllCopyTiling::PostTiling()
{
    context_->SetBlockDim(calcInfo_.useCoreNum);

    OP_CHECK_IF(
        tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "actual",
                                                   (std::to_string(tilingData_.GetDataSize()) + ", " +
                                                    std::to_string(context_->GetRawTilingData()->GetCapacity()))
                                                       .c_str(),
                                                   "actual all copy tiling data size > context tiling data size"),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ResizeBicubicV2Grad, ResizeBicubicV2GradAllCopyTiling, TILING_PRIORITY_ALL_COPY);
} // namespace optiling
