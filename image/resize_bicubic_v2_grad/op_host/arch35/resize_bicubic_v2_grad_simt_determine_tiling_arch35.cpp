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
 * \file resize_bicubic_v2_grad_simt_determine_tiling_arch35.cpp
 * \brief resize_bicubic_v2_grad_simt_determine_tiling_arch35
 */
#include "resize_bicubic_v2_grad_tiling_arch35.h"
#include "op_api/error_report_util.h"

namespace optiling {

constexpr uint64_t TILING_KEY_SIMT_DETERMINE = 20000;
constexpr uint64_t TILING_KEY_SIMT_DETERMINE_IDX64 = 20001;
constexpr uint64_t TILING_KEY_SIMT_DETERMINE_SPLITK = 20002;
constexpr uint64_t TILING_KEY_SIMT_DETERMINE_SPLITK_IDX64 = 20003;
constexpr uint64_t TILING_PRIORITY_SIMT_DETERMINE = 2000;

// Must match the kernel-side SIMT_DETERMINE_THREAD_NUM_INT32/INT64.
constexpr int64_t SIMT_DETERMINE_THREAD_NUM_INT32 = 512;
constexpr int64_t SIMT_DETERMINE_THREAD_NUM_INT64 = 256;
// Only take the split-K path when the gather K-domain (H rows contributing to each
// output) is far larger than the number of output elements, i.e. the op is badly
// under-parallelized (few outputs, huge per-output serial reduction). Threshold keeps
// all normal upsample/downsample/equal cases on the untouched 20000/20001 path.
constexpr int64_t SPLITK_KDOMAIN_THRESHOLD = 4096;

bool ResizeBicubicV2GradSimtDetermineTiling::IsCapable()
{
    if (compileInfo_.isDetermine) {
        calcInfo_.isMatchDetermine = 1;
        return true;
    } else {
        calcInfo_.isMatchDetermine = 0;
        if (inputInfo_.lenSrcH < inputInfo_.lenDstH || inputInfo_.lenSrcW < inputInfo_.lenDstW) {
            calcInfo_.isMatchDetermine = 1;
            return true;
        }
        this->SetScales();
        if ((calcInfo_.scaleH > 0.0f && calcInfo_.scaleH < 1.0f) ||
            (calcInfo_.scaleW > 0.0f && calcInfo_.scaleW < 1.0f)) {
            calcInfo_.isMatchDetermine = 1;
            return true;
        }
    }
    return false;
}

void ResizeBicubicV2GradSimtDetermineTiling::SetTilingData()
{
    tilingData_.set_lenC(inputInfo_.lenC);
    tilingData_.set_lenSrcH(inputInfo_.lenSrcH);
    tilingData_.set_lenSrcW(inputInfo_.lenSrcW);
    tilingData_.set_lenDstH(inputInfo_.lenDstH);
    tilingData_.set_lenDstW(inputInfo_.lenDstW);
    tilingData_.set_format(inputInfo_.format);
    tilingData_.set_alignCorners(inputInfo_.alignCorners);
    tilingData_.set_useCoreNum(calcInfo_.useCoreNum);
    tilingData_.set_coreFactor(calcInfo_.coreFactor);
    tilingData_.set_coreTailFactor(calcInfo_.coreTailFactor);
    tilingData_.set_scaleH(calcInfo_.scaleH);
    tilingData_.set_scaleW(calcInfo_.scaleW);
    tilingData_.set_inverseScaleH(calcInfo_.inverseScaleH);
    tilingData_.set_inverseScaleW(calcInfo_.inverseScaleW);
    tilingData_.set_splitK(calcInfo_.splitK);
    tilingData_.set_coresPerOutput(calcInfo_.coresPerOutput);
    tilingData_.set_segsPerOutput(calcInfo_.segsPerOutput);
}

void ResizeBicubicV2GradSimtDetermineTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "ResizeBicubicV2Grad tilingData: lenC is %ld, lenSrcH is %ld, lenSrcW is %ld, lenDstH is %ld, lenDstW is %ld, "
        "format is %ld, alignCorners is %ld, useCoreNum is %ld, coreFactor is %ld, coreTailFactor is %ld, "
        "scaleH is %f, scaleW is %f, inverseScaleH is %f, inverseScaleW is %f, splitK is %ld, "
        "coresPerOutput is %ld, segsPerOutput is %ld",
        tilingData_.get_lenC(), tilingData_.get_lenSrcH(), tilingData_.get_lenSrcW(), tilingData_.get_lenDstH(),
        tilingData_.get_lenDstW(), tilingData_.get_format(), tilingData_.get_alignCorners(),
        tilingData_.get_useCoreNum(), tilingData_.get_coreFactor(), tilingData_.get_coreTailFactor(),
        tilingData_.get_scaleH(), tilingData_.get_scaleW(), tilingData_.get_inverseScaleH(),
        tilingData_.get_inverseScaleW(), tilingData_.get_splitK(), tilingData_.get_coresPerOutput(),
        tilingData_.get_segsPerOutput());
    return;
}

ge::graphStatus ResizeBicubicV2GradSimtDetermineTiling::DoOpTiling()
{
    this->SetScales();

    calcInfo_.useCoreNum = calcInfo_.yShapeSize < compileInfo_.coreNum ? calcInfo_.yShapeSize : compileInfo_.coreNum;
    calcInfo_.coreFactor = Ops::Base::FloorDiv(calcInfo_.yShapeSize, calcInfo_.useCoreNum);
    calcInfo_.coreTailFactor = calcInfo_.yShapeSize - calcInfo_.coreFactor * calcInfo_.useCoreNum;

    // --- split-K deterministic path decision ---
    // Trigger only when: (a) outputs under-parallelize the cores (yShapeSize < coreNum, so
    // the normal path leaves cores idle while each active thread serially scans a huge H
    // gather domain), and (b) that per-output H gather domain (lenDstH) is far larger than
    // the number of outputs. This precisely targets the extreme-upsample-backward timeout
    // and leaves every normal case on the untouched 20000/20001 path.
    calcInfo_.splitK = 0;
    calcInfo_.coresPerOutput = 1;
    calcInfo_.segsPerOutput = 1;
    if (calcInfo_.yShapeSize > 0 && calcInfo_.yShapeSize < compileInfo_.coreNum &&
        inputInfo_.lenDstH >= SPLITK_KDOMAIN_THRESHOLD) {
        int64_t coresPerOutput = Ops::Base::FloorDiv(compileInfo_.coreNum, calcInfo_.yShapeSize);
        if (coresPerOutput > inputInfo_.lenDstH) {
            coresPerOutput = inputInfo_.lenDstH; // never more H-segments than H rows
        }
        if (coresPerOutput > 1) {
            int64_t threadNum = this->IsUseIdx32() ? SIMT_DETERMINE_THREAD_NUM_INT32 : SIMT_DETERMINE_THREAD_NUM_INT64;
            calcInfo_.splitK = 1;
            calcInfo_.coresPerOutput = coresPerOutput;
            calcInfo_.segsPerOutput = coresPerOutput * threadNum;
            // useCoreNum now spans all (output, coreSeg) pairs.
            calcInfo_.useCoreNum = calcInfo_.yShapeSize * coresPerOutput;
        }
    }

    SetTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t ResizeBicubicV2GradSimtDetermineTiling::GetTilingKey() const
{
    bool useIdx32 = this->IsUseIdx32();
    if (calcInfo_.splitK) {
        return useIdx32 ? TILING_KEY_SIMT_DETERMINE_SPLITK : TILING_KEY_SIMT_DETERMINE_SPLITK_IDX64;
    }
    return useIdx32 ? TILING_KEY_SIMT_DETERMINE : TILING_KEY_SIMT_DETERMINE_IDX64;
}

ge::graphStatus ResizeBicubicV2GradSimtDetermineTiling::PostTiling()
{
    context_->SetBlockDim(calcInfo_.useCoreNum);

    OP_CHECK_IF(
        tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "actual",
                                                   (std::to_string(tilingData_.GetDataSize()) + ", " +
                                                    std::to_string(context_->GetRawTilingData()->GetCapacity()))
                                                       .c_str(),
                                                   "actual simt determine tiling data size > context tiling data size"),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ResizeBicubicV2Grad, ResizeBicubicV2GradSimtDetermineTiling,
                             TILING_PRIORITY_SIMT_DETERMINE);
} // namespace optiling
