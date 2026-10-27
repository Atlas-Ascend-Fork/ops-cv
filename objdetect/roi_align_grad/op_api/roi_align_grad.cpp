/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// check
#include "roi_align_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
static const size_t DIM_0 = 0;
static const size_t DIM_1 = 1;
static const size_t DIM_2 = 2;
static const size_t DIM_3 = 3;

OP_TYPE_REGISTER(ROIAlignGrad);

const aclTensor* ROIAlignGrad(const aclTensor* gradOutput, const aclTensor* boxes, const aclTensor* roisN,
                              const aclIntArray* inputShape, int64_t pooledHeight, int64_t pooledWidth,
                              float spatialScale, int64_t samplingRatio, int64_t roiEndMode, aclOpExecutor* executor)
{
    L0_DFX(ROIAlignGrad, gradOutput, boxes, inputShape, pooledHeight, pooledWidth, spatialScale, samplingRatio,
           roiEndMode);

    // A5(Regbase)不转私有格式，直接用 4D ViewShape + ND 格式；其他芯片经 TransData 转 5D，沿用 StorageShape
    bool is950 = (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510);
    if (is950) {
        op::Shape outShape = gradOutput->GetViewShape();
        outShape.SetDim(DIM_0, (*inputShape)[DIM_0]);
        outShape.SetDim(DIM_2, (*inputShape)[DIM_2]);
        outShape.SetDim(DIM_3, (*inputShape)[DIM_3]);
        auto gradInput = executor->AllocTensor(outShape, gradOutput->GetDataType(), gradOutput->GetStorageFormat());
        if (gradInput == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradInput tensor failed");
            return nullptr;
        }
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
            ROIAlignGrad, OP_INPUT(gradOutput, boxes, roisN), OP_OUTPUT(gradInput),
            OP_ATTR(inputShape, pooledWidth, pooledHeight, spatialScale, samplingRatio, roiEndMode));
        OP_CHECK(ret == ACLNN_SUCCESS,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ROIAlignGrad ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
        return gradInput;
    }
    op::Shape outStorageShape = gradOutput->GetStorageShape();
    op::Shape outOriginalShape = gradOutput->GetOriginalShape();
    outStorageShape.SetDim(DIM_0, (*inputShape)[DIM_0]);
    outStorageShape.SetDim(DIM_2, (*inputShape)[DIM_2]);
    outStorageShape.SetDim(DIM_3, (*inputShape)[DIM_3]);
    outOriginalShape.SetDim(DIM_0, (*inputShape)[DIM_0]);
    outOriginalShape.SetDim(DIM_2, (*inputShape)[DIM_2]);
    outOriginalShape.SetDim(DIM_3, (*inputShape)[DIM_3]);
    auto gradInput = executor->AllocTensor(outStorageShape, outOriginalShape, gradOutput->GetDataType(),
                                           gradOutput->GetStorageFormat(), gradOutput->GetOriginalFormat());
    if (gradInput == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradInput tensor failed");
        return nullptr;
    }

    // 调用device的RoiAlignGrad算子
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ROIAlignGrad, OP_INPUT(gradOutput, boxes), OP_OUTPUT(gradInput),
        OP_ATTR(inputShape, pooledWidth, pooledHeight, spatialScale, samplingRatio, roiEndMode));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ROIAlignGrad ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);

    return gradInput;
}
} // namespace l0op
