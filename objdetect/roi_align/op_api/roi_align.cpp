/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "roi_align.h"
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
static const size_t DIM_0 = 0;
static const size_t DIM_1 = 1;
static const size_t DIM_2 = 2;
static const size_t DIM_3 = 3;

OP_TYPE_REGISTER(ROIAlign);

const aclTensor* ROIAlign(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                          float spatialScale, int outputHeight, int outputWidth, int samplingRatio, const char* mode,
                          aclOpExecutor* executor)
{
    L0_DFX(ROIAlign, self, rois, batchIndices, spatialScale, outputHeight, outputWidth, samplingRatio, mode);

    op::Shape outStorageShape = self->GetStorageShape();
    op::Shape outOriginalShape = self->GetOriginalShape();
    op::Shape roisStorageShape = rois->GetStorageShape();
    auto numRois = roisStorageShape.GetDim(0);
    outStorageShape.SetDim(DIM_0, numRois);
    outStorageShape.SetDim(DIM_2, outputHeight);
    outStorageShape.SetDim(DIM_3, outputWidth);
    outOriginalShape.SetDim(DIM_0, numRois);
    outOriginalShape.SetDim(DIM_2, outputHeight);
    outOriginalShape.SetDim(DIM_3, outputWidth);

    auto out = executor->AllocTensor(outStorageShape, outOriginalShape, self->GetDataType(), self->GetStorageFormat(),
                                     self->GetOriginalFormat());
    if (out == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                               "ROIAlign AllocTensor returned nullptr; propagate allocator first error");
        return nullptr;
    }

    // 调用device的RoiAlign算子
    ADD_TO_LAUNCHER_LIST_AICORE(ROIAlign, OP_INPUT(self, rois, batchIndices), OP_OUTPUT(out),
                                OP_ATTR(spatialScale, outputHeight, outputWidth, samplingRatio, 0, mode));

    return out;
}

const aclTensor* ROIAlignV2(const aclTensor* self, const aclTensor* boxes, float spatialScale, int64_t outputHeight,
                            int64_t outputWidth, int64_t samplingRatio, const char* mode, int64_t roiEndMode,
                            aclOpExecutor* executor)
{
    L0_DFX(ROIAlignV2, self, boxes, spatialScale, outputHeight, outputWidth, samplingRatio, mode, roiEndMode);

    op::Shape outStorageShape = self->GetStorageShape();
    op::Shape outOriginalShape = self->GetOriginalShape();
    op::Shape roisStorageShape = boxes->GetStorageShape();
    auto numRois = roisStorageShape.GetDim(0);
    outStorageShape.SetDim(DIM_0, numRois);
    outStorageShape.SetDim(DIM_2, outputHeight);
    outStorageShape.SetDim(DIM_3, outputWidth);
    outOriginalShape.SetDim(DIM_0, numRois);
    outOriginalShape.SetDim(DIM_2, outputHeight);
    outOriginalShape.SetDim(DIM_3, outputWidth);

    auto out = executor->AllocTensor(outStorageShape, outOriginalShape, self->GetDataType(), self->GetStorageFormat(),
                                     self->GetOriginalFormat());
    if (out == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                               "ROIAlignV2 AllocTensor returned nullptr; propagate allocator first error");
        return nullptr;
    }

    // 调用device的RoiAlign算子
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ROIAlign, OP_INPUT(self, boxes), OP_OUTPUT(out),
        OP_ATTR(spatialScale, outputHeight, outputWidth, samplingRatio, roiEndMode, mode));
    if (ret != ACLNN_SUCCESS) {
        const std::string reason = "ADD_TO_LAUNCHER_LIST_AICORE failed, rc=" +
                                   std::to_string(static_cast<int32_t>(ret)) + ". Kernel is ROIAlign.";
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "%s", reason.c_str());
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"ROIAlign", reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
        return nullptr;
    }

    return out;
}
} // namespace l0op
