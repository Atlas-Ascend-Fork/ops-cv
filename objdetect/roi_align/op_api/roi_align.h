/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_ROI_ALIGN_H_
#define OP_API_INC_LEVEL0_ROI_ALIGN_H_

#include "opdev/op_executor.h"

namespace l0op {
// used in aclnnRoiAlign
const aclTensor* ROIAlign(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                          float spatialScale, int outputHeight, int outputWidth, int samplingRatio, const char* mode,
                          aclOpExecutor* executor);

// used in aclnnRoiAlignV2
const aclTensor* ROIAlignV2(const aclTensor* self, const aclTensor* boxes, float spatialScale, int64_t outputHeight,
                            int64_t outputWidth, int64_t samplingRatio, const char* mode, int64_t roiEndMode,
                            aclOpExecutor* executor);

const aclTensor* ROIAlignV2(const aclTensor* self, const aclTensor* boxes, const aclTensor* batchIndices,
                            float spatialScale, int64_t outputHeight, int64_t outputWidth, int64_t samplingRatio,
                            const char* mode, int64_t roiEndMode, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_ROI_ALIGN_H_
