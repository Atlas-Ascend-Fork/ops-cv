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
 * \file roi_pooling_grad_with_arg_max.cpp
 * \brief
 */

#include "roi_pooling_grad_with_arg_max.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RoiPoolingGradWithArgMax);

const aclTensor* RoiPoolingGradWithArgMax(const aclTensor* grad, const aclTensor* x, const aclTensor* rois,
                                          const aclTensor* roiActualNumOptional, const aclTensor* argmax,
                                          int64_t pooledH, int64_t pooledW, double spatialScaleH, double spatialScaleW,
                                          int64_t poolChannel, aclOpExecutor* executor)
{
    L0_DFX(RoiPoolingGradWithArgMax, grad, x, rois, roiActualNumOptional, argmax, pooledH, pooledW, spatialScaleH,
           spatialScaleW, poolChannel);
    auto out = executor->AllocTensor(grad->GetDataType(), grad->GetStorageFormat(), grad->GetOriginalFormat());
    INFER_SHAPE(RoiPoolingGradWithArgMax, OP_INPUT(grad, x, rois, roiActualNumOptional, argmax), OP_OUTPUT(out),
                OP_ATTR(pooledH, pooledW, spatialScaleH, spatialScaleW, poolChannel));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RoiPoolingGradWithArgMax,
                                           OP_INPUT(grad, x, rois, roiActualNumOptional, argmax), OP_OUTPUT(out),
                                           OP_ATTR(pooledH, pooledW, spatialScaleH, spatialScaleW, poolChannel));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                                    "RoiPoolingGradWithArgMaxAiCore ADD_TO_LAUNCHER_LIST_AICORE failed; "
                                    "propagate the first error reported by the failing stage."),
             return nullptr);
    return out;
}
} // namespace l0op
