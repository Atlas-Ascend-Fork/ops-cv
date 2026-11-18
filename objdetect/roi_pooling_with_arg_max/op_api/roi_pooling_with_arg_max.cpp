/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_with_arg_max.cpp
 * \brief Level0 op implementation for RoiPoolingWithArgMax
 */
#include "roi_pooling_with_arg_max.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
using namespace op;

namespace l0op {
static const size_t DIM_0 = 0;
static const size_t DIM_1 = 1;
static const size_t DIM_2 = 2;
static const size_t DIM_3 = 3;

OP_TYPE_REGISTER(RoiPoolingWithArgMax);

const aclTensor* RoiPoolingWithArgMax(const aclTensor* x, const aclTensor* rois, const aclTensor* roi_actual_num,
                                      int64_t pooled_h, int64_t pooled_w, float spatial_scale_h, float spatial_scale_w,
                                      int64_t pool_channel, aclOpExecutor* executor, const aclTensor** out_y,
                                      const aclTensor** out_argmax)
{
    L0_DFX(RoiPoolingWithArgMax, x, rois, roi_actual_num, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w,
           pool_channel);

    const auto& xShape = x->GetViewShape();
    const auto& roisShape = rois->GetViewShape();
    int64_t numRois = static_cast<int64_t>(roisShape.GetDim(DIM_0));

    op::Shape yStorageShape = x->GetStorageShape();
    op::Shape yOriginalShape = x->GetOriginalShape();
    yStorageShape.SetDim(DIM_0, numRois);
    yStorageShape.SetDim(DIM_1, xShape.GetDim(DIM_1));
    yStorageShape.SetDim(DIM_2, pooled_h);
    yStorageShape.SetDim(DIM_3, pooled_w);
    yOriginalShape.SetDim(DIM_0, numRois);
    yOriginalShape.SetDim(DIM_1, xShape.GetDim(DIM_1));
    yOriginalShape.SetDim(DIM_2, pooled_h);
    yOriginalShape.SetDim(DIM_3, pooled_w);

    auto y = executor->AllocTensor(yStorageShape, yOriginalShape, x->GetDataType(), x->GetStorageFormat(),
                                   x->GetOriginalFormat());
    if (y == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                               "RoiPoolingWithArgMax alloc y tensor failed; propagate the first error "
                               "reported by the failing stage.");
        return nullptr;
    }

    op::Shape argmaxStorageShape = yStorageShape;
    op::Shape argmaxOriginalShape = yOriginalShape;

    auto argmax = executor->AllocTensor(argmaxStorageShape, argmaxOriginalShape, op::DataType::DT_INT32,
                                        op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    if (argmax == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                               "RoiPoolingWithArgMax alloc argmax tensor failed; propagate the first error "
                               "reported by the failing stage.");
        return nullptr;
    }

    INFER_SHAPE(RoiPoolingWithArgMax, OP_INPUT(x, rois, roi_actual_num), OP_OUTPUT(y, argmax),
                OP_ATTR(pooled_h, pooled_w, spatial_scale_h, spatial_scale_w, pool_channel));
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RoiPoolingWithArgMax, OP_INPUT(x, rois, roi_actual_num),
                                           OP_OUTPUT(y, argmax),
                                           OP_ATTR(pooled_h, pooled_w, spatial_scale_h, spatial_scale_w, pool_channel));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "RoiPoolingWithArgMax ADD_TO_LAUNCHER_LIST_AICORE failed; "
                                                        "propagate the first error reported by the failing stage."),
        return nullptr);

    *out_y = y;
    *out_argmax = argmax;
    return y;
}
} // namespace l0op
