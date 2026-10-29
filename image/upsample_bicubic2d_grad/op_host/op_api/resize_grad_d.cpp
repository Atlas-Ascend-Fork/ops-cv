/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "resize_grad_d.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeGradD);

static const string ALIGN_CORNERS = "align_corners";
static const string HALF_PIXEL = "half_pixel";
static const string ROUND_PREFER_FLOOR = "round_prefer_floor";
static const string DATA_FORMAT = "HWNC";
static const float CUBIC_COEFF_A = -0.75f;
static const float EXTRAPOLATION_VALUE = 0.0;
static const int64_t EXCLUDE_OUTSIDE = 0;

inline const string& GetCoordinateTransformationModeStr(const bool alignCorners)
{
    if (alignCorners) {
        return ALIGN_CORNERS;
    }
    return HALF_PIXEL;
}

const aclTensor* ResizeGradD(const aclTensor* grads, const aclIntArray* inputSize, const aclFloatArray* scales,
                             const bool alignCorners, const aclTensor* y, const std::string& mode,
                             aclOpExecutor* executor)
{
    L0_DFX(ResizeGradD, grads, inputSize, scales, alignCorners, mode);

    aclTensor* out = executor->AllocTensor(y->GetViewShape(), y->GetDataType(), y->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);

    const int64_t roi[0] = {};
    aclIntArray* roiArray = executor->AllocIntArray(roi, 0);
    CHECK_RET(roiArray != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeGradD, OP_INPUT(grads), OP_OUTPUT(out),
        OP_ATTR(inputSize, roiArray, scales, GetCoordinateTransformationModeStr(alignCorners), CUBIC_COEFF_A,
                EXCLUDE_OUTSIDE, EXTRAPOLATION_VALUE, mode, ROUND_PREFER_FLOOR, DATA_FORMAT));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "ResizeGradDAiCore ADD_TO_LAUNCHER_LIST_AICORE failed; "
                                                        "propagate the first error reported by the failing stage."),
        return nullptr);
    return out;
}
} // namespace l0op
