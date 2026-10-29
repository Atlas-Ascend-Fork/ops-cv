/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_d.cpp
 * \brief
 */
#include "resize_d.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ResizeD);

static const string ALIGN_CORNERS = "align_corners";
static const string HALF_PIXEL = "half_pixel";
static const string LINEAR_MODE = "linear";
static const string ROUND_PREFER_FLOOR = "round_prefer_floor";
static const string DATA_FORMAT = "HWNC";
static const float CUBIC_COEFF_A = -0.75f;
static const float EXTRAPOLATION_VALUE = 0.0;
static const int64_t EXCLUDE_OUTSIDE = 0;
static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DIM_THREE = 3;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

inline const string& GetCoordinateTransformationModeStr(const bool alignCorners)
{
    if (alignCorners) {
        return ALIGN_CORNERS;
    }
    return HALF_PIXEL;
}

const aclTensor* ResizeD(const aclTensor* x, const aclIntArray* size, const bool alignCorners, const aclTensor* y,
                         const aclFloatArray* scales, const std::string& mode, aclOpExecutor* executor)
{
    L0_DFX(ResizeD, x, size, alignCorners, scales, mode);

    const int64_t roi[0] = {};
    aclIntArray* roiArray = executor->AllocIntArray(roi, 0);
    CHECK_RET(roiArray != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ResizeD, OP_INPUT(x), OP_OUTPUT(y),
        OP_ATTR(size, scales, roiArray, GetCoordinateTransformationModeStr(alignCorners), CUBIC_COEFF_A,
                EXCLUDE_OUTSIDE, EXTRAPOLATION_VALUE, mode, ROUND_PREFER_FLOOR, DATA_FORMAT));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "ResizeDAiCore ADD_TO_LAUNCHER_LIST_AICORE failed; propagate "
                                                        "the first error reported by the failing stage."),
        return nullptr);
    return y;
}
} // namespace l0op
