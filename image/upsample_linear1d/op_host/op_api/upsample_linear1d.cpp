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
 * \file upsample_linear1d.cpp
 * \brief
 */
#include "upsample_linear1d.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/common_types.h"

#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleLinear1d);

static const string LINEAR_MODE = "linear";
static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DIM_THREE = 3;

const aclTensor* UpsampleLinear1dNcdhw(const aclTensor* x, const aclTensor* outputSize, const bool alignCorners,
                                       const aclTensor* y, const double scale, aclOpExecutor* executor)
{
    L0_DFX(UpsampleLinear1dNcdhw, x, outputSize, alignCorners, scale);

    float realScale = 0.0f;
    if (scale > 0) {
        realScale = static_cast<float>(1.0 / scale);
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleLinear1d, OP_INPUT(x, outputSize), OP_OUTPUT(y),
                                           OP_ATTR(alignCorners, realScale));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "UpsampleLinear1dAICORE ADD_TO_LAUNCHER_LIST_AICORE failed; "
                                                        "propagate the first error reported by the failing stage."),
        return nullptr);
    return y;
}
} // namespace l0op
