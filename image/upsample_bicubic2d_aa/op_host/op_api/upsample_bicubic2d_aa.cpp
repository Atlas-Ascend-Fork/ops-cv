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
 * \file upsample_bicubic2d_aa.cpp
 * \brief
 */
#include "upsample_bicubic2d_aa.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBicubic2dAA);

const aclTensor* UpsampleBicubic2dAA(const aclTensor* x, const aclIntArray* size, const bool alignCorners,
                                     const aclTensor* y, const float scalesH, const float scalesW,
                                     aclOpExecutor* executor)
{
    L0_DFX(UpsampleBicubic2dAA, x, size, alignCorners, scalesH, scalesW);

    aclTensor* out = executor->AllocTensor(y->GetViewShape(), x->GetDataType(), y->GetViewFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBicubic2dAA, OP_INPUT(x), OP_OUTPUT(out),
                                           OP_ATTR(size, alignCorners, scalesH, scalesW));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "UpsampleBicubic2dAAAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}
} // namespace l0op
