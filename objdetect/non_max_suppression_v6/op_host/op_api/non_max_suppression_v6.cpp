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
 * \file non_max_suppression.cpp
 * \brief
 */
#include "non_max_suppression_v6.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NonMaxSuppressionV6);

const aclTensor* NonMaxSuppressionV6(const aclTensor* boxes, const aclTensor* scores,
                                     aclIntArray* maxOutputBoxesPerClass, const aclFloatArray* iouThreshold,
                                     aclFloatArray* scoreThreshold, int centerPointBox, int maxBoxesSize,
                                     const aclTensor* selectedIndices, aclOpExecutor* executor)
{
    L0_DFX(NonMaxSuppressionV6, boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox,
           maxBoxesSize);

    auto maxOutputBoxesSize = executor->ConvertToTensor(maxOutputBoxesPerClass, op::ToOpDataType(ACL_INT32));
    auto iouThd = executor->ConvertToTensor(iouThreshold, op::ToOpDataType(ACL_FLOAT));
    auto scoreThd = executor->ConvertToTensor(scoreThreshold, op::ToOpDataType(ACL_FLOAT));

    aclTensor* out = executor->AllocTensor(selectedIndices->GetViewShape(), selectedIndices->GetDataType(),
                                           selectedIndices->GetViewFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed");
        return nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NonMaxSuppressionV6,
                                           OP_INPUT(boxes, scores, maxOutputBoxesSize, iouThd, scoreThd),
                                           OP_OUTPUT(out), OP_ATTR(centerPointBox, maxBoxesSize));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "NonMaxSuppressionV6AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}
} // namespace l0op
