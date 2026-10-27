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
 * \file non_max_suppression.cpp
 * \brief
 */
#include "non_max_suppression_v6.h"
#include <string>
#include <vector>
#include "base/err_msg.h"
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
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                               "NonMaxSuppressionV6 AllocTensor returned nullptr; propagate allocator first error");
        return nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NonMaxSuppressionV6,
                                           OP_INPUT(boxes, scores, maxOutputBoxesSize, iouThd, scoreThd),
                                           OP_OUTPUT(out), OP_ATTR(centerPointBox, maxBoxesSize));
    if (ret != ACLNN_SUCCESS) {
        const std::string reason = "ADD_TO_LAUNCHER_LIST_AICORE failed, rc=" +
                                   std::to_string(static_cast<int32_t>(ret)) +
                                   ". Kernel is NonMaxSuppressionV6, format is ND.";
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "%s", reason.c_str());
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"NonMaxSuppressionV6", reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
        return nullptr;
    }
    return out;
}
} // namespace l0op
