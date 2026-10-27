/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ciou.h"
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
OP_TYPE_REGISTER(CIoU);

const std::tuple<aclTensor*, aclTensor*> CIoU(const aclTensor* bBoxes, const aclTensor* gtBoxes, bool trans,
                                              bool isCross, const char* mode, bool atanSubFlag, aclOpExecutor* executor)
{
    L0_DFX(CIoU, bBoxes, gtBoxes, trans, isCross, mode, atanSubFlag);
    op::Shape outShape;
    if (isCross == false) {
        outShape = {1, bBoxes->GetViewShape().GetDim(1)};
    } else {
        outShape = {gtBoxes->GetViewShape().GetDim(1), bBoxes->GetViewShape().GetDim(1)};
    }
    auto overlap = executor->AllocTensor(outShape, bBoxes->GetDataType(), op::Format::FORMAT_ND);
    auto atan_sub = executor->AllocTensor(outShape, bBoxes->GetDataType(), op::Format::FORMAT_ND);
    if (overlap == nullptr || atan_sub == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                               "CIoU AllocTensor returned nullptr; propagate allocator first error");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CIoU, OP_INPUT(bBoxes, gtBoxes), OP_OUTPUT(overlap, atan_sub),
                                           OP_ATTR(trans, isCross, mode, atanSubFlag));
    if (ret != ACL_SUCCESS) {
        const std::string reason = "ADD_TO_LAUNCHER_LIST_AICORE failed, rc=" +
                                   std::to_string(static_cast<int32_t>(ret)) + ". Kernel is CIoU, format is ND.";
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "%s", reason.c_str());
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"CIoU", reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(overlap, atan_sub);
}
} // namespace l0op
