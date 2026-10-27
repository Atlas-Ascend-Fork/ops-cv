/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "iou_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IouV2);

const aclTensor* IouV2(const aclTensor* bBoxes, const aclTensor* gtBoxes, const char* mode, float eps, bool aligned,
                       aclOpExecutor* executor)
{
    L0_DFX(IouV2, bBoxes, gtBoxes, mode, eps, aligned);

    // 根据算子语义，推导算子输出shape
    op::Shape outShape;
    if (aligned) {
        outShape = {bBoxes->GetViewShape().GetDim(1), 1};
    } else {
        outShape = {gtBoxes->GetViewShape().GetDim(0), bBoxes->GetViewShape().GetDim(0)};
    }

    auto out = executor->AllocTensor(outShape, bBoxes->GetDataType(), op::Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IouV2 alloc out tensor failed");
        return nullptr;
    }

    // 调用device的IouV2算子
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IouV2, OP_INPUT(bBoxes, gtBoxes), OP_OUTPUT(out),
                                           OP_ATTR(mode, eps, aligned));
    OP_LOGI("IouV2 ret:%d, out:%p", ret, out);

    return out;
}
} // namespace l0op
