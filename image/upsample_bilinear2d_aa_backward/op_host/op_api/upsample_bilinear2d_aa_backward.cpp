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
 * \file upsample_bilinear2d_aa_backward.cpp
 * \brief
 */
#include "upsample_bilinear2d_aa_backward.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBilinear2dAABackward);

const aclTensor* UpsampleBilinear2dAABackward(const aclTensor* gradOutput, const aclIntArray* outputSize,
                                              const aclIntArray* inputSize, aclTensor* output, bool alignCorners,
                                              float scales_h, float scales_w, aclOpExecutor* executor)
{
    L0_DFX(UpsampleBilinear2dAABackward, gradOutput, outputSize, inputSize, output, alignCorners, scales_h, scales_w);

    auto out = executor->AllocTensor(output->GetViewShape(), gradOutput->GetDataType(), output->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBilinear2dAABackward, OP_INPUT(gradOutput), OP_OUTPUT(out),
                                           OP_ATTR(outputSize, inputSize, alignCorners, scales_h, scales_w));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                                    "UpsampleBilinear2dAABackwardAiCore ADD_TO_LAUNCHER_LIST_AICORE failed; propagate "
                                    "the first error reported by the failing stage."),
             return nullptr);

    return out;
}
} // namespace l0op
