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
 * \file upsample_bicubic2d.cpp
 * \brief
 */
#include "upsample_bicubic2d.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UpsampleBicubic2d);

const aclTensor* UpsampleBicubic2d(const aclTensor* input, const aclIntArray* output_size, const bool align_corners,
                                   const float scales_h, const float scales_w, const aclTensor* output,
                                   aclOpExecutor* executor)
{
    L0_DFX(UpsampleBicubic2d, input, output_size, align_corners, scales_h, scales_w, output);

    aclTensor* out = nullptr;
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2002 || curArch == NpuArch::DAV_3002) {
        out = executor->AllocTensor(output->GetViewShape(), output->GetDataType(), output->GetViewFormat());
    } else {
        out = executor->AllocTensor(output->GetViewShape(), op::DataType::DT_FLOAT, output->GetViewFormat());
    }
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBicubic2d, OP_INPUT(input), OP_OUTPUT(out),
                                           OP_ATTR(output_size, align_corners, scales_h, scales_w));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "UpsampleBicubic2dAiCore ADD_TO_LAUNCHER_LIST_AICORE failed; "
                                                        "propagate the first error reported by the failing stage."),
        return nullptr);
    return out;
}
} // namespace l0op
