/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

#include "gaussian_filter_3_d.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "opdev/op_def.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GaussianFilter3D);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                          DataType::DT_BF16};

static inline const aclTensor* GaussianFilter3DAiCore(const aclTensor* x, float sigma, const char* mode, aclTensor* out,
                                                      aclOpExecutor* executor)
{
    L0_DFX(GaussianFilter3DAiCore, x, sigma, mode, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore GaussianFilter3D算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GaussianFilter3D, OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(sigma, mode));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GaussianFilter3DAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

const aclTensor* GaussianFilter3D(const aclTensor* x, double sigma, const char* mode, aclOpExecutor* executor)
{
    L0_DFX(GaussianFilter3D, x, sigma, mode);

    if (!CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GaussianFilter3D dtype not support.");
        return nullptr;
    }

    // 根据输入shape申请输出tensor（与输入同shape同dtype同layout）
    auto out = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    return GaussianFilter3DAiCore(x, static_cast<float>(sigma), mode, out, executor);
}
} // namespace l0op
