/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file integral_image.cpp
 * \brief IntegralImage L0 层（l0op）实现：拼装输入/输出/属性并下发 launcher
 */

#include "integral_image.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(IntegralImage);

namespace {

const aclTensor* LaunchIntegralImage(const aclTensor* image, int64_t sdepth, aclTensor* satOut,
                                     aclTensor* sqsumOutOptional, aclTensor* tiltedOutOptional, aclOpExecutor* executor)
{
    // sqsumOutOptional / tiltedOutOptional 传 nullptr 表示「本形态不使能该输出」：
    // 框架按 optional 输出未提供的语义处理，tiling 侧读到 null shape 后置
    // sqsumEnabled / tiltedEnabled = 0，kernel 直接跳过对应分支。
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IntegralImage, OP_INPUT(image),
                                           OP_OUTPUT(satOut, sqsumOutOptional, tiltedOutOptional), OP_ATTR(sdepth));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IntegralImage ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return satOut;
}

} // namespace

const aclTensor* IntegralImage(const aclTensor* image, int64_t sdepth, aclTensor* satOut, aclTensor* sqsumOutOptional,
                               aclTensor* tiltedOutOptional, aclOpExecutor* executor)
{
    L0_DFX(IntegralImage, image, sdepth, satOut, sqsumOutOptional, tiltedOutOptional);
    return LaunchIntegralImage(image, sdepth, satOut, sqsumOutOptional, tiltedOutOptional, executor);
}

} // namespace l0op
