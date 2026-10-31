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
 * \file integral_image.h
 * \brief IntegralImage L0 层（l0op）接口声明
 */

#ifndef OP_API_INC_LEVEL0_OP_INTEGRAL_IMAGE_OP_H_
#define OP_API_INC_LEVEL0_OP_INTEGRAL_IMAGE_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief 下发 IntegralImage（积分图）算子到 AI Core 执行器。
 * @param [in]  image              输入图像，(H,W) 或 (H,W,C) HWC 布局，ND，uint8/float16/float32
 * @param [in]  sdepth             输出深度选择：-1 自动、0 int32（仅 uint8）、1 float32
 * @param [out] satOut             必选输出：积分图 (H+1,W+1[,C])
 * @param [out] sqsumOutOptional   可选输出：平方和积分图，传 nullptr 表示本次不计算
 * @param [out] tiltedOutOptional  可选输出：倾斜三角区域累加，传 nullptr 表示本次不计算
 * @param [in]  executor           执行器
 * @return 成功返回 satOut，失败返回 nullptr
 */
const aclTensor* IntegralImage(const aclTensor* image, int64_t sdepth, aclTensor* satOut, aclTensor* sqsumOutOptional,
                               aclTensor* tiltedOutOptional, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_INTEGRAL_IMAGE_OP_H_
