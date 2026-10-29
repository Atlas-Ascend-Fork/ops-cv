/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ssim_proto.h
 * \brief ssim operator graph interface
 */
#ifndef OPS_CV_IMAGE_SSIM_PROTO_H_
#define OPS_CV_IMAGE_SSIM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Ssim operator interface implementation. \n
*  calculating the structural similarity (SSIM) between two images. \n
*  MSSIM = mean( l(x1,x2) * c(x1,x2) * s(x1,x2) ) over the valid region
*
* @par Inputs
* Two inputs, including:
* @li x1: A Tensor. Support dtype: [uint8, float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [uint8, float32], support format: [ND].

* @par Outputs
* One output, including:
* @li yOut: A Tensor. Support dtype: [float32], support format: [ND]. The MSSIM scalar.
*/
REG_OP(Ssim)
    .INPUT(x1, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_UINT8, DT_FLOAT}))
    .OUTPUT(yOut, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Ssim)

} // namespace ge

#endif // OPS_CV_IMAGE_SSIM_PROTO_H_
