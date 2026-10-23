/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

/*!
 * \file grid_sampler3_d_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GRID_SAMPLE3DGRAD_OPS_H_
#define OPS_OP_PROTO_INC_GRID_SAMPLE3DGRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the gradients of GridSampler3D.

* @par Inputs:
* @li grad: 5-D Tensor with shape `[batch, channels, depth, height, width]`.
* @li x: 5-D Tensor with shape `[batch, channels, depth, height, width]`.
* @li grid: flow field grid, 5-D Tensor with shape `[batch, depth, height, width, 2]`.

* @par Attributes:
* @li interpolation_mode: An optional string specifying the interpolation method.
* "bilinear" and "nearest" are supported. Defaults to "bilinear".
* @li padding_mode: An optional string specifying the pad method.
* One of "zeros", "border", or "reflection". Defaults to "zeros".
* @li align_corners: An optional bool. If "true", the centers of the corner
* pixels of the input and output tensors are aligned. Defaults to "false" .

* @par Outputs:
* dx: Returns 5-D Tensor with the same dtype and shape as `x`.
* dgrid: Returns 5-D Tensor with the same dtype and shape as `grid`.

* @par Third-party framework compatibility
* Compatible with pytorch GridSampler3DGrad operator.
*/
REG_OP(GridSampler3DGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(grid, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(dgrid, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(interpolation_mode, String, "bilinear")
    .ATTR(padding_mode, String, "zeros")
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(GridSampler3DGrad)

} // namespace ge

#endif
