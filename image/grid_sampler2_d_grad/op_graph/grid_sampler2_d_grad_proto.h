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
 * \file grid_sampler2_d_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GRID_SAMPLE2DGRAD_OPS_H_
#define OPS_OP_PROTO_INC_GRID_SAMPLE2DGRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the gradients of GridSampler2D.

* @par Inputs:
* @li grad: A 4-D tensor with shape `[batch, out_height, out_width, channels]`.
* The gradient of the output of the previous layer during backpropagation.
* Must be one of the following types: `[float16, float32, bfloat16, double]`.
* The format must be NHWC.
* @li x: A 4-D tensor with shape `[batch, height, width, channels]`.
* Input tensor representing backpropagation.
* Must be one of the following types: `[float16, float32, bfloat16, double]`.
* The format must be NHWC.
* @li grid: Flow field grid, 4-D tensor with shape `[batch, out_height, out_width, 2]`. Must be one of the following
* types:
* `[float16, float32, bfloat16, double]`.
* The format must be ND.

* @par Attributes:
* @li interpolation_mode: An optional string specifying the interpolation method.
* 'bilinear' and 'nearest' are supported now. Defaults to "bilinear".
* @li padding_mode: An optional string specifying the pad method: "zeros", "border", or "reflection".
* Defaults to "zeros".
* @li align_corners: An optional bool. Indicates the mapping mode between the coordinates of the feature map and the
feature value.
* If "true", the centers of the corner pixels of the input and output tensors are aligned. Defaults to false.

* @par Outputs:
* @li dx: Indicates the output gradient of backpropagation. Returns 4-D tensor with the same dtype, format and shape as
* `x`.
* @li dgrid: Indicates the grid gradient. Returns 4-D tensor with the same dtype, format and shape as `grid`.

* @par Third-party framework compatibility
* Compatible with pytorch GridSampler2DGrad operator.
*/
REG_OP(GridSampler2DGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(grid, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OUTPUT(dgrid, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(interpolation_mode, String, "bilinear")
    .ATTR(padding_mode, String, "zeros")
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(GridSampler2DGrad)

} // namespace ge

#endif
