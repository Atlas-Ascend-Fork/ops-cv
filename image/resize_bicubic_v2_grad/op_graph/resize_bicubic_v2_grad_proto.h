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
 * \file resize_bicubic_v2_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RESIZE_BICUBIC_V2_GRAD_H_
#define OPS_OP_PROTO_INC_RESIZE_BICUBIC_V2_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Backwards calculation of ResizeBicubicV2.

*@par Inputs:
*@li grads: A 4D tensor, represents the gradient of output of ResizeBicubicV2. Format must be NCHW or NHWC. Data dtype
     must be float32, float16 or bfloat16.
*@li original_image: A 4D tensor, represents the resized image of ResizeBicubicV2. Format and data dtype must be the
     same as grads.The N, C dimension must be the same as grads.

*@par Outputs:
*y: A 4D tensor, represents the gradient of original_image. Format and data dtype and shape must be the same as
original_image.

*@par Attributes:
*@li align_corners: An optional bool. If true, the centers of the 4 corner pixels of the input and output tensors are
     aligned, preserving the values at the corner pixels. If false, calculate interpolation using half pixel centers.
     Defaults to false.
*@li scales: An optional listfloat with two elements, the scale factors for the H and W axes, respectively. When
     align_corners is false, a value greater than 0 is used for the corresponding axis; otherwise, its scale is
     calculated from the shapes of grads and y. When align_corners is true, scales are calculated from the tensor
     shapes and this attribute does not take effect. Defaults to {0.0f, 0.0f}.

*@par Third-party framework compatibility
*Compatible with PyTorch upsample_bicubic2d_backward operator.
*/
REG_OP(ResizeBicubicV2Grad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(original_image, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeBicubicV2Grad)
} // namespace ge

#endif // OPS_OP_PROTO_INC_RESIZE_BICUBIC_V2_GRAD_H_
