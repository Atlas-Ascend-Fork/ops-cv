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
 * \file resize_bilinear_v2_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RESIZE_BILINEAR_V2_GRAD_H_
#define OPS_OP_PROTO_INC_RESIZE_BILINEAR_V2_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**

*@brief Backwards calculation of ResizeBilinearV2.

*@par Inputs:
*@li grads: A 4D Tensor, represents the gradient of output of ResizeBilinearV2. Format must be NCHW or NHWC.
     Dtype must be float32, float16 or bfloat16.
*@li original_image: A 4D Tensor, represents the resized image of ResizeBilinearV2. Format must be the same as grads.
     When the dtype of grads is float16 or bfloat16, the dtype of original_image must be the same as grads, when the
     dtype of grads is float32, the dtype of original_image can be float32, float16 or bfloat16. The N, C dimension
     must be the same as grads.

*@par Outputs:
*y: A 4D Tensor, represents the gradient of original_image. Format and dtype and shape must be the same as
original_image.

*@par Attributes:
*@li align_corners: An optional bool. If true, the centers of the 4 corner pixels of the input and output tensors are
     aligned, preserving the values at the corner pixels. Defaults to false.
*@li half_pixel_centers: An optional bool. If true, the center of pixels locate in [0.5, 0.5]. Defaults to false.
*@li scales: An optional listfloat which has two elements. The first element represents the ratio of the H axis
     subscripts of pixels in grads to the H axis subscripts of pixels in y, only takes effect when align_corners is
     false and the value is greater than 0. The second element represents the ratio of the W axis subscripts of pixels
in grads to the W axis subscripts of pixels in y, only takes effect when align_corners is false and the value is greater
than 0.0f. Defaults to {0.0f, 0.0f}.

*@par Third-party framework compatibility
*Compatible with tensorflow and pytorch ResizeBilinearV2Grad operator.
*/
REG_OP(ResizeBilinearV2Grad)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(original_image, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeBilinearV2Grad)
} // namespace ge

#endif // OPS_OP_PROTO_INC_RESIZE_BILINEAR_V2_GRAD_H_
