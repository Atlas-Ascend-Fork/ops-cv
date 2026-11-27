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
     false and the value is greater than 0.0f. The second element represents the ratio of the W axis subscripts of
     pixels in grads to the W axis subscripts of pixels in y, only takes effect when align_corners is false and the
     value is greater than 0.0f. When align_corners is true, both elements are ignored and the scaling ratio is derived
     from the H/W extents of grads and y. Defaults to {0.0f, 0.0f}.
*@li Note: align_corners and half_pixel_centers must not be true at the same time, otherwise the operator reports an
     invalid-value error.

*@par Restrictions:
*@li Empty tensors are not supported. The N, H and W extents of grads and y must all be greater than zero; a
     non-positive N, H or W extent is rejected during tiling with a shape error.
*@li Dynamic shape and dynamic rank are declared as supported at registration (DynamicShapeSupportFlag,
     DynamicRankSupportFlag), so -1 (unknown dimension) and -2 (unknown rank) placeholders are accepted at compile
     time; at execution time the real shape must be a determined 4-D shape satisfying the restrictions above.

*@par Third-party framework compatibility
*The operator interface follows the semantics of the tensorflow ResizeBilinearGrad and pytorch
*upsample_bilinear2d_backward operators. The delivered runtime binding of this repository covers the GE graph mode
*(this IR) and the aclnn entry aclnnUpsampleBilinear2dBackward only; no TensorFlow or PyTorch end-to-end adapter is
*delivered here.
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
