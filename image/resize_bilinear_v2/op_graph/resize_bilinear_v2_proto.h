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
 * \file resize_bilinear_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RESIZE_BILINEAR_V2_H_
#define OPS_OP_PROTO_INC_RESIZE_BILINEAR_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Resize images to size using bilinear interpolation . \n

*@par Inputs:
*Input images must be a 4-D tensor. Inputs include:
*@li x: 4-D tensor. Must set the format,
supported format list ["NCHW, NHWC"]. Empty tensor is not supported: the op is
rejected when the element count of x is 0. Must be a contiguous tensor.
*@li size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new
size for the images. Both elements must be positive . \n

*@par Attributes:
* @li align_corners: An optional bool. If true, the centers of the 4 corner pixels of the input and
output tensors are aligned, preserving the values at the corner pixels.
Defaults to false .
* @li half_pixel_centers: An optional bool. If true, the center of pixels locate in [0.5, 0.5]. When "align_corners" is
true, "half_pixel_centers" cannot be true. Defaults to False .
* @li dtype: An optional Type attr, support type list [uint8, float32, float16, bfloat16].
The data type of output y.
Defaults to float32 .
* @li scales: An optional listfloat of exactly 2 elements {scale_h, scale_w},
overriding the scale factor used by the coordinate mapping for the spatial
dimensions H and W. Defaults to {0.0f, 0.0f}, meaning no override.
It does NOT change the output shape: the spatial size of y is always derived from
the "size" input by shape inference, and "size" is never ignored. An element is
treated as an override only when it is greater than 0.0; otherwise the scale falls
back to the derived value H_in / H_out (resp. W_in / W_out). When "align_corners"
is true the override is not applied, because the scale is then fixed to
(H_in - 1) / (H_out - 1) (resp. for W). The two elements are ordered
{scale_h, scale_w}, each being the ratio of output to input length, so the
coordinate mapping uses 1 / scale_h (resp. 1 / scale_w) .
*@par Outputs:
*y: 4-D tensor, format must be the same as x.
support format list ["NCHW", "NHWC"].
When the dtype of y is float32, the dtype of x can be float32, float16 or bfloat16. When the dtype of y is
float16 or bfloat16, then the dtype of y must be the same as x.The N, C dimension must be the same as x.
Empty tensor is not supported: the op is rejected when the element count of y is 0.
Must be a contiguous tensor.

*@par Third-party framework compatibility
*Compatible with TensorFlow and PyTorch ResizeBilinearV2 operator.
*/

REG_OP(ResizeBilinearV2)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                          DT_BF16}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_UINT8, DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(dtype, Type, DT_FLOAT)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeBilinearV2)
} // namespace ge

#endif // OPS_OP_PROTO_INC_RESIZE_BILINEAR_V2_H_
