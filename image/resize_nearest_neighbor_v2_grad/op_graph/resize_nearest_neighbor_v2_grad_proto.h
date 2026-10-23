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
 * \file resize_nearest_neighbor_v2_grad_proto.h
 * \brief
 */
#ifndef RESIZE_NEAREST_NEIGHBOR_V2_GRAD_PROTO_H
#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes the gradient of ResizeNearestNeighborV2 by accumulating output gradients at their source pixels. \n

* @par Inputs:
* Inputs include:
* @li grads: A 4-D tensor. Represents gradients of the resized image.
* Must set the format, supported format list ["NCHW", "NHWC"].
* Must be one of the following types: int8, uint8, int16, uint16, int32, int64, float16, float32,
* double, bfloat16.
* @li size: A 1-D int32 tensor of 2 elements: orig_height, orig_width.
* Specifies the original image height and width, which determine the spatial dimensions of the output gradient.
* Must be the type int32. \n

* @par Attributes:
* @li align_corners: An optional bool. Determines whether to align the corners of the input and output images.
* If set to True, the corner pixels of the input and output images are aligned,
* preserving the value of the corner pixels. When set to false,
* the scaling process scales according to proportions and does not strictly align the corners.
* Defaults to false.
* @li half_pixel_centers: An optional bool. Determines the pixel center position during interpolation.
* If this parameter is set to True, the interpolation algorithm considers the center point of the pixel
* to estimate the pixel value more accurately. When set to false, the pixel center is on the integer coordinate point.
* Defaults to false. \n

* @li scales: An optional listfloat. Multiplier for spatial size. Defaults to {0.0f, 0.0f} .
* @par Outputs:
* y: A 4-D tensor. Contains accumulated gradients of the original image. Has the same type and format as "grads".
     The N and C dimensions match grads; spatial dimensions are specified by size. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow ResizeNearestNeighborGrad operator.
*/

REG_OP(ResizeNearestNeighborV2Grad)
    .INPUT(grads, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT,
                              DT_DOUBLE, DT_BF16}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
                           DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
    .ATTR(scales, ListFloat, {0.0f, 0.0f})
    .OP_END_FACTORY_REG(ResizeNearestNeighborV2Grad)
} // namespace ge

#endif
