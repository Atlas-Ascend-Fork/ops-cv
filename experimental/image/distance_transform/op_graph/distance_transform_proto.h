/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPS_CV_EXPERIMENTAL_IMAGE_DISTANCE_TRANSFORM_PROTO_H_
#define OPS_CV_EXPERIMENTAL_IMAGE_DISTANCE_TRANSFORM_PROTO_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Computes the distance from every foreground pixel to the nearest background pixel.
 *
 * @par Inputs:
 * @li x: A uint8 ND tensor containing a binary image. Zero denotes background and nonzero denotes foreground.
 *
 * Supported shapes are (H, W), (N, H, W), and (N, 1, H, W).
 *
 * @par Attributes:
 * @li distance_type: An optional string selected from "L1", "L2", and "CHESSBOARD". Defaults to "L2".
 * @li mask_size: An optional integer selected from 0, 3, and 5. Defaults to 0. For L2, 0 selects the precise
 *
 * algorithm while 3 and 5 select the corresponding chamfer mask. L1 and CHESSBOARD always use a 3 x 3
 * neighborhood
 * for every valid mask_size value.
 *
 * @par Outputs:
 * @li y: A float32 ND tensor with the same shape as x. Background pixels are zero; foreground pixels contain
 * their
 * distance to the nearest background pixel.
 *
 * @attention Constraints:
 * The rank of x must be 2, 3, or 4. A rank-4
 * x must have shape (N, 1, H, W). N, H, and W must be positive.
 * H must not exceed INT32_MAX, and W must not exceed
 * 1073741824 (2^30).
 * distance_type is case-sensitive. The total number of elements must not exceed UINT32_MAX.
 *
 *
 * @par Third-party framework compatibility
 * Compatible with OpenCV cv2.distanceTransform for inputs that contain at
 * least one background pixel.
 */
REG_OP(DistanceTransform)
    .INPUT(x, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(distance_type, String, "L2")
    .ATTR(mask_size, Int, 0)
    .OP_END_FACTORY_REG(DistanceTransform)

} // namespace ge

#endif // OPS_CV_EXPERIMENTAL_IMAGE_DISTANCE_TRANSFORM_PROTO_H_
