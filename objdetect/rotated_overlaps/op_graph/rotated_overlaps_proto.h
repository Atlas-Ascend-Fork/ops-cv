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
 * \file rotated_overlaps_proto.h
 * \brief RotatedOverlaps 图模式（GE IR）算子原型定义
 */
#ifndef OPS_OP_PROTO_INC_ROTATED_OVERLAPS_H_
#define OPS_OP_PROTO_INC_ROTATED_OVERLAPS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the overlap area between two groups of rotated boxes. \n

* @par Inputs:
* @li boxes: A 3D Tensor of type float32 with shape (B, 5, N). "N" indicates
* the number of boxes, and the value "5" refers to [x, y, w, h, theta] or
* [x1, y1, x2, y2, theta].
* @li query_boxes: A 3D Tensor of type float32 with shape (B, 5, K). "K"
* indicates the number of boxes, and the value "5" refers to [x, y, w, h,
* theta] or [x1, y1, x2, y2, theta].

* @par Attributes:
* @li trans: An optional attr. If true, the coordinate format is xyxyt;
* otherwise, it is xywht. Defaults to false.

* @par Outputs:
* overlaps: A 3D Tensor of type float32 with shape [B, N, K].

* @par Dynamic Shape:
* Supports GE IR dynamic shape. Runtime tiling uses the instantiated concrete
* shape.

* @attention Constraints:
* Invalid boxes (non-finite coordinates or non-positive width/height) may appear
* at any position; each pair containing one produces zero overlap.
*/
#ifndef OPS_PROTO_DEF_ROTATEDOVERLAPS
#define OPS_PROTO_DEF_ROTATEDOVERLAPS
REG_OP(RotatedOverlaps)
    .INPUT(boxes, TensorType({DT_FLOAT}))
    .INPUT(query_boxes, TensorType({DT_FLOAT}))
    .OUTPUT(overlaps, TensorType({DT_FLOAT}))
    .ATTR(trans, Bool, false)
    .OP_END_FACTORY_REG(RotatedOverlaps)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_ROTATED_OVERLAPS_H_
