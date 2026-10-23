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
 * \file rotated_box_decode_proto.h
 * \brief RotatedBoxDecode 算子原型定义，将网络预测偏移量叠加到锚框上解码出旋转检测框。
 */
#ifndef OPS_OP_PROTO_INC_ROTATED_BOX_DECODE_H_
#define OPS_OP_PROTO_INC_ROTATED_BOX_DECODE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief RotatedBoxDecode. \n

*@par Inputs:
* Two inputs, including:
*@li anchor_box: A 3D Tensor of float32 (float16) with shape (B, 5, N).
* "B" indicates the number of batch size
* "N" indicates the number of bounding boxes, and the value "5" refers to
* "lx", "ly", "rx", "ry" and "angle" (in degrees).
*@li deltas: A 3D Tensor of float32 (float16) with shape (B, 5, N).
* "B" indicates the number of batch size
* "N" indicates the number of bounding boxes, and the value "5" refers to
* "dx", "dy", "dw", "dh" and "dt". \n

*@par Attributes:
*@li weight: A float list corresponding to "dx", "dy", "dw", "dh" and "dt",
* defaults to [1.0, 1.0, 1.0, 1.0, 1.0].

*@par Outputs:
*@li y: A 3D Tensor of type float32 (float16) with shape (B, 5, N),
* containing the decoded rotated boxes: "lx'", "ly'", "rx'", "ry'" and
* "theta_t" (in degrees).
*/
REG_OP(RotatedBoxDecode)
    .INPUT(anchor_box, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(deltas, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(weight, ListFloat, {1.0, 1.0, 1.0, 1.0, 1.0})
    .OP_END_FACTORY_REG(RotatedBoxDecode);

} // namespace ge

#endif // OPS_OP_PROTO_INC_ROTATED_BOX_DECODE_H_
