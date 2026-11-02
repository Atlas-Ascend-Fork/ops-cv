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
 * \file check_valid_proto.h
 * \brief check whether bounding boxes are within the valid image region.
 */
#ifndef OPS_OP_PROTO_INC_CHECK_VALID_H_
#define OPS_OP_PROTO_INC_CHECK_VALID_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Judges whether the bounding box is valid. It is a customized
* FasterRcnn operator .

*@par Inputs:
* Two inputs, including:
* @li bbox_tensor: Bounding box. A 2D Tensor of type float16 or float32 with shape (N, 4).
* "N" indicates the number of bounding boxes, the value "4" indicates "x0",
* "y0", "x1", and "y1".
* @li img_metas: Valid boundary value of the image. A 1D Tensor of type float16 or float32
* with shape (G,), where G >= 3. \n

*@par Outputs:
*valid_tensor: An int8 tensor with shape (N, 1), specifying whether an input anchor is
* in an image. "1" indicates valid, while "0" indicates invalid . \n

*@attention Constraints:
* img_metas must be a 1D tensor with at least three elements.
* The first three elements are (height, width, ratio); remaining elements are ignored.
*/
REG_OP(CheckValid)
    .INPUT(bbox_tensor, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(img_metas, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(valid_tensor, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(CheckValid)

} // namespace ge

#endif
