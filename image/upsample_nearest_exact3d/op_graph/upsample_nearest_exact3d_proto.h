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
 * \file upsample_nearest_exact3d_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_UPSAMPLE_NEAREST_EXACT3D_H_
#define OPS_OP_PROTO_INC_UPSAMPLE_NEAREST_EXACT3D_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Upsample the 3-D data with the nearest neighbor ​interpolation algorithm. \n

*@par Inputs:
*One inputs, including: \n
*x: A 5-D input tensor. The format must be NCDHW. Must be one of the following types:
* float16, float32, bfloat16. \n

*@par Attributes:
*@li output_size: A required listInt. Defaults to none.
* Must contain exactly 3 elements: output_depth, output_height, and output_width. \n
*@li scale_d: An optional float. Defaults to 0.0. Specifies the depth sampling scale. \n
*@li scale_h: An optional float. Defaults to 0.0. Specifies the height sampling scale. \n
*@li scale_w: An optional float. Defaults to 0.0. Specifies the width sampling scale. \n
* The three scale attributes are applied only if all of them are greater than zero; otherwise, the scales are derived
* from the input shape and output_size. \n
*
*@par Outputs:
*y: A 5-D tensor. The format must be NCDHW. Must be one of the following types:
* float16, float32, bfloat16.
* Has the same type as input x. The N and C dimensions match x; output D, H, and W are specified by output_size. \n
*
*/

REG_OP(UpsampleNearestExact3d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(scale_d, Float, 0.0f)
    .ATTR(scale_h, Float, 0.0f)
    .ATTR(scale_w, Float, 0.0f)
    .OP_END_FACTORY_REG(UpsampleNearestExact3d)

} // namespace ge

#endif // OPS_OP_PROTO_INC_UPSAMPLE_NEAREST_EXACT3D_H_
