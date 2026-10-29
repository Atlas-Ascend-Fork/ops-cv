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
 * \file upsample_nearest3d_proto.h
 * \brief UpsampleNearest3d operator proto. Ascend 950PR/Ascend 950DT (A5) does not support GEIR.
 */
#ifndef OPS_OP_PROTO_INC_UPSAMPLE_NEAREST3D_H_
#define OPS_OP_PROTO_INC_UPSAMPLE_NEAREST3D_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Upsample the 3-D data with the nearest neighbor ​interpolation algorithm. \n

*@par Inputs:
*One inputs, including: \n
*x: A 5-D input tensor. The format must be NCDHW. Must be one of the following dtype:
* float16, float32, double, uint8, bfloat16. \n

*@par Attributes:
*@li output_size: An optional listInt. Defaults to none.
* Contain 3 elements: output_depth, output_height, output_width. The number of elements of 'output_size'
* must be three.And it must have the same d h w dimensions as the output y. \n
*@li scales: An optional listFloat. Defaults to none.
* The scale array along each dimension, contain 3 elements: scale_depth, scale_height, scale_width.
* The number of elements of 'scales' must be three.
*At least one of scales and output_size must exist; both cannot be None.

*@par Outputs:
*y: A 5-D tensor. The format must be NCDHW. Must be one of the following dtypes:
* float16, float32, double, uint8, bfloat16.
* Has the same dtype and format as input x, shape depends on x and output_size/scales. \n

*/

REG_OP(UpsampleNearest3d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BF16}))
    .ATTR(output_size, ListInt, {})
    .ATTR(scales, ListFloat, {})
    .OP_END_FACTORY_REG(UpsampleNearest3d)

} // namespace ge

#endif // OPS_OP_PROTO_INC_UPSAMPLE_NEAREST3D_H_
