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
 * \file lut3_d_proto.h
 * \brief LUT3D operator proto definition
 */
#ifndef LUT3_D_PROTO_H_
#define LUT3_D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief LUT3D
* Find the corresponding optimal pixel value for the pixel values in the input img. \n

*@par Inputs:
* Two inputs, including:
*@li img: A 3D or 4D Tensor of type uint8 or float32, with shape [H,W,C] or [N,H,W,C] respectively.
* The format of the tensor is ND or NHWC. The range of values for elements within the tensor is [0, 255].
*@li lut_table: A 4D Tensor of the same type as "img", with shape [lut_table_n, lut_table_n, lut_table_n, 3].
 * The format of the tensor is ND. The value of lut_table_n is limited to 20. \n

*@par Outputs:
* lut_img: A 3D or 4D Tensor of type float32. Has the same shape as "img".
* The format of the tensor is ND or NHWC. \n
*/
REG_OP(LUT3D)
    .INPUT(img, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(lut_table, TensorType({DT_UINT8, DT_FLOAT}))
    .OUTPUT(lut_img, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(LUT3D)

} // namespace ge

#endif // LUT3_D_PROTO_H_
