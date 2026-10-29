/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_BILATERAL_FILTER_3D_H_
#define OPS_OP_PROTO_INC_BILATERAL_FILTER_3D_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief 3D bilateral filter (reference MONAI; boundary handling differs, see design.md 1.1).
 *
 * y = sum(w * v) / sum(w) over the valid kernel_size^3 neighborhood,
 * w = exp(-(|off|^2 / (2*sigma_space^2) + (v - c)^2 / (2*sigma_color^2)));
 * boundary = valid-neighborhood truncation with renormalization.
 *
 * @par Inputs:
 * @li x: Rank 3/4/5 ND tensor, float16/float32. The last three dims are D/H/W.
 *
 * @par Outputs:
 * @li y: Same shape and dtype as x.
 *
 * @par Attributes:
 * @li kernel_size: Required odd size in {3, 5, 7, 9}.
 * @li sigma_space: Required positive finite float.
 * @li sigma_color: Required positive finite float.
 */
REG_OP(BilateralFilter3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(kernel_size, Int)
    .REQUIRED_ATTR(sigma_space, Float)
    .REQUIRED_ATTR(sigma_color, Float)
    .OP_END_FACTORY_REG(BilateralFilter3D)
} // namespace ge

#endif // OPS_OP_PROTO_INC_BILATERAL_FILTER_3D_H_
