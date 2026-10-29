/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_GAMMA_CORRECTION_H_
#define OPS_OP_PROTO_INC_GAMMA_CORRECTION_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Applies image gamma correction: y = pow(x, gamma) plus optional
 *        normalize / clamp / sRGB image post-processing.
 *
 * @par Inputs:
 * @li x: Rank 2/3/4 ND tensor, bfloat16/float16/float32/uint8/int8/uint16/int16.
 *
 * @par Outputs:
 * @li y: Same shape and dtype as x.
 *
 * @par Attributes:
 * @li gamma: Required finite positive exponent.
 * @li correction_mode (optional, default 0): 0 = pure pow, 1 = normalize+pow+clamp,
 *     2 = sRGB encoding curve.
 * @li max_value (optional, default 0): value-range upper bound; derived from dtype
 *     when omitted (1.0 for floats, 255.0 for 8-bit, 65535.0 for 16-bit integers).
 */
REG_OP(GammaCorrection)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16}))
    .REQUIRED_ATTR(gamma, Float)
    .ATTR(correction_mode, Int, 0)
    .ATTR(max_value, Float, 0.0)
    .OP_END_FACTORY_REG(GammaCorrection)
} // namespace ge

#endif // OPS_OP_PROTO_INC_GAMMA_CORRECTION_H_
