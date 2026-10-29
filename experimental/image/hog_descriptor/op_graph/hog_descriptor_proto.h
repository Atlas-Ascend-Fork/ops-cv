/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_HOG_DESCRIPTOR_H_
#define OPS_OP_PROTO_INC_HOG_DESCRIPTOR_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Histogram of Oriented Gradients feature descriptor with replicate
 * padding and trilinear cell voting; see README.md for the operator semantics.
 *
 * Pipeline: optional gamma(sqrt) compression -> center-difference gradient
 * (C=3 keeps the max-magnitude channel) -> trilinear cell histogram voting
 * (2 orientation bins x 4 cells) -> block normalization (L1/L1-sqrt/L2/L2-Hys)
 * -> dense descriptor concatenation.
 *
 * @par Inputs:
 * @li x: [H,W]/[H,W,C]/[N,H,W]/[N,H,W,C], C in {1,3}, float16/float32.
 *
 * @par Outputs:
 * @li descriptor: float32, [D] (single image) or [N,D],
 *    D = orientations * b_h * b_w * n_blocks_h * n_blocks_w.
 *
 * @par Attributes:
 * @li pixels_per_cell: [cell_h, cell_w], 1..256.
 * @li cells_per_block: [block_h, block_w], 1..8.
 * @li block_stride: [stride_h, stride_w] in cells, 1..8.
 * @li orientations: 2..36.
 * @li block_norm: "L1" | "L1-sqrt" | "L2" | "L2-Hys".
 * @li l2_hys_threshold: 0..1, L2-Hys only.
 * @li gamma_correction: sqrt power-law compression.
 * @li signed_gradient: false -> [0,180), true -> [0,360).
 * @li derivative_aperture: only 1 ([-1,0,1]) is supported.
 * @li eps: normalization guard, > 0.
 */
REG_OP(HOGDescriptor)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(descriptor, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(pixels_per_cell, ListInt)
    .REQUIRED_ATTR(cells_per_block, ListInt)
    .REQUIRED_ATTR(block_stride, ListInt)
    .REQUIRED_ATTR(orientations, Int)
    .REQUIRED_ATTR(block_norm, String)
    .REQUIRED_ATTR(l2_hys_threshold, Float)
    .REQUIRED_ATTR(gamma_correction, Bool)
    .REQUIRED_ATTR(signed_gradient, Bool)
    .REQUIRED_ATTR(derivative_aperture, Int)
    .REQUIRED_ATTR(eps, Float)
    .OP_END_FACTORY_REG(HOGDescriptor)
} // namespace ge

#endif // OPS_OP_PROTO_INC_HOG_DESCRIPTOR_H_
