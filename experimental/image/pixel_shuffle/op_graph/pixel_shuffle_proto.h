/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_PIXEL_SHUFFLE_H_
#define OPS_OP_PROTO_INC_PIXEL_SHUFFLE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(PixelShuffle)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .ATTR(scale_factor, Int, 2)
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(PixelShuffle)

} // namespace ge

#endif // OPS_OP_PROTO_INC_PIXEL_SHUFFLE_H_
