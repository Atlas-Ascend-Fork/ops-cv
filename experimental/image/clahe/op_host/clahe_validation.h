/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CLAHE_VALIDATION_H
#define CLAHE_VALIDATION_H
#include <cstddef>
namespace ClaheHost {
template <typename Shape>
bool ValidShape(const Shape& shape, bool dynamic)
{
    const size_t rank = shape.GetDimNum();
    if (dynamic && rank == 1 && shape.GetDim(0) == -2) {
        return true;
    }
    if (rank < 2 || rank > 4) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        const auto dim = shape.GetDim(i);
        if (dim <= 0 && !(dynamic && dim == -1)) {
            return false;
        }
    }
    if (rank >= 3 && shape.GetDim(0) > 256) {
        return false;
    }
    return rank != 4 || shape.GetDim(1) == 1 || (dynamic && shape.GetDim(1) == -1);
}
} // namespace ClaheHost
#endif
