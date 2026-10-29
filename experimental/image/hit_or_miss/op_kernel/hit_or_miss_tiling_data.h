/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HIT_OR_MISS_TILING_DATA_H
#define HIT_OR_MISS_TILING_DATA_H

#include <cstdint>

struct HitOrMissTilingData {
    int64_t totalLength;
    int64_t blockLength;
    int64_t axis0Stride;
    int64_t axis1Stride;
    int64_t axis0Dim;
    int64_t axis1Dim;
    int32_t kernelH;
    int32_t kernelW;
    int32_t anchorH;
    int32_t anchorW;
    int32_t borderValue;
    int32_t usedCoreNum;
};

#endif
