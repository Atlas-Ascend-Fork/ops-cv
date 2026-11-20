/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STACK_GROUP_POINTS_ARCH35_TILING_H_
#define STACK_GROUP_POINTS_ARCH35_TILING_H_

#include "platform/platform_info.h"

struct StackGroupPointsTilingData {
    int64_t m;
    int64_t c;
    int64_t nsample;
    int64_t b;
    int64_t n;
    int64_t totalElements;
    int64_t needCoreNum;
    int64_t zeroFill;
};

struct StackGroupPointsCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

#endif // STACK_GROUP_POINTS_ARCH35_TILING_H_
