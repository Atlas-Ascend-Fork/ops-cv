/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STACK_GROUP_POINTS_TILING_DATA_H_
#define STACK_GROUP_POINTS_TILING_DATA_H_

struct StackGroupPointsTilingData {
    int64_t m = 0;
    int64_t c = 0;
    int64_t nsample = 0;
    int64_t b = 0;
    int64_t n = 0;
    int64_t totalElements = 0;
    int64_t needCoreNum = 0;
    int64_t zeroFill = 0;
};

#endif // STACK_GROUP_POINTS_TILING_DATA_H_
