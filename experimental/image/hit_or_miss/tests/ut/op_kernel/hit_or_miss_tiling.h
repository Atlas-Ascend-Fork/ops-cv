/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HIT_OR_MISS_UT_TILING_H_
#define HIT_OR_MISS_UT_TILING_H_

#include <cstring>
#include "../../../op_kernel/hit_or_miss_tiling_data.h"

#ifndef __aicore__
#define __aicore__
#endif

inline void InitTilingData(uint8_t* tiling, HitOrMissTilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(HitOrMissTilingData));
}

#ifdef GET_TILING_DATA_WITH_STRUCT
#undef GET_TILING_DATA_WITH_STRUCT
#endif
#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif
