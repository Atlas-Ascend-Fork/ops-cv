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
 * \file integral_image_tiling.h
 * \brief CPU 仿真（tikicpulib）下 tiling 结构解析宏，替代编译期生成的 tiling data 头文件
 */

#ifndef INTEGRAL_IMAGE_UT_TILING_H_
#define INTEGRAL_IMAGE_UT_TILING_H_

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/integral_image_tiling_data.h"

#ifndef __aicore__
#define __aicore__
#endif

inline void InitTilingData(uint8_t* tiling, IntegralImageTilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(IntegralImageTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif // INTEGRAL_IMAGE_UT_TILING_H_
