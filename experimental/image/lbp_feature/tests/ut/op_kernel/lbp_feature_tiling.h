/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lbp_feature_tiling.h
 * \brief kernel UT 用的轻量 tiling 适配头（ut.cmake 检测到本文件会优先 -include 它，
 *        替代从 tiling 库生成的同名头）。提供 CPU 仿真下的 LbpFeatureTilingData 与
 *        GET_TILING_DATA 宏；设备侧不受影响（本文件不参与算子包构建）。
 */

#ifndef LBP_FEATURE_UT_TILING_H
#define LBP_FEATURE_UT_TILING_H

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/lbp_feature_tiling_data.h"

template <typename T>
inline void LbpFeatureUtInitTilingData(const uint8_t* tilingArg, T* out)
{
    (void)std::memcpy(out, tilingArg, sizeof(T));
}

#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    LbpFeatureTilingData tiling_data;            \
    LbpFeatureUtInitTilingData(reinterpret_cast<const uint8_t*>(tiling_arg), &tiling_data);

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                   \
    tiling_struct tiling_data;                                              \
    LbpFeatureUtInitTilingData(reinterpret_cast<const uint8_t*>(tiling_arg), &tiling_data);

#endif // LBP_FEATURE_UT_TILING_H
