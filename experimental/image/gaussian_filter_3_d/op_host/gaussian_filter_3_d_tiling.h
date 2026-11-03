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
 * \file gaussian_filter_3_d_tiling.h
 * \brief GaussianFilter3D tiling数据结构定义
 */
#ifndef OPS_NN_FILTER_GAUSSIAN_FILTER_3D_TILING_H
#define OPS_NN_FILTER_GAUSSIAN_FILTER_3D_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
// 核半径上限：r = int(4*sigma + 0.5)，sigma <= 100 时 r <= 400。
// 同时保证 TilingData（12标量 + 801权重 ≈ 3.2KB）不超过运行时4096B tiling缓冲上限。
constexpr uint32_t GAUSSIAN_FILTER_3D_MAX_RADIUS = 400;
// 一维高斯核最多 2*r+1 个权重
constexpr uint32_t GAUSSIAN_FILTER_3D_MAX_TAPS = 2 * GAUSSIAN_FILTER_3D_MAX_RADIUS + 1;

// 边界模式（对齐 scipy.ndimage）
// scipy REFLECT = 重复边缘像素（torch symmetric）；scipy MIRROR = 不重复边缘（torch reflect）
constexpr uint32_t GAUSSIAN_MODE_REFLECT = 0;
constexpr uint32_t GAUSSIAN_MODE_CONSTANT = 1;
constexpr uint32_t GAUSSIAN_MODE_NEAREST = 2;
constexpr uint32_t GAUSSIAN_MODE_MIRROR = 3;
constexpr uint32_t GAUSSIAN_MODE_WRAP = 4;

BEGIN_TILING_DATA_DEF(GaussianFilter3DTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);    // N*C（前导维合并）
TILING_DATA_FIELD_DEF(uint32_t, depth);    // D
TILING_DATA_FIELD_DEF(uint32_t, height);   // H
TILING_DATA_FIELD_DEF(uint32_t, width);    // W
TILING_DATA_FIELD_DEF(uint32_t, radius);   // 核半径 r = int(4*sigma + 0.5)
TILING_DATA_FIELD_DEF(uint32_t, mode);     // 边界模式 0..4
TILING_DATA_FIELD_DEF(uint32_t, tileLen);  // 每个tile沿W的输出元素数（8对齐，<=W）
TILING_DATA_FIELD_DEF(uint32_t, chunkNum); // 每行的tile数 = ceil(W / tileLen)
TILING_DATA_FIELD_DEF(uint32_t, lineNum);  // 输出行数 = batch*D*H
TILING_DATA_FIELD_DEF(uint32_t, coreNum);  // 参与计算的AIV数
TILING_DATA_FIELD_DEF(uint32_t, tasksPerCore);
TILING_DATA_FIELD_DEF(uint32_t, pongOffset); // pong缓冲在workspace中的字节偏移
TILING_DATA_FIELD_DEF_ARR(uint32_t, GAUSSIAN_FILTER_3D_MAX_TAPS, weightBits); // 高斯权重fp32位模式，共2r+1个
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GaussianFilter3D, GaussianFilter3DTilingData)

struct GaussianFilter3DCompileInfo {};
} // namespace optiling
#endif // OPS_NN_FILTER_GAUSSIAN_FILTER_3D_TILING_H
