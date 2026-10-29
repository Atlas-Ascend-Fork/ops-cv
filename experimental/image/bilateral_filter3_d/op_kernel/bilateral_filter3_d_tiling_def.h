/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_KERNEL_BILATERAL_FILTER_3D_TILING_DEF_H_
#define OPS_OP_KERNEL_BILATERAL_FILTER_3D_TILING_DEF_H_

#include "adv_api/kernel_tiling.h"

namespace optiling {
// Tiling key 只编码输入 dtype(与宿主侧 BilateralFilter3DTilingData 保持一致)。
// 注意: TILING_KEY_IS 预编译期只认数字字面量, 不能引用这里的具名常量。
constexpr uint64_t BF3D_TILING_KEY_FP16 = 0;
constexpr uint64_t BF3D_TILING_KEY_FP32 = 1;

// 行内分块的宽度上限(fp32 元素)。宿主侧用同一上限推导 chunkLen 并校验 UB 预算,
// 两侧必须一致。
constexpr uint32_t BF3D_CHUNK_MAX = 1024;

// Device-side POD mirror. Field order and types must match host TilingData exactly:
// 设备侧按 uint32_t 字长逐字拷贝, 顺序错位不会报错, 只会读到错误数值。
struct BilateralFilter3DTilingData {
    uint64_t totalRows; // nVolume * D * H, 每行 = W 个连续体素
    uint32_t d;
    uint32_t h;
    uint32_t w;
    uint32_t kernelSize;
    uint32_t usedCoreNum;
    uint32_t chunkLen; // 行内分块宽度, 8 的倍数, <= align8(W) 且 <= BF3D_CHUNK_MAX
    float inv2ss;      // 0.5 / sigma_space^2, 钳制到 (0, 1e30]
    float inv2sc;      // 0.5 / sigma_color^2, 同上
};
} // namespace optiling

#endif // OPS_OP_KERNEL_BILATERAL_FILTER_3D_TILING_DEF_H_
