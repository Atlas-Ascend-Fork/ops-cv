/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_HOST_BILATERAL_FILTER_3D_TILING_H_
#define OPS_OP_HOST_BILATERAL_FILTER_3D_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
// Tiling keys encode only the input dtype.
constexpr uint64_t BF3D_TILING_KEY_FP16 = 0;
constexpr uint64_t BF3D_TILING_KEY_FP32 = 1;

// 行内分块宽度上限(fp32 元素)。与 op_kernel/bilateral_filter3_d_tiling_def.h 中的
// BF3D_CHUNK_MAX 必须一致。
constexpr int64_t BF3D_CHUNK_MAX = 1024;

// Field order and types must match the device-side POD mirror in
// op_kernel/bilateral_filter3_d_tiling_def.h exactly.
BEGIN_TILING_DATA_DEF(BilateralFilter3DTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalRows); // nVolume * D * H, 每行 = W 个连续体素
TILING_DATA_FIELD_DEF(uint32_t, d);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, w);
TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, chunkLen); // 行内分块宽度, 8 的倍数
TILING_DATA_FIELD_DEF(float, inv2ss);      // 0.5 / sigma_space^2, 钳制到 (0, 1e30]
TILING_DATA_FIELD_DEF(float, inv2sc);      // 0.5 / sigma_color^2, 同上
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BilateralFilter3D, BilateralFilter3DTilingData)

struct BilateralFilter3DCompileInfo {
    int64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

#endif // OPS_OP_HOST_BILATERAL_FILTER_3D_TILING_H_
