/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_KERNEL_GAMMA_CORRECTION_TILING_DEF_H_
#define OPS_OP_KERNEL_GAMMA_CORRECTION_TILING_DEF_H_

#include "adv_api/kernel_tiling.h"

namespace optiling {
constexpr uint64_t GC_TILING_KEY_FP16 = 0;
constexpr uint64_t GC_TILING_KEY_BF16 = 1;
constexpr uint64_t GC_TILING_KEY_FP32 = 2;
constexpr uint64_t GC_TILING_KEY_UINT8 = 3;
constexpr uint64_t GC_TILING_KEY_INT8 = 4;
constexpr uint64_t GC_TILING_KEY_UINT16 = 5;
constexpr uint64_t GC_TILING_KEY_INT16 = 6;

constexpr uint32_t GC_MODE_GENERIC = 0;
constexpr uint32_t GC_MODE_IDENTITY = 1;
constexpr uint32_t GC_MODE_SQUARE = 2;
constexpr uint32_t GC_MODE_SQRT = 3;
constexpr uint32_t GC_MODE_INTEGER = 4;

constexpr uint32_t GC_CORRECTION_PURE = 0;
constexpr uint32_t GC_CORRECTION_CLAMP = 1;
constexpr uint32_t GC_CORRECTION_SRGB = 2;

// Device-side POD mirror. Field order and types must match host TilingData.
struct GammaCorrectionTilingData {
    uint64_t totalNum;
    uint64_t blockFactor;
    uint64_t ubFactor;
    uint32_t usedCoreNum;
    uint32_t gammaMode;
    uint32_t integerExponent;
    uint32_t correctionMode;
    float gamma;
    float maxValue;
    float invMax;
};
} // namespace optiling

#endif // OPS_OP_KERNEL_GAMMA_CORRECTION_TILING_DEF_H_
