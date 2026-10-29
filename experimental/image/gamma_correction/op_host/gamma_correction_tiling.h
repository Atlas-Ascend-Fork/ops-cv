/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_HOST_GAMMA_CORRECTION_TILING_H_
#define OPS_OP_HOST_GAMMA_CORRECTION_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
// Tiling keys encode only the input dtype; correctionMode is carried as a field.
constexpr uint64_t GC_TILING_KEY_FP16 = 0;
constexpr uint64_t GC_TILING_KEY_BF16 = 1;
constexpr uint64_t GC_TILING_KEY_FP32 = 2;
constexpr uint64_t GC_TILING_KEY_UINT8 = 3;
constexpr uint64_t GC_TILING_KEY_INT8 = 4;
constexpr uint64_t GC_TILING_KEY_UINT16 = 5;
constexpr uint64_t GC_TILING_KEY_INT16 = 6;

// gammaMode selects the pow fast path (exponent shape), independent of dtype.
constexpr uint32_t GC_MODE_GENERIC = 0;
constexpr uint32_t GC_MODE_IDENTITY = 1;
constexpr uint32_t GC_MODE_SQUARE = 2;
constexpr uint32_t GC_MODE_SQRT = 3;
constexpr uint32_t GC_MODE_INTEGER = 4;

// correctionMode selects the image pipeline around the pow core.
constexpr uint32_t GC_CORRECTION_PURE = 0;  // y = pow(x, gamma)
constexpr uint32_t GC_CORRECTION_CLAMP = 1; // normalize + pow + denormalize + clamp
constexpr uint32_t GC_CORRECTION_SRGB = 2;  // sRGB encoding piecewise curve

BEGIN_TILING_DATA_DEF(GammaCorrectionTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalNum);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubFactor);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, gammaMode);
TILING_DATA_FIELD_DEF(uint32_t, integerExponent);
TILING_DATA_FIELD_DEF(uint32_t, correctionMode);
TILING_DATA_FIELD_DEF(float, gamma);
TILING_DATA_FIELD_DEF(float, maxValue);
TILING_DATA_FIELD_DEF(float, invMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GammaCorrection, GammaCorrectionTilingData)

struct GammaCorrectionCompileInfo {
    int64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

#endif // OPS_OP_HOST_GAMMA_CORRECTION_TILING_H_
