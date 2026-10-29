/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SSIM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SSIM_H_
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SsimTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalOutRows);    // N * (H - 10)
TILING_DATA_FIELD_DEF(uint32_t, perImageOutRows); // H - 10
TILING_DATA_FIELD_DEF(uint32_t, perImageInRows);  // H
TILING_DATA_FIELD_DEF(uint32_t, outW);            // W - 10
TILING_DATA_FIELD_DEF(uint32_t, inW);             // W
TILING_DATA_FIELD_DEF(uint32_t, inH);             // H
TILING_DATA_FIELD_DEF(uint32_t, channels);        // C
TILING_DATA_FIELD_DEF(uint32_t, batch);           // N
TILING_DATA_FIELD_DEF(uint32_t, blockRows);       // output rows per core (except last)
TILING_DATA_FIELD_DEF(uint32_t, lastBlockRows);   // output rows for the last core
TILING_DATA_FIELD_DEF(uint32_t, colTile);         // output columns per tile
TILING_DATA_FIELD_DEF(uint32_t, totalPixels);     // total internal pixels to average
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(float, invTotalPixels);
TILING_DATA_FIELD_DEF(float, c1);
TILING_DATA_FIELD_DEF(float, c2);
TILING_DATA_FIELD_DEF(uint32_t, isUint8);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Ssim, SsimTilingData)

struct SsimCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SSIM_H_
