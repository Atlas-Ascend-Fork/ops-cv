/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_HOST_HOG_DESCRIPTOR_TILING_H_
#define OPS_OP_HOST_HOG_DESCRIPTOR_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
// Tiling keys encode only the input dtype (must stay numeric literals in the kernel).
// Keys match the float16/float32 op_list entries in hog_descriptor_binary.json.
constexpr uint64_t HOG_TILING_KEY_FP16 = 0;
constexpr uint64_t HOG_TILING_KEY_FP32 = 1;

// v1 vector-kernel layout constants (single full-width strip per task).
constexpr uint32_t HOG_V1_MAX_STRIP_PIX = 1024; // padded width cap, see tiling.cpp UB budget

BEGIN_TILING_DATA_DEF(HogDescriptorTilingData)
TILING_DATA_FIELD_DEF(uint64_t, dimD); // descriptor length per image
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, w);
TILING_DATA_FIELD_DEF(uint32_t, c);
TILING_DATA_FIELD_DEF(uint32_t, pH);
TILING_DATA_FIELD_DEF(uint32_t, pW);
TILING_DATA_FIELD_DEF(uint32_t, bH);
TILING_DATA_FIELD_DEF(uint32_t, bW);
TILING_DATA_FIELD_DEF(uint32_t, sH);
TILING_DATA_FIELD_DEF(uint32_t, sW);
TILING_DATA_FIELD_DEF(uint32_t, nBins);
TILING_DATA_FIELD_DEF(uint32_t, nCellsH);
TILING_DATA_FIELD_DEF(uint32_t, nCellsW);
TILING_DATA_FIELD_DEF(uint32_t, nBlocksH);
TILING_DATA_FIELD_DEF(uint32_t, nBlocksW);
TILING_DATA_FIELD_DEF(uint32_t, padH);     // replicate-padded pixel height nCellsH * pH
TILING_DATA_FIELD_DEF(uint32_t, padW);     // replicate-padded pixel width  nCellsW * pW
TILING_DATA_FIELD_DEF(uint32_t, nBinsPad); // align8(orientations)
TILING_DATA_FIELD_DEF(uint32_t, bandLen);  // block rows per band (last band may be shorter)
TILING_DATA_FIELD_DEF(uint32_t, bandsPerImg);
TILING_DATA_FIELD_DEF(uint32_t, taskCnt); // n * bandsPerImg
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, normMode); // 0=L1 1=L1-sqrt 2=L2 3=L2-Hys
TILING_DATA_FIELD_DEF(uint32_t, gammaFlag);
TILING_DATA_FIELD_DEF(uint32_t, signedFlag);
TILING_DATA_FIELD_DEF(float, binScale); // nBins / (pi or 2*pi)
TILING_DATA_FIELD_DEF(float, l2Hys);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HOGDescriptor, HogDescriptorTilingData)

struct HogDescriptorCompileInfo {
    int64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

#endif // OPS_OP_HOST_HOG_DESCRIPTOR_TILING_H_
