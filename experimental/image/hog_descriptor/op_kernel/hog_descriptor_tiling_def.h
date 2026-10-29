/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_KERNEL_HOG_DESCRIPTOR_TILING_DEF_H_
#define OPS_OP_KERNEL_HOG_DESCRIPTOR_TILING_DEF_H_

#include "adv_api/kernel_tiling.h"

namespace optiling {
// Tiling keys encode only the input dtype; TILING_KEY_IS only accepts numeric literals.
// Keys match the float16/float32 op_list entries in hog_descriptor_binary.json.
constexpr uint64_t HOG_TILING_KEY_FP16 = 0;
constexpr uint64_t HOG_TILING_KEY_FP32 = 1;

// Device-side POD mirror. Field order and types must match host HogDescriptorTilingData.
struct HogDescriptorTilingData {
    uint64_t dimD;
    uint32_t n;
    uint32_t h;
    uint32_t w;
    uint32_t c;
    uint32_t pH;
    uint32_t pW;
    uint32_t bH;
    uint32_t bW;
    uint32_t sH;
    uint32_t sW;
    uint32_t nBins;
    uint32_t nCellsH;
    uint32_t nCellsW;
    uint32_t nBlocksH;
    uint32_t nBlocksW;
    uint32_t padH;
    uint32_t padW;
    uint32_t nBinsPad;
    uint32_t bandLen;
    uint32_t bandsPerImg;
    uint32_t taskCnt;
    uint32_t usedCoreNum;
    uint32_t normMode;
    uint32_t gammaFlag;
    uint32_t signedFlag;
    float binScale;
    float l2Hys;
    float eps;
};
} // namespace optiling

#endif // OPS_OP_KERNEL_HOG_DESCRIPTOR_TILING_DEF_H_
