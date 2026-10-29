/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_HOST_HOG_DESCRIPTOR_UB_H_
#define OPS_OP_HOST_HOG_DESCRIPTOR_UB_H_

#include <cstdint>

namespace optiling {
inline uint64_t HogAlignUb(uint64_t bytes) { return (bytes + 31U) / 32U * 32U; }

// Mirrors InitBuffer in HogDescriptorVectorKernel::Init, excluding the histogram
// plane. Include 8 KiB of free space for high-level vector API temporary storage.
inline uint64_t HogFixedUbBytes(uint32_t padW, uint32_t channels, uint32_t inputBytes, uint32_t cellColumns,
                                uint32_t binsPad, uint32_t blockH, uint32_t blockW)
{
    const uint64_t strip = HogAlignUb(static_cast<uint64_t>(padW) * sizeof(float));
    const uint64_t row = HogAlignUb(static_cast<uint64_t>(padW) * channels * inputBytes);
    const uint64_t out = HogAlignUb(static_cast<uint64_t>(binsPad) * sizeof(float));
    const uint64_t segment = HogAlignUb(static_cast<uint64_t>((cellColumns + 7U) / 8U * 8U) * sizeof(float));
    const uint64_t stage = HogAlignUb(static_cast<uint64_t>(blockH) * blockW * binsPad * sizeof(float));
    constexpr uint64_t API_SCRATCH_BYTES = 8U * 1024U;
    return 22U * strip + 2U * row + 4U * out + 2U * segment + 3U * stage + 2U * HogAlignUb(8U * sizeof(float)) +
           API_SCRATCH_BYTES;
}
} // namespace optiling

#endif // OPS_OP_HOST_HOG_DESCRIPTOR_UB_H_
