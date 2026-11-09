/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_BOX_ENCODE_TILING_ARCH35_H
#define ROTATED_BOX_ENCODE_TILING_ARCH35_H

#include <cstdint>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../../op_kernel/arch35/rotated_box_encode_tiling_data.h"

namespace optiling {

// ===========================================================================
// Design constants (DESIGN §9.4 / §9.5 / §9.1 + DESIGN-BRANCH-0/1.md §2).
// ===========================================================================
constexpr int64_t BOX_CHANNELS = 5;        // spec.yaml shape[1]==5
constexpr int64_t MIN_TILING_BITS = 32768; // §2.2 4KB/core (bits)
constexpr int64_t MAX_RANK = 3;            // §9.3.2 rank==3

// Branch-0 (fp16-upcast): perBoxBytes=70, elemStride=10, alignFactor=128
constexpr int64_t BRANCH0_BOX_CHANNELS = 5;
constexpr int64_t BRANCH0_ALIGN_256_BYTES = 256;
constexpr int64_t BRANCH0_ELEM_ALIGN_FACTOR = 512;
constexpr int64_t BRANCH0_PER_BOX_BYTES = 70;
constexpr int64_t BRANCH0_ELEM_BYTES = 2;
constexpr int64_t BRANCH0_ELEM_STRIDE = BRANCH0_BOX_CHANNELS * BRANCH0_ELEM_BYTES; // 10
constexpr int64_t BRANCH0_ALIGN_FACTOR = 128;

// Branch-1 (fp32-direct): perBoxBytes=60, elemStride=20, alignFactor=64
constexpr int64_t BRANCH1_BOX_CHANNELS = 5;
constexpr int64_t BRANCH1_ALIGN_256_BYTES = 256;
constexpr int64_t BRANCH1_ELEM_ALIGN_FACTOR = 512;
constexpr int64_t BRANCH1_PER_BOX_BYTES = 60;
constexpr int64_t BRANCH1_ELEM_BYTES = 4;
constexpr int64_t BRANCH1_ELEM_STRIDE = BRANCH1_BOX_CHANNELS * BRANCH1_ELEM_BYTES; // 20
constexpr int64_t BRANCH1_ALIGN_FACTOR = 64;

// ===========================================================================
// Branch input structs: parsed inputs for the branch tiling formulas.
// ===========================================================================
struct RotatedBoxEncodeBranch0Inputs {
    int64_t dim0;
    int64_t N;
    uint32_t coreNumAiv;
    uint64_t ubSize;
    float weight[BRANCH0_BOX_CHANNELS];
};

struct RotatedBoxEncodeBranch1Inputs {
    int64_t dim0;
    int64_t N;
    uint32_t coreNumAiv;
    uint64_t ubSize;
    float weight[BRANCH1_BOX_CHANNELS];
};

// ===========================================================================
// Branch pure tiling computations (DESIGN-BRANCH-0/1.md §2).
// Caller MUST memset `out` to 0 before calling.
// ===========================================================================
ge::graphStatus ComputeBranch0Tiling(const RotatedBoxEncodeBranch0Inputs& in, RotatedBoxEncodeTilingData& out);

ge::graphStatus ComputeBranch1Tiling(const RotatedBoxEncodeBranch1Inputs& in, RotatedBoxEncodeTilingData& out);

// ===========================================================================
// TilingRotatedBoxEncode: runtime tiling entry (DESIGN §9.9).
// Platform parameters are resolved at runtime; no compile-time parser is used.
// ===========================================================================
ge::graphStatus TilingRotatedBoxEncode(gert::TilingContext* ctx);

} // namespace optiling

#endif // ROTATED_BOX_ENCODE_TILING_ARCH35_H
