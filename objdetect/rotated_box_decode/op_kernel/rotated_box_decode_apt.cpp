/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// rotated_box_decode_package/op_kernel/rotated_box_decode_apt.cpp
// =============================================================================
//
// ROLE: Ascend C kernel entry point for RotatedBoxDecode.
//   核函数签名与 docs/rotated_box_decode/develop/proto.md §3 kernel 函数签名一致:
//     template<int COPY_MODE, int UB_AXIS_SEL>
//     __global__ __aicore__ void rotated_box_decode(
//         GM_ADDR anchor_box, GM_ADDR deltas, GM_ADDR y,
//         GM_ADDR workspace, GM_ADDR tiling)
//
//   TPL 机制 (ASCENDC_TPL_SEL in struct.h) 据注册组合实例化本模板:
//     key=0: COPY_MODE=NDDMA(0) + UB_AXIS_SEL=UB_AXIS_N(0)  — multi-core along N
//     key=1: COPY_MODE=NDDMA(0) + UB_AXIS_SEL=UB_AXIS_B(1)  — fullload along B
//   DTYPE_ANCHOR_BOX macro selects dtype (half/bfloat16_t/float per dtype combo).
//
//   TilingData: §7 non-template struct (host writes, kernel reads — same layout).
//   workspace param retained but unused (DESIGN §9.7.2: no operator-level workspace).
//
// =============================================================================

#include "kernel_operator.h"                       // Ascend C kernel framework
#include "arch35/rotated_box_decode_struct.h"      // TPL declarations (COPY_MODE, UB_AXIS_SEL)
#include "arch35/rotated_box_decode_tiling_data.h" // RotatedBoxDecodeTilingData (§7, non-template)
#include "arch35/rotated_box_decode_kernel.h"      // RotatedBoxDecodeKernel + VF (§10)
#include "arch35/rotated_box_decode_simt.h"        // N==1 SIMT branch (UB_AXIS_SEL=I)

// ===========================================================================
// __global__ __aicore__ void rotated_box_decode(anchorBox, deltas, y, workspace, tiling)
//
// Kernel entry signature per docs/rotated_box_decode/develop/proto.md §3.
// Instantiates RotatedBoxDecodeKernel<DTYPE_ANCHOR_BOX, COPY_MODE, UB_AXIS_SEL>
// and runs Init / Process per DESIGN §10.1 / §10.4 / §10.5.
// UB_AXIS_SEL=I (N==1, tiling key 512) routes to the per-box scalar SIMT
// pipeline (rbd_simt::RotatedBoxDecodeSimtProcess) instead — the VF kernel
// class is not instantiated on that branch.
// ===========================================================================
template <int COPY_MODE, int UB_AXIS_SEL>
__global__ __aicore__ void rotated_box_decode(GM_ADDR anchorBox, GM_ADDR deltas, GM_ADDR y, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    AscendC::SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RotatedBoxDecodeTilingData);

    // TilingData: §7 non-template struct (host writes, kernel reads — same layout)
    GET_TILING_DATA_WITH_STRUCT(RotatedBoxDecodeTilingData, td, tiling);

    // System-level workspace (DESIGN §9.7.2: no operator-level workspace)
    (void)workspace;

    // DTYPE_ANCHOR_BOX macro injected by codegen (same_as_first_input, §6.1)
    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_I) {
        rbd_simt::RotatedBoxDecodeSimtProcess<DTYPE_ANCHOR_BOX>(anchorBox, deltas, y, &td);
    } else {
        rbd_kernel::RotatedBoxDecodeKernel<DTYPE_ANCHOR_BOX, COPY_MODE, UB_AXIS_SEL> kernel;
        kernel.Init(anchorBox, deltas, y, &td);
        kernel.Process();
    }
}
