/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nms_with_mask_apt.cpp
 * \brief nms_with_mask
 */

#include "arch35/nms_with_mask_regbase_multiprocess.h"

using namespace AscendC;
using namespace NMSWithMaskOp;

#define TILING_KEY_FOR_MULTICORE 10000

extern "C" __global__ __aicore__ void nms_with_mask(GM_ADDR box_scores, GM_ADDR selected_boxes, GM_ADDR selected_idx,
                                                    GM_ADDR selected_mask, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(NMSWithMaskTilingData);
    GET_TILING_DATA_WITH_STRUCT(NMSWithMaskTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_FOR_MULTICORE)) {
        NMSWithMaskRegbaseMultiProcess<DTYPE_BOX_SCORES> op(&pipe);
        op.Init(box_scores, selected_boxes, selected_idx, selected_mask, workspace, tilingData);
        op.Process();
        return;
    }
}
