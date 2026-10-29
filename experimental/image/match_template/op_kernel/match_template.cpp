/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file match_template.cpp
 * \brief
 */

#include "match_template.h"
#include "match_template_tiling_key.h"

template <uint32_t schMode, uint32_t method>
__global__ __aicore__ void match_template(GM_ADDR image, GM_ADDR templateImage, GM_ADDR score, GM_ADDR bestScore,
                                          GM_ADDR bestLocation, GM_ADDR workspace, GM_ADDR tiling)
{
    // Hardware SyncAll requires the Vector kernel to enable the scheduler
    // through the __mix__(0, 1) task type.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(MatchTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatchTemplateTilingData, tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    // Tiling workspace offsets are relative to the user workspace, after the system-reserved region.
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    if constexpr (schMode == MATCH_TEMPLATE_SCH_MODE_HALO_TILE) {
        NsMatchTemplate::KernelMatchTemplate<DTYPE_IMAGE, method> op;
        op.Init(image, templateImage, score, bestScore, bestLocation, userWorkspace, &tilingData);
        op.Process();
    }
}
