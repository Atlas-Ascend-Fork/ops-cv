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
 * \file guided_filter.cpp
 * \brief
 */

#include "guided_filter.h"

// kernel UT（ICPU）不传宏时给一个可用默认值，真机编译由 build 按 op def 的 dtype 组合注入
#ifndef DTYPE_GUIDE
#define DTYPE_GUIDE float
#endif
#ifndef DTYPE_SRC
#define DTYPE_SRC float
#endif
#ifndef DTYPE_DST
#define DTYPE_DST float
#endif

template <uint32_t schMode>
__global__ __aicore__ void guided_filter(GM_ADDR guide, GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    // 内核用无参硬件 SyncAll 做阶段 A/B 的跨核同步，按 MIX_AIV_1_0 口径声明 task type
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(GuidedFilterTilingData);
    GET_TILING_DATA_WITH_STRUCT(GuidedFilterTilingData, tilingData, tiling);

    if constexpr (schMode == GUIDED_FILTER_TPL_SCH_MODE_0) {
        NsGuidedFilter::GuidedFilter<DTYPE_GUIDE, DTYPE_SRC, DTYPE_DST> op;
        op.Init(guide, src, dst, workspace, &tilingData); // 算子kernel实例初始化
        op.Process();                                     // 算子kernel实例执行
    }
}
