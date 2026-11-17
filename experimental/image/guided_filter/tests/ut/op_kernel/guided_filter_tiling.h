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
 * \file guided_filter_tiling.h
 * \brief GuidedFilter 的 kernel UT 本地 tiling 头
 *
 * 真机构建时由 gen_tiling_head_file.sh 生成等价的头文件；kernel UT 直接复用算子手写的
 * tiling 数据结构（op_kernel/guided_filter_tiling_data.h），这样 UT 与内核读到的字段
 * 完全同源，不会出现"A/B 两份结构体各改一半"的漂移。
 *
 * 注意：本文件被 CMake 以 `-include` 强制注入到内核 .cpp 与测试 .cpp 之前，
 * 因此必须是自包含且可重复包含的。
 */

#ifndef _I_GUIDED_FILTER_TILING_H_
#define _I_GUIDED_FILTER_TILING_H_

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/guided_filter_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"

// CPU 侧没有 __aicore__ 属性，TILING 侧由内核编译选项提供
#define __aicore__

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, GuidedFilterTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (size_t i = 0; i < sizeof(GuidedFilterTilingData) / sizeof(uint32_t); i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, GuidedFilterTilingData* constData)
{
    memcpy(constData, tiling, sizeof(GuidedFilterTilingData));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)              \
    __ubuf__ tilingStruct* tilingDataPointer = reinterpret_cast<__ubuf__ tilingStruct*>( \
        (__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    GuidedFilterTilingData tilingData;         \
    InitTilingData(tilingArg, &tilingData)

#endif // _I_GUIDED_FILTER_TILING_H_
