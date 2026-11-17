# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    # op_host 的 tiling 源码在这里只是为了配合 AddOpTestCase 生成 tiling 头；
    # 本目录同时提供了手写的 guided_filter_tiling.h（-include 优先用它），
    # 所以 kernel UT 不依赖构建期生成物。
    set(guided_filter_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/guided_filter_tiling.cpp
        )

    # 内核入口是模板，真机构建时 dtype 由 op def 注入；UT 固定 float32 口径。
    # 单核执行：内核用无参硬件 SyncAll 做阶段 A/B 的跨核同步，ICPU 模拟下屏障退化
    # 为空操作、block 串行执行，只有 blockDim == 1 才能保证
    # 「全部 tile 的 alpha/beta 落盘后再开始读」的语义与真机一致。
    AddOpTestCase(guided_filter "ascend910B1"
        "-DDTYPE_GUIDE=float -DDTYPE_SRC=float -DDTYPE_DST=float"
        "${guided_filter_tiling_files}")
endif()
