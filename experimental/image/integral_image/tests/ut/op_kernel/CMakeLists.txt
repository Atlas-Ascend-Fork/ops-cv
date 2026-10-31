# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的 tiling 源码（kernel 侧需要复用 host tiling 中的常量与结构定义）
    set(integral_image_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/integral_image_tiling.cpp
        )
    # param1：算子名称；param2：soc 版本（ascend910B1 为 910B 系列 CPU 仿真产品型号）；
    # param3：自定义编译选项；param4：该算子依赖的所有 tiling 源码文件
    AddOpTestCase(integral_image "ascend910B1" "" "${integral_image_tiling_files}")
endif()
