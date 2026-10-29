# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if (UT_TEST_ALL OR OP_KERNEL_UT)
    # Tiling sources that the kernel UT depends on.
    set(ssim_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/ssim_tiling.cpp
        )
    # param1: operator name, named in kernel style
    # param2: soc version, multiple separated by semicolon
    # param3: custom compile options
    # param4: all tiling source files this operator depends on
    AddOpTestCase(ssim "ascend910B1" "" "${ssim_tiling_files}")
endif()
