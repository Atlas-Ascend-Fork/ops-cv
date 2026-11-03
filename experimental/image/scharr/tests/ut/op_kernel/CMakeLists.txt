# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的 tiling 文件路径（构建系统用它生成 tiling 相关产物）
    set(scharr_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/scharr_tiling.cpp
        )
    # param2 填实际硬件型号
    # 同一 (op, SoC) 只能注册一个 kernel UT 目标（目标名唯一）；dtype 变体在
    # test_scharr.cpp 内按 dtypeId 参数化实例化，全部用例在同一可执行内运行。
    AddOpTestCase(scharr "Ascend910B4" "-DDTYPE_INPUT=uint8" "${scharr_tiling_files}")
endif()
