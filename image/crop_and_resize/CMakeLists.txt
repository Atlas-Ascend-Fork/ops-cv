# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
set(ASCEND_OP_NAME "" CACHE STRING "Ascend op names to compile")
set(OP_TYPE "crop_and_resize")

skip_aicpu_kernel("${OP_TYPE}" "${ASCEND_OP_NAME}")

if(SKIP_FLAG)
  return()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

set(SUPPORT_COMPUTE_UNIT "ascend950" "ascend350")
set(SUPPORT_TILING_DIR "arch35" "arch35")

add_all_modules_sources(OPTYPE crop_and_resize ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
                        TILING_DIR ${SUPPORT_TILING_DIR} HOSTCPU TRUE)

# def.cpp 编入 tiling 库：注册宏在 OP_TILING_LIB 下向 V1 注册表 OpCheckFuncRegistry 登记
# check_supported 回调 CheckIfAICoreSupported，FE 引擎分配阶段查询；缺失则
# needCheckSupport 为 true 时所有输入含合规均被误拒 fallback AiCpu，已实测，勿删
if (TARGET ${OPHOST_NAME}_tiling_obj)
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE op_host/crop_and_resize_def.cpp)
endif()
