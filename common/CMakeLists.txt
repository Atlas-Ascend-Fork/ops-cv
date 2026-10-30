# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  npu_op_library(${COMMON_NAME}_obj TILING)
else()
  add_library(${COMMON_NAME}_obj OBJECT)
endif()

file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/op_host/*.cpp")

target_sources(${COMMON_NAME}_obj PRIVATE ${CPP_SOURCES})

target_include_directories(${COMMON_NAME}_obj
  PRIVATE
  ${C_SEC_INCLUDE}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${TILINGAPI_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${OPS_CV_DIR}/common/inc
  ${ASCEND_DIR}/pkg_inc
)

target_link_libraries(${COMMON_NAME}_obj
  PRIVATE
  $<BUILD_INTERFACE:intf_pub_cxx17>
  tiling_api
)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  add_subdirectory(stub)
endif()

# FE 兼容融合注册适配层（预埋休眠）：默认 OFF，本目录不参与构建，产物与 master 一致；
# 首个真实消费者接入时置 ON 并在 fusion_pass_common.cpp 登记名单
option(OP_GRAPH_FE_REGISTRY "编译 FE 兼容融合注册适配层（首个真实消费者接入时启用）" OFF)
if(OP_GRAPH_FE_REGISTRY AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/op_graph/CMakeLists.txt")
  add_subdirectory(src/op_graph)
endif()

if(ENABLE_TEST)
  add_subdirectory(tests)
endif()
