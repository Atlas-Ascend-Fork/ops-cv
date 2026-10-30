# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# FE 兼容融合注册适配层：把桩注册编译单元接入 op_graph_cv_obj。
# 参照 ops-math PR#5140 的接入方式，目标名/平台宏/名单均为本仓独立维护；
# 入口由 OP_GRAPH_FE_REGISTRY 选项门控（默认 OFF，见 common/CMakeLists.txt）。

file(GLOB OP_GRAPH_COMMON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(OP_GRAPH_COMMON_SRCS)
  add_op_graph_modules()
  target_include_directories(${OP_GRAPH_NAME}_obj PRIVATE
    ${PROJECT_SOURCE_DIR}/common/stub/inc/op_graph
  )
  # CANN 外部的 version/ge-compiler_version.h 宏守卫含连字符，告警在 -Werror 下变错误；
  # CANN 目录标记为 SYSTEM（-isystem）抑制之，本仓代码仍保持 -Werror（同 math 处理）。
  target_include_directories(${OP_GRAPH_NAME}_obj SYSTEM PRIVATE
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/include/external
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/external
  )
  # 仅编译 ascend950 时注入平台宏，桩注册内容才编入；非目标平台完全不编入。
  # 注意：用 ASCEND_COMPUTE_UNIT（本次构建选择），非 ASCEND_ALL_COMPUTE_UNIT（全平台常量）。
  if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    set_source_files_properties(${OP_GRAPH_COMMON_SRCS} PROPERTIES
      COMPILE_DEFINITIONS OP_GRAPH_CV_ASCEND950=1
    )
  endif()
  target_sources(${OP_GRAPH_NAME}_obj PRIVATE ${OP_GRAPH_COMMON_SRCS})
endif()
