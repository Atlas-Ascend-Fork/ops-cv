# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------


# 为target link依赖库 usage: add_modules(MODE SUB_LIBS EXTERNAL_LIBS) SUB_LIBS 为内部创建的target, EXTERNAL_LIBS为外部依赖的target
function(add_modules)
  set(oneValueArgs MODE)
  set(multiValueArgs TARGETS SUB_LIBS EXTERNAL_LIBS)

  cmake_parse_arguments(ARGS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(target ${ARGS_TARGETS})
    if(TARGET ${target} AND TARGET ${ARGS_SUB_LIBS})
      target_link_libraries(${target} ${ARGS_MODE} ${ARGS_SUB_LIBS})
    endif()
    if(TARGET ${target} AND ARGS_EXTERNAL_LIBS)
      target_link_libraries(${target} ${ARGS_MODE} ${ARGS_EXTERNAL_LIBS})
    endif()
  endforeach()
endfunction()

# 添加opbase object
function(add_opbase_modules)
  if(TARGET opbase_infer_objs OR TARGET opbase_tiling_objs OR TARGET opbase_util_objs)
    return()
  endif()
  file(GLOB_RECURSE OPS_BASE_INFER_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_broadcast_util.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_elewise_util.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_reduce_util.cpp
  )

  file(GLOB_RECURSE OPS_BASE_TILING_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/elewise/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/broadcast/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/reduce/*.cpp
  )

  file(GLOB_RECURSE OPS_BASE_UTIL_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/util/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/log/*.cpp
  )

  if(OPS_BASE_INFER_SRC)
    add_library(opbase_infer_objs OBJECT ${OPS_BASE_INFER_SRC})
    target_include_directories(opbase_infer_objs PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_options(opbase_infer_objs
        PRIVATE
        $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
        -fvisibility=hidden
    )
    target_link_libraries(
      opbase_infer_objs
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()

  if(OPS_BASE_TILING_SRC)
    add_library(opbase_tiling_objs OBJECT ${OPS_BASE_TILING_SRC})
    target_include_directories(opbase_tiling_objs PRIVATE ${OP_TILING_INCLUDE})
    target_compile_options(opbase_tiling_objs
        PRIVATE
        $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                        -fvisibility=hidden -fno-strict-aliasing
    )
    target_link_libraries(
      opbase_tiling_objs
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              tiling_api
      )
  endif()

  if(OPS_BASE_UTIL_SRC)
    add_library(opbase_util_objs OBJECT ${OPS_BASE_UTIL_SRC})
    target_include_directories(opbase_util_objs PRIVATE ${OP_TILING_INCLUDE} ${ASCEND_DIR}/pkg_inc)
    target_compile_options(opbase_util_objs
        PRIVATE
        $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
        -fvisibility=hidden
    )
    target_link_libraries(
      opbase_util_objs
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()
endfunction()

# 添加infer object
function(add_infer_modules)
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OPHOST_NAME}_infer_obj GRAPH)
    else()
      add_library(${OPHOST_NAME}_infer_obj OBJECT)
    endif()
    target_include_directories(${OPHOST_NAME}_infer_obj PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_infer_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO" OP_SUBMOD_NAME="OPS_CV"
                                       $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPSPROTO_UT> LOG_CPP
      )
    target_compile_options(
      ${OPHOST_NAME}_infer_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    target_link_libraries(
      ${OPHOST_NAME}_infer_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      )
  endif()
endfunction()

# 添加tiling object
function(add_tiling_modules)
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OPHOST_NAME}_tiling_obj TILING)
    else()
      add_library(${OPHOST_NAME}_tiling_obj OBJECT)
    endif()
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${OP_TILING_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_tiling_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING" OP_SUBMOD_NAME="OPS_CV"
                                        $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPTILING_UT> LOG_CPP
      )
    target_compile_options(
      ${OPHOST_NAME}_tiling_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                        -fvisibility=hidden -fno-strict-aliasing
      )
    target_link_libraries(
      ${OPHOST_NAME}_tiling_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
              $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
              tiling_api
      )
  endif()
endfunction()

# 添加opapi object
function(add_opapi_modules)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OPHOST_NAME}_opapi_obj ACLNN)
    else()
      add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    endif()
    unset(opapi_ut_depends_inc)
    if(ENABLE_TEST)
      set(opapi_ut_depends_inc ${UT_PATH}/op_api/stub)
    endif()
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
            ${opapi_ut_depends_inc}
            ${OPAPI_INCLUDE})
    target_compile_options(${OPHOST_NAME}_opapi_obj PRIVATE -Dgoogle=ascend_private -DACLNN_LOG_FMT_CHECK)
    target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE LOG_CPP)
    target_link_libraries(
      ${OPHOST_NAME}_opapi_obj
      PUBLIC $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
      PRIVATE $<BUILD_INTERFACE:adump_headers> $<BUILD_INTERFACE:dlog_headers>
      $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
      )
  endif()
endfunction()

# determine whether aicpu kernels skip the processing.
function(skip_aicpu_kernel op_type ascend_op_name)
  set(SKIP_FLAG FALSE PARENT_SCOPE)

  if(DISABLE_AICPU)
    message(STATUS "disable aicpu kernel ${op_type}, skip it.")
    set(SKIP_FLAG TRUE PARENT_SCOPE)
    return()
  endif()

  if(NOT "${ascend_op_name}" STREQUAL "")
    set(_ascend_op_name_tmp "${ascend_op_name}")
    separate_arguments(_ascend_op_name_tmp)

    list(FIND _ascend_op_name_tmp "${op_type}" _index)
    if(_index EQUAL -1)
      message(STATUS "[${op_type}] skipped, not in ascend_op_name list: ${ascend_op_name}")
      set(SKIP_FLAG TRUE PARENT_SCOPE)
      return()
    else()
      message(STATUS "[${op_type}] selected, in ascend_op_name list: ${ascend_op_name}")
    endif()
  endif()
endfunction()

# AICPU built-in package manifest: check whether <op_name> is allowed into the built-in package.
# Only gates device-side AICPU packaging (op_def/json + kernel .so); host constant folding
# and UT are not affected. ENABLE_CUSTOM bypasses the manifest.
# Sets _AICPU_BUILTIN_PKG_ALLOWED (TRUE/FALSE) in the caller's scope.
function(check_aicpu_builtin_pkg_allowed op_name)
  set(_allowed FALSE)
  if(AICPU_BUILTIN_PACKAGE_FORCE_ALL)
    set(_allowed TRUE)
  else()
    foreach(_entry ${AICPU_BUILTIN_PACKAGE_MANIFEST})
      if(_entry STREQUAL "${op_name}=ON")
        set(_allowed TRUE)
        break()
      endif()
    endforeach()
  endif()
  if(_allowed)
    message(STATUS "[${op_name}] allowed in AICPU built-in package by manifest")
  endif()
  set(_AICPU_BUILTIN_PKG_ALLOWED ${_allowed} PARENT_SCOPE)
endfunction()

function(add_aicpu_kernel_modules)
  message(STATUS "add_aicpu_kernel_modules")
  if(NOT TARGET ${OPHOST_NAME}_aicpu_obj)
    add_library(${OPHOST_NAME}_aicpu_obj OBJECT)
    set_target_properties(${OPHOST_NAME}_aicpu_obj PROPERTIES CANN_GLIBCXX_USE_CXX11_ABI 1)
    target_include_directories(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_aicpu_obj PRIVATE _FORTIFY_SOURCE=2 google=ascend_private
                                       $<$<BOOL:${ENABLE_TEST}>:ASCEND_AICPU_UT>
      )
    target_compile_options(
      ${OPHOST_NAME}_aicpu_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${OPHOST_NAME}_aicpu_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()
endfunction()

function(add_aicpu_cust_kernel_modules target_name)
  message(STATUS "add_aicpu_cust_kernel_modules for ${target_name}")
  if(NOT TARGET ${target_name})
    add_library(${target_name} OBJECT)
    set_target_properties(${target_name} PROPERTIES CANN_GLIBCXX_USE_CXX11_ABI 1)
    target_include_directories(${target_name} PRIVATE ${AICPU_INCLUDE})
    target_compile_definitions(
      ${target_name} PRIVATE
                    _FORTIFY_SOURCE=2
                    google=ascend_private
                    $<$<BOOL:${ENABLE_TEST}>:ASCEND_AICPU_UT>
      )
    target_compile_options(
      ${target_name} PRIVATE
                    $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                    -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${target_name}
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              -Wl,--no-whole-archive
              Eigen3::Eigen
      )
    if (NOT ${target_name} IN_LIST AICPU_CUST_OBJ_TARGETS)
      set(AICPU_CUST_OBJ_TARGETS ${AICPU_CUST_OBJ_TARGETS} ${target_name} CACHE INTERNAL "All aicpu cust obj targets")
    endif()
  endif()
endfunction()

# Compiles aicpu source as OBJECT for host side.
# Collects into AICPU_HOST_OBJ_TARGETS; linking into libopconstant_folding_cv.so
# is done in symbol.cmake gen_aicpu_const_symbol().
function(add_aicpu_host_kernel_modules host_target_name)
  message(STATUS "add aicpu host kernel modules for ${host_target_name}")
  if(NOT TARGET ${host_target_name})
    add_library(${host_target_name} OBJECT)
    target_include_directories(${host_target_name} PRIVATE
        ${AICPU_INCLUDE}
        ${CANN_3RD_LIB_PATH}/eigen
    )
    target_compile_definitions(
      ${host_target_name} PRIVATE
                    _FORTIFY_SOURCE=2
                    google=ascend_private
                    OPS_CV_AICPU_HOST_KERNEL
      )
    target_compile_options(
      ${host_target_name} PRIVATE
                    -Dgoogle=ascend_private
                    -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${host_target_name}
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              Eigen3::Eigen
      )
    if (NOT ${host_target_name} IN_LIST AICPU_HOST_OBJ_TARGETS)
      set(AICPU_HOST_OBJ_TARGETS
          ${AICPU_HOST_OBJ_TARGETS} ${host_target_name}
          CACHE INTERNAL "All aicpu host builtin obj targets")
    endif()
  endif()
endfunction()

function(add_aicpu_utils_modules)
  add_aicpu_cust_kernel_modules(utils_cust_obj)
  target_sources(utils_cust_obj PRIVATE ${OPS_CV_DIR}/common/src/common/allocator_utils.cpp)
endfunction()

function(add_op_graph_modules)
  if(NOT TARGET ${OP_GRAPH_NAME}_obj)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${OP_GRAPH_NAME}_obj GRAPH)
    else()
      add_library(${OP_GRAPH_NAME}_obj OBJECT)
    endif()
    add_dependencies(${OP_GRAPH_NAME}_obj json)
    target_include_directories(${OP_GRAPH_NAME}_obj PRIVATE
      ${OP_PROTO_INCLUDE}
      ${PROJECT_SOURCE_DIR}/common/inc
      ${JSON_INCLUDE_DIR}
      ${JSON_INSTALL_PATH}/include
      ${ASCEND_DIR}/include
      ${ASCEND_DIR}/include/external
      ${ASCEND_DIR}/include/exe_graph
      ${ASCEND_DIR}/include/base/context_builder
      ${ASCEND_DIR}/include/ge
    )
    target_compile_definitions(${OP_GRAPH_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_GRAPH" LOG_CPP)

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      target_compile_options(
        ${OP_GRAPH_NAME}_obj PRIVATE -Dgoogle=ascend_private -fvisibility=hidden
      )
    else()
      target_compile_options(
        ${OP_GRAPH_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    endif()
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      target_link_libraries(
      ${OP_GRAPH_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
              metadef
              graph
              register
              ge_compiler
      )
    else()
      target_link_libraries(
      ${OP_GRAPH_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      )
    endif()
  endif()
endfunction()

# usage: add_modules_sources(OPTYPE ACLNNTYPE) ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude OPTYPE 和 ACLNNTYPE
# 需一一对应
macro(add_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_opbase_modules()
  # opapi l0 默认全部编译
  file(GLOB OPAPI_L0_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  list(FILTER OPAPI_L0_SRCS EXCLUDE REGEX "aclnn_")
  if(OPAPI_L0_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L0_SRCS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)

  if("add_example_aicpu" IN_LIST ASCEND_OP_NAME AND "${OP_NAME}" STREQUAL "add_example")
    set(INDEX 0)
  endif()

  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
    # ASCEND_OP_NAME 为空表示全部编译
    return()
  endif()

  if(OP_NAME IN_LIST COMPILED_OPS)
    # 已经编译过，忽略
    message(STATUS "already compiled ${OP_NAME}, skip")
    return()
  endif()

  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如image/crop_and_resize
  set(COMPILED_OPS
      ${COMPILED_OPS} ${OP_NAME}
      CACHE STRING "Compiled Ops" FORCE
    )
  set(COMPILED_OP_DIRS
      ${COMPILED_OP_DIRS} ${PARENT_DIR}
      CACHE STRING "Compiled Ops Dirs" FORCE
    )

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if(OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  file(GLOB OPAPI_L2_SRCS ${SOURCE_DIR}/op_api/aclnn_*.cpp)
  if(OPAPI_L2_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L2_SRCS})
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if(OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB OPTILING_SRCS ${SOURCE_DIR}/*_tiling*.cpp)
  if(OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
    pypto_force_include_tiling("${OPTILING_SRCS}")
  endif()

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/*_aicpu.cpp)
  if(AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if(${AclnnType} STREQUAL "aclnn"
         OR ${AclnnType} STREQUAL "aclnn_inner"
         OR ${AclnnType} STREQUAL "aclnn_exclude"
        )
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)
        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSUPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(
        FATAL_ERROR
          "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
          "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
          "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
        )
    endif()
  endif()
endmacro()

function(add_dependent_ops dependent_ops)
  foreach(dep_op ${dependent_ops})
    # 查询依赖算子所在目录
    file(GLOB
      dep_op_path_list
      "${CMAKE_CURRENT_SOURCE_DIR}/../../image/${dep_op}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../objdetect/${dep_op}"
    )
    # 检查依赖存在
    if(NOT dep_op_path_list)
      message(FATAL_ERROR "dependent operator(${dep_op}) not exists")
    endif()
    list(LENGTH ${dep_op_path_list} find_dep_ops_count)
    if(find_dep_ops_count GREATER 1)
      message(FATAL_ERROR "dependent operator(${dep_op}) is not unique, the found operators:${dep_op_path_list}")
    endif()

    # 不在编译列表则加入，ASCEND_OP_NAME 为空表示全部编译，则不单独编译
    if(ASCEND_OP_NAME AND (NOT (dep_op IN_LIST ASCEND_OP_NAME)))
      # 加入依赖并去重
      set(NEW_OP_NAMES ${ASCEND_OP_NAME} ${dep_op})
      list(REMOVE_DUPLICATES NEW_OP_NAMES)
      set(ASCEND_OP_NAME
          ${NEW_OP_NAMES}
          CACHE STRING "Ascend op names to compile" FORCE
        )
      # 添加目录
      get_filename_component(dep_op_path "${dep_op_path_list}" ABSOLUTE)
      get_filename_component(dep_op_parent "${dep_op_path}" DIRECTORY)
      get_filename_component(parent_path "${dep_op_parent}" NAME)
      message(STATUS "add dependent operator: ${parent_path}/${dep_op}, path: ${dep_op_path}")
      add_subdirectory("${dep_op_path}" "${CMAKE_BINARY_DIR}/dependent-ops/${parent_path}/${dep_op}")
    endif()
  endforeach()
endfunction()

# 从两个长度一致的列表中查找相同位置的元素
function(find_value_by_key key_list value_list search_key result)
  list(LENGTH key_list key_list_length)
  list(LENGTH value_list value_list_length)
  if(NOT ${key_list_length} EQUAL ${value_list_length})
    message(FATAL_ERROR "key_list length is ${key_list_length}, value_list length is ${value_list_length}, not equal")
  endif()
  set(found_value "")
  if(key_list_length GREATER 0)
    list(FIND key_list ${search_key} index)
    if(NOT ${index} EQUAL -1)
      list(GET value_list ${index} found_value)
    endif()
  endif()
  set(${result} ${found_value} PARENT_SCOPE)
endfunction()

###################################################################################################
# get npuArch by socVersion, mapping is consistent with get_simulator_chip_version in scripts/build_example.sh
# 映射列表 ASCEND_ALL_COMPUTE_UNIT/SOC_TO_ARCH_VALUES 定义在 soc_arch_variables.cmake，两者按位置一一对应
# usage: get_npu_arch_by_soc_version(<soc_version> <out_var>)
# e.g.: get_npu_arch_by_soc_version("ascend910b" npu_arch) -> npu_arch is dav_2201
# 未匹配的 socVersion（如 ascend610lite/ascend630 等暂无 npuArch 的 soc）返回空串
###################################################################################################
function(get_npu_arch_by_soc_version soc_version npu_arch)
    string(TOLOWER "${soc_version}" soc_version_lower)
    list(FIND ASCEND_ALL_COMPUTE_UNIT ${soc_version_lower} index)
    if(NOT index EQUAL -1)
        list(GET SOC_TO_ARCH_VALUES ${index} mapped_value)
        set(${npu_arch} ${mapped_value} PARENT_SCOPE)
    else()
        set(${npu_arch} "" PARENT_SCOPE)
    endif()
endfunction()

function(get_soc_version_by_npu_arch npu_arch soc_version)
    list(LENGTH SOC_TO_ARCH_VALUES arch_list_length)
    set(mapped_list)
    set(index 0)
    # SOC_TO_ARCH_VALUES 含空占位元素（foreach 展开会丢弃空元素），须用 list(GET) 按位遍历
    while(index LESS arch_list_length)
        list(GET SOC_TO_ARCH_VALUES ${index} arch)
        if("${npu_arch}" STREQUAL "${arch}")
            list(GET ASCEND_ALL_COMPUTE_UNIT ${index} mapped_value)
            list(APPEND mapped_list ${mapped_value})
        endif()
        math(EXPR index "${index} + 1")
    endwhile()
    if(mapped_list)
        set(${soc_version} ${mapped_list} PARENT_SCOPE)
    else()
        set(${soc_version} "" PARENT_SCOPE)
    endif()
endfunction()

# ------------------------------------------------------------------------------------------------------------
# require_pypto_pro(<op_name>)
#   Require pypto_pro for the current operator. Call at the beginning of an operator's CMakeLists.txt, before
#   registering any host or kernel build rules. If pypto_pro is unavailable in ASCEND_PYTHON_EXECUTABLE, warn
#   and return from the caller's CMakeLists.txt so the whole operator is skipped.
# ------------------------------------------------------------------------------------------------------------
macro(require_pypto_pro op_name)
  get_property(_pypto_pro_checked GLOBAL PROPERTY PYPTO_PRO_CHECKED)
  if(NOT _pypto_pro_checked)
    execute_process(
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} -c
              "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pypto_pro') else 1)"
      RESULT_VARIABLE _pypto_pro_check_result
      OUTPUT_QUIET
      ERROR_QUIET
      )
    if(_pypto_pro_check_result EQUAL 0)
      set_property(GLOBAL PROPERTY PYPTO_PRO_AVAILABLE TRUE)
    else()
      set_property(GLOBAL PROPERTY PYPTO_PRO_AVAILABLE FALSE)
    endif()
    set_property(GLOBAL PROPERTY PYPTO_PRO_CHECKED TRUE)
  endif()

  get_property(_pypto_pro_available GLOBAL PROPERTY PYPTO_PRO_AVAILABLE)
  if(NOT _pypto_pro_available)
    message(WARNING "pypto_pro is unavailable in ${ASCEND_PYTHON_EXECUTABLE}; skip operator ${op_name}.")
    return()
  endif()

  unset(_pypto_pro_checked)
  unset(_pypto_pro_check_result)
  unset(_pypto_pro_available)
endmacro()

# ------------------------------------------------------------------------------------------------------------
# enable_pypto_kernel(<op_file>)
#   Mark an operator whose kernel is written in PyPTO. Call ONCE from the op's CMakeLists.txt, and BEFORE
#   add_all_modules_sources(...) / add_modules_sources(...) (so the tiling logic can read PYPTO_GEN_DIR).
#   Works from both ops-cv call sites: the op root CMakeLists.txt (op_kernel/<op_file>.py) and an
#   op_host/CMakeLists.txt (../op_kernel/<op_file>.py).
#   It runs pypto codegen at configure time into a pypto-exclusive dir, exposes PYPTO_GEN_DIR to the caller
#   scope (for the host tiling force-include), and records cache flags consumed by kernel_src_copy /
#   prepare_compile_from_config to install the generated artifacts into the kernel staging dirs.
# ------------------------------------------------------------------------------------------------------------
function(enable_pypto_kernel op_file)
  set(_compute_units ${ASCEND_COMPUTE_UNIT})
  list(FILTER _compute_units INCLUDE REGEX ".+")
  if(NOT _compute_units)
    message(FATAL_ERROR "enable_pypto_kernel: ASCEND_COMPUTE_UNIT is empty")
  endif()
  list(GET _compute_units 0 _soc_unit)

  # ops-cv has two op layouts: add_all_modules_sources() from the op root, and add_modules_sources() from
  # op_host/. Accept both so enable_pypto_kernel() sits next to whichever one the operator uses.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/${op_file}.py")
    set(_op_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/${op_file}.py")
    get_filename_component(_op_dir "${CMAKE_CURRENT_SOURCE_DIR}/.." REALPATH)
  else()
    message(FATAL_ERROR "enable_pypto_kernel: kernel python file not found: "
                        "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/${op_file}.py")
  endif()
  set(_py "${_op_dir}/op_kernel/${op_file}.py")
  get_filename_component(_op_name "${_op_dir}" NAME)

  # Same op selection as add_modules_sources / add_all_modules_sources: an empty ASCEND_OP_NAME means
  # build everything. Bail out before the codegen for an op this build did not ask for, so an unrelated
  # --ops=... build neither pays for the codegen nor picks up --pypto-ops / PYPTO_JIT_ARCH.
  list(FIND ASCEND_OP_NAME ${_op_name} _op_index)
  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND _op_index EQUAL -1)
    return()
  endif()

  set(_gen "${CMAKE_BINARY_DIR}/binary/${_soc_unit}/${op_file}_pypto_gen")

  message(STATUS "pypto codegen: ${op_file} -> ${_gen}")
  execute_process(
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${OPS_CV_DIR}/scripts/util/pypto_codegen.py
            --py-file "${_py}" --out-dir "${_gen}"
            --op-file "${op_file}" --soc "${_soc_unit}"
    RESULT_VARIABLE _rc
    )
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "pypto codegen failed for ${op_file} (exit ${_rc})")
  endif()

  # Re-run configure (hence codegen) when the kernel python source changes.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_py})

  # Read by add_tiling_sources / add_modules_sources (they run in this caller scope) to force-include the
  # generated tiling headers into this op's tiling cpp.
  set(PYPTO_GEN_DIR ${_gen} PARENT_SCOPE)
  # Read by kernel_src_copy and prepare_compile_from_config (separate scopes), keyed by the op dir name so
  # they can stage the generated artifacts next to the copied op_kernel sources.
  set(_gen_dirs ${${_op_name}_pypto_gen_dirs})
  list(APPEND _gen_dirs ${_gen})
  list(REMOVE_DUPLICATES _gen_dirs)
  set(${_op_name}_pypto_gen_dirs "${_gen_dirs}" CACHE INTERNAL "pypto gen dirs for op ${_op_name}")
  # Collected by add_ops_impl_target and passed to ascendc_impl_build.py as --pypto-ops so those ops get the
  # pypto wrapper variant (it calls pypto_compile_op on the DSL .py). Keyed by op_file (matches
  # op_desc.op_file).
  set_property(GLOBAL APPEND PROPERTY PYPTO_ENABLED_OPS ${op_file})
endfunction()

# ------------------------------------------------------------------------------------------------------------
# pypto_soc_to_arch(<soc> <out_var>)
#   Map an ASCEND_COMPUTE_UNIT to the pypto codegen arch token, mirroring _ARCH_MAP in
#   scripts/util/pypto_codegen.py. Sets <out_var> to the empty string for an unmapped soc, which leaves
#   pypto on its own default.
# ------------------------------------------------------------------------------------------------------------
function(pypto_soc_to_arch soc out_var)
  string(TOLOWER "${soc}" _soc)
  if(_soc STREQUAL "ascend910b" OR _soc STREQUAL "ascend910c")
    set(${out_var} "a3" PARENT_SCOPE)
  elseif(_soc STREQUAL "ascend950")
    set(${out_var} "a5" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

# ------------------------------------------------------------------------------------------------------------
# pypto_jit_arch_env(<soc> <out_var>)
#   Build the `export PYPTO_JIT_ARCH=<arch> &&` prefix for the kernel-compile commands of ops marked by
#   enable_pypto_kernel. pypto_compile_op() reads PYPTO_JIT_ARCH in the opc process and otherwise defaults
#   to a5, which emits instructions the other socs cannot assemble. Sets <out_var> to the empty string when
#   no pypto op is being built or the soc has no mapping (non-pypto ops never read the variable).
# ------------------------------------------------------------------------------------------------------------
function(pypto_jit_arch_env soc out_var)
  set(${out_var} "" PARENT_SCOPE)
  get_property(_pypto_ops GLOBAL PROPERTY PYPTO_ENABLED_OPS)
  if(NOT _pypto_ops)
    return()
  endif()
  pypto_soc_to_arch("${soc}" _arch)
  if(NOT _arch)
    message(WARNING "pypto: no arch mapping for soc '${soc}'; leaving PYPTO_JIT_ARCH unset.")
    return()
  endif()
  set(${out_var} export PYPTO_JIT_ARCH=${_arch} && PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------------------------------------
# pypto_force_include_tiling(<tiling_srcs>)
#   The op was marked by enable_pypto_kernel (which set PYPTO_GEN_DIR): force-include the generated headers
#   into this op's tiling cpp, so the source needs no explicit include. Scoped by file list, so it only hits
#   this op's tiling sources and does not leak onto other operators sharing ${OPHOST_NAME}_tiling_obj.
#   Order: *_tilingkey.h first (its template_argument.h pulls <cstdint>), *_tiling.h after (it uses int64_t).
# ------------------------------------------------------------------------------------------------------------
function(pypto_force_include_tiling tiling_srcs)
  if(NOT DEFINED PYPTO_GEN_DIR)
    return()
  endif()
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(WARNING "pypto force-include of generated tiling headers needs CMake >= 3.18 "
                    "(TARGET_DIRECTORY); got ${CMAKE_VERSION}. Tiling sources must include them explicitly.")
    return()
  endif()
  file(GLOB _pypto_key_h ${PYPTO_GEN_DIR}/*_tilingkey.h)
  file(GLOB _pypto_data_h ${PYPTO_GEN_DIR}/*_tiling.h)
  set(_pypto_force_opts "")
  foreach(_h ${_pypto_key_h} ${_pypto_data_h})
    list(APPEND _pypto_force_opts -include ${_h})
  endforeach()
  if(NOT _pypto_force_opts)
    return()
  endif()
  set_source_files_properties(${tiling_srcs}
                              TARGET_DIRECTORY ${OPHOST_NAME}_tiling_obj
                              PROPERTIES COMPILE_OPTIONS "${_pypto_force_opts}")
endfunction()

function(add_tiling_sources tiling_dir disable_in_opp)
  if(NOT disable_in_opp)
    set(disable_in_opp FALSE)
  endif()
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND disable_in_opp)
    message(STATUS "don't need add tiling sources")
    return()
  endif()

  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  if("${tiling_dir}" STREQUAL "")
    file(GLOB OPTILING_SRCS ${SOURCE_DIR}/op_host/*_tiling*.cpp)
  else()
    file(GLOB OPTILING_SRCS ${SOURCE_DIR}/op_host/*_tiling*.cpp ${SOURCE_DIR}/op_host/${tiling_dir}/*_tiling*.cpp)
  endif()

  if(OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
    pypto_force_include_tiling("${OPTILING_SRCS}")
  endif()
endfunction()

# usage: add_all_modules_sources(OPTYPE ACLNNTYPE DEPENDENCIES COMPUTE_UNIT NPU_ARCH TILING_DIR DISABLE_IN_OPP HOSTCPU)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应
# DEPENDENCIES 指定依赖的算子名称列表
# COMPUTE_UNIT 兼容参数：设置支持芯片版本号（soc列表），必须与TILING_DIR一一对应，缺省为配置所有的soc
# NPU_ARCH 设置支持的npuArch列表，必须与TILING_DIR一一对应，示例：dav_2201 dav_3510；与COMPUTE_UNIT互斥，
#          配置后当前构建芯片转换为npuArch统一按arch匹配
# TILING_DIR 设置所支持芯片类型对应的tiling文件目录，必须与COMPUTE_UNIT（或NPU_ARCH）一一对应，示例：arch22 arch35
# DISABLE_IN_OPP 设置是否在opp包中编译tiling文件，布尔类型：TRUE，FALSE
# HOSTCPU 设置是否编译host侧常量折叠OBJECT，布尔类型：TRUE，FALSE，仅在built-in包生效
macro(add_all_modules_sources)
  set(oneValueArgs DISABLE_IN_OPP HOSTCPU)
  set(multiValueArgs OPTYPE ACLNNTYPE DEPENDENCIES COMPUTE_UNIT NPU_ARCH TILING_DIR)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_opbase_modules()
  # opapi l0 默认全部编译
  file(GLOB OPAPI_L0_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  list(FILTER OPAPI_L0_SRCS EXCLUDE REGEX "aclnn_")
  if(OPAPI_L0_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L0_SRCS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(OP_NAME ${SOURCE_DIR} NAME)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
    # ASCEND_OP_NAME 为空表示全部编译
    return()
  endif()

  if(OP_NAME IN_LIST COMPILED_OPS)
    # 已经编译过，忽略
    message(STATUS "already compiled ${OP_NAME}, skip")
    return()
  endif()
  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如image/resize_bilinear_v2
  set(COMPILED_OPS
      ${COMPILED_OPS} ${OP_NAME}
      CACHE STRING "Compiled Ops" FORCE
    )
  set(COMPILED_OP_DIRS
      ${COMPILED_OP_DIRS} ${SOURCE_DIR}
      CACHE STRING "Compiled Ops Dirs" FORCE
    )

  # 添加依赖算子
  add_dependent_ops("${MODULE_DEPENDENCIES}")

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if(OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()
  file(GLOB OPAPI_L2_SRCS ${SOURCE_DIR}/op_api/aclnn_*.cpp)
  if(OPAPI_L2_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L2_SRCS})
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/op_host/*_infershape*.cpp)
  if(OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  # 添加tiling文件：NPU_ARCH 与 TILING_DIR 按位置一一配对，当前构建芯片转换为 npuArch 后统一按 arch 匹配；
  # 未配置 NPU_ARCH 时走原 COMPUTE_UNIT 路径，按 soc 名匹配（保持既有行为不变）；
  # 构建芯片无 npuArch 映射（如 ascend610lite/ascend630）时与旧路径对不支持 soc 的行为一致，回退默认 tiling 目录
  set(tiling_dir "")
  if(MODULE_NPU_ARCH)
    get_npu_arch_by_soc_version("${ASCEND_COMPUTE_UNIT}" _cur_key)
    set(_cur_keys "${MODULE_NPU_ARCH}")
  else()
    set(_cur_key "${ASCEND_COMPUTE_UNIT}")
    set(_cur_keys "${MODULE_COMPUTE_UNIT}")
  endif()
  if(_cur_key)
    find_value_by_key("${_cur_keys}" "${MODULE_TILING_DIR}" "${_cur_key}" tiling_dir)
  endif()
  add_tiling_sources("${tiling_dir}" "${MODULE_DISABLE_IN_OPP}")

  # Some existing TF plugin sources in ops-cv still depend on non-public GE headers
  # and cannot be built with the installed CANN package yet. Keep automatic
  # collection disabled by default and use add_tf_plugin_sources() for explicit
  # verification until those files are adapted.
  if(ENABLE_AUTO_TF_PLUGIN_SOURCES AND BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_TEST)
    file(GLOB TF_PLUGIN_SRCS ${SOURCE_DIR}/framework/*_tf_plugin.cpp)
    if(TF_PLUGIN_SRCS)
      target_sources(${TF_PLUGIN_NAME}_obj PRIVATE ${TF_PLUGIN_SRCS})
    endif()
  endif()

  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    file(GLOB AICPU_SRCS ${SOURCE_DIR}/op_kernel_aicpu/*_aicpu*.cpp)
    if(AICPU_SRCS)
      add_aicpu_kernel_modules()
      target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
    endif()
  endif()

  # Host-side constant folding: compile aicpu source as x86 OBJECT for libopconstant_folding_cv.so
  # Only enabled when HOSTCPU TRUE is passed to add_all_modules_sources
  if(MODULE_HOSTCPU AND BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_CUSTOM)
    file(GLOB AICPU_HOST_SRCS ${SOURCE_DIR}/op_kernel_aicpu/*_aicpu.cpp)
    if(AICPU_HOST_SRCS AND NOT DISABLE_AICPU)
      set(HOST_OBJ_NAME ${OP_NAME}_host_const_obj)
      add_aicpu_host_kernel_modules(${HOST_OBJ_NAME})
      target_sources(${HOST_OBJ_NAME} PRIVATE ${AICPU_HOST_SRCS})
    endif()
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One (cv op: ${OP_NAME})")
    endif()

    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)

      if(${AclnnType} STREQUAL "aclnn"
         OR ${AclnnType} STREQUAL "aclnn_inner"
         OR ${AclnnType} STREQUAL "aclnn_exclude"
        )
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/op_host/${OpType}_def*.cpp)
        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSUPPORTED (cv op: ${OP_NAME})! Only support aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/op_host/*_def*.cpp)
    if(OPDEF_SRCS)
      message(
        FATAL_ERROR
          "cv op ${OP_NAME}: Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
          "usage: add_all_modules_sources(OPTYPE [op1 op2] ACLNNTYPE [type1 type2])\n"
          "example: add_all_modules_sources(OPTYPE resize_bilinear_v2 ACLNNTYPE aclnn_exclude)"
        )
    endif()
  endif()

  file(GLOB OP_GRAPH_SRCS ${SOURCE_DIR}/op_graph/*_graph_*.cpp ${SOURCE_DIR}/op_graph/*_fallback.cpp ${SOURCE_DIR}/op_graph/fusion_pass/*_pass.cpp)
  if(OP_GRAPH_SRCS)
    add_op_graph_modules()
    target_sources(${OP_GRAPH_NAME}_obj PRIVATE ${OP_GRAPH_SRCS})
  endif()

  file(GLOB OP_GRAPH_PROTO_HEADERS ${SOURCE_DIR}/op_graph/*_proto*.h)
  if(OP_GRAPH_PROTO_HEADERS)
    target_sources(${OP_GRAPH_NAME}_proto_headers INTERFACE ${OP_GRAPH_PROTO_HEADERS})
  endif()
endmacro()


# usage: add_op_graph_sources()
macro(add_op_graph_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(DEFINED ASCEND_OP_NAME
     AND NOT "${ASCEND_OP_NAME}" STREQUAL ""
    )
    if(NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
      return()
    endif()
  endif()

  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    file(GLOB OP_GRAPH_SRCS ${SOURCE_DIR}/*_graph*.cpp ${SOURCE_DIR}/fusion_pass/*_pass.cpp)
  else()
    file(GLOB OP_GRAPH_SRCS ${SOURCE_DIR}/*_graph*.cpp)
  endif()
  if(OP_GRAPH_SRCS)
    add_op_graph_modules()
    target_sources(${OP_GRAPH_NAME}_obj PRIVATE ${OP_GRAPH_SRCS})
  endif()

  file(GLOB OP_GRAPH_PROTO_HEADERS ${SOURCE_DIR}/*_proto*.h)
  if(OP_GRAPH_PROTO_HEADERS)
    target_sources(${OP_GRAPH_NAME}_proto_headers INTERFACE ${OP_GRAPH_PROTO_HEADERS})
  endif()
endmacro()

# ######################################################################################################################
# get operating system info
# ######################################################################################################################
function(get_system_info SYSTEM_INFO)
  if(UNIX)
    execute_process(
      COMMAND
        grep -i ^id= /etc/os-release
      OUTPUT_VARIABLE TEMP
      )
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO}
        ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
        PARENT_SCOPE
      )
  elseif(WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif()
endfunction()

# ######################################################################################################################
# add compile options, e.g.: -g -O0
# ######################################################################################################################
function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${OPS_CV_DIR}/scripts/util/ascendc_gen_options.py
      ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT} ${OP_COMPILE_OPTIONS}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR
    )
  if(${EXEC_RESULT})
    message("add ops compile options info: ${EXEC_INFO}")
    message("add ops compile options error: ${EXEC_ERROR}")
    message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

# ######################################################################################################################
# check whether the compiled operators meet expectations
# ######################################################################################################################
function(check_compiled_ops)
  # framework ONNX 插件（common/src/framework）可能没有独立算子实现目录，不走 image/objdetect
  # 的目录扫描，故 COMPILED_OPS 不会含它们；但插件源码与 UT 仍存在，--ops=<插件算子> 应生效。
  # 这里在比对前主动扫描 common/src/framework 的 *_onnx_plugin.cpp，从文件名提取算子名注入。
  # 注意：本函数在 CMakeLists.txt 末尾调用，无论 ENABLE_EXPERIMENTAL 与否都会执行
  # （experimental 模式下 common/src/framework 不被 add_subdirectory，宏注入路径不可达，必须在此注入）。
  file(GLOB _fw_onnx_plugin_srcs ${OPS_CV_DIR}/common/src/framework/*_onnx_plugin.cpp)
  foreach(_plugin_src ${_fw_onnx_plugin_srcs})
    get_filename_component(_plugin_name ${_plugin_src} NAME_WE)
    string(REGEX REPLACE "_onnx_plugin$" "" _op_name "${_plugin_name}")
    if(_op_name AND NOT _op_name IN_LIST COMPILED_OPS)
      set(COMPILED_OPS
          ${COMPILED_OPS} ${_op_name}
          CACHE STRING "Compiled Ops" FORCE)
    endif()
  endforeach()

  message(STATUS "Ops for this compilation contains: ${COMPILED_OPS}")
  if(COMPILED_OPS STREQUAL "")
    message(FATAL_ERROR "Specified ops not found in this depository, please check --ops parameter")
  endif()

  # 未指定算子，全部编译
  if(NOT NEED_COMPILE_OPS)
    return()
  endif()

  # 指定了但未参与编译的算子，即为无效算子名，应该报错
  set(not_compiled_ops)
  foreach(op_name IN LISTS NEED_COMPILE_OPS)
    if(NOT op_name IN_LIST COMPILED_OPS)
      list(APPEND not_compiled_ops ${op_name})
    endif()
  endforeach()

  if(NOT not_compiled_ops)
    return()
  endif()

  list(JOIN not_compiled_ops "," not_compiled_ops_str)
  if(ENABLE_EXPERIMENTAL)
    message(FATAL_ERROR
        "Specified ops(${not_compiled_ops_str}) not found in experimental, please check --ops parameter"
    )
  else()
    message(FATAL_ERROR
        "Specified ops(${not_compiled_ops_str}) not found in this depository, please check --ops parameter"
    )
  endif()
endfunction()

function(protobuf_generate_external comp c_var h_var)
  if (NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
    return()
  endif()

  set(${c_var})
  set(${h_var})
  set(_add_target FALSE)

  set(extra_option "")
  foreach(arg ${ARGN})
    if ("${arg}" MATCHES "--proto_path")
      set(extra_option ${arg})
    endif()
  endforeach()

  foreach(file ${ARGN})
    if ("${file}" STREQUAL "TARGET")
      set(_add_target TRUE)
      continue()
    endif()

    if ("${file}" MATCHES "--proto_path")
      continue()
    endif()

    get_filename_component(abs_file ${file} ABSOLUTE)
    get_filename_component(file_name ${file} NAME_WE)
    get_filename_component(file_dir ${abs_file} PATH)
    get_filename_component(parent_subdir ${file_dir} NAME)

    if ("${parent_subdir}" STREQUAL "proto")
      set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
    else()
      set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
    endif()
    list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
    list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

    add_custom_command(
      OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
      COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
      COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I${file_dir} ${extra_option} --cpp_out=${proto_output_path} ${abs_file}
      DEPENDS ${abs_file} host_protoc
      COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
  endforeach()

    if (_add_target)
      add_custom_target(
        ${comp} DEPENDS ${${c_var}} ${${h_var}})
    endif()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()


function(add_onnx_plugin_modules)
  if (NOT TARGET ${ONNX_PLUGIN_NAME}_obj)
    set(ge_onnx_proto_srcs
      ${ASCEND_DIR}/include/proto/ge_onnx.proto)

    protobuf_generate_external(onnx ge_onnx_proto_cc ge_onnx_proto_h ${ge_onnx_proto_srcs})

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      npu_op_library(${ONNX_PLUGIN_NAME}_obj GRAPH ${ge_onnx_proto_h})
    else()
      add_library(${ONNX_PLUGIN_NAME}_obj OBJECT ${ge_onnx_proto_h})
    endif()
    # 为特定目标设置C++14标准
    set_target_properties(${ONNX_PLUGIN_NAME}_obj PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_include_directories(${ONNX_PLUGIN_NAME}_obj
      PRIVATE
      ${OP_PROTO_INCLUDE}
      ${CMAKE_BINARY_DIR}/bin
      ${CMAKE_BINARY_DIR}/proto
      ${ONNX_PLUGIN_COMMON_INCLUDE}
      ${JSON_INCLUDE_DIR}
      ${ABS_INSTALL_DIR}
    )
    target_compile_definitions(${ONNX_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="ONNX_PLUGIN" LOG_CPP)

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      target_compile_options(
        ${ONNX_PLUGIN_NAME}_obj PRIVATE -Dgoogle=ascend_private -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
      )
    else()
      target_compile_options(
        ${ONNX_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
      )
    endif()

    target_link_libraries(
      ${ONNX_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx14,intf_pub_cxx14>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
              ascend_protobuf_static
              json
      )
  endif()
endfunction()

macro(add_onnx_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  file(GLOB ONNX_PLUGIN_SRCS ${SOURCE_DIR}/*_onnx_plugin.cpp)
  add_onnx_plugin_modules()
  target_sources(${ONNX_PLUGIN_NAME}_obj PRIVATE ${ONNX_PLUGIN_SRCS})
endmacro()

function(add_tf_plugin_modules)
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG OR ENABLE_TEST)
    return()
  endif()

  if(NOT TARGET ${TF_PLUGIN_NAME}_obj)
    set(tf_proto_srcs
      ${ASCEND_DIR}/include/proto/ge_ir.proto
    )
    protobuf_generate_external(tf tf_proto_cc tf_proto_h ${tf_proto_srcs})
    npu_op_library(${TF_PLUGIN_NAME}_obj GRAPH ${tf_proto_h})
    set_target_properties(${TF_PLUGIN_NAME}_obj PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_include_directories(${TF_PLUGIN_NAME}_obj
      PRIVATE
      ${OP_PROTO_INCLUDE}
      ${HOST_PROTOC_SRC}
      ${HOST_PROTOC_PATH}
      ${PROTOBUF_INCLUDE_DIRS}
      ${CMAKE_BINARY_DIR}/proto
      ${TF_PLUGIN_COMMON_INCLUDE}
    )
    target_compile_definitions(${TF_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="TF_PLUGIN" LOG_CPP)
    target_compile_options(
      ${TF_PLUGIN_NAME}_obj PRIVATE -Dgoogle=ascend_private -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
    )
    target_link_libraries(
      ${TF_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
              ascend_protobuf_static
    )
  endif()
endfunction()

function(init_tf_plugin_modules)
  add_tf_plugin_modules()
endfunction()

function(add_tf_plugin_sources)
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG OR ENABLE_TEST)
    return()
  endif()

  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  file(GLOB TF_PLUGIN_SRCS ${SOURCE_DIR}/*_tf_plugin.cpp)
  if(TF_PLUGIN_SRCS)
    target_sources(${TF_PLUGIN_NAME}_obj PRIVATE ${TF_PLUGIN_SRCS})
  endif()
endfunction()


# 删除以下重复函数，使用 cmake 仓版本:
# - replace_cur_major_minor_ver
# - set_package / set_cann_package
# - set_build_dependencies / set_cann_build_dependencies
# - set_run_dependencies / set_cann_run_dependencies
# - check_pkg_build_deps / check_cann_pkg_build_deps
# - add_version_info_targets / add_cann_version_info_targets
# 以上重复函数已删除，使用 cmake 仓版本
