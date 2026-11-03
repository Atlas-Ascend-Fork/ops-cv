# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(ACLNN_EXAMPLE)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")

set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(test_aclnn_scharr test_aclnn_scharr.cpp)
add_executable(benchmark_aclnn_scharr benchmark_aclnn_scharr.cpp)

# ASCEND_PATH（如遇 CANN 包路径有误，请根据实际路径修改）
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

# 获取自定义算子包名称：优先 custom_cv（本仓 vendor 名），否则取最后一个自定义包
set(VENDORS_DIR "${ASCEND_PATH}/opp/vendors")
if(IS_DIRECTORY "${VENDORS_DIR}/custom_cv")
    set(TARGET_SUBDIR "${VENDORS_DIR}/custom_cv")
else()
    file(GLOB CUSTOM_DIRS "${VENDORS_DIR}/*")
    foreach(CUSTOM_DIR ${CUSTOM_DIRS})
        if(IS_DIRECTORY ${CUSTOM_DIR})
            set(TARGET_SUBDIR ${CUSTOM_DIR})
        endif()
    endforeach()
endif()

if(NOT DEFINED TARGET_SUBDIR)
    message(FATAL_ERROR "在路径${ASCEND_PATH}中未找到自定义算子包")
endif()

message(STATUS "使用自定义算子包: ${TARGET_SUBDIR}")

set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${TARGET_SUBDIR}/op_api/include
)

target_link_libraries(test_aclnn_scharr PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so
)
target_link_libraries(benchmark_aclnn_scharr PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so
)
target_link_options(test_aclnn_scharr PRIVATE
    "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib"
)
target_link_options(benchmark_aclnn_scharr PRIVATE
    "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib"
)

install(TARGETS test_aclnn_scharr DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(TARGETS benchmark_aclnn_scharr DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
