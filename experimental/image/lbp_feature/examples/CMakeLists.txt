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
# 设置工程名
project(ACLNN_EXAMPLE)

# 设置C++编译标准
add_compile_options(-std=c++11)

# 设置编译输出目录为当前目录下的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

# 设置调试和发布模式的编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 添加可执行文件（已替换为 lbp_feature 实际文件名）
add_executable(test_aclnn_lbp_feature
test_aclnn_lbp_feature.cpp)

# 添加性能基准可执行文件
add_executable(benchmark_aclnn_lbp_feature
benchmark_aclnn_lbp_feature.cpp)

# ASCEND_PATH（如遇CANN包路径有误，请根据实际路径修改）
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

# 设置头文件路径（aclnn_lbp_feature.h 在 op_api/include 下）
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${TARGET_SUBDIR}/op_api/include
)
include_directories(
    ${INCLUDE_BASE_DIR}
)

# 链接所需的动态库（已替换为 lbp_feature 实际目标名）
target_link_libraries(test_aclnn_lbp_feature PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so      # 链接自定义算子库文件
)
target_link_options(test_aclnn_lbp_feature PRIVATE
    "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib"
)

target_link_libraries(benchmark_aclnn_lbp_feature PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so
)
target_link_options(benchmark_aclnn_lbp_feature PRIVATE
    "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib"
)

# 安装目标文件到bin目录
install(TARGETS test_aclnn_lbp_feature DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(TARGETS benchmark_aclnn_lbp_feature DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
