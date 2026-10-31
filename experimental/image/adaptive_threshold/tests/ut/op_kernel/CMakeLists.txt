# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
if(NOT (UT_TEST_ALL OR OP_KERNEL_UT))
    return()
endif()
if(ASCEND_OP_NAME AND NOT "adaptive_threshold" IN_LIST ASCEND_OP_NAME)
    return()
endif()

# The kernel uses its native tiling struct. Register the CPU tests directly;
# legacy tiling-header generation expects a registered TilingDataBase class.
foreach(socVersion IN ITEMS Ascend910B1 Ascend910_9391)
    if(ASCEND_COMPUTE_UNIT)
        if(socVersion STREQUAL "Ascend910B1" AND NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
            continue()
        endif()
        if(socVersion STREQUAL "Ascend910_9391" AND NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93")
            continue()
        endif()
    endif()

    set(caseTarget adaptive_threshold_${socVersion}_cases_obj)
    add_library(${caseTarget} OBJECT test_adaptive_threshold.cpp)
    target_compile_options(${caseTarget} PRIVATE -fno-access-control -include ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_threshold_tiling.h)
    target_include_directories(${caseTarget} PRIVATE
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api)
    target_link_libraries(${caseTarget} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> tikicpulib::${socVersion} GTest::gtest)

    set(moduleTarget ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
    if(NOT TARGET ${moduleTarget})
        add_library(${moduleTarget} OBJECT $<TARGET_OBJECTS:${caseTarget}>)
    else()
        target_sources(${moduleTarget} PRIVATE $<TARGET_OBJECTS:${caseTarget}>)
    endif()
    target_link_libraries(${moduleTarget} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> $<TARGET_OBJECTS:${caseTarget}>)

    list(APPEND fastOpTestSocVersions ${socVersion})
    list(REMOVE_DUPLICATES fastOpTestSocVersions)
    set(fastOpTestSocVersions "${fastOpTestSocVersions}" CACHE STRING "fastOp Test SocVersions" FORCE)
endforeach()
