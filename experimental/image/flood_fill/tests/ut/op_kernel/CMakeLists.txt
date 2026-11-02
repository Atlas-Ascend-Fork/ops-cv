# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(UT_TEST_ALL OR OP_KERNEL_UT)
    # FloodFill shares a plain C++ tiling struct with its host. Register the
    # test object directly: the legacy AddOpTestCase reflection generator
    # expects a registered TilingData class and non-experimental kernel paths.
    foreach(ff_soc IN ITEMS Ascend910B1 Ascend910_9391)
        find_value_by_key("${SHORT_NAME_LIST}" "${FULL_NAME_LIST}" "${ASCEND_COMPUTE_UNIT}" ff_selected)
        string(TOLOWER "${ff_soc}" ff_lower)
        string(TOLOWER "${ff_selected}" ff_selected_lower)
        string(TOLOWER "${ASCEND_COMPUTE_UNIT}" ff_compute_lower)
        if(ASCEND_COMPUTE_UNIT AND NOT ff_lower STREQUAL ff_selected_lower AND NOT ff_lower STREQUAL ff_compute_lower)
            continue()
        endif()
        set(ff_target flood_fill_${ff_soc}_cases_obj)
        add_library(${ff_target} OBJECT test_flood_fill_kernel.cpp)
        target_include_directories(${ff_target} PRIVATE ${PROJECT_SOURCE_DIR} ${OP_TILING_INCLUDE})
        target_link_libraries(${ff_target} PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> tikicpulib::${ff_soc} GTest::gtest)
        set(ff_cases ${OP_KERNEL_MODULE_NAME}_${ff_soc}_cases_obj)
        if(NOT TARGET ${ff_cases})
            add_library(${ff_cases} OBJECT)
        endif()
        target_sources(${ff_cases} PRIVATE $<TARGET_OBJECTS:${ff_target}>)
        target_link_libraries(${ff_cases} PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> $<TARGET_OBJECTS:${ff_target}>)
        list(APPEND fastOpTestSocVersions ${ff_soc})
    endforeach()
    list(REMOVE_DUPLICATES fastOpTestSocVersions)
    set(fastOpTestSocVersions ${fastOpTestSocVersions} CACHE STRING "fastOp Test SocVersions" FORCE)
endif()
