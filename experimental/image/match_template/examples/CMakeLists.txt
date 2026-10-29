# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(MATCH_TEMPLATE_CANN_ROOT "${ASCEND_HOME_PATH}" CACHE PATH "CANN installation root")
set(MATCH_TEMPLATE_VENDOR_ROOT "" CACHE PATH "Installed custom vendor root")

if(NOT MATCH_TEMPLATE_CANN_ROOT)
    message(FATAL_ERROR "Set MATCH_TEMPLATE_CANN_ROOT or source the CANN environment first")
endif()
if(NOT MATCH_TEMPLATE_VENDOR_ROOT)
    message(FATAL_ERROR "Set MATCH_TEMPLATE_VENDOR_ROOT to the installed vendor directory")
endif()

find_path(MATCH_TEMPLATE_ACL_INCLUDE_DIR acl/acl.h
    PATHS "${MATCH_TEMPLATE_CANN_ROOT}/include"
          "${MATCH_TEMPLATE_CANN_ROOT}/aarch64-linux/include"
    NO_DEFAULT_PATH)
find_path(MATCH_TEMPLATE_OP_API_INCLUDE_DIR aclnn_match_template.h
    PATHS "${MATCH_TEMPLATE_VENDOR_ROOT}/op_api/include"
    NO_DEFAULT_PATH)
find_library(MATCH_TEMPLATE_CUST_OPAPI_LIBRARY cust_opapi
    PATHS "${MATCH_TEMPLATE_VENDOR_ROOT}/op_api/lib"
    NO_DEFAULT_PATH)
find_library(MATCH_TEMPLATE_ASCENDCL_LIBRARY ascendcl
    PATHS "${MATCH_TEMPLATE_CANN_ROOT}/lib64"
          "${MATCH_TEMPLATE_CANN_ROOT}/aarch64-linux/lib64"
    NO_DEFAULT_PATH)
find_library(MATCH_TEMPLATE_NNOPBASE_LIBRARY nnopbase
    PATHS "${MATCH_TEMPLATE_CANN_ROOT}/lib64"
          "${MATCH_TEMPLATE_CANN_ROOT}/aarch64-linux/lib64"
    NO_DEFAULT_PATH)

foreach(required_var IN ITEMS MATCH_TEMPLATE_ACL_INCLUDE_DIR MATCH_TEMPLATE_OP_API_INCLUDE_DIR
                              MATCH_TEMPLATE_CUST_OPAPI_LIBRARY MATCH_TEMPLATE_ASCENDCL_LIBRARY
                              MATCH_TEMPLATE_NNOPBASE_LIBRARY)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "" OR
       "${${required_var}}" MATCHES "-NOTFOUND$")
        message(FATAL_ERROR "Could not locate ${required_var}")
    endif()
endforeach()

set(MATCH_TEMPLATE_EXAMPLE_INCLUDE_DIRS
    "${MATCH_TEMPLATE_ACL_INCLUDE_DIR}"
    "${MATCH_TEMPLATE_OP_API_INCLUDE_DIR}")
set(MATCH_TEMPLATE_EXAMPLE_LIBRARIES
    "${MATCH_TEMPLATE_CUST_OPAPI_LIBRARY}"
    "${MATCH_TEMPLATE_ASCENDCL_LIBRARY}"
    "${MATCH_TEMPLATE_NNOPBASE_LIBRARY}")

add_executable(match_template_aclnn_test test_aclnn_match_template.cpp)
target_compile_features(match_template_aclnn_test PRIVATE cxx_std_17)
target_include_directories(match_template_aclnn_test PRIVATE ${MATCH_TEMPLATE_EXAMPLE_INCLUDE_DIRS})
target_link_libraries(match_template_aclnn_test PRIVATE ${MATCH_TEMPLATE_EXAMPLE_LIBRARIES})

add_executable(match_template_aclnn_benchmark benchmark_aclnn_match_template.cpp)
target_compile_features(match_template_aclnn_benchmark PRIVATE cxx_std_17)
target_include_directories(match_template_aclnn_benchmark PRIVATE ${MATCH_TEMPLATE_EXAMPLE_INCLUDE_DIRS})
target_link_libraries(match_template_aclnn_benchmark PRIVATE ${MATCH_TEMPLATE_EXAMPLE_LIBRARIES})

add_executable(match_template_aclnn_case_runner run_aclnn_match_template_case.cpp)
target_compile_features(match_template_aclnn_case_runner PRIVATE cxx_std_17)
target_include_directories(match_template_aclnn_case_runner PRIVATE ${MATCH_TEMPLATE_EXAMPLE_INCLUDE_DIRS})
target_link_libraries(match_template_aclnn_case_runner PRIVATE ${MATCH_TEMPLATE_EXAMPLE_LIBRARIES})

add_custom_target(match_template_examples
    DEPENDS match_template_aclnn_test match_template_aclnn_benchmark match_template_aclnn_case_runner)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME match_template_aclnn_functional
             COMMAND match_template_aclnn_test)
    set_tests_properties(match_template_aclnn_functional
                         PROPERTIES LABELS "match_template;aclnn;functional")
endif()
