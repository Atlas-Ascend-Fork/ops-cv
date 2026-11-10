# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------


include_guard(GLOBAL)
add_cann_third_party(gtest)

if(UT_TEST_ALL OR OP_HOST_UT)
  set(OP_TILING_MODULE_NAME
      ${PKG_NAME}_op_tiling_ut
      CACHE STRING "op_tiling ut module name" FORCE
    )
  set(OP_INFERSHAPE_MODULE_NAME
      ${PKG_NAME}_op_infershape_ut
      CACHE STRING "op_infershape ut module name" FORCE
    )
  set(OP_FRAMEWORK_MODULE_NAME
      ${PKG_NAME}_op_framework_ut
      CACHE STRING "op_framework ut module name" FORCE
    )
  function(add_optiling_ut_modules OP_TILING_MODULE_NAME)
    # add optiling ut common object: cv_op_tiling_ut_common_obj
    add_library(${OP_TILING_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_TILING_UT_COMMON_SRC ${UT_COMMON_INC}/tiling_context_faker.cpp
         ${UT_COMMON_INC}/tiling_case_executor.cpp
      )
    target_sources(${OP_TILING_MODULE_NAME}_common_obj PRIVATE ${OP_TILING_UT_COMMON_SRC})
    target_include_directories(
      ${OP_TILING_MODULE_NAME}_common_obj PRIVATE ${JSON_INCLUDE_DIR} ${GTEST_INCLUDE}
                                                  ${ASCEND_DIR}/include/base/context_builder ${ASCEND_DIR}/pkg_inc
      )
    target_link_libraries(
      ${OP_TILING_MODULE_NAME}_common_obj PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<BUILD_INTERFACE:dlog_headers>
      json GTest::gtest c_sec
      )

    # add optiling ut cases object: cv_op_tiling_ut_cases_obj
    if(NOT TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
      add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_include_directories(
      ${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${UT_COMMON_INC} ${GTEST_INCLUDE} ${OP_TILING_INCLUDE}
                                                 ${JSON_INCLUDE_DIR} ${ASCEND_DIR}/include/base/context_builder
                                                 ${PROJECT_SOURCE_DIR}/common/inc
                                                 ${ASCEND_DIR}/pkg_inc
                                                 ${ASCEND_DIR}/pkg_inc/base
      )
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> GTest::gtest json)

    # add op tiling ut cases static lib: libcv_op_tiling_ut_cases.a
    add_library(${OP_TILING_MODULE_NAME}_cases STATIC)
    target_link_libraries(
      ${OP_TILING_MODULE_NAME}_cases PRIVATE ${OP_TILING_MODULE_NAME}_common_obj ${OP_TILING_MODULE_NAME}_cases_obj
      )
  endfunction()

  function(add_infershape_ut_modules OP_INFERSHAPE_MODULE_NAME)
    # add opinfershape ut common object: cv_op_infershape_ut_common_obj
    add_library(${OP_INFERSHAPE_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_INFERSHAPE_UT_COMMON_SRC ${UT_COMMON_INC}/infershape_context_faker.cpp
      ${UT_COMMON_INC}/infershape_case_executor.cpp
      )
    target_sources(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE ${OP_INFERSHAPE_UT_COMMON_SRC})
    target_include_directories(
      ${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
      ${GTEST_INCLUDE} ${ASCEND_DIR}/include/base/context_builder
      ${ASCEND_DIR}/pkg_inc
    )
    target_link_libraries(
      ${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> GTest::gtest c_sec
      )

    # add opinfershape ut cases object: cv_op_infershape_ut_cases_obj
    if(NOT TARGET ${OP_INFERSHAPE_MODULE_NAME}_cases_obj)
      add_library(${OP_INFERSHAPE_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_include_directories(
      ${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE ${UT_COMMON_INC} ${GTEST_INCLUDE} ${OP_PROTO_INCLUDE}
                                                     ${ASCEND_DIR}/include/base/context_builder
                                                     ${ASCEND_DIR}/pkg_inc
                                                     ${ASCEND_DIR}/pkg_inc/base
      )
    target_link_libraries(
      ${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<BUILD_INTERFACE:dlog_headers>
      GTest::gtest
      )

    # add op infershape ut cases static lib: libcv_op_infershape_ut_cases.a
    add_library(${OP_INFERSHAPE_MODULE_NAME}_cases STATIC)
    target_link_libraries(
      ${OP_INFERSHAPE_MODULE_NAME}_cases PRIVATE ${OP_INFERSHAPE_MODULE_NAME}_common_obj
                                                 ${OP_INFERSHAPE_MODULE_NAME}_cases_obj
      )
  endfunction()

  function(add_framework_ut_modules OP_FRAMEWORK_MODULE_NAME)
    if(NOT TARGET ${OP_FRAMEWORK_MODULE_NAME}_cases_obj)
      add_library(${OP_FRAMEWORK_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_include_directories(
      ${OP_FRAMEWORK_MODULE_NAME}_cases_obj PRIVATE ${GTEST_INCLUDE} ${JSON_INCLUDE_DIR} ${ONNX_PLUGIN_COMMON_INCLUDE}
                                                    ${OP_PROTO_INCLUDE} ${ASCEND_DIR}/pkg_inc ${ASCEND_DIR}/pkg_inc/base
                                                    ${CMAKE_BINARY_DIR}/bin ${CMAKE_BINARY_DIR}/proto
      )
    target_compile_options(${OP_FRAMEWORK_MODULE_NAME}_cases_obj PRIVATE -fno-access-control -Dgoogle=ascend_private
                                                                         -Wno-shadow -Wno-unused-parameter)
    target_link_libraries(
      ${OP_FRAMEWORK_MODULE_NAME}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                                                   $<BUILD_INTERFACE:dlog_headers> GTest::gtest json
      )

    add_library(${OP_FRAMEWORK_MODULE_NAME}_cases STATIC)
    target_link_libraries(${OP_FRAMEWORK_MODULE_NAME}_cases PRIVATE ${OP_FRAMEWORK_MODULE_NAME}_cases_obj)
  endfunction()
endif()

if(UT_TEST_ALL OR OP_API_UT)
  set(OP_API_MODULE_NAME
      ${PKG_NAME}_op_api_ut
      CACHE STRING "op_api ut module name" FORCE
    )
  function(add_opapi_ut_modules OP_API_MODULE_NAME)
    # add opapi ut L2 obj
    if(NOT TARGET ${OP_API_MODULE_NAME}_cases_obj)
      add_library(${OP_API_MODULE_NAME}_cases_obj OBJECT)
    endif()
    target_sources(${OP_API_MODULE_NAME}_cases_obj PRIVATE ${UT_PATH}/op_api/stub/opdev/platform.cpp)
    file(GLOB ACLNN_SRC_DIRS
          ${PROJECT_SOURCE_DIR}/image/*/op_host/op_api
          ${PROJECT_SOURCE_DIR}/objdetect/*/op_host/op_api
          ${PROJECT_SOURCE_DIR}/image/*/op_api
          ${PROJECT_SOURCE_DIR}/objdetect/*/op_api
        )
    target_include_directories(
      ${OP_API_MODULE_NAME}_cases_obj
      PRIVATE ${JSON_INCLUDE_DIR} ${HI_PYTHON_INC_TEMP} ${UT_PATH}/op_api/stub ${OP_API_UT_COMMON_INC}
              ${ASCEND_DIR}/include ${ASCEND_DIR}/include/aclnn ${ASCEND_DIR}/include/aclnnop
              ${OPAPI_INCLUDE}
      )
    target_link_libraries(${OP_API_MODULE_NAME}_cases_obj PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<BUILD_INTERFACE:dlog_headers>
      GTest::gtest
      )
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
  set(AICPU_OP_KERNEL_MODULE_NAME ${PKG_NAME}_aicpu_op_kernel_ut CACHE STRING "aicpu_op_kernel ut module name" FORCE)
  message("******************* ${AICPU_OP_KERNEL_MODULE_NAME}" )
  function(add_aicpu_opkernel_ut_modules AICPU_OP_KERNEL_MODULE_NAME)
    ## add opkernel ut common object: cv_aicpu_op_kernel_ut_common_obj
    add_library(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_KERNEL_UT_COMMON_SRC
        ./stub/*.cpp
        ${PROJECT_SOURCE_DIR}/common/src/common/*.cpp
    )

    file(GLOB OP_KERNEL_AICPU_UT_UTILS_SRC
        utils/*.cpp
    )

    target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE ${OP_KERNEL_UT_COMMON_SRC})
    target_include_directories(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        ${GTEST_INCLUDE}
        ${PROJECT_SOURCE_DIR}/common/inc/op_api
        ${ASCEND_DIR}/pkg_inc/base
        ${AICPU_INC_DIRS}
    )
    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        GTest::gtest
        c_sec
    )

    ## add opkernel ut cases object: cv_aicpu_op_kernel_ut_cases_obj
    if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
        add_library(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE gcov -ldl Eigen3::Eigen)
    target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE ${OP_KERNEL_AICPU_UT_UTILS_SRC})

    ## add opkernel ut cases shared lib: libcv_aicpu_op_kernel_ut_cases.so
    add_library(${AICPU_OP_KERNEL_MODULE_NAME}_cases SHARED
        $<TARGET_OBJECTS:${AICPU_OP_KERNEL_MODULE_NAME}_common_obj>
        $<TARGET_OBJECTS:${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj>
    )
    message(STATUS ">>>> ut.cmake Defined targets: ${AICPU_OP_KERNEL_MODULE_NAME}_cases, ${ASCEND_DIR}")

    # 链接静态库时使用 whole-archive，保证 RegistCpuKernel 被拉入
    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        GTest::gtest
        c_sec
        -ldl
        -Wl,--whole-archive
            ${ASCEND_DIR}/lib64/libaicpu_context_host.a
            ${ASCEND_DIR}/lib64/libaicpu_nodedef_host.a
            ${ASCEND_DIR}/lib64/libhost_ascend_protobuf.a
        -Wl,--no-whole-archive
        -Wl,-Bsymbolic
        -Wl,--exclude-libs=libhost_ascend_protobuf.a
        Eigen3::Eigen
        ${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
        ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj
    )
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_UT)
  set(OP_KERNEL_MODULE_NAME
      ${PKG_NAME}_op_kernel_ut
      CACHE STRING "op_kernel ut module name" FORCE
    )

  # ######################################################################################################################
  # get op_type from *_binary.json
  # ######################################################################################################################
  function(get_op_type_from_binary_json BINARY_JSON OP_TYPE)
  execute_process(
    COMMAND
      grep op_type ${BINARY_JSON}
    OUTPUT_VARIABLE op_type
    )
  string(REGEX REPLACE "\"op_type\"" "" op_type ${op_type})
  string(REGEX MATCH "\".+\"" op_type ${op_type})
  string(REGEX REPLACE "\"" "" op_type ${op_type})

  set(OP_TYPE
      ${op_type}
      PARENT_SCOPE
    )
  endfunction()

  function(add_opkernel_ut_modules OP_KERNEL_MODULE_NAME)
    # add opkernel ut common object: cv_op_kernel_ut_common_obj
    add_library(${OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_KERNEL_UT_COMMON_SRC ${UT_COMMON_INC}/tiling_context_faker.cpp
         ${UT_COMMON_INC}/tiling_case_executor.cpp ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel/data_utils.cpp
      )
    target_sources(${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE ${OP_KERNEL_UT_COMMON_SRC})
    target_include_directories(
      ${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE ${JSON_INCLUDE_DIR} ${GTEST_INCLUDE}
                                                  ${ASCEND_DIR}/include/base/context_builder ${ASCEND_DIR}/pkg_inc
      )
    target_link_libraries(
      ${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json GTest::gtest c_sec
      )

    foreach(socVersion ${fastOpTestSocVersions})
      # add op kernel ut cases obj: cv_op_tiling_ut_${socVersion}_cases
      if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
        add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT)
      endif()
      target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE gcov)

      # add op kernel ut cases dynamic lib: libcv_op_tiling_ut_${socVersion}_cases.so
      add_library(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases SHARED
        $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_common_obj>
        $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj>
        )
      target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases
        PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> ${OP_KERNEL_MODULE_NAME}_common_obj
                ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
        )
    endforeach()
  endfunction()
endif()

if(UT_TEST_ALL OR OP_GRAPH_UT)
  function(add_op_graph_ut_modules OP_GRAPH_MODULE_NAME)
    if(NOT TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
      add_library(${OP_GRAPH_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_include_directories(
      ${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
      ${PROJECT_SOURCE_DIR}/tests/ut/common
      ${JSON_INCLUDE_DIR}
      ${GTEST_INCLUDE}
      ${OP_PROTO_INCLUDE}
      ${PROJECT_SOURCE_DIR}/common/inc
      ${ASCEND_DIR}/include
      ${ASCEND_DIR}/include/external
      ${ASCEND_DIR}/include/exe_graph
      ${ASCEND_DIR}/include/base/context_builder
      ${ASCEND_DIR}/include/ge
    )
    target_link_libraries(
      ${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<BUILD_INTERFACE:dlog_headers>
      GTest::gtest
      json
    )
    target_compile_options(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE -fno-access-control)
    target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE LOG_CPP)

    add_library(${OP_GRAPH_MODULE_NAME}_static_lib STATIC)
    target_link_libraries(
      ${OP_GRAPH_MODULE_NAME}_static_lib PRIVATE
      ${OP_GRAPH_MODULE_NAME}_cases_obj
    )
  endfunction()
endif()

if(UT_TEST_ALL
   OR OP_HOST_UT
   OR OP_API_UT
   OR OP_GRAPH_UT
  )
  function(add_modules_ut_sources)
    set(options OPTION_RESERVED)
    set(oneValueArgs UT_NAME MODE DIR)
    set(multiValueArgs TILING_DIR MULIT_RESERVED)
    cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if("${MODULE_UT_NAME}" STREQUAL "${OP_TILING_MODULE_NAME}")
      get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
      get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
      get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
      get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
      list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
      # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
      if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
      endif()

      if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
        add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
      endif()
      set(_ut_tiling_globs ${MODULE_DIR}/test_*_tiling.cpp)
      foreach(_tiling_sub_dir IN LISTS MODULE_TILING_DIR)
        if(NOT "${_tiling_sub_dir}" STREQUAL "")
          list(APPEND _ut_tiling_globs ${MODULE_DIR}/${_tiling_sub_dir}/test_*_tiling*.cpp)
        endif()
      endforeach()
      file(GLOB OPHOST_TILING_CASES_SRC ${_ut_tiling_globs})
      target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPHOST_TILING_CASES_SRC})
    endif()

    if("${MODULE_UT_NAME}" STREQUAL "${OP_INFERSHAPE_MODULE_NAME}")
      get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
      get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
      get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
      get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
      list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
      # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
      if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
      endif()

      if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
        add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
      endif()
      file(GLOB OPHOST_INFERSHAPE_CASES_SRC ${MODULE_DIR}/test_*_infershape.cpp)
      target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPHOST_INFERSHAPE_CASES_SRC})
    endif()

    if("${MODULE_UT_NAME}" STREQUAL "${OP_FRAMEWORK_MODULE_NAME}")
      if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
        add_library(${MODULE_UT_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
      endif()
      # framework UT 位于 common/tests/ut/framework/，目录路径推导不出具体算子名
      # （会得到 OP_NAME=common），因此按测试文件名逐文件过滤：test_<op_name>_onnx_plugin.cpp -> <op_name>
      file(GLOB OPFRAMEWORK_CASES_SRC ${MODULE_DIR}/test_*_onnx_plugin.cpp)
      set(OPFRAMEWORK_FILTERED_SRC "")
      foreach(_src ${OPFRAMEWORK_CASES_SRC})
        get_filename_component(_src_name ${_src} NAME_WE)
        string(REGEX REPLACE "^test_(.*)_onnx_plugin$" "\\1" _op_name "${_src_name}")
        list(FIND ASCEND_OP_NAME "${_op_name}" _idx)
        # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND _idx EQUAL -1)
          continue()
        endif()
        list(APPEND OPFRAMEWORK_FILTERED_SRC ${_src})
      endforeach()
      target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPFRAMEWORK_FILTERED_SRC})
    endif()

    if("${MODULE_UT_NAME}" STREQUAL "${OP_API_MODULE_NAME}")
      get_filename_component(OP_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
      get_filename_component(UT_DIR ${OP_HOST_DIR} DIRECTORY)
      get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
      get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
      get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
      list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
      # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
      if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
      endif()

      if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
        add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
      endif()
      file(GLOB OPAPI_CASES_SRC ${MODULE_DIR}/test_aclnn_*.cpp ${OPS_CV_DIR}/common/stub/*)
      target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPAPI_CASES_SRC})
    endif()

    if("${MODULE_UT_NAME}" STREQUAL "${OP_GRAPH_MODULE_NAME}")
      get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
      get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
      get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
      get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
      list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
      if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
        return()
      endif()

      if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
        add_library(${MODULE_UT_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
      endif()
      file(GLOB OPGRAPH_CASES_SRC ${MODULE_DIR}/test_*_pass.cpp)
      file(GLOB OPGRAPH_PASS_SRC ${MODULE_DIR}/../../../op_graph/fusion_pass/*_pass.cpp)
      target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPGRAPH_CASES_SRC} ${OPGRAPH_PASS_SRC})
    endif()
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_UT)
  set(fastOpTestSocVersions
      ""
      CACHE STRING "fastOp Test SocVersions"
    )
  function(AddOpTestCase opName supportedSocVersion otherCompileOptions tilingSrcFiles)
    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    # find kernel file
    file(GLOB KernelFile "${PROJECT_SOURCE_DIR}/*/${opName}/op_kernel/${opName}.cpp")

    # standardize opType
    set(OP_TYPE "")
    file(GLOB jsonFiles "${PROJECT_SOURCE_DIR}/*/${opName}/op_host/config/*/${opName}_binary.json")
    list(LENGTH jsonFiles numFiles)
    if(numFiles EQUAL 0)
        string(REPLACE "_" ";" opTypeTemp "${opName}")
        foreach(word IN LISTS opTypeTemp)
            string(SUBSTRING "${word}" 0 1 firstLetter)
            string(SUBSTRING "${word}" 1 -1 restOfWord)
            string(TOUPPER "${firstLetter}" firstLetter)
            string(TOLOWER "${restOfWord}" restOfWord)
            set(OP_TYPE "${OP_TYPE}${firstLetter}${restOfWord}")
        endforeach()
    endif()
    if(NOT numFiles EQUAL 0)
        foreach(jsonFile ${jsonFiles})
            get_op_type_from_binary_json(${jsonFile} OP_TYPE)
            message(STATUS "Current file OP_TYPE: ${OP_TYPE}")
        endforeach()
    endif()

    # standardize tiling files
    string(REPLACE "," ";" tilingSrc "${tilingSrcFiles}")

    foreach(oriSocVersion ${supportedSocVersion})
      # check if the case should be compiled according to soc
      if(ASCEND_COMPUTE_UNIT)
        string(TOLOWER "${oriSocVersion}" lowerSocVersion)
        string(TOLOWER "${ASCEND_COMPUTE_UNIT}" lowerComputeUnit)
        # map compute unit to full name (e.g. ascend910b -> ascend910B1)
        find_value_by_key("${SHORT_NAME_LIST}" "${FULL_NAME_LIST}" "${ASCEND_COMPUTE_UNIT}" computeUnitLong)
        string(TOLOWER "${computeUnitLong}" lowerComputeUnitLong)
        if(NOT ("${lowerComputeUnitLong}" STREQUAL "${lowerSocVersion}" OR "${lowerComputeUnit}" STREQUAL "${lowerSocVersion}"))
          continue()
        endif()
      endif()
      # standardize socVersion
      string(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")

      # skip duplicated targets: op with DEPENDENCIES may be processed twice
      # (dependent-ops path via add_dependent_ops, then main traversal)
      if(TARGET ${opName}_${socVersion}_tiling_tmp)
        continue()
      endif()

      # add tiling tmp so: ${opName}_${socVersion}_tiling_tmp.so
      add_library(${opName}_${socVersion}_tiling_tmp SHARED ${tilingSrc} $<TARGET_OBJECTS:${COMMON_NAME}_obj>
                  $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
                  $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>)
      target_include_directories(
        ${opName}_${socVersion}_tiling_tmp
        PRIVATE ${OP_TILING_INCLUDE}
                ${PROJECT_SOURCE_DIR}
        )
      target_compile_definitions(${opName}_${socVersion}_tiling_tmp PRIVATE LOG_CPP)
      target_link_libraries(
        ${opName}_${socVersion}_tiling_tmp PRIVATE
        $<BUILD_INTERFACE:dlog_headers>
        -Wl,--whole-archive
        tiling_api rt2_registry_static
        -Wl,--no-whole-archive
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        register
        opp_registry
        )

      # gen ascendc tiling head files
      set(tilingFile ${CMAKE_CURRENT_BINARY_DIR}/${opName}_tiling_data.h)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling.h")
        set(compileOptions -include "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling.h")
      else()
        set(compileOptions -include ${tilingFile})
      endif()
      set(CUSTOM_TILING_DATA_KEYS "")
      string(REGEX MATCH "-DUT_CUSTOM_TILING_DATA_KEYS=([^ ]+)" matchedPart "${otherCompileOptions}")
      if(CMAKE_MATCH_1)
        set(CUSTOM_TILING_DATA_KEYS ${CMAKE_MATCH_1})
        string(REGEX REPLACE "-DUT_CUSTOM_TILING_DATA_KEYS=[^ ]+" "" modifiedString ${otherCompileOptions})
        set(otherCompileOptions ${modifiedString})
      endif()
      string(REPLACE " " ";" options "${otherCompileOptions}")
      foreach(option IN LISTS options)
        set(compileOptions ${compileOptions} ${option})
      endforeach()
      message("compileOptions: ${compileOptions}")
      set(gen_tiling_head_file ${OPS_CV_DIR}/tests/ut/op_kernel/scripts/gen_tiling_head_file.sh)
      set(gen_tiling_so_path ${CMAKE_CURRENT_BINARY_DIR}/lib${opName}_${socVersion}_tiling_tmp.so)
      set(gen_tiling_head_tag ${opName}_${socVersion}_gen_head)
      set(gen_cmd bash "${gen_tiling_head_file}" "${OP_TYPE}" "${opName}" "${gen_tiling_so_path}")
      if(ENABLE_ASAN)
        execute_process(
          COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
          OUTPUT_VARIABLE tiling_asan_runtime
          OUTPUT_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE tiling_asan_result
        )
        if(NOT tiling_asan_result EQUAL 0 OR NOT EXISTS "${tiling_asan_runtime}")
          message(FATAL_ERROR "Cannot locate ASan runtime for tiling header generation")
        endif()
        set(gen_cmd
            ${CMAKE_COMMAND} -E env
            "LD_PRELOAD=${tiling_asan_runtime}:$ENV{LD_PRELOAD}"
            "ASAN_OPTIONS=$ENV{ASAN_OPTIONS}:detect_leaks=0"
            ${gen_cmd})
      endif()
      if(CUSTOM_TILING_DATA_KEYS)
        list(APPEND gen_cmd "${CUSTOM_TILING_DATA_KEYS}")
      endif()
      message("gen tiling head file to ${tilingFile}, command:")
      message("${gen_cmd}")
      add_custom_command(
        OUTPUT ${tilingFile}
        COMMAND rm -f ${tilingFile}
        COMMAND ${gen_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${opName}_${socVersion}_tiling_tmp
        )
      add_custom_target(${gen_tiling_head_tag} ALL DEPENDS ${tilingFile})

      # add object: ${opName}_${socVersion}_cases_obj
      file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)
      add_library(${opName}_${socVersion}_cases_obj OBJECT ${KernelFile} ${OPKERNEL_CASES_SRC})
      add_dependencies(${opName}_${socVersion}_cases_obj ${gen_tiling_head_tag})
      target_compile_options(
        ${opName}_${socVersion}_cases_obj PRIVATE -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}"
        )
      target_include_directories(
          ${opName}_${socVersion}_cases_obj
        PRIVATE
          ${GTEST_INCLUDE}
          ${ASCEND_DIR}/include/base/context_builder
          ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
          ${PROJECT_SOURCE_DIR}/tests/ut/common
          ${OP_TILING_INCLUDE}
          ${ASCEND_DIR}/pkg_inc/base
          ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/
          ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/
          ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/
          ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/)
      target_link_libraries(
        ${opName}_${socVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> tikicpulib::${socVersion}
                                                  GTest::gtest
        )

      # add object: cv_op_kernel_ut_${oriSocVersion}_cases_obj
      if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj)
        add_library(
          ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj OBJECT
          $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
          )
      else()
        target_sources(${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj PRIVATE $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>)
      endif()
      target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                                                                    $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
        )

      list(FIND fastOpTestSocVersions "${oriSocVersion}" index)
      if(index EQUAL -1)
        set(fastOpTestSocVersions
            ${fastOpTestSocVersions} ${oriSocVersion}
            CACHE STRING "fastOp Test SocVersions" FORCE
          )
      endif()
    endforeach()
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
  function(AddAicpuOpTestCase opName)
    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${UT_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    ## find kernel file
    file(GLOB KernelFile "${PROJECT_SOURCE_DIR}/*/${opName}/op_kernel_aicpu/${opName}_aicpu.cpp")

    ## add object: ${opName}_cases_obj
    file(GLOB OPKERNEL_CASES_SRC ${UT_DIR}/tests/ut/op_kernel_aicpu/test_${opName}*.cpp)

    message(STATUS "aicpu kernel info: ${opName}, ${KernelFile}, ${OPKERNEL_CASES_SRC}")

    add_library(${opName}_cases_obj OBJECT
            ${KernelFile}
            ${OPKERNEL_CASES_SRC}
            )
    target_compile_options(${opName}_cases_obj PRIVATE
            -g
            )
    target_include_directories(${opName}_cases_obj PRIVATE
            ${AICPU_INCLUDE}
            ${OPBASE_INC_DIRS}
            ${AICPU_INC_DIRS}
            ${ASCEND_DIR}/pkg_inc/base
            ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel_aicpu
            )

    target_link_libraries(${opName}_cases_obj PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            $<BUILD_INTERFACE:dlog_headers>
            -ldl
            GTest::gtest
            c_sec
            Eigen3::Eigen
            )

    ## add object: cv_op_kernel_ut_cases_obj
    if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
      add_library(
        ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT
        $<TARGET_OBJECTS:${opName}_cases_obj>
        )
    else()
      target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE $<TARGET_OBJECTS:${opName}_cases_obj>)
    endif()

    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan>
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            -ldl
            $<TARGET_OBJECTS:${opName}_cases_obj>
            GTest::gtest
            c_sec
            Eigen3::Eigen
            )
  endfunction()
endif()
