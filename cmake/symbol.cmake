# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

function(gen_common_symbol)
  install(
    DIRECTORY ${OPS_CV_COMMON_INC_HEADERS}
    DESTINATION ${COMMON_INC_INSTALL_DIR}
    )
  if(ENABLE_STATIC)
    install(
      DIRECTORY ${OPS_CV_COMMON_INC_HEADERS}
      DESTINATION ${STATIC_COMMON_INC_INSTALL_DIR}
    )
  endif()
endfunction()

# ophost shared
function(gen_ophost_symbol)
  add_library(
    ${OPHOST_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
    $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
    )

  target_link_libraries(
    ${OPHOST_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            tiling_api
            -Wl,-Bsymbolic
            unified_dlog
            mmpa
    )

  target_link_directories(${OPHOST_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  if(NOT ENABLE_CUSTOM)
    install(
      TARGETS ${OPHOST_NAME}
      LIBRARY DESTINATION ${OPHOST_LIB_INSTALL_PATH}
      )
  endif()
endfunction()

# gen es_cv
function(gen_es_cv_lib_ready)
  # 合并proto.h生成ops_proto_cv.h和ops_proto_cv.cpp
  merge_graph_headers(TARGET merge_ops_proto_${PKG_NAME} OUT_DIR ${ASCEND_GRAPH_CONF_DST})
  add_library(
    proto_${PKG_NAME} SHARED
    ${ASCEND_GRAPH_CONF_DST}/ops_proto_cv.cpp
  )
  add_dependencies(proto_${PKG_NAME} merge_ops_proto_${PKG_NAME})
  target_link_libraries(
        proto_${PKG_NAME}
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                -Wl,--no-as-needed
                register
                -Wl,--as-needed
        )
  target_link_directories(proto_${PKG_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  # 生成 es_cv
  add_es_library_and_whl(
    ES_LINKABLE_AND_ALL_TARGET es_${PKG_NAME}
    OPP_PROTO_TARGET proto_${PKG_NAME}
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_packages
  )
  install(
    FILES ${CMAKE_BINARY_DIR}/es_packages/lib64/libes_cv.so
    DESTINATION ${VERSION_INFO_INSTALL_DIR}/lib64
    OPTIONAL
    )
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/es_packages/include/es_cv
    DESTINATION ${VERSION_INFO_INSTALL_DIR}/include/es
    OPTIONAL
    )
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/es_packages/whl/
    DESTINATION ${WHL_INSTALL_DIR}/es_packages/whl
    OPTIONAL
    )
endfunction()

# gen es_cv for custom
function(gen_es_cv_lib_ready_cust)
  # 合并proto.h生成ops_proto_cv.h和ops_proto_cv.cpp
  merge_graph_headers(TARGET merge_ops_proto_${PKG_NAME}_cust OUT_DIR ${ASCEND_GRAPH_CONF_DST})
  add_library(
    proto_${PKG_NAME}_cust SHARED
    ${ASCEND_GRAPH_CONF_DST}/ops_proto_cv.cpp
  )
  add_dependencies(proto_${PKG_NAME}_cust merge_ops_proto_${PKG_NAME}_cust)
  target_link_libraries(
        proto_${PKG_NAME}_cust
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                -Wl,--no-as-needed
                register
                -Wl,--as-needed
        )
  target_link_directories(proto_${PKG_NAME}_cust PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  # 生成 es_cv
  add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_${PKG_NAME}
    OPP_PROTO_TARGET proto_${PKG_NAME}_cust
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_packages
  )
  install(
    DIRECTORY ${CMAKE_BINARY_DIR}/es_packages/include/es_${PKG_NAME}/
    DESTINATION ${ES_INC_INSTALL_DIR}
    OPTIONAL
  )
  install(
    FILES ${CMAKE_BINARY_DIR}/es_packages/lib64/libes_${PKG_NAME}.so
    DESTINATION ${ES_LIB_INSTALL_DIR}
    OPTIONAL
  )
endfunction()


# graph_plugin shared
function(gen_opgraph_symbol)
  gen_es_cv_lib_ready()

  if(TARGET ${OP_GRAPH_NAME}_obj)
    unset(GRAPH_SOURCE)
    get_target_property(GRAPH_SOURCE ${OP_GRAPH_NAME}_obj SOURCES)
    if(GRAPH_SOURCE)
      add_dependencies(${OP_GRAPH_NAME}_obj
        build_es_math
        build_es_cv
      )
      target_link_libraries(${OP_GRAPH_NAME}_obj
        PRIVATE
        es_math
        es_cv
      )
      add_library(
        ${OPGRAPH_NAME} SHARED
        $<$<TARGET_EXISTS:${OP_GRAPH_NAME}_obj>:$<TARGET_OBJECTS:${OP_GRAPH_NAME}_obj>>
        $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
        $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      )

      target_link_libraries(
        ${OPGRAPH_NAME}
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                -Wl,--no-as-needed
                register
                -Wl,--as-needed
                -Wl,--whole-archive
                rt2_registry_static
                -Wl,--no-whole-archive
                -Wl,-Bsymbolic
                ge_compiler
                ge_common
                es_math
                es_cv
                unified_dlog
                ascendalog
        )

      target_link_directories(${OPGRAPH_NAME} PRIVATE
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
        ${CMAKE_BINARY_DIR}/es_packages/lib64
      )
      set_target_properties(${OPGRAPH_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/opp/built-in/op_proto
      )
      install(
        TARGETS ${OPGRAPH_NAME}
        LIBRARY DESTINATION ${OPGRAPH_LIB_INSTALL_DIR}
        )
    endif()
  endif()
  install(
    FILES ${ASCEND_GRAPH_CONF_DST}/ops_proto_cv.h
    DESTINATION ${OPGRAPH_INC_INSTALL_DIR}
    OPTIONAL
    )

endfunction()

function(gen_opapi_symbol)
  # opapi shared
  add_library(
    ${OPAPI_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
    $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )

  target_link_libraries(
    ${OPAPI_NAME}
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    PRIVATE c_sec nnopbase -Wl,-Bsymbolic
    $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:opapi_math>>
    )

  install(
    TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
    )
endfunction()

function(gen_cust_opapi_symbol)
  # op_api
  npu_op_library(cust_opapi ACLNN)
  target_sources(
    cust_opapi
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
           $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )
  target_link_libraries(
    cust_opapi
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:opapi_math>>
    )
endfunction()

function(gen_cust_optiling_symbol)
  # op_tiling
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    return()
  endif()
  npu_op_library(cust_opmaster TILING)
  target_sources(
    cust_opmaster
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
           $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
           $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
           $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
    )
  target_link_libraries(
    cust_opmaster
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    )
endfunction()

function(gen_cust_proto_symbol)
  # op_proto
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    return()
  endif()
  npu_op_library(cust_proto GRAPH)

  set(NEED_LINK_ES OFF)
  if(TARGET ${OP_GRAPH_NAME}_obj)
    unset(GRAPH_SOURCE)
    get_target_property(GRAPH_SOURCE ${OP_GRAPH_NAME}_obj SOURCES)
    if(GRAPH_SOURCE)
      # 添加obj依赖es
      gen_es_cv_lib_ready_cust()
      add_dependencies(${OP_GRAPH_NAME}_obj
        build_es_math
        build_es_cv
      )
      target_link_libraries(${OP_GRAPH_NAME}_obj
        PRIVATE
        es_math
        es_cv
      )
      set(NEED_LINK_ES ON)
    endif()
  endif()

  set(NEED_MERGE_PROTO OFF)
  if(TARGET ${OP_GRAPH_NAME}_proto_headers)
    get_target_property(_proto_headers ${OP_GRAPH_NAME}_proto_headers INTERFACE_SOURCES)
    if(_proto_headers)
      set(NEED_MERGE_PROTO ON)
    endif()
  endif()

  if(NEED_MERGE_PROTO AND NOT NEED_LINK_ES)
    merge_graph_headers(TARGET merge_ops_proto_${PKG_NAME}_cust OUT_DIR ${ASCEND_GRAPH_CONF_DST})
  endif()

  if(NEED_MERGE_PROTO)
    target_sources(cust_proto PRIVATE ${ASCEND_GRAPH_CONF_DST}/ops_proto_cv.cpp)
    add_dependencies(cust_proto merge_ops_proto_${PKG_NAME}_cust)
  endif()

  target_sources(
    cust_proto
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
           $<$<TARGET_EXISTS:${OP_GRAPH_NAME}_obj>:$<TARGET_OBJECTS:${OP_GRAPH_NAME}_obj>>
           $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
           $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
    )
  target_link_libraries(
    cust_proto
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    ge_compiler
    ge_common
    )

  if(NEED_LINK_ES)
    add_dependencies(cust_proto build_es_math build_es_cv)

    target_link_directories(cust_proto
      PRIVATE
        ${CMAKE_BINARY_DIR}/es_packages/lib64
        ${ES_LIB_INSTALL_DIR}
    )
    target_link_libraries(cust_proto
      PRIVATE
        -Wl,--no-as-needed
        es_math
        es_cv
        -Wl,--as-needed
    )
  endif()

  file(GLOB_RECURSE proto_headers ${ASCEND_AUTOGEN_PATH}/*_proto.h)
  install(
    FILES ${proto_headers}
    DESTINATION ${OPPROTO_INC_INSTALL_DIR}
    OPTIONAL
    )
endfunction()

function(gen_aicpu_op_def_symbol)
  message(STATUS "enter gen_aicpu_op_def_symbol")
  get_property(ALL_AICPU_OP_DEF_FILES GLOBAL PROPERTY AICPU_OP_DEF_FILES)
  if(NOT ALL_AICPU_OP_DEF_FILES)
    message(STATUS "No aicpu op def files, skipping.")
    return()
  endif()
  message(STATUS "ALL_AICPU_OP_DEF_FILES:${ALL_AICPU_OP_DEF_FILES}")
 	# 调用opbuild命令生成固定文件 aicpu_kernel.ini
 	opbuild_aicpu_ini(OPS_SRC ${ALL_AICPU_OP_DEF_FILES}
 	                  OUT_DIR ${ASCEND_AUTOGEN_PATH}
 	)
endfunction()

function(gen_aicpu_json_symbol enable_built_in)
  get_property(ALL_AICPU_JSON_FILES GLOBAL PROPERTY AICPU_JSON_FILES)
  get_property(ALL_AICPU_OP_DEF_FILES GLOBAL PROPERTY AICPU_OP_DEF_FILES)
  if(NOT ALL_AICPU_OP_DEF_FILES AND NOT ALL_AICPU_JSON_FILES)
    message(STATUS "No aicpu opdef/json files to merge, skipping.")
    return()
  endif()

  set(MERGED_JSON ${CMAKE_BINARY_DIR}/cust_aicpu_kernel.json)
  if(enable_built_in)
    set(MERGED_JSON ${CMAKE_BINARY_DIR}/aicpu_cv.json)
  endif()

  set(INPUT_JSONS "")
  if(ALL_AICPU_OP_DEF_FILES)
    gen_aicpu_op_def_symbol()
    set(OPDEF_JSON ${CMAKE_BINARY_DIR}/aicpu_opdef_kernel.json)
    if(enable_built_in)
      set(OPDEF_JSON ${CMAKE_BINARY_DIR}/aicpu_cv_opdef_kernel.json)
    endif()

    set(cmd python3 "${CMAKE_SOURCE_DIR}/scripts/kernel/binary_script/parser_ini.py" "${ASCEND_AUTOGEN_PATH}/aicpu_kernel.ini" "${OPDEF_JSON}")
    message("cmd: ${cmd}")

    execute_process(
        COMMAND ${cmd}
        WORKING_DIRECTORY ${ASCEND_AUTOGEN_PATH}
        OUTPUT_VARIABLE result
        ERROR_VARIABLE error
        RESULT_VARIABLE code
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
 		message("convert ini to json finish.")
    if(NOT code EQUAL 0)
      message(FATAL_ERROR "parser_ini.py failed with code ${code}: ${error}")
    endif()
    list(APPEND INPUT_JSONS ${OPDEF_JSON})
  endif()

  if(ALL_AICPU_JSON_FILES)
    list(APPEND INPUT_JSONS ${ALL_AICPU_JSON_FILES})
  endif()

  execute_process(
      COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/util/merge_aicpu_info_json.sh" "${CMAKE_SOURCE_DIR}" "${MERGED_JSON}" ${INPUT_JSONS}
      RESULT_VARIABLE merge_code
  )
  message("merge aicpu json finish, code: ${merge_code}")
  if(NOT merge_code EQUAL 0)
    message(FATAL_ERROR "merge_aicpu_info_json.sh failed with code ${merge_code}; aicpu json ${MERGED_JSON} is incomplete and will not be installed")
  endif()

  install(
      FILES ${MERGED_JSON}
      DESTINATION ${AICPU_JSON_CONFIG}
      OPTIONAL
  )
endfunction()

function(gen_aicpu_kernel_symbol enable_built_in)
  if(NOT AICPU_CUST_OBJ_TARGETS)
    message(STATUS "No aicpu cust obj targets found, skipping.")
    return()
  endif()

  set(ARM_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  set(ARM_SO_OUTPUT ${CMAKE_BINARY_DIR}/libcv_aicpu_kernels.so)

  set(ALL_OBJECTS "")
  foreach(tgt IN LISTS AICPU_CUST_OBJ_TARGETS)
    list(APPEND ALL_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  endforeach()

  message(STATUS "Linking aicpu_kernels with ARM toolchain: ${ARM_CXX_COMPILER}")
  message(STATUS "Objects: ${ALL_OBJECTS}")
  message(STATUS "Output: ${ARM_SO_OUTPUT}")

  add_custom_command(
    OUTPUT ${ARM_SO_OUTPUT}
    COMMAND ${ARM_CXX_COMPILER} -shared ${ALL_OBJECTS}
      -Wl,--whole-archive
      ${ASCEND_DIR}/lib64/libaicpu_context.a
      ${ASCEND_DIR}/lib64/libbase_ascend_protobuf.a
      -Wl,--no-whole-archive
      -Wl,-Bsymbolic
      -Wl,--exclude-libs=libbase_ascend_protobuf.a
      -Wl,-z,now
      -s
      -o ${ARM_SO_OUTPUT}
    DEPENDS ${AICPU_CUST_OBJ_TARGETS}
    COMMENT "Linking aicpu_kernels.so using ARM toolchain"
    COMMAND_EXPAND_LISTS
  )

  add_custom_target(aicpu_kernels ALL DEPENDS ${ARM_SO_OUTPUT})
  install(
    FILES ${ARM_SO_OUTPUT}
    DESTINATION ${AICPU_KERNEL_IMPL}
    OPTIONAL
  )
endfunction()

# Collects AICPU_HOST_OBJ_TARGETS (registered by add_aicpu_host_kernel_modules in func.cmake)
# and links all host OBJECT files into one SHARED library libopconstant_folding_cv.so.
# Reference: same pattern as gen_aicpu_kernel_symbol (device side).
function(gen_aicpu_const_symbol)
  if(NOT AICPU_HOST_OBJ_TARGETS)
    message(STATUS "No builtin host aicpu targets found, skipping.")
    return()
  endif()

  set(CONST_SO_OUTPUT ${CMAKE_BINARY_DIR}/libopconstant_folding_cv.so)

  set(ALL_OBJECTS "")
  foreach(tgt IN LISTS AICPU_HOST_OBJ_TARGETS)
    list(APPEND ALL_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  endforeach()

  message(STATUS "Linking libopconstant_folding_cv.so with host compiler")
  message(STATUS "AICPU host all objects: ${ALL_OBJECTS}")
  message(STATUS "AICPU host so output: ${CONST_SO_OUTPUT}")

  set(AICPU_HOST_CONST_LIBS
    -lc_sec
    -lgraph
    -lexe_graph
    -lregister
    -lpthread
    -ldl
  )

  add_custom_command(
    OUTPUT ${CONST_SO_OUTPUT}
    COMMAND ${CMAKE_CXX_COMPILER} -shared ${ALL_OBJECTS}
      -Wl,--whole-archive
          ${ASCEND_DIR}/lib64/libaicpu_context_host.a
          ${ASCEND_DIR}/lib64/libaicpu_nodedef_host.a
          ${ASCEND_DIR}/lib64/libhost_ascend_protobuf.a
      -Wl,--no-whole-archive
      -Wl,-Bsymbolic
      -Wl,--exclude-libs=libhost_ascend_protobuf.a
      -Wl,-z,now
      -s

      -L${ASCEND_DIR}/lib64
      ${AICPU_HOST_CONST_LIBS}

      -o ${CONST_SO_OUTPUT}
    DEPENDS ${AICPU_HOST_OBJ_TARGETS}
    COMMENT "Linking libopconstant_folding_cv.so using host compiler okay."
    COMMAND_EXPAND_LISTS
  )

  add_custom_target(opconstant_folding_cv_builtin ALL DEPENDS ${CONST_SO_OUTPUT})
  install(
    FILES ${CONST_SO_OUTPUT}
    DESTINATION ${AICPU_HOST_KERNEL_IMPL}
    OPTIONAL
  )
endfunction()

function(gen_onnx_plugin_symbol)
  if (NOT TARGET ${ONNX_PLUGIN_NAME}_obj)
    return()
  endif()

  add_library(
    ${ONNX_PLUGIN_NAME} SHARED
    $<$<TARGET_EXISTS:${ONNX_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${ONNX_PLUGIN_NAME}_obj>>
  )

  target_link_libraries(
    ${ONNX_PLUGIN_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
            c_sec
            -Wl,--no-as-needed
            register
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            unified_dlog
            ascendalog
            $<$<CONFIG:Release>:-s>
    )

  target_link_directories(${ONNX_PLUGIN_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  install(
    TARGETS ${ONNX_PLUGIN_NAME}
    LIBRARY DESTINATION ${ONNX_PLUGIN_LIB_INSTALL_DIR}
    )

endfunction()

function(gen_tf_plugin_symbol)
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG OR ENABLE_TEST)
    return()
  endif()

  if(TARGET ${TF_PLUGIN_NAME}_obj)
    unset(TF_PLUGIN_SOURCE)
    get_target_property(TF_PLUGIN_SOURCE ${TF_PLUGIN_NAME}_obj SOURCES)
    if(TF_PLUGIN_SOURCE)
      list(FILTER TF_PLUGIN_SOURCE INCLUDE REGEX "_tf_plugin\\.(cpp|cc|cxx)$")
    endif()
  endif()

  if(NOT TF_PLUGIN_SOURCE)
    message(STATUS "No tf_plugin sources found for ${TF_PLUGIN_NAME}_obj, skipping tf_plugin library creation")
    return()
  endif()

  add_library(
    ${TF_PLUGIN_NAME} SHARED
    $<TARGET_OBJECTS:${TF_PLUGIN_NAME}_obj>
  )

  target_link_libraries(
    ${TF_PLUGIN_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
            c_sec
            -Wl,--no-as-needed
            register
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            unified_dlog
            ascendalog
            $<$<CONFIG:Release>:-s>
    )

  target_link_directories(${TF_PLUGIN_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  install(
    TARGETS ${TF_PLUGIN_NAME}
    LIBRARY DESTINATION ${TF_PLUGIN_LIB_INSTALL_DIR}
    OPTIONAL
    )
endfunction()

function(gen_norm_symbol)
  gen_ophost_symbol()

  gen_opgraph_symbol()

  gen_opapi_symbol()

  gen_aicpu_const_symbol()

  gen_onnx_plugin_symbol()

  gen_tf_plugin_symbol()

  if(NOT DISABLE_AICPU AND ENABLE_PACKAGE)
    gen_aicpu_json_symbol(TRUE)
    gen_aicpu_kernel_symbol(TRUE)
  endif()
endfunction()

function(gen_cust_symbol)
  gen_ophost_symbol()

  gen_cust_opapi_symbol()

  gen_cust_optiling_symbol()

  gen_cust_proto_symbol()

  gen_aicpu_json_symbol(FALSE)

  gen_aicpu_kernel_symbol(FALSE)
endfunction()
