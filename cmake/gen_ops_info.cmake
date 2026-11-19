# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------


# ######################################################################################################################
# copy kernel src to tbe/ascendc path
# ######################################################################################################################
function(kernel_src_copy)
  set(oneValueArgs TARGET DST_DIR)
  set(multiValueArgs IMPL_DIR)
  cmake_parse_arguments(KNCPY "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_custom_target(${KNCPY_TARGET})
  foreach(OP_DIR ${KNCPY_IMPL_DIR})
    get_filename_component(OP_NAME ${OP_DIR} NAME)
    message(STATUS "start copy kernel file: ${OP_NAME} to ${KNCPY_DST_DIR}")
    if(NOT TARGET ${OP_NAME}_src_copy)
      set(SRC_DIR ${OP_DIR}/op_kernel)
      if(NOT EXISTS ${SRC_DIR})
        continue()
      endif()
      add_custom_target(
        ${OP_NAME}_src_copy
        COMMAND
          ${CMAKE_COMMAND} -E make_directory ${KNCPY_DST_DIR}/${OP_NAME}
        COMMAND
          find "${SRC_DIR}" -mindepth 1 -maxdepth 1 -exec cp -r {} "${KNCPY_DST_DIR}/${OP_NAME}" \;
        VERBATIM
        )
      # pypto-pro ops (marked by enable_pypto_kernel): stage the configure-time generated artifacts
      # (*_tiling.h / *_tilingkey.h / *_pypto_infer.cpp) next to the copied op_kernel sources, so they
      # travel with the kernel source both in tbe/ascendc and in the installed package.
      foreach(PYPTO_GEN_SRC ${${OP_NAME}_pypto_gen_dirs})
        if(PYPTO_GEN_SRC AND EXISTS ${PYPTO_GEN_SRC})
          add_custom_command(
            TARGET ${OP_NAME}_src_copy
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${PYPTO_GEN_SRC} ${KNCPY_DST_DIR}/${OP_NAME}
            VERBATIM
            )
          if(ENABLE_PACKAGE)
            install(DIRECTORY ${PYPTO_GEN_SRC}/ DESTINATION ${IMPL_INSTALL_DIR}/${OP_NAME})
          endif()
        endif()
      endforeach()
      add_dependencies(${KNCPY_TARGET} ${OP_NAME}_src_copy)
      if(ENABLE_PACKAGE)
        install(DIRECTORY ${SRC_DIR}/ DESTINATION ${IMPL_INSTALL_DIR}/${OP_NAME})
      endif()
    endif()
  endforeach()

  # common install
  add_custom_target(
    atvoss_src_copy
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${KNCPY_DST_DIR}/common
    COMMAND
      cp -r "${OPBASE_SOURCE_PATH}/pkg_inc/op_common/atvoss" "${KNCPY_DST_DIR}/common"
    COMMAND
      cp -r "${OPBASE_SOURCE_PATH}/include/op_common/op_kernel" "${KNCPY_DST_DIR}/common"
    VERBATIM
  )
  add_dependencies(${KNCPY_TARGET} atvoss_src_copy)
  if(ENABLE_PACKAGE)
    install(DIRECTORY ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/atvoss DESTINATION ${IMPL_INSTALL_DIR}/common)
    install(DIRECTORY ${OPBASE_SOURCE_PATH}/include/op_common/op_kernel DESTINATION ${IMPL_INSTALL_DIR}/common)
  endif()
endfunction()

function(get_op_type_and_validate OP_DIR compute_unit op_name_var op_type_var is_valid_var)
  get_filename_component(op_name "${OP_DIR}" NAME)
  set(${op_name_var} "${op_name}" PARENT_SCOPE)
  set(cache_key "OP_CACHE_${op_name}_${compute_unit}")
  if(DEFINED ${cache_key})
    set(cached_op_type "${${cache_key}}")
    if(cached_op_type)
      set(${op_type_var} "${${cache_key}}" PARENT_SCOPE)
      set(${is_valid_var} TRUE PARENT_SCOPE)
      return()
    else()
      set(${is_valid_var} FALSE PARENT_SCOPE)
      return()
    endif()
  endif()

  set(op_type "")
  set(is_valid FALSE)
  set(${op_type_var} "" PARENT_SCOPE)
  set(${is_valid_var} FALSE PARENT_SCOPE)
  set(binary_json ${OP_DIR}/op_host/config/${compute_unit}/${op_name}_binary.json)

  if(EXISTS ${binary_json})
    get_op_type_from_binary_json("${binary_json}" OP_TYPE)
    set(op_type ${OP_TYPE})
    message(STATUS "[INFO] On [${compute_unit}], [${op_name}] compile binary with self config.")
    if(NOT op_type)
      set(${cache_key} "" CACHE INTERNAL "")
      return()
    endif()
  else()
    get_op_type_from_op_name("${op_name}" op_type)
    if(NOT op_type)
      message(STATUS "[INFO] On [${compute_unit}], [${op_name}] not need to compile.")
      set(${cache_key} "" CACHE INTERNAL "")
      return()
    endif()

    set(check_op_supported_result)
    check_op_supported("${op_name}" "${OP_DIR}" "${compute_unit}" check_op_supported_result)
    if(NOT check_op_supported_result)
      message(STATUS "[INFO] On [${compute_unit}], [${op_name}] not supported.")
      set(${cache_key} "" CACHE INTERNAL "")
      return()
    endif()
    message(STATUS "[INFO] On [${compute_unit}], [${op_name}] compile binary with def config.")
  endif()

  set(is_valid TRUE)
  set(${op_type_var} "${op_type}" PARENT_SCOPE)
  set(${is_valid_var} TRUE PARENT_SCOPE)
  set(${cache_key} "${op_type}" CACHE INTERNAL "Cached op_type for ${op_name} on ${compute_unit}")
endfunction()

# ######################################################################################################################
# generate operator dynamic python script for compile, generenate out path ${CMAKE_BINARY_DIR}/tbe, and install to
# packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
# ######################################################################################################################
function(add_ops_impl_target)
  set(oneValueArgs TARGET OPS_INFO_DIR IMPL_DIR OUT_DIR INSTALL_DIR DEPENDS)
  cmake_parse_arguments(OPIMPL "" "${oneValueArgs}" "OPS_BATCH;OPS_ITERATE" ${ARGN})

  # pypto-pro ops (marked by enable_pypto_kernel, global property PYPTO_ENABLED_OPS keyed by op_file):
  # forward so ascendc_impl_build.py emits the pypto wrapper variant (calls pypto_compile_op).
  # Comma-joined so it stays a single argv token.
  get_property(_pypto_adapt_ops GLOBAL PROPERTY PYPTO_ENABLED_OPS)
  set(PYPTO_ADAPT_OPS_OPTION "")
  set(PYPTO_OPS_ENV "")
  if(_pypto_adapt_ops)
    list(JOIN _pypto_adapt_ops "," _pypto_adapt_ops_str)
    set(PYPTO_ADAPT_OPS_OPTION --pypto-ops ${_pypto_adapt_ops_str})
    # gen_compile_option.sh calls write_scripts() directly rather than through the CLI, so it reads the
    # same list from PYPTO_OPS.
    set(PYPTO_OPS_ENV export PYPTO_OPS=${_pypto_adapt_ops_str} &&)
  endif()

  add_custom_command(
    OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND
      mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_impl_build.py \"\" \"${OPIMPL_OPS_BATCH}\"
      \"${OPIMPL_OPS_ITERATE}\" ${OPIMPL_IMPL_DIR} ${OPIMPL_OUT_DIR}/dynamic ${ASCEND_AUTOGEN_PATH} --opsinfo-dir
      ${OPIMPL_OPS_INFO_DIR} ${OPIMPL_OPS_INFO_DIR}/inner ${OPIMPL_OPS_INFO_DIR}/exc ${PYPTO_ADAPT_OPS_OPTION}
    COMMAND
      rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND
      touch ${OPIMPL_OUT_DIR}/.impl_timestamp
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_impl_build.py
            ${OPIMPL_DEPENDS}
    )
  add_custom_target(${OPIMPL_TARGET} ALL DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)

  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    set(compute_unit_op_cache "${compute_unit}_ALL_COMPUTE_PAIRS")
    if(NOT DEFINED ${compute_unit_op_cache})
      set(all_op_pairs)
      foreach(OP_DIR ${COMPILED_OP_DIRS})
        get_op_type_and_validate("${OP_DIR}" "${compute_unit}" op_name op_type is_valid)
        if(NOT is_valid)
          continue()
        endif()
        list(APPEND all_op_pairs "${op_type}:${compute_unit}")
      endforeach()
      set(${compute_unit_op_cache} ${all_op_pairs} CACHE STRING "compute_unit:${compute_unit}")
    endif()

    set(cur_op_pairs ${${compute_unit_op_cache}})
    add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/${compute_unit}/.gen_timestamp
      COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/${compute_unit}
      COMMAND ${PYPTO_OPS_ENV} bash ${CMAKE_SOURCE_DIR}/scripts/util/gen_compile_option.sh ${cur_op_pairs}
      COMMAND rm -rf ${OPIMPL_OUT_DIR}/${compute_unit}/.gen_timestamp
      COMMAND touch ${OPIMPL_OUT_DIR}/${compute_unit}/.gen_timestamp
      DEPENDS merge_ini_${compute_unit} ${OPIMPL_OUT_DIR}/.impl_timestamp
    )
    add_custom_target(gen_compile_options_${compute_unit} ALL
      DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp ${OPIMPL_OUT_DIR}/${compute_unit}/.gen_timestamp)
    add_dependencies(${OPIMPL_TARGET} gen_compile_options_${compute_unit})
  endforeach()

  file(GLOB dynamic_impl ${OPIMPL_OUT_DIR}/dynamic/*.py)
  if(ENABLE_PACKAGE)
    install(
      FILES ${dynamic_impl}
      DESTINATION ${OPIMPL_INSTALL_DIR}
      OPTIONAL
      )
  endif()
endfunction()

# ######################################################################################################################
# generate aic-${compute_unit}-ops-info.json from aic-${compute_unit}-ops-info.ini generate outpath:
# ${CMAKE_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/ install path:
# packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/config/${compute_unit}
# ######################################################################################################################
function(add_ops_info_target)
  set(oneValueArgs TARGET OPS_INFO_DIR COMPUTE_UNIT OUTPUT INSTALL_DIR)
  cmake_parse_arguments(OPINFO "" "${oneValueArgs}" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(
    OUTPUT ${OPINFO_OUTPUT}
    COMMAND
      mkdir -p ${opinfo_file_path}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/parse_ini_to_json.py
      ${OPINFO_OPS_INFO_DIR}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini
      ${OPINFO_OPS_INFO_DIR}/inner/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini
      ${OPINFO_OPS_INFO_DIR}/exc/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini ${OPINFO_OUTPUT}
    DEPENDS opbuild_custom_gen_aclnn_all
    )
  add_custom_target(${OPINFO_TARGET} ALL DEPENDS ${OPINFO_OUTPUT})

  if(ENABLE_PACKAGE)
    install(
      FILES ${OPINFO_OUTPUT}
      DESTINATION ${OPINFO_INSTALL_DIR}
      OPTIONAL
      )
  endif()
endfunction()

# ######################################################################################################################
# merge ops info ini in aclnn/aclnn_inner/aclnn_exc to a total ini file srcpath: ${ASCEND_AUTOGEN_PATH} generate
# outpath: ${CMAKE_BINARY_DIR}/tbe/config
# ######################################################################################################################
function(merge_ini_files)
  set(oneValueArgs TARGET OPS_INFO_DIR COMPUTE_UNIT)
  cmake_parse_arguments(MGINI "" "${oneValueArgs}" "" ${ARGN})
  add_custom_command(
    OUTPUT ${ASCEND_KERNEL_CONF_DST}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
    COMMAND
      touch ${MGINI_OPS_INFO_DIR}/aic-merged-${MGINI_COMPUTE_UNIT}-ops-info.ini
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${OPS_KERNEL_BINARY_SCRIPT}/merge_ini_files.py
      ${MGINI_OPS_INFO_DIR}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
      ${MGINI_OPS_INFO_DIR}/inner/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
      ${MGINI_OPS_INFO_DIR}/exc/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini --output-file
      ${ASCEND_KERNEL_CONF_DST}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini
    DEPENDS opbuild_custom_gen_aclnn_all
    )
  add_custom_target(${MGINI_TARGET} ALL DEPENDS ${ASCEND_KERNEL_CONF_DST}/aic-${MGINI_COMPUTE_UNIT}-ops-info.ini)
endfunction()

# ######################################################################################################################
# merge ops proto headers in aclnn/aclnn_inner/aclnn_exc to a total proto file srcpath: ${ASCEND_AUTOGEN_PATH} generate
# outpath: ${CMAKE_BINARY_DIR}/tbe/graph
# ######################################################################################################################
function(merge_graph_headers)
  set(oneValueArgs TARGET OUT_DIR)
  cmake_parse_arguments(MGPROTO "" "${oneValueArgs}" "" ${ARGN})
  get_target_property(proto_headers ${OP_GRAPH_NAME}_proto_headers INTERFACE_SOURCES)
  set(proto_headers ${proto_headers} ${CMAKE_SOURCE_DIR}/common/inc/op_graph/op_cv_proto_extend.h)
  add_custom_command(
    OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_cv.h
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/merge_proto.py ${proto_headers} --output-file
      ${MGPROTO_OUT_DIR}/ops_proto_cv.h
  )
  add_custom_command(
    OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_cv.cpp
    COMMAND ${CMAKE_COMMAND} -E copy
      ${MGPROTO_OUT_DIR}/ops_proto_cv.h
      ${MGPROTO_OUT_DIR}/ops_proto_cv.cpp
    DEPENDS ${MGPROTO_OUT_DIR}/ops_proto_cv.h
  )
  add_custom_target(
    ${MGPROTO_TARGET} ALL
    DEPENDS
      ${MGPROTO_OUT_DIR}/ops_proto_cv.h
      ${MGPROTO_OUT_DIR}/ops_proto_cv.cpp
  )
endfunction()

# ######################################################################################################################
# generate binary compile shell script and binary json srcpath: ${ASCEND_AUTOGEN_PATH} outpath:
# ${CMAKE_BINARY_DIR}/binary/${compute_unit}
# ######################################################################################################################
function(generate_bin_scripts)
  set(oneValueArgs TARGET OP_NAME OPS_INFO_DIR COMPUTE_UNIT OUT_DIR)
  cmake_parse_arguments(GENBIN "" "${oneValueArgs}" "" ${ARGN})
  file(MAKE_DIRECTORY ${GENBIN_OUT_DIR}/gen)
  file(MAKE_DIRECTORY ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME})
  message(STATUS "start generate_bin_scripts for op: ${GENBIN_OP_NAME}")
  add_custom_target(
    generate_bin_scripts_${GENBIN_COMPUTE_UNIT}_${GENBIN_OP_NAME}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_bin_param_build.py
      ${GENBIN_OPS_INFO_DIR}/aic-${GENBIN_COMPUTE_UNIT}-ops-info.ini ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME}
      ${GENBIN_COMPUTE_UNIT} --opc-config-file ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS} --ops ${GENBIN_OP_NAME}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_bin_param_build.py
      ${GENBIN_OPS_INFO_DIR}/inner/aic-${GENBIN_COMPUTE_UNIT}-ops-info.ini ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME}
      ${GENBIN_COMPUTE_UNIT} --opc-config-file ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS} --ops ${GENBIN_OP_NAME}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_bin_param_build.py
      ${GENBIN_OPS_INFO_DIR}/exc/aic-${GENBIN_COMPUTE_UNIT}-ops-info.ini ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME}
      ${GENBIN_COMPUTE_UNIT} --opc-config-file ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS} --ops ${GENBIN_OP_NAME}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${OPS_KERNEL_BINARY_SCRIPT}/merge_ops_config_json.py
          ${GENBIN_OUT_DIR}/gen/${GENBIN_OP_NAME}
      DEPENDS ascendc_impl_gen  # this target depend ascendc_impl_gen to gen asc-xxx-ops-info.ini
    )
  if(NOT TARGET ${GENBIN_TARGET})
    add_custom_target(${GENBIN_TARGET})
  endif()
  add_dependencies(${GENBIN_TARGET} generate_bin_scripts_${GENBIN_COMPUTE_UNIT}_${GENBIN_OP_NAME})
endfunction()

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

# ######################################################################################################################
# get op_type from def cpp
# ######################################################################################################################
function(get_op_type_from_op_name OP_NAME OP_TYPE)
  execute_process(
    COMMAND
      find ${CMAKE_CURRENT_SOURCE_DIR} -name ${OP_NAME}_def.cpp -exec grep OP_ADD {} \;
    OUTPUT_VARIABLE op_type
    )
  if(NOT op_type)
    set(op_type "")
  else()
    string(REGEX REPLACE "OP_ADD\\(" "" op_type ${op_type})
    string(REGEX REPLACE "\\).*$" "" op_type ${op_type})
  endif()
  set(${OP_TYPE}
      ${op_type}
      PARENT_SCOPE
    )
endfunction()

# ######################################################################################################################
# copy binary config from op_host/config to tbe/config path
# ######################################################################################################################
function(binary_config_copy)
  set(oneValueArgs TARGET OP_NAME CONF_DIR DST_DIR COMPUTE_UNIT)
  cmake_parse_arguments(CNFCPY "" "${oneValueArgs}" "" ${ARGN})
  file(MAKE_DIRECTORY ${CNFCPY_DST_DIR}/${CNFCPY_COMPUTE_UNIT}/${CNFCPY_OP_NAME})
  add_custom_target(
    ${CNFCPY_TARGET}
    COMMAND
      rm -rf ${CNFCPY_DST_DIR}/${CNFCPY_COMPUTE_UNIT}/${CNFCPY_OP_NAME}/*
    COMMAND
      cp -r ${CNFCPY_CONF_DIR}/${CNFCPY_COMPUTE_UNIT}/* ${CNFCPY_DST_DIR}/${CNFCPY_COMPUTE_UNIT}/${CNFCPY_OP_NAME}
    )
endfunction()

# ######################################################################################################################
# compile binary from op_host/config binary json files generate outpath: ${CMAKE_BINARY_DIR}/binary/${compute_unit}/bin
# install path: ${BIN_KERNEL_INSTALL_DIR}/${compute_unit}
# ######################################################################################################################
function(prepare_compile_from_config)
  set(oneValueArgs TARGET OP_NAME OP_TYPE BINARY_JSON OPS_INFO_DIR IMPL_DIR CONFIG_DIR OP_PYTHON_DIR OUT_DIR INSTALL_DIR
                   COMPUTE_UNIT
    )
  cmake_parse_arguments(CONFCMP "" "${oneValueArgs}" "" ${ARGN})
  file(MAKE_DIRECTORY ${CONFCMP_OUT_DIR}/src)
  file(MAKE_DIRECTORY ${CONFCMP_OUT_DIR}/bin)
  file(MAKE_DIRECTORY ${CONFCMP_OUT_DIR}/gen)
  message(STATUS "start to compile op: ${CONFCMP_OP_NAME}, op_type: ${CONFCMP_OP_TYPE}")
  # add Environment Variable Configurations of python & ccache
  set(_ASCENDC_ENV_VAR)
  list(APPEND _ASCENDC_ENV_VAR export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} &&)
  # pypto-pro ops: pin the codegen arch to the soc being built (pypto_compile_op defaults to a5).
  pypto_jit_arch_env("${CONFCMP_COMPUTE_UNIT}" _pypto_arch_env)
  if(_pypto_arch_env)
    list(APPEND _ASCENDC_ENV_VAR ${_pypto_arch_env})
  endif()
  # whether need judging CMAKE_C_COMPILER_LAUNCHER
  if("${CMAKE_CXX_COMPILER_LAUNCHER}" MATCHES "ccache$")
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CMAKE_CXX_COMPILER_LAUNCHER} &&)
  endif()

  if (EXISTS ${CONFCMP_BINARY_JSON})
    # copy binary config file to tbe/config
    binary_config_copy(
          TARGET bin_conf_${CONFCMP_OP_NAME}_${CONFCMP_COMPUTE_UNIT}_copy OP_NAME ${CONFCMP_OP_NAME} CONF_DIR
          ${CONFCMP_CONFIG_DIR} DST_DIR ${ASCEND_KERNEL_CONF_DST} COMPUTE_UNIT ${CONFCMP_COMPUTE_UNIT}
    )
  else()
    file(MAKE_DIRECTORY ${ASCEND_KERNEL_CONF_DST}/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME})
    add_custom_target(bin_conf_${CONFCMP_OP_NAME}_${CONFCMP_COMPUTE_UNIT}_copy
          COMMAND cp ${CMAKE_BINARY_DIR}/binary/${CONFCMP_COMPUTE_UNIT}/gen/${CONFCMP_OP_NAME}/${CONFCMP_OP_NAME}_binary.json  ${ASCEND_KERNEL_CONF_DST}/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}
          COMMENT "cp ${CMAKE_BINARY_DIR}/binary/${CONFCMP_COMPUTE_UNIT}/gen/${CONFCMP_OP_NAME}/${CONFCMP_OP_NAME}_binary.json  ${ASCEND_KERNEL_CONF_DST}/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}"
    )
  endif()

  if(NOT TARGET gen_opc_info_${CONFCMP_COMPUTE_UNIT})
    add_custom_target(gen_opc_info_${CONFCMP_COMPUTE_UNIT}
            COMMAND ${_ASCENDC_ENV_VAR} bash ${OPS_KERNEL_BINARY_SCRIPT}/build_binary_gen_opc_info.sh
            ${CONFCMP_COMPUTE_UNIT}
            WORKING_DIRECTORY ${OPS_KERNEL_BINARY_SCRIPT}
            DEPENDS ${ASCEND_KERNEL_CONF_DST}/aic-${CONFCMP_COMPUTE_UNIT}-ops-info.ini
    )
    add_dependencies(gen_opc_info_${CONFCMP_COMPUTE_UNIT} merge_ini_${compute_unit})
  endif()
  add_custom_target(
    config_compile_${CONFCMP_COMPUTE_UNIT}_${CONFCMP_OP_NAME}
    COMMAND
      ${_ASCENDC_ENV_VAR} bash ${OPS_KERNEL_BINARY_SCRIPT}/build_binary_opc.sh ${CONFCMP_OP_TYPE} ${CONFCMP_COMPUTE_UNIT}
      ${CONFCMP_OUT_DIR}/bin ${ENABLE_MSSANITIZER} ${CMAKE_BUILD_TYPE} ${ENABLE_OOM} ${ENABLE_DUMP_CCE} bisheng_flags=${BISHENG_FLAGS} kernel_template_input=${KERNEL_TEMPLATE_INPUT}
    WORKING_DIRECTORY ${OPS_KERNEL_BINARY_SCRIPT}
    DEPENDS ${ASCEND_KERNEL_CONF_DST}/aic-${CONFCMP_COMPUTE_UNIT}-ops-info.ini ascendc_kernel_src_copy
            bin_conf_${CONFCMP_OP_NAME}_${CONFCMP_COMPUTE_UNIT}_copy
          gen_opc_info_${CONFCMP_COMPUTE_UNIT}
    )
    add_dependencies(config_compile_${CONFCMP_COMPUTE_UNIT}_${CONFCMP_OP_NAME} ascendc_impl_gen)

  if(NOT TARGET prepare_binary_compile_${CONFCMP_COMPUTE_UNIT})
    add_custom_target(prepare_binary_compile_${CONFCMP_COMPUTE_UNIT})
  endif()
  add_custom_target(
    ${CONFCMP_TARGET}
    COMMAND
      cp -r ${CONFCMP_IMPL_DIR}/*.* ${CONFCMP_OUT_DIR}/src
    COMMAND
      cp ${CONFCMP_OP_PYTHON_DIR}/${CONFCMP_OP_NAME}*.py ${CONFCMP_OUT_DIR}/src
    )
  # pypto-pro ops: the generated artifacts live in the configure-time gen dir, not in the op's
  # op_kernel source dir copied above, so stage them into the binary src dir as well.
  foreach(PYPTO_GEN_SRC ${${CONFCMP_OP_NAME}_pypto_gen_dirs})
    if(PYPTO_GEN_SRC AND EXISTS ${PYPTO_GEN_SRC})
      add_custom_command(
        TARGET ${CONFCMP_TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PYPTO_GEN_SRC} ${CONFCMP_OUT_DIR}/src
        VERBATIM
        )
    endif()
  endforeach()
  add_dependencies(prepare_binary_compile_${CONFCMP_COMPUTE_UNIT} config_compile_${CONFCMP_COMPUTE_UNIT}_${CONFCMP_OP_NAME} ${CONFCMP_TARGET})

  if(ENABLE_PACKAGE)
    if(ENABLE_CUSTOM)
      set(_KERNEL_BIN_INSTALL_DIR ${BIN_KERNEL_INSTALL_DIR}/${CONFCMP_COMPUTE_UNIT})
      set(_KERNEL_CONFIG_INSTALL_DIR ${BIN_KERNEL_CONFIG_INSTALL_DIR}/${CONFCMP_COMPUTE_UNIT})
    else()
      set(_KERNEL_BIN_INSTALL_DIR ${BIN_KERNEL_INSTALL_DIR}/${CONFCMP_COMPUTE_UNIT}/ops_cv)
      set(_KERNEL_CONFIG_INSTALL_DIR ${BIN_KERNEL_CONFIG_INSTALL_DIR}/${CONFCMP_COMPUTE_UNIT}/ops_cv)
    endif()
    install(
      DIRECTORY ${CONFCMP_OUT_DIR}/bin/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}
      DESTINATION ${_KERNEL_BIN_INSTALL_DIR}
      OPTIONAL
      )
    install(
      FILES ${CONFCMP_OUT_DIR}/bin/config/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}.json
      DESTINATION ${_KERNEL_CONFIG_INSTALL_DIR}
      OPTIONAL
      )
    install(
      DIRECTORY ${CONFCMP_OUT_DIR}/bin/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}_apt
      DESTINATION ${_KERNEL_BIN_INSTALL_DIR}
      OPTIONAL
      )
    install(
      FILES ${CONFCMP_OUT_DIR}/bin/config/${CONFCMP_COMPUTE_UNIT}/${CONFCMP_OP_NAME}_apt.json
      DESTINATION ${_KERNEL_CONFIG_INSTALL_DIR}
      OPTIONAL
      )
  endif()
endfunction()

###################################################################################################
# compile binary from op_host/config binary json files
# generate outpath: ${CMAKE_BINARY_DIR}/binary/${compute_unit}/bin
# install path: ${BIN_KERNEL_INSTALL_DIR}/${compute_unit}
###################################################################################################
function(compile_from_config)
  set(oneValueArgs TARGET OUT_DIR INSTALL_DIR COMPUTE_UNIT)
  cmake_parse_arguments(CONFCMP "" "${oneValueArgs}" "" ${ARGN})
  message(STATUS "start to compile unit: ${CONFCMP_COMPUTE_UNIT}")
  set(_ASCENDC_ENV_VAR)
  list(APPEND _ASCENDC_ENV_VAR export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} &&)
  # pypto-pro ops: pin the codegen arch to the soc being built (pypto_compile_op defaults to a5).
  pypto_jit_arch_env("${CONFCMP_COMPUTE_UNIT}" _pypto_arch_env)
  if(_pypto_arch_env)
    list(APPEND _ASCENDC_ENV_VAR ${_pypto_arch_env})
  endif()
  if(CCACHE_PROGRAM)
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CCACHE_PROGRAM} &&)
  endif()

  if(NOT TARGET binary)
    add_custom_target(binary)
  endif()

  add_custom_target(exe_compile_${CONFCMP_COMPUTE_UNIT}_out
          COMMAND ${_ASCENDC_ENV_VAR} bash ${OPS_KERNEL_BINARY_SCRIPT}/build_binary_op_exe_task_out.sh ${CONFCMP_OUT_DIR}/bin
          WORKING_DIRECTORY ${OPS_KERNEL_BINARY_SCRIPT}
  )

  set(OPC_NUM_UNIT "OPC_NUM_${CONFCMP_COMPUTE_UNIT}")
  message(STATUS "[INFO] Binary compile tasks: ${OPC_NUM_UNIT}")
  foreach(idx RANGE 1 ${${OPC_NUM_UNIT}})
    set(_BUILD_COMMAND)
    List(APPEND _BUILD_COMMAND export TILINGKEY_PAR_COMPILE=1 &&)
    List(APPEND _BUILD_COMMAND export BIN_FILENAME_HASHED=1 &&)
    List(APPEND _BUILD_COMMAND export ASCEND_SLOG_PRINT_TO_STDOUT=1 &&)
    List(APPEND _BUILD_COMMAND ${_ASCENDC_ENV_VAR} bash ${OPS_KERNEL_BINARY_SCRIPT}/build_binary_op_exe_task.sh ${CONFCMP_OUT_DIR}/bin ${idx})
    List(APPEND _BUILD_COMMAND && echo $(MAKE))
    add_custom_target(exe_compile_${CONFCMP_COMPUTE_UNIT}_${idx}
            COMMAND ${_BUILD_COMMAND}
            WORKING_DIRECTORY ${OPS_KERNEL_BINARY_SCRIPT}
    )
    add_dependencies(exe_compile_${CONFCMP_COMPUTE_UNIT}_out exe_compile_${CONFCMP_COMPUTE_UNIT}_${idx})
  endforeach()

  add_dependencies(binary exe_compile_${CONFCMP_COMPUTE_UNIT}_out)

endfunction()

# ######################################################################################################################
# generate binary_info_config.json generate outpath: ${CMAKE_BINARY_DIR}/binary/${compute_unit}/bin/config install path:
# packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel/config
# ######################################################################################################################
function(gen_binary_info_config_json)
  set(oneValueArgs TARGET BIN_DIR COMPUTE_UNIT)
  cmake_parse_arguments(GENBIN_INFOCFG "" "${oneValueArgs}" "" ${ARGN})

  add_custom_command(
    OUTPUT ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}/binary_info_config.json
    COMMAND
      ${CMAKE_COMMAND} -E make_directory
      ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE}
      ${OPS_KERNEL_BINARY_SCRIPT}/gen_binary_info_config.py
      ${GENBIN_INFOCFG_BIN_DIR}/bin
      ${GENBIN_INFOCFG_COMPUTE_UNIT}
    )

  add_custom_target(
    ${GENBIN_INFOCFG_TARGET}
    DEPENDS ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}/binary_info_config.json
    )

  if(NOT TARGET gen_bin_info_config)
    add_custom_target(gen_bin_info_config)
  endif()
  add_dependencies(gen_bin_info_config ${GENBIN_INFOCFG_TARGET})

  if(ENABLE_PACKAGE)
    if(ENABLE_CUSTOM)
      set(_KERNEL_CONFIG_INSTALL_DIR ${BIN_KERNEL_CONFIG_INSTALL_DIR}/${GENBIN_INFOCFG_COMPUTE_UNIT})
    else()
      set(_KERNEL_CONFIG_INSTALL_DIR ${BIN_KERNEL_CONFIG_INSTALL_DIR}/${GENBIN_INFOCFG_COMPUTE_UNIT}/ops_cv)
    endif()
    install(
      FILES ${GENBIN_INFOCFG_BIN_DIR}/bin/config/${GENBIN_INFOCFG_COMPUTE_UNIT}/binary_info_config.json
      DESTINATION ${_KERNEL_CONFIG_INSTALL_DIR}
      OPTIONAL
      )
  endif()
endfunction()

# ######################################################################################################################
# get op_type from *_def.cpp
# ######################################################################################################################
function(check_op_supported OP_NAME OP_DIR COMPUTE_UNIT OP_SUPPORTED_COMPUTE_UNIT)
  set(_check_npu_arch_upper "")
  if(COMMAND get_npu_arch_by_soc_version)
    get_npu_arch_by_soc_version("${COMPUTE_UNIT}" _check_npu_arch)
    if(_check_npu_arch)
      string(TOUPPER "${_check_npu_arch}" _check_npu_arch_upper)
    endif()
  endif()
  # 旧风格注册：按行匹配，与基线行为保持一致
  execute_process(
    COMMAND find "${OP_DIR}" -name "${OP_NAME}_def.cpp" -exec grep "\\.AddConfig(\\s*\"${COMPUTE_UNIT}\"" {} \;
    OUTPUT_VARIABLE op_supported_compute_unit
    )
  # AddConfigByArch 注册：grep 按行匹配无法覆盖换行写法，固定行数窗口又会把调用结束后的
  # 无关架构值误算为注册项（或漏判更长跨行调用）。改为提取完整注册调用（自函数名起至语句
  # 结束分号，可跨任意行）后在调用参数内匹配架构枚举值，不含调用结束后的代码；匹配前先
  # 去除行注释与块注释，避免注释中的架构枚举值造成误判
  if(NOT op_supported_compute_unit AND _check_npu_arch_upper)
    file(GLOB_RECURSE _def_files "${OP_DIR}/${OP_NAME}_def.cpp" "${OP_DIR}/*/${OP_NAME}_def.cpp")
    foreach(_def_file IN LISTS _def_files)
      file(READ "${_def_file}" _def_src)
      string(REGEX REPLACE "//[^\n]*" "" _def_src "${_def_src}")
      set(_prev_src "")
      while(NOT "${_def_src}" STREQUAL "${_prev_src}")
        set(_prev_src "${_def_src}")
        string(REGEX REPLACE "/\\*[^*]*\\*/" "" _def_src "${_def_src}")
      endwhile()
      string(REGEX MATCHALL "AddConfigByArch[^\;]*\;" _arch_calls "${_def_src}")
      foreach(_arch_call IN LISTS _arch_calls)
        if(_arch_call MATCHES "${_check_npu_arch_upper}")
          set(op_supported_compute_unit TRUE)
        endif()
      endforeach()
    endforeach()
  endif()
  if(NOT op_supported_compute_unit)
    set(op_supported_compute_unit FALSE)
  else()
    set(op_supported_compute_unit TRUE)
  endif()
  set(${OP_SUPPORTED_COMPUTE_UNIT}
      ${op_supported_compute_unit}
      PARENT_SCOPE
    )
endfunction()

# binary compile
function(gen_ops_info_and_python)
  gen_aclnn_with_opdef()
  if(NOT TARGET opbuild_custom_gen_aclnn_all)
    message(STATUS "no need build binary, for all the ops do not have any operator def")
    return()
  endif()

  kernel_src_copy(
    TARGET ascendc_kernel_src_copy OP_LIST ${COMPILED_OPS} IMPL_DIR ${COMPILED_OP_DIRS} DST_DIR
    ${ASCEND_KERNEL_SRC_DST}
    )

  merge_graph_headers(TARGET merge_ops_proto ALL OUT_DIR ${ASCEND_GRAPH_CONF_DST})

  set(ascendc_impl_gen_depends ascendc_kernel_src_copy opbuild_custom_gen_aclnn_all)
  foreach(compute_unit ${ASCEND_ALL_COMPUTE_UNIT})
    # generate aic-${compute_unit}-ops-info.json, operator infos
    if(ENABLE_CUSTOM)
      set(ops_info_suffix "ops-info.json")
    else()
      set(ops_info_suffix "ops-info-cv.json")
    endif()
    add_ops_info_target(
      TARGET ops_info_gen_${compute_unit} OUTPUT
      ${CMAKE_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-${ops_info_suffix} OPS_INFO_DIR
      ${ASCEND_AUTOGEN_PATH} COMPUTE_UNIT ${compute_unit} INSTALL_DIR ${OPS_INFO_INSTALL_DIR}/${compute_unit}
      )

    # merge ops info ini files
    merge_ini_files(TARGET merge_ini_${compute_unit} OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH} COMPUTE_UNIT ${compute_unit})
    list(APPEND ascendc_impl_gen_depends ops_info_gen_${compute_unit})
  endforeach()

  add_ops_impl_target(
    TARGET ascendc_impl_gen
    OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
    IMPL_DIR ${ASCEND_KERNEL_SRC_DST}
    OUT_DIR ${CMAKE_BINARY_DIR}/tbe
    INSTALL_DIR ${IMPL_DYNAMIC_INSTALL_DIR}
    DEPENDS ${ascendc_impl_gen_depends}
  )
  add_dependencies(ascendc_impl_gen ${ascendc_impl_gen_depends})

  if(ENABLE_BINARY OR ENABLE_CUSTOM)
    foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
      set(HAS_OP_COMPILE_OF_COMPUTE_UNIT FALSE)
      foreach(OP_DIR ${COMPILED_OP_DIRS})
        get_filename_component(op_name ${OP_DIR} NAME)
        set(op_type)
        get_op_type_from_op_name("${op_name}" op_type)
        if(NOT op_type)
          message(STATUS "[INFO] On [${compute_unit}], [${op_name}] not need to compile.")
          continue()
        endif()

        set(check_op_supported_result)
        check_op_supported("${op_name}" "${OP_DIR}" "${compute_unit}" check_op_supported_result)
        if(NOT check_op_supported_result)
          message(STATUS "[INFO] On [${compute_unit}], [${op_name}] not supported.")
          continue()
        endif()

        set(HAS_OP_COMPILE_OF_COMPUTE_UNIT TRUE)

        # generate opc shell scripts for autogen binary config ops
        generate_bin_scripts(
          TARGET gen_bin_scripts OP_NAME ${op_name} OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH} COMPUTE_UNIT ${compute_unit}
          OUT_DIR ${CMAKE_BINARY_DIR}/binary/${compute_unit}
          )
        set(binary_json ${OP_DIR}/op_host/config/${compute_unit}/${op_name}_binary.json)
        if(EXISTS ${binary_json})
          # binary compile from binary json config
          message(STATUS "[INFO] On [${compute_unit}], [${op_name}] compile binary with self config.")
        else()
          message(STATUS "[INFO] On [${compute_unit}], [${op_name}] compile binary with auto gen config.")
        endif()
        prepare_compile_from_config(
              TARGET
              ascendc_bin_${compute_unit}_${op_name}
              OP_NAME
              ${op_name}
              OP_TYPE
              ${op_type}
              BINARY_JSON
              ${binary_json}
              OPS_INFO_DIR
              ${ASCEND_AUTOGEN_PATH}
              IMPL_DIR
              ${OP_DIR}/op_kernel
              CONFIG_DIR
              ${OP_DIR}/op_host/config
              OP_PYTHON_DIR
              ${CMAKE_BINARY_DIR}/tbe/dynamic
              OUT_DIR
              ${CMAKE_BINARY_DIR}/binary/${compute_unit}
              INSTALL_DIR
              ${BIN_KERNEL_INSTALL_DIR}
              COMPUTE_UNIT
              ${compute_unit}
        )
        add_dependencies(ascendc_bin_${compute_unit}_${op_name} merge_ini_${compute_unit} ascendc_impl_gen gen_bin_scripts)
      endforeach()
      # binary compile from binary json config

      message(STATUS "[INFO] On [${compute_unit}], [${op_name}] compile binary from config.")
      compile_from_config(
              TARGET ascendc_bin_${compute_unit}_${op_name}
              OUT_DIR ${CMAKE_BINARY_DIR}/binary/${compute_unit}
              INSTALL_DIR ${BIN_KERNEL_INSTALL_DIR}
              COMPUTE_UNIT ${compute_unit}
      )

      if(HAS_OP_COMPILE_OF_COMPUTE_UNIT)
        # generate binary_info_config.json
        gen_binary_info_config_json(
          TARGET gen_bin_info_config_${compute_unit} BIN_DIR ${CMAKE_BINARY_DIR}/binary/${compute_unit} COMPUTE_UNIT
          ${compute_unit}
          )
      else()
        message(WARNING "[WARNING] There is no operator support for ${compute_unit}.")
      endif()
      set(LOCK_CLEANUP_CMD "rm -f '${CMAKE_BINARY_DIR}/binary/${compute_unit}/bin/.*.cleaned.lock'")
      add_custom_command(TARGET binary
              POST_BUILD
              COMMAND sh -c "${LOCK_CLEANUP_CMD}"
              COMMENT "Executing cleanup: ${LOCK_CLEANUP_CMD}")
    endforeach()
  endif()
endfunction()
