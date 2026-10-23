#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Example build and run functions.

build_example() {
  echo $dotted_line
  echo "Start to run examples,name:${EXAMPLE_NAME} mode:${EXAMPLE_MODE}"

  mkdir -p "${BUILD_PATH}"
  cd "${BUILD_PATH}" || exit 1

  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    build_example_eager
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    build_example_graph
  else
    usage
    exit 1
  fi
}

#######################################
# eager mode
#######################################
build_example_eager() {
  local file
  local sim_lib_path
  sim_lib_path=$(get_simulator_args)
  local ret=$?
  if [ $ret -ne 0 ]; then
    exit 1
  fi

  if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
    file=$(find ../experimental -path "*/${EXAMPLE_NAME}/examples/*" -name test_aclnn_*.cpp)
  else
    file=$(find ../ -path "*/${EXAMPLE_NAME}/examples/test_aclnn_*.cpp" -not -path "*/experimental/*")
    if [[ "$COMPUTE_UNIT" == "ascend950" || "$COMPUTE_UNIT" == "ascend350" || "$COMPUTE_UNIT" == "mc62" ]]; then
      local arch35_file
      arch35_file=$(find ../ -path "*/${EXAMPLE_NAME}/examples/arch35/test_aclnn_*.cpp" -not -path "*/experimental/*")
      [[ -n "$arch35_file" ]] && file="$arch35_file"
    fi
  fi

  if [ -z "$file" ]; then
    echo "ERROR: ${EXAMPLE_NAME} do not have eager examples"
    exit 1
  fi

  for f in $file; do
    if [[ "${PKG_MODE}" == "cust" && "$f" == *add_example_aicpu* && "$f" == *opgen* ]]; then
      continue
    fi

    echo "Start compile and run examples file: $f"

    if [[ "${PKG_MODE}" == "" ]]; then
      if [[ -n "$sim_lib_path" ]]; then
        export LD_LIBRARY_PATH=${sim_lib_path}:${LD_LIBRARY_PATH}
        ln -sf ${sim_lib_path}/libruntime_camodel.so ${sim_lib_path}/libruntime.so
        ln -sf ${sim_lib_path}/libnpu_drv_camodel.so ${sim_lib_path}/libascend_hal.so
        g++ ${f} \
          -I ${INCLUDE_PATH} \
          -I ${ACLNN_INCLUDE_PATH} \
          -L ${EAGER_LIBRARY_PATH} \
          -lopapi_cv -lascendcl -lnnopbase \
          -L ${sim_lib_path} \
          -lruntime_camodel -lnpu_drv_camodel \
          -o test_aclnn_${EXAMPLE_NAME} \
          -Wl,-rpath=${sim_lib_path}
      else
        g++ ${f} \
          -I ${INCLUDE_PATH} \
          -I ${ACLNN_INCLUDE_PATH} \
          -L ${EAGER_LIBRARY_PATH} \
          -lopapi_cv -lascendcl -lnnopbase \
          -o test_aclnn_${EXAMPLE_NAME}
      fi

    elif [[ "${PKG_MODE}" == "cust" ]]; then
      echo "pkg_mode:${PKG_MODE} vendor_name:${VENDOR}"

      local cust_include_flags=""
      local cust_library_flags=""
      local cust_rpath_flags=""
      local cust_aclnnop_paths=""

      if [[ -n "${ASCEND_CUSTOM_OPP_PATH}" ]]; then
        IFS=':' read -ra PATH_ARRAY <<< "${ASCEND_CUSTOM_OPP_PATH}"
        for path in "${PATH_ARRAY[@]}"; do
          cust_include_flags="${cust_include_flags} -I ${path}/op_api/include"
          cust_library_flags="${cust_library_flags} -L ${path}/op_api/lib"
          cust_rpath_flags="${cust_rpath_flags}:${path}/op_api/lib"
          cust_aclnnop_paths="${cust_aclnnop_paths} ${path}/op_api/include/aclnnop"
        done
        cust_rpath_flags="${cust_rpath_flags#:}"
      else
        cust_include_flags="-I ${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/include"
        cust_library_flags="-L ${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/lib"
        cust_rpath_flags="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/lib"
        cust_aclnnop_paths="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/include/aclnnop"
      fi

      for aclnnop_path in ${cust_aclnnop_paths}; do
        local include_dir=$(dirname ${aclnnop_path})
        local include_dir_mode=$(stat -c %a ${include_dir} 2>/dev/null)
        if [ ! -L ${aclnnop_path} ]; then
          chmod u+w ${include_dir} 2>/dev/null
          ln -s ${include_dir} ${aclnnop_path} 2>/dev/null
        fi
      done

      if [[ -n "$sim_lib_path" ]]; then
        export LD_LIBRARY_PATH=${sim_lib_path}:${LD_LIBRARY_PATH}
        ln -sf ${sim_lib_path}/libruntime_camodel.so ${sim_lib_path}/libruntime.so
        ln -sf ${sim_lib_path}/libnpu_drv_camodel.so ${sim_lib_path}/libascend_hal.so
        g++ ${f} \
          ${cust_include_flags} \
          -I ${INCLUDE_PATH} \
          -I ${INCLUDE_PATH}/aclnnop \
          ${cust_library_flags} \
          -L ${EAGER_LIBRARY_PATH} \
          -lcust_opapi -lascendcl -lnnopbase \
          -L ${sim_lib_path} \
          -lruntime_camodel -lnpu_drv_camodel \
          -o test_aclnn_${EXAMPLE_NAME} \
          -Wl,-rpath=${cust_rpath_flags}:${sim_lib_path}
      else
        g++ ${f} \
          ${cust_include_flags} \
          -I ${INCLUDE_PATH} \
          -I ${INCLUDE_PATH}/aclnnop \
          ${cust_library_flags} \
          -L ${EAGER_LIBRARY_PATH} \
          -lcust_opapi -lascendcl -lnnopbase \
          -o test_aclnn_${EXAMPLE_NAME} \
          -Wl,-rpath=${cust_rpath_flags}
      fi

      for aclnnop_path in ${cust_aclnnop_paths}; do
        if [ -L ${aclnnop_path} ]; then
          local include_dir=$(dirname ${aclnnop_path})
          local include_dir_mode=$(stat -c %a ${include_dir} 2>/dev/null)
          rm ${aclnnop_path} 2>/dev/null
          chmod ${include_dir_mode} ${include_dir} 2>/dev/null
        fi
      done

    else
      echo "Error: pkg_mode(${PKG_MODE}) must be cust."
      exit 1
    fi

    if [[ -n "$sim_lib_path" ]]; then
      ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0} \
        ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3} \
        ./test_aclnn_${EXAMPLE_NAME}
    else
      ./test_aclnn_${EXAMPLE_NAME}
    fi

    if [ $? -eq 0 ]; then
      echo "run test_aclnn_${EXAMPLE_NAME}, execute samples success"
    else
      echo "run test_aclnn_${EXAMPLE_NAME}, execute samples failed"
      exit 1
    fi
  done
}

get_simulator_chip_version() {
  local soc=$1
  # soc->npuArch 映射统一维护在 cmake/soc_arch_variables.cmake，经 query.cmake 查询，替代原先的 case 硬编码
  local arch
  arch=$(cmake -P ${BASE_PATH}/cmake/query.cmake Func get_npu_arch_by_soc_version ${soc} 2>&1)
  if [[ -z "$arch" ]]; then
    echo "[ERROR] Unsupported soc version for simulator: $soc" >&2
    return 1
  fi
  echo "$arch"
}

get_simulator_args() {
  if [[ "$ENABLE_SIMULATOR" == "FALSE" ]]; then
    return 0
  fi
  if [[ "$ENABLE_SIMULATOR" == "TRUE" ]] && [[ -n "$COMPUTE_UNIT" ]]; then
    local chip_version
    chip_version=$(get_simulator_chip_version "$COMPUTE_UNIT")
    if [[ $? -ne 0 ]]; then
      exit 1
    fi
    if [[ -n "$chip_version" ]]; then
      local sim_lib_path="${ASCEND_HOME_PATH}/tools/simulator/${chip_version}/lib"
      if [[ ! -d "$sim_lib_path" ]]; then
        echo "[ERROR] Simulator lib path not found: $sim_lib_path" >&2
        exit 1
      else
        echo "[INFO] Successfully linked simulator libraries: ${sim_lib_path}/libruntime_camodel.so, ${sim_lib_path}/libnpu_drv_camodel.so" >&2
      fi
      echo "$sim_lib_path"
      return 0
    fi
  fi
  echo ""
  return 1
}

#######################################
# graph mode
#######################################
build_example_graph() {
  local file

  if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
    file=$(find ../experimental -path "*/${EXAMPLE_NAME}/examples/*" -name test_geir_*.cpp)
  else
    file=$(find ../ -path "*/${EXAMPLE_NAME}/examples/test_geir_*.cpp" -not -path "*/experimental/*")
    if [[ "$COMPUTE_UNIT" == "ascend950" || "$COMPUTE_UNIT" == "ascend350" || "$COMPUTE_UNIT" == "mc62" ]]; then
      local arch35_file
      arch35_file=$(find ../ -path "*/${EXAMPLE_NAME}/examples/arch35/test_geir_*.cpp" -not -path "*/experimental/*")
      [[ -n "$arch35_file" ]] && file="$arch35_file"
    fi
  fi

  if [ -z "$file" ]; then
    echo "ERROR: ${EXAMPLE_NAME} do not have graph examples"
    exit 1
  fi

  for f in $file; do
    echo "Start compile and run examples file: $f"

    g++ ${f} \
      -I ${GRAPH_INCLUDE_PATH} \
      -I ${GE_INCLUDE_PATH} \
      -I ${INCLUDE_PATH} \
      -I ${INC_INCLUDE_PATH} \
      -I ${BASE_PATH}/common/inc \
      -I ${CP_GRAPH_INCLUDE_PATH} \
      -I ${CP_GE_INCLUDE_PATH} \
      -I ${CP_GE_EXTERNAL_INCLUDE_PATH} \
      -L ${GRAPH_LIBRARY_STUB_PATH} \
      -L ${GRAPH_LIBRARY_PATH} \
      -L ${CP_GRAPH_LIBRARY_STUB_PATH} \
      -L ${CP_GRAPH_LIBRARY_PATH} \
      -L ${CP_EXECUTOR_LIBRARY_PATH} \
      -lgraph -lge_runner -lgraph_base -lge_compiler \
      -o test_geir_${EXAMPLE_NAME}

    ./test_geir_${EXAMPLE_NAME}

    if [ $? -eq 0 ]; then
      echo "run test_geir_${EXAMPLE_NAME}, execute samples success"
    else
      echo "run test_geir_${EXAMPLE_NAME}, execute samples failed"
      exit 1
    fi
  done
}
