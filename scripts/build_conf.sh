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
# Global config: variables, paths, utility functions shared across all build scripts.

RELEASE_TARGETS=("ophost" "opapi" "opgraph" "opkernel" "opkernel_aicpu" "onnxplugin" "tfplugin")
SUPPORTED_UT_TARGETS=("ophost_test" "opapi_test" "opgraph_test" "opkernel_test" "opkernel_aicpu_test")
# 支持的芯片列表从 cmake/soc_arch_variables.cmake 统一获取（经 query.cmake 按行输出，bash 直接分词）
SUPPORT_COMPUTE_UNIT_SHORT=($(cmake -P ${BASE_PATH}/cmake/query.cmake Value ASCEND_ALL_COMPUTE_UNIT 2>&1))

# 所有支持的短选项
SUPPORTED_SHORT_OPTS="hj:vO:uf:-:"

# 所有支持的长选项
SUPPORTED_LONG_OPTS=(
  "help" "ops=" "soc=" "vendor_name=" "build-type=" "cov" "noexec" "aicpu" "noaicpu" "opkernel" "opkernel_aicpu" "jit"
  "pkg" "asan" "valgrind" "make_clean" "static" "simulator"
  "ophost" "opapi" "opgraph" "ophost_test" "opapi_test" "opgraph_test" "opkernel_test" "opkernel_aicpu_test"
  "run_example" "genop=" "genop_aicpu=" "cann_3rd_lib_path"  "experimental" "mssanitizer" "oom" "onnxplugin" "tfplugin" "dump_cce"
  "bisheng_flags=" "kernel_template_input=" "rule_launch=" "ccache=" "pkg-type="
)

dotted_line="----------------------------------------------------------------"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
REPOSITORY_NAME="cv"

CORE_NUMS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || grep -c "processor" /proc/cpuinfo 2>/dev/null || echo 8)
ARCH_INFO=$(uname -m)
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"

if [ -z "$ASCEND_INSTALL_PATH" ]; then
  if [ -n "$ASCEND_HOME_PATH" ]; then
    ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
  else
    ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
  fi
fi

# Base paths
export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export LIB_PATH="${ASCEND_HOME_PATH}/lib64"

# Include paths
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export GRAPH_INCLUDE_PATH="${INCLUDE_PATH}/graph"
export CP_GRAPH_INCLUDE_PATH="${INCLUDE_PATH}/graph"
export GE_INCLUDE_PATH="${INCLUDE_PATH}/ge"
export CP_GE_INCLUDE_PATH="${INCLUDE_PATH}/ge"
export CP_GE_EXTERNAL_INCLUDE_PATH="${INCLUDE_PATH}/external"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"

# Library paths
export EAGER_LIBRARY_PATH="${LIB_PATH}"
export GRAPH_LIBRARY_PATH="${LIB_PATH}"
export CP_GRAPH_LIBRARY_PATH="${LIB_PATH}"
export CP_EXECUTOR_LIBRARY_PATH="${LIB_PATH}"

# Stub paths
export GRAPH_LIBRARY_STUB_PATH="${LIB_PATH}/stub"
export CP_GRAPH_LIBRARY_STUB_PATH="${LIB_PATH}/stub"

in_array() {
  local needle="$1"
  shift
  local haystack=("$@")
  for item in "${haystack[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

check_pkg_type() {
  local pkg_type="$1"
  if [[ "$pkg_type" != "run" && "$pkg_type" != "rpm" && "$pkg_type" != "deb" && "$pkg_type" != "all" ]]; then
    echo "[ERROR] --pkg-type only supports run/rpm/deb/all, got: $pkg_type"
    exit 1
  fi
}

normalize_compute_unit() {
  local compute_unit="$1"
  local compute_unit_lower
  compute_unit_lower=$(echo "$compute_unit" | tr '[:upper:]' '[:lower:]')
  if [[ "$compute_unit_lower" =~ ^ascend950[0-9a-z_-]*$ ]]; then
    echo "ascend950"
  else
    echo "$compute_unit_lower"
  fi
}

check_option_validity() {
  local arg="$1"

  if [[ "$arg" =~ ^-[^-] ]]; then
    local opt_chars=${arg:1}

    local needs_arg_opts=$(echo "$SUPPORTED_SHORT_OPTS" | grep -o "[a-zA-Z]:" | tr -d ':')

    local i=0
    while [ $i -lt ${#opt_chars} ]; do
      local char="${opt_chars:$i:1}"

      if [[ ! "$SUPPORTED_SHORT_OPTS" =~ "$char" ]]; then
        echo "[ERROR] Invalid short option: -$char"
        return 1
      fi

      if [[ "$needs_arg_opts" =~ "$char" ]]; then
        while [ $i -lt ${#opt_chars} ] && [[ "${opt_chars:$i:1}" =~ [0-9a-zA-Z] ]]; do
          i=$((i + 1))
        done
      else
        i=$((i + 1))
      fi
    done
    return 0
  fi

  if [[ "$arg" =~ ^-- ]]; then
    local long_opt="${arg:2}"
    local opt_name="${long_opt%%=*}"

    for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
      # with "=" in long options
      if [[ "$supported_opt" =~ =$ ]]; then
        local base_opt="${supported_opt%=}"
        if [[ "$opt_name" == "$base_opt" ]]; then
          return 0
        fi
      else
        # without "=" in long options
        if [[ "$opt_name" == "$supported_opt" ]]; then
          return 0
        fi
      fi
    done

    echo "[ERROR] Invalid long option: --$opt_name"
    return 1
  fi

  return 0
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "\033[31m[ERROR] ${msg}\033[0m"
  echo $dotted_line
  echo
}
