#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

logging() {
  echo "[INFO] $@"
}

mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}"
  logging "Created ${create_dir}"
}

version_ge() {
  local v1="$1"
  local v2="$2"

  local v1_major v1_minor v2_major v2_minor
  v1_major=$(echo "$v1" | cut -d. -f1)
  v1_minor=$(echo "$v1" | cut -d. -f2)
  v2_major=$(echo "$v2" | cut -d. -f1)
  v2_minor=$(echo "$v2" | cut -d. -f2)

  if [[ $v1_major -gt $v2_major ]]; then
    return 0
  elif [[ $v1_major -eq $v2_major && $v1_minor -ge $v2_minor ]]; then
    return 0
  else
    return 1
  fi
}

# 获取 lcov 版本号
get_lcov_version() {
  lcov --version 2>/dev/null | grep -oP 'version \K[0-9]+\.[0-9]+' | head -1
}

# 根据 lcov 版本构建 --ignore-errors 参数
build_ignore_errors_opt() {
  local version
  version=$(get_lcov_version)

  if [[ -n "$version" ]] && version_ge "$version" "2.0"; then
    echo "--ignore-errors mismatch,gcov,source,empty,negative"
  else
    echo "--ignore-errors gcov"
  fi
}

build_ignore_errors_remove_opt() {
  local version
  version=$(get_lcov_version)

  if [[ -n "$version" ]] && version_ge "$version" "2.0"; then
    echo "--ignore-errors empty,mismatch,unused,negative,source,inconsistent"
  else
    echo "--ignore-errors inconsistent"
  fi
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  if [[ -z "${_source_dir}" ]]; then
    logging "directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    logging "directory does not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    logging "using default file name to generate coverage"
  fi

  which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mk_dir "${_path_to_gen}"
  fi

  # 根据 lcov 版本选择合适的忽略错误选项
  local _ignore_opt=$(build_ignore_errors_opt)

  # 生成基础覆盖率文件
  lcov --capture \
    --directory "${_source_dir}" \
    --output-file "${_coverage_file}" \
    --rc geninfo_unexecuted_blocks=1 \
    ${_ignore_opt}

  logging "generated coverage file ${_coverage_file}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"
  local _ut_type="$3"    # op_kernel/op_host/op_api/op_kernel_aicpu/all
  local _op_names="$4"   # optional semicolon/comma separated

  if [[ ! -f "${_coverage_file}" ]]; then
    logging "coverage data file required"
    exit 1
  fi

  which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  # 根据 lcov 版本选择合适的忽略错误选项
  local _ignore_opt=$(build_ignore_errors_remove_opt)

  # 构建移除列表
  echo "ASCEND_PARENT_PATH: ${ASCEND_PARENT_PATH}"
  local remove_patterns=("${ASCEND_PARENT_PATH}/*" '/usr/include/*' )
  # 保留第三方 headers 还是视情况而定；对测试和公共代码我们默认去掉
  remove_patterns+=( '*/third_party/*' '*/tests/*' )

  # 针对具体 ut type 增加额外移除规则（例如运行 op_kernel 测试时剔除 op_host 库）
  case "${_ut_type}" in
    op_kernel)
      remove_patterns+=( '*/op_host/*' '*/op_api/*' )
      ;;
    op_host)
      remove_patterns+=( '*/op_kernel/*' )
      ;;
    op_api)
      remove_patterns+=( '*/op_host/*' '*/op_kernel/*' )
      ;;
    op_kernel_aicpu)
      remove_patterns+=( '*/op_host/*' '*/op_api/*' )
      ;;
    *)
      ;; # all 或未指定，不额外过滤
  esac

  # 执行过滤
  lcov --remove "${_coverage_file}" "${remove_patterns[@]}" -o "${_filtered_file}" ${_ignore_opt}

  # op_name filtering: if specified, keep only coverage records whose source file
  # path contains one of the given operator names.  Use lcov --extract to filter
  # by operator directories.
  if [[ -n "${_op_names}" ]]; then
    IFS=';, ' read -r -a ops <<< "${_op_names}"
    extract_patterns=""
    for op in "${ops[@]}"; do
      if [[ -z "${extract_patterns}" ]]; then
        extract_patterns="*/${op}/*"
      else
        extract_patterns="${extract_patterns} */${op}/*"
      fi
    done
    logging "Extracting coverage for operators: ${_op_names}"
    # Use lcov --extract to keep only files matching the operator patterns
    lcov --extract "${_filtered_file}" ${extract_patterns} -o "${_filtered_file}.tmp" ${_ignore_opt}
    mv "${_filtered_file}.tmp" "${_filtered_file}"
  fi
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  if [[ ! -d "${_out_path}" ]]; then
    mk_dir "${_out_path}"
  fi

  # 根据 lcov 版本选择合适的忽略错误选项
  local _lcov_version=$(get_lcov_version)
  local _ignore_opt=""
  if [[ -n "${_lcov_version}" ]] && version_ge "${_lcov_version}" "2.0"; then
    _ignore_opt="--ignore-errors empty"
  fi

  if [[ ! -s "${_filtered_file}" ]]; then
    logging "empty coverage data"
  else
    genhtml "${_filtered_file}" -o "${_out_path}" ${_ignore_opt}
  fi
}

if [[ $# -lt 3 ]]; then
  logging "Usage: $0 DIR COV_FILE OUT_PATH [UT_TYPE] [OP_NAMES]"
  logging "  UT_TYPE one of op_kernel,op_host,op_api,op_kernel_aicpu,all"
  logging "  OP_NAMES optional semicolon/comma-separated operator list (e.g. grid_sample)"
  exit 1
fi

_src="$1"
_cov_file="$2"
_out="$3"
_ut_type="${4:-all}"
_op_names="${5:-}"

if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  logging "ASCEND_HOME_PATH is not set"
  exit 1
fi

ASCEND_PARENT_PATH=$(dirname "${ASCEND_HOME_PATH}")
if [[ -z "${ASCEND_PARENT_PATH}" ]]; then
  logging "ASCEND_HOME_PATH is not set"
  exit 1
fi

if [ -d "${_src}" ] && [ -n "$(find "${_src}" -name "*.gcda" -print -quit 2>/dev/null)" ]; then
  generate_coverage "${_src}" "${_cov_file}"
  filter_coverage "${_cov_file}" "${_cov_file}_filtered" "${_ut_type}" "${_op_names}"
  generate_html "${_cov_file}_filtered" "${_out}"
else
  logging "No coverage data found"
fi
