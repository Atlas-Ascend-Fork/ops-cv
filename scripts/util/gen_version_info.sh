#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

if [ "$#" -lt 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
  echo "[ERROR] Usage: $0 <ascend_install_dir> <gen_file_dir>" >&2
  exit 1
fi

ascend_install_dir="$1"
gen_file_dir="$2"
compiler_version_file="${ascend_install_dir}/compiler/version.info"

if [ ! -f "${compiler_version_file}" ]; then
  echo "[ERROR] compiler version.info not found: ${compiler_version_file}" >&2
  exit 1
fi

mkdir -p "${gen_file_dir}"

# create version.info
compiler_version=$(grep -w "Version" "${compiler_version_file}" | head -n 1 | awk -F = '{print $2}')
if [ -z "${compiler_version}" ]; then
  echo "[ERROR] Version field not found in ${compiler_version_file}" >&2
  exit 1
fi
echo "custom_opp_compiler_version=${compiler_version}" > "${gen_file_dir}/version.info"
