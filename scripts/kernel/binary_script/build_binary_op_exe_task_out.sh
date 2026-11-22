#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

main() {
  echo "[INFO]execute file: $0"
  if [ $# -lt 1 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {out_path}"
    exit 1
  fi
  local output_path="$1"
  local task_path=$output_path/opc_cmd

  source build_env.sh
  out_cmd_file="${task_path}/${OUT_TASK_NAME}"

  if [ ! -f "${out_cmd_file}" ]; then
    echo "[ERROR]output task file not found: ${out_cmd_file}"
    exit 1
  fi
  # step2: gen output one by one
  while IFS= read -r cmd_task || [ -n "${cmd_task}" ]; do
    [ -z "${cmd_task}" ] && continue
    echo "[INFO]exe_task: begin to gen kernel list with cmd: ${cmd_task}."
    ${cmd_task}
  done < "${out_cmd_file}"
}
set -o pipefail
main "$@"
