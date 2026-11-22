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

if [ "$#" -lt 3 ]; then
    echo "[ERROR] Usage: $0 <top_dir> <output_json> <first_json> [more_json ...]" >&2
    exit 1
fi

echo "$@"
top_dir="$1"
output_json="$2"
first_json="$3"
shift 3

merge_json_tool="${top_dir}/scripts/util/insert_op_info.py"

if [[ ! -f "${first_json}" ]]; then
    echo "[ERROR] ${first_json} is not a file" >&2
    exit 1
fi

if [[ ! "${first_json}" -ef "${output_json}" ]]; then
    if ! cp -f "${first_json}" "${output_json}"; then
        echo "[ERROR] failed to copy ${first_json} to ${output_json}" >&2
        rm -f "${output_json}"
        exit 1
    fi
fi

for single_json in "$@"; do
    if [[ -f "${single_json}" ]]; then
        if ! python3 "${merge_json_tool}" "${single_json}" "${output_json}"; then
            echo "[ERROR] insert_op_info.py failed for ${single_json}" >&2
            rm -f "${output_json}"
            exit 1
        fi
    else
        echo "[ERROR] ${single_json} is not a file" >&2
        rm -f "${output_json}"
        exit 1
    fi
done
