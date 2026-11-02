#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
FF_TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FF_CANN=${ASCEND_HOME_PATH:?Load CANN set_env.sh first}
FF_VENDOR=${1:?Usage: bash run.sh /path/to/vendors/custom_cv [device_id]}
FF_DEVICE=${2:-0}
FF_TEST_BUILD=$(mktemp -d "${TMPDIR:-/tmp}/flood-fill-st.XXXXXX")
trap 'rm -rf -- "$FF_TEST_BUILD"' EXIT

g++ -O2 -std=c++17 -Wall -Wextra -Werror "$FF_TEST_DIR/test_aclnn_flood_fill_regression.cpp" \
    -I "$FF_CANN/include" -I "$FF_CANN/include/aclnnop" -I "$FF_VENDOR/op_api/include" \
    -L "$FF_VENDOR/op_api/lib" -L "$FF_CANN/lib64" -lcust_opapi -lascendcl -lnnopbase \
    -Wl,-rpath="$FF_VENDOR/op_api/lib:$FF_CANN/lib64" -o "$FF_TEST_BUILD/flood_fill_regression"

export ASCEND_CUSTOM_OPP_PATH="$FF_VENDOR"
export LD_LIBRARY_PATH="$FF_VENDOR/op_api/lib:$FF_CANN/lib64:${LD_LIBRARY_PATH:-}"
"$FF_TEST_BUILD/flood_fill_regression" "$FF_DEVICE"
