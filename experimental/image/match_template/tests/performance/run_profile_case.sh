#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

benchmark_bin="${MATCH_TEMPLATE_BENCHMARK_BIN:?MATCH_TEMPLATE_BENCHMARK_BIN is required}"
case_name="${MATCH_TEMPLATE_PROFILE_CASE:?MATCH_TEMPLATE_PROFILE_CASE is required}"
exec "${benchmark_bin}" "${case_name}"
