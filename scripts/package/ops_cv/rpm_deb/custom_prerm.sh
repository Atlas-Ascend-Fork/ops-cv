#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

sourcedir="${INSTALL_PATH}"
WHL_INSTALL_DIR_PATH="${sourcedir}/python/site-packages"
export PYTHONPATH="${WHL_INSTALL_DIR_PATH}"
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip() { python3 -m pip "$@" || pip3 "$@"; }
run_pip uninstall -y es_cv >/dev/null 2>&1 || true

rm -rf "${WHL_INSTALL_DIR_PATH}/es_cv" 2>/dev/null
rm -rf "${WHL_INSTALL_DIR_PATH}/es_cv-"*.dist-info 2>/dev/null

# remove __init__.py
built_in_impl_path="${sourcedir}/opp/built-in/op_impl/ai_core/tbe/impl/ops_cv"
if [ -d "${built_in_impl_path}" ]; then
    rm -f "${built_in_impl_path}/__init__.py" 2>/dev/null
    rm -f "${built_in_impl_path}/dynamic/__init__.py" 2>/dev/null
fi

# remove cross-arch so files
pkg_arch_name="${PKG_ARCH_NAME}"
actual_arch=$(uname -m)
if [ -n "${pkg_arch_name}" ] && [ "${actual_arch}" != "${pkg_arch_name}" ]; then
    graph_so_path="${sourcedir}/opp/built-in/op_graph/lib/linux/${actual_arch}/libopgraph_cv.so"
    host_so_path="${sourcedir}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${actual_arch}/libophost_cv.so"
    for so_path in "${graph_so_path}" "${host_so_path}"; do
        so_dir=$(dirname "${so_path}")
        if [ -f "${so_path}" ]; then
            rm -f "${so_path}"
        fi
        if [ -d "${so_dir}" ] && [ -z "$(ls -A "${so_dir}")" ]; then
            rmdir "${so_dir}" 2>/dev/null || true
        fi
    done
fi

# clean up whl and empty directories
rm -f "${sourcedir}"/ops_cv/es_packages/whl/*.whl 2>/dev/null
rmdir "${sourcedir}"/ops_cv/es_packages/whl 2>/dev/null || true
rmdir "${sourcedir}"/ops_cv/es_packages 2>/dev/null || true
rmdir "${sourcedir}"/ops_cv 2>/dev/null || true

rmdir "${WHL_INSTALL_DIR_PATH}" 2>/dev/null || true
parent=$(dirname "${WHL_INSTALL_DIR_PATH}")
[ -d "${parent}" ] && rmdir "${parent}" 2>/dev/null || true
