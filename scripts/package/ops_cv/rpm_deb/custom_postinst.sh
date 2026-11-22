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
unset PYTHONPATH
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip() { python3 -m pip "$@" || pip3 "$@"; }

# install es_cv whl
whl_dir="${sourcedir}/ops_cv/es_packages/whl"
if [ -d "${whl_dir}" ]; then
    for whl in "${whl_dir}"/*.whl; do
        if [ -f "${whl}" ]; then
            echo "[ops-cv] installing ${whl}"
            chmod u+w "${WHL_INSTALL_DIR_PATH}" 2>/dev/null
            run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall -t "${WHL_INSTALL_DIR_PATH}" "${whl}" \
                && rm -f "${whl}" || true
        fi
    done
fi

# clean up ops_cv source directory after whl install
if [ -d "${sourcedir}/ops_cv" ]; then
    rm -rf "${sourcedir}/ops_cv"
fi

# touch __init__.py for ops_cv impl
built_in_impl_path="${sourcedir}/opp/built-in/op_impl/ai_core/tbe/impl/ops_cv"
if [ -d "${built_in_impl_path}" ]; then
    if [ "$(id -u)" != "0" ] && [ ! -w "${built_in_impl_path}" ]; then
        chmod u+w -R "${built_in_impl_path}" 2>/dev/null
    fi
    touch "${built_in_impl_path}/__init__.py"
    [ -d "${built_in_impl_path}/dynamic" ] && touch "${built_in_impl_path}/dynamic/__init__.py"
fi

# cross-arch so copy: copy graph/host so to actual arch dir when arch mismatches
pkg_arch_name="${PKG_ARCH_NAME}"
actual_arch=$(uname -m)
if [ -n "${pkg_arch_name}" ] && [ "${actual_arch}" != "${pkg_arch_name}" ]; then
    graph_so_src="${sourcedir}/opp/built-in/op_graph/lib/linux/${pkg_arch_name}/libopgraph_cv.so"
    graph_so_dst_dir="${sourcedir}/opp/built-in/op_graph/lib/linux/${actual_arch}"
    host_so_src="${sourcedir}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${pkg_arch_name}/libophost_cv.so"
    host_so_dst_dir="${sourcedir}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${actual_arch}"

    for so_src in "${graph_so_src}" "${host_so_src}"; do
        if [ -f "${so_src}" ]; then
            dst_dir=$(dirname "${so_src}" | sed "s|/${pkg_arch_name}$|/${actual_arch}|")
            if [ ! -d "${dst_dir}" ]; then
                mkdir -p "${dst_dir}"
            fi
            cp -f "${so_src}" "${dst_dir}/"
            chmod 755 "${dst_dir}/$(basename "${so_src}")"
        fi
    done
fi
