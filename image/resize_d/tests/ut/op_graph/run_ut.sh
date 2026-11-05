#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# ResizeDFusionPass UT 独立编译运行脚本
# 用途：ops-cv 仓 opgraph UT 构建目标尚未建设，此脚本以 g++ 直接编译运行融合 pass 单元测试
# 注意：必须使用 -D_GLIBCXX_USE_CXX11_ABI=0 与 CANN 库（libplatform/libes_cv 等）的 ABI 保持一致，
#       否则 PlatformInfoManager 的 STL 对象跨库操作会触发 invalid pointer 崩溃。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPS_CV_DIR="${SCRIPT_DIR}/../../../../.."   # ops-cv 仓根
source /home/developer/Ascend/cann-9.2.0/set_env.sh 2>/dev/null || true

ARCH_DIR="$(uname -m)-linux"
CANN_INC="${ASCEND_HOME_PATH}/${ARCH_DIR}/include"
CANN_PKG_INC="${ASCEND_HOME_PATH}/${ARCH_DIR}/pkg_inc"
CANN_LIB="${ASCEND_HOME_PATH}/${ARCH_DIR}/lib64"
ES_CV_INC="${OPS_CV_DIR}/build/es_packages/include/es_cv"
ES_MATH_INC="${OPS_CV_DIR}/build/es_packages_math/include/es_math"
ES_CV_LIB="${OPS_CV_DIR}/build/es_packages/lib64"
ES_MATH_LIB="${OPS_CV_DIR}/build/es_packages_math/lib64"
OPBASE_INC="${OPS_CV_DIR}/third_party/opbase/include/op_common"
OP_GRAPH_INC="${OPS_CV_DIR}/common/inc"

OUT_DIR="${SCRIPT_DIR}/build"
mkdir -p "${OUT_DIR}"

# gtest must be compiled with the same CXX11 ABI as the CANN libraries (ABI=0), otherwise the
# std::string based gtest internals fail to link. Build it from the system googletest sources
# when no ABI=0 static library is available.
GTEST_LIB_DIR="${OUT_DIR}/gtest_abi0"
if [ ! -f "${GTEST_LIB_DIR}/lib/libgtest.a" ]; then
    GTEST_SRC="/usr/src/googletest"
    if [ ! -d "${GTEST_SRC}/googletest" ]; then
        echo "[ERROR] googletest sources not found at ${GTEST_SRC}, install libgtest-dev first."
        exit 1
    fi
    mkdir -p "${GTEST_LIB_DIR}/build" "${GTEST_LIB_DIR}/include" "${GTEST_LIB_DIR}/lib"
    g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -O0 -w \
        -I"${GTEST_SRC}/googletest/include" -I"${GTEST_SRC}/googletest" \
        -c "${GTEST_SRC}/googletest/src/gtest-all.cc" -o "${GTEST_LIB_DIR}/build/gtest-all.o"
    g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -O0 -w \
        -I"${GTEST_SRC}/googletest/include" -I"${GTEST_SRC}/googletest" \
        -c "${GTEST_SRC}/googletest/src/gtest_main.cc" -o "${GTEST_LIB_DIR}/build/gtest_main.o"
    ar rcs "${GTEST_LIB_DIR}/lib/libgtest.a" "${GTEST_LIB_DIR}/build/gtest-all.o"
    ar rcs "${GTEST_LIB_DIR}/lib/libgtest_main.a" "${GTEST_LIB_DIR}/build/gtest_main.o"
    cp -r "${GTEST_SRC}/googletest/include/gtest" "${GTEST_LIB_DIR}/include/"
fi
GTEST_INC_DIR="${GTEST_LIB_DIR}/include"

g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -g -O0 -w -fno-access-control \
    -I/usr/include \
    -I"${GTEST_INC_DIR}" \
    -I"${CANN_INC}" \
    -I"${CANN_PKG_INC}" \
    -I"${CANN_PKG_INC}/base" \
    -I"${CANN_INC}/external" \
    -I"${CANN_INC}/exe_graph" \
    -I"${CANN_INC}/ge" \
    -I"${ES_CV_INC}" \
    -I"${ES_MATH_INC}" \
    -I"${OPS_CV_DIR}/tests/ut/common" \
    -I"${OPBASE_INC}" \
    -I"${OP_GRAPH_INC}" \
    -I"${OPS_CV_DIR}/image/resize_d/op_graph/fusion_pass" \
    "${OPS_CV_DIR}/image/resize_d/op_graph/fusion_pass/resize_d_fusion_pass.cpp" \
    "${OPS_CV_DIR}/image/resize_d/tests/ut/op_graph/test_resize_d_fusion_pass.cpp" \
    -o "${OUT_DIR}/test_resize_d_fusion_pass" \
    -L"${ES_CV_LIB}" -les_cv \
    -L"${ES_MATH_LIB}" -les_math \
    -L"${CANN_LIB}" \
    -L"${GTEST_LIB_DIR}/lib" \
    -Wl,--no-as-needed \
    -lplatform -lge_compiler -lgraph -lgraph_base -lregister -lmetadef -lge_common \
    -lascendalog -lunified_dlog -lc_sec \
    -leager_style_graph_builder_base \
    -lgtest -lgtest_main -lpthread \
    -Wl,--as-needed \
    -Wl,--disable-new-dtags \
    -Wl,-rpath,"${ES_CV_LIB}:${ES_MATH_LIB}:${CANN_LIB}"

echo "==== build done, running UT ===="
cd "${OUT_DIR}"
# LD_LIBRARY_PATH (from set_env.sh) 优先于 DT_RUNPATH，会将 CANN 自带的 libes_cv.so（缺少本仓
# 生成的 EsResizeTrilinear/EsResizeD 等符号）优先加载。这里显式把仓内 es 库目录前置，
# 并以 --disable-new-dtags 生成 DT_RPATH，确保链接的是与本 UT 编译头文件一致的仓内 es 库。
LD_LIBRARY_PATH="${ES_CV_LIB}:${ES_MATH_LIB}:${CANN_LIB}:${LD_LIBRARY_PATH}" ./test_resize_d_fusion_pass
