#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# lbp_feature aclnn 真机验证：uint8 + float32 两轮，均与 UT 同源 golden 对拍

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _P=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _P=$ASCEND_HOME_PATH
else
    _P="/usr/local/Ascend/cann"
fi
source ${_P}/bin/setenv.bash

VENDOR_DIR=$(find ${_P}/opp/vendors -maxdepth 2 -type d -name custom_cv | head -1)
echo "VENDOR_DIR=${VENDOR_DIR}"
export LD_LIBRARY_PATH=${VENDOR_DIR}/op_api/lib:${LD_LIBRARY_PATH}

UT_DATA_DIR=/ddhome/operators_zny/ops-cv/build/tests/ut/op_kernel/lbp_feature_data

# ── 编译 ──
rm -rf build && mkdir build && cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE || exit 1
make || exit 1
cd bin

run_one() {
    local dtype=$1   # uint8 | float32
    echo "================ dtype=${dtype} ================"

    # 清理本算子旧 output bin，避免 compare 文件配对错乱
    rm -f ${UT_DATA_DIR}/*_output_lbp_feature.bin \
          ${UT_DATA_DIR}/*_output_lbp_feature.bin.ut.bak

    # 关键修复：cd 到数据目录执行 gen_data.py，确保 bin 生成在 UT_DATA_DIR
    ( cd ${UT_DATA_DIR} && python3 ./gen_data.py \
        --shape 8,130 --dtype ${dtype} --radius 1 --neighbors 8 --seed 42 ) || return 1

    # 按时间取最新生成的输入 bin（防止万一有多个）
    local INPUT_BIN
    INPUT_BIN=$(ls -t ${UT_DATA_DIR}/input_lbp_feature*.bin 2>/dev/null | head -1)
    if [ -z "${INPUT_BIN}" ]; then
        echo "[ERROR] ${UT_DATA_DIR} 下未发现输入 bin"
        return 1
    fi
    echo "使用输入文件: ${INPUT_BIN} ($(stat -c%s ${INPUT_BIN}) 字节)"
    cp "${INPUT_BIN}" ./input_lbp_feature.bin || return 1

    # 真机执行（fp32 传第三参）
    local extra=""
    [ "${dtype}" = "float32" ] && extra="float32"
    ./test_aclnn_lbp_feature ./input_lbp_feature.bin ./output_npu.bin ${extra} || return 1

    # 输出/golden 恒为 int32（与输入 dtype 无关），统一对拍
    cp output_npu.bin ${UT_DATA_DIR}/int32_output_lbp_feature.bin || return 1
    ( cd ${UT_DATA_DIR} && python3 compare_data.py "int32" )
}

run_one uint8
RET8=$?
run_one float32
RETF=$?

echo "=================================================="
[ ${RET8} -eq 0 ] && echo "[PASS] uint8   真机对拍逐位一致" || echo "[FAIL] uint8   真机对拍失败"
[ ${RETF} -eq 0 ] && echo "[PASS] float32 真机对拍逐位一致" || echo "[FAIL] float32 真机对拍失败"
echo "=================================================="
exit $(( RET8 + RETF ))
