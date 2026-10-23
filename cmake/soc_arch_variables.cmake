# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# socVersion 与 npuArch 的映射关系（与 scripts/build_example.sh 仿真器目录约定一致）
# ASCEND_ALL_COMPUTE_UNIT 与 SOC_TO_ARCH_VALUES 按位置一一对应；暂无 npuArch 的 soc 以空串占位，
# 查询（get_npu_arch_by_soc_version）对未匹配的 socVersion 返回空串
# 注意：SOC_TO_ARCH_VALUES 含空占位元素，list 命令在 CMP0007 OLD 下会丢弃空元素导致按位错位，
# 调用方须保证 CMP0007 为 NEW（工程模式由 cmake_minimum_required(VERSION 3.16) 保证，
# 脚本模式由 query.cmake 顶层 cmake_policy(SET CMP0007 NEW) 保证）
set(ASCEND_ALL_COMPUTE_UNIT "ascend031" "ascend035" "ascend310b" "ascend310p" "ascend610lite" "ascend630"
                             "ascend910" "ascend350" "ascend910b" "ascend910_93" "ascend950" "ascend960dt"
                             "ascend960pr" "mc62" "kirinx90" "kirin9030" "ascend5162a")
set(SOC_TO_ARCH_VALUES "dav_3004" "dav_3505" "dav_3002" "dav_2002" "dav_3102" "dav_5102" "dav_1001" "dav_3510" "dav_2201"
                       "dav_2201" "dav_3510" "dav_9201" "dav_9202" "dav_5102" "dav_3003" "dav_3113" "dav_5162")
# npuArch 集合（由映射值去重得到，排除空占位）
set(NPU_ARCH_LIST ${SOC_TO_ARCH_VALUES})
list(FILTER NPU_ARCH_LIST EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES NPU_ARCH_LIST)
