/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lbp_feature_ut_helper.h
 * \brief 供 kernel UT 调用真实 tiling 的辅助接口（独立 TU，避免 host 宏与 kernel 头冲突）。
 */

#ifndef LBP_FEATURE_UT_HELPER_H
#define LBP_FEATURE_UT_HELPER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace LbpUt {

struct TilingResult {
    bool ok = false;
    uint64_t tilingKey = 0;
    uint32_t blockNum = 0;
    size_t workspaceSize = 0;
    std::vector<uint8_t> tilingData;
};

// dtypeId: 0 = uint8, 1 = float32
TilingResult RunTiling(const std::vector<int64_t>& shape, int dtypeId, int64_t radius, int64_t neighbors);

} // namespace LbpUt

#endif // LBP_FEATURE_UT_HELPER_H
