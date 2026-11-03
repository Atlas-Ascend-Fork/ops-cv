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
 * \file test_scharr_tiling_helper.cpp
 * \brief 独立 TU：调用真实 TilingFunc 生成 kernel UT 所需的 tiling 数据。
 *        注意：本文件不包含 kernel 源码（kernel_type.h 在 host 侧会把 DT_* 定义为宏，
 *        与 ge::DT_* 冲突），因此 tiling 组装单独放在此 TU。
 */

#include "scharr_ut_helper.h"

#include <string>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

static constexpr uint32_t UT_CORE_NUM = 40U;
static constexpr uint64_t UT_UB_SIZE = 196608ULL;
static constexpr uint32_t UT_TILING_CAPACITY = 4096U;

namespace ScharrUt {

static ge::DataType DTypeFromId(int dtypeId)
{
    switch (dtypeId) {
        case 0:
            return ge::DT_UINT8;
        case 1:
            return ge::DT_FLOAT16;
        default:
            return ge::DT_FLOAT;
    }
}

TilingResult RunTiling(const std::vector<int64_t>& shape, int dtypeId, int dirIdx, int borderIdx)
{
    TilingResult result;
    static const char* DIRS[2] = {"X", "Y"};
    static const char* BORDERS[3] = {"REFLECT_101", "REPLICATE", "CONSTANT"};
    if (dirIdx < 0 || dirIdx > 1 || borderIdx < 0 || borderIdx > 2) {
        return result;
    }

    struct ScharrCompileInfo {
    } compileInfo;
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(shape.size());
    ss.MutableStorageShape().SetDimNum(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        ss.MutableOriginShape().SetDim(i, shape[i]);
        ss.MutableStorageShape().SetDim(i, shape[i]);
    }
    gert::TilingContextPara::TensorDescription xDesc(ss, DTypeFromId(dtypeId), ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription yDesc(ss, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para(
        "Scharr", {xDesc}, {yDesc},
        {gert::TilingContextPara::OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<std::string>(BORDERS[borderIdx])),
         gert::TilingContextPara::OpAttr("direction", Ops::Cv::AnyValue::CreateFrom<std::string>(DIRS[dirIdx]))},
        &compileInfo, "", UT_CORE_NUM, UT_UB_SIZE, UT_TILING_CAPACITY);

    TilingInfo tilingInfo;
    if (!ExecuteTiling(para, tilingInfo)) {
        return result;
    }
    result.ok = true;
    result.tilingKey = static_cast<uint64_t>(tilingInfo.tilingKey);
    result.blockNum = static_cast<uint32_t>(tilingInfo.blockNum);
    result.workspaceSize = tilingInfo.workspaceSizes.empty() ? 0U : static_cast<size_t>(tilingInfo.workspaceSizes[0]);
    result.tilingData.assign(tilingInfo.tilingData.get(), tilingInfo.tilingData.get() + tilingInfo.tilingDataSize);
    return result;
}

} // namespace ScharrUt
