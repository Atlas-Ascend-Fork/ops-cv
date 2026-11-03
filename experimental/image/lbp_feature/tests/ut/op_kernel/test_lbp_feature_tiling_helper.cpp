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
 * \file test_lbp_feature_tiling_helper.cpp
 * \brief 独立 TU：调用真实 TilingFunc 生成 kernel UT 所需的 tiling 数据。
 *        注意：本文件不包含 kernel 源码（kernel_type.h 在 host 侧会把 DT_* 定义为宏，
 *        与 ge::DT_* 冲突），因此 tiling 组装单独放在此 TU。
 */

#include "lbp_feature_ut_helper.h"

#include <string>
#include <vector>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

static constexpr uint32_t UT_CORE_NUM = 40U;
static constexpr uint64_t UT_UB_SIZE = 196608ULL;
static constexpr uint32_t UT_TILING_CAPACITY = 4096U;

namespace LbpUt {

static ge::DataType DTypeFromId(int dtypeId) { return (dtypeId == 0) ? ge::DT_UINT8 : ge::DT_FLOAT; }

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(dims.size());
    ss.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        ss.MutableOriginShape().SetDim(i, dims[i]);
        ss.MutableStorageShape().SetDim(i, dims[i]);
    }
    return ss;
}

// 输出 shape：前导维不变，最后两维减去 2*radius
static std::vector<int64_t> OutShape(const std::vector<int64_t>& shape, int64_t radius)
{
    std::vector<int64_t> out = shape;
    if (out.size() >= 2) {
        out[out.size() - 2] -= 2 * radius;
        out[out.size() - 1] -= 2 * radius;
    }
    return out;
}

TilingResult RunTiling(const std::vector<int64_t>& shape, int dtypeId, int64_t radius, int64_t neighbors)
{
    TilingResult result;

    struct LbpFeatureCompileInfo {
    } compileInfo;
    gert::TilingContextPara::TensorDescription xDesc(MakeStorageShape(shape), DTypeFromId(dtypeId), ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription yDesc(MakeStorageShape(OutShape(shape, radius)), ge::DT_INT32,
                                                     ge::FORMAT_ND);
    gert::TilingContextPara para(
        "LbpFeature", {xDesc}, {yDesc},
        {gert::TilingContextPara::OpAttr("radius", Ops::Cv::AnyValue::CreateFrom<int64_t>(radius)),
         gert::TilingContextPara::OpAttr("neighbors", Ops::Cv::AnyValue::CreateFrom<int64_t>(neighbors))},
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

} // namespace LbpUt
