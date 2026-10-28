/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/clahe_tiling_data.h"

namespace {
struct CompileInfo {};
gert::TilingContextPara MakePara(const std::vector<int64_t>& dims, ge::DataType dtype, float clip,
                                 const std::vector<int64_t>& grid, void* info)
{
    gert::Shape shape;
    for (auto dim : dims) {
        shape.AppendDim(dim);
    }
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return gert::TilingContextPara(
        "Clahe", {{storage, dtype, ge::FORMAT_ND}}, {{storage, dtype, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("clip_limit", Ops::Cv::AnyValue::CreateFrom<float>(clip)),
         gert::TilingContextPara::OpAttr("tile_grid_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(grid))},
        info, "Ascend910b", 40, 196608, sizeof(ClaheTilingData));
}
} // namespace

TEST(ClaheTiling, Uint8WorkspaceAndCores)
{
    CompileInfo info;
    auto para = MakePara({64, 64}, ge::DT_UINT8, 40.0f, {8, 8}, &info);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(para, result));
    EXPECT_EQ(result.tilingKey, 0U);
    EXPECT_EQ(result.blockNum, 40U);
    ASSERT_EQ(result.workspaceSizes.size(), 1U);
    EXPECT_EQ(result.workspaceSizes[0], 16U * 1024 * 1024 + 64U * 256 * 2);
}

TEST(ClaheTiling, Uint16BatchedWorkspace)
{
    CompileInfo info;
    auto para = MakePara({3, 1, 65, 64}, ge::DT_UINT16, 2.0f, {4, 8}, &info);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(para, result));
    EXPECT_EQ(result.workspaceSizes[0], 16U * 1024 * 1024 + 3U * 32 * 65536 * 2);
}

TEST(ClaheTiling, WideHistogramWorkspace)
{
    CompileInfo info;
    auto para = MakePara({257, 257}, ge::DT_UINT16, 0.0f, {1, 1}, &info);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(para, result));
    EXPECT_EQ(result.blockNum, 4U);
    EXPECT_EQ(result.workspaceSizes[0], 16U * 1024 * 1024 + 65536U * 2 + 4U * 65536 * 4);
}

TEST(ClaheTiling, RejectInvalidShapes)
{
    CompileInfo info;
    for (const auto& dims :
         std::vector<std::vector<int64_t>>{{1}, {1, 2, 3, 4, 5}, {0, 64}, {-1, 64}, {257, 8, 8}, {1, 3, 8, 8}}) {
        auto para = MakePara(dims, ge::DT_UINT8, 40.0f, {8, 8}, &info);
        TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(para, result));
    }
}

TEST(ClaheTiling, RejectInvalidAttributesAndType)
{
    CompileInfo info;
    for (const auto& grid :
         std::vector<std::vector<int64_t>>{{}, {8}, {1, 2, 3}, {0, 8}, {-1, 8}, {INT64_MAX, 2}, {65536, 65536}}) {
        auto para = MakePara({64, 64}, ge::DT_UINT8, 40.0f, grid, &info);
        TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(para, result));
    }
    for (float clip : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        auto para = MakePara({64, 64}, ge::DT_UINT8, clip, {8, 8}, &info);
        TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(para, result));
    }
    auto para = MakePara({64, 64}, ge::DT_FLOAT, 40.0f, {8, 8}, &info);
    TilingInfo result;
    EXPECT_FALSE(ExecuteTiling(para, result));
}

TEST(ClaheTiling, PositiveAndNonpositiveClipLimits)
{
    CompileInfo info;
    for (float clip : {-1.0f, 0.0f, 0.001f, 40.0f, std::numeric_limits<float>::max()}) {
        auto para = MakePara({1, 256, 3}, ge::DT_UINT8, clip, {5, 7}, &info);
        TilingInfo result;
        EXPECT_TRUE(ExecuteTiling(para, result));
    }
}
