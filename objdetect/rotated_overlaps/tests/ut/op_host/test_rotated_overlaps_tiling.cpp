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

#include <cstddef>
#include <limits>

#include "any_value.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/arch35/rotated_overlaps_tiling_data.h"
#include "../../../op_kernel/arch35/rotated_overlaps_tiling_key.h"

namespace {
struct CompileInfo {};

gert::TilingContextPara MakeContext(int64_t batch, int64_t boxes, int64_t queries, uint64_t ubSize = 262144)
{
    static CompileInfo compileInfo;
    return gert::TilingContextPara(
        "RotatedOverlaps",
        {
            {gert::StorageShape({batch, 5, boxes}, {batch, 5, boxes}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({batch, 5, queries}, {batch, 5, queries}), ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gert::StorageShape({batch, boxes, queries}, {batch, boxes, queries}), ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("trans", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, "Ascend950", 64, ubSize);
}

const RotatedOverlapsTilingData* GetTilingData(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingDataSize, sizeof(RotatedOverlapsTilingData));
    return reinterpret_cast<const RotatedOverlapsTilingData*>(info.tilingData.get());
}
} // namespace

TEST(RotatedOverlapsTiling, accepts_regular_input_address_products)
{
    auto context = MakeContext(2, 7, 31);
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(context, info));
}

TEST(RotatedOverlapsTiling, rejects_input_bytes_that_overflow_uint64)
{
    auto context = MakeContext(1000000000LL, 1000000000LL, 1);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(RotatedOverlapsTiling, uses_vector_fallback_and_64_bit_indices_for_more_than_uint32_pairs)
{
    constexpr int64_t kBatch = 1;
    constexpr int64_t kBoxes = 2147484;
    constexpr int64_t kQueries = 2000;
    constexpr uint64_t kCoordinateCount = 5;
    constexpr uint64_t kBoxElements = static_cast<uint64_t>(kBatch) * kCoordinateCount * kBoxes;
    constexpr uint64_t kQueryElements = static_cast<uint64_t>(kBatch) * kCoordinateCount * kQueries;
    constexpr uint64_t kTotalPairs = static_cast<uint64_t>(kBatch) * kBoxes * kQueries;
    constexpr uint64_t kBytesPerFloat = sizeof(float);
    static_assert(kQueries <= 2000);
    static_assert(kTotalPairs > std::numeric_limits<uint32_t>::max());
    static_assert(kBoxElements <= std::numeric_limits<std::size_t>::max() / kBytesPerFloat);
    static_assert(kQueryElements <= std::numeric_limits<std::size_t>::max() / kBytesPerFloat);
    static_assert(kTotalPairs <= std::numeric_limits<std::size_t>::max() / kBytesPerFloat);

    auto context = MakeContext(kBatch, kBoxes, kQueries);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_EQ(tiling->totalPairs, kTotalPairs);
    EXPECT_EQ(tiling->usePairParallelSimt, 0U);
    EXPECT_EQ(tiling->use32Bit, 0U);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(ROTATED_OVERLAPS_TPL_XYWHT, ROTATED_OVERLAPS_TPL_INDEX_64));
}

TEST(RotatedOverlapsTiling, keeps_contiguous_query_vectorization_when_core_waves_are_equal)
{
    auto context = MakeContext(1, 3, 127);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_EQ(tiling->vectorizeBoxes, 0U);
    EXPECT_EQ(tiling->totalTasks, 3U);
    EXPECT_EQ(tiling->tasksPerCore, 1U);
}

TEST(RotatedOverlapsTiling, vectorizes_boxes_when_it_removes_core_waves)
{
    auto context = MakeContext(1, 127, 3);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_EQ(tiling->vectorizeBoxes, 1U);
    EXPECT_EQ(tiling->totalTasks, 3U);
    EXPECT_EQ(tiling->tasksPerCore, 1U);
}

TEST(RotatedOverlapsTiling, splits_multiple_query_tiles_into_independent_tasks)
{
    auto context = MakeContext(1, 65, 129);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* tiling = GetTilingData(info);
    EXPECT_EQ(tiling->vectorizeBoxes, 0U);
    EXPECT_EQ(tiling->tilesPerOuter, 2U);
    EXPECT_EQ(tiling->totalTasks, 130U);
}

TEST(RotatedOverlapsTiling, rejects_insufficient_ub_for_minimum_tile)
{
    // 36 KiB UB minus the 32 KiB SIMT Data Cache reserve leaves only 4 KiB,
    // which cannot hold even the minimum 8-element vector tile plus math
    // workspace (8 * 153 * 4 + sincos tmp + 256-byte mask reserve).
    auto context = MakeContext(1, 4, 4, 36U * 1024U);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}
