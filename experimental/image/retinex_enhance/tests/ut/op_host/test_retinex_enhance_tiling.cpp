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

#include "experimental/image/retinex_enhance/op_kernel/retinex_enhance_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16U * 1024U * 1024U;

gert::TilingContextPara MakeContext(const gert::StorageShape& inputShape, const gert::StorageShape& outputShape,
                                    uint64_t coreNum = 64, uint64_t ubSize = 262144)
{
    static uint8_t compileInfo = 0;
    return gert::TilingContextPara("RetinexEnhance", {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo, "Ascend910B", coreNum,
                                   ubSize, sizeof(RetinexEnhanceTilingData));
}
} // namespace

TEST(RetinexEnhanceTiling, GeneratesExpectedNonalignedLayout)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{63, 65, 3}, {63, 65, 3}}, {{63, 65, 3}, {63, 65, 3}}), info));
    ASSERT_EQ(info.blockNum, 3U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + 63U * 65U * 3U * sizeof(float));
    ASSERT_EQ(info.tilingDataSize, sizeof(RetinexEnhanceTilingData));

    const auto* tiling = reinterpret_cast<const RetinexEnhanceTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->height, 63U);
    EXPECT_EQ(tiling->width, 65U);
    EXPECT_EQ(tiling->channels, 3U);
    EXPECT_EQ(tiling->tilePixels, 32U);
    EXPECT_EQ(tiling->stripeCount, 3U);
    EXPECT_EQ(tiling->rowElements, 195U);
    EXPECT_EQ(tiling->rowElementsAligned, 4600U);
    EXPECT_EQ(tiling->tileElementsAligned, 96U);
    EXPECT_EQ(tiling->verticalTileRows, 32U);
    EXPECT_EQ(tiling->verticalTileElementsAligned, 3072U);
    EXPECT_EQ(tiling->verticalIndexElementsAligned, 1568U);
    EXPECT_EQ(tiling->radii[0], 45U);
    EXPECT_EQ(tiling->radii[1], 240U);
    EXPECT_EQ(tiling->radii[2], 750U);
}

TEST(RetinexEnhanceTiling, SupportsGrayscaleAndRgbaLayouts)
{
    TilingInfo grayscaleInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{17, 19, 1}, {17, 19, 1}}, {{17, 19, 1}, {17, 19, 1}}), grayscaleInfo));
    const auto* grayscale = reinterpret_cast<const RetinexEnhanceTilingData*>(grayscaleInfo.tilingData.get());
    EXPECT_EQ(grayscale->channels, 1U);
    EXPECT_EQ(grayscale->rowElements, 19U);
    EXPECT_GE(grayscale->rowElementsAligned, grayscale->maxHaloElementsAligned);
    EXPECT_GE(grayscale->tileElementsAligned, 64U);
    EXPECT_EQ(grayscaleInfo.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + 17U * 19U * sizeof(float));

    TilingInfo rgbaInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{17, 19, 4}, {17, 19, 4}}, {{17, 19, 4}, {17, 19, 4}}), rgbaInfo));
    const auto* rgba = reinterpret_cast<const RetinexEnhanceTilingData*>(rgbaInfo.tilingData.get());
    EXPECT_EQ(rgba->channels, 4U);
    EXPECT_EQ(rgba->rowElements, 76U);
    EXPECT_EQ(rgbaInfo.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + 17U * 19U * 4U * sizeof(float));
}

TEST(RetinexEnhanceTiling, LimitsBlocksToAvailableStripes)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{17, 19, 3}, {17, 19, 3}}, {{17, 19, 3}, {17, 19, 3}}, 64), info));
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + 17U * 19U * 3U * sizeof(float));
}

TEST(RetinexEnhanceTiling, AdaptsToDifferentRuntimeSpatialShapes)
{
    TilingInfo smallInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{17, 19, 3}, {17, 19, 3}}, {{17, 19, 3}, {17, 19, 3}}), smallInfo));
    TilingInfo largeInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{480, 640, 3}, {480, 640, 3}}, {{480, 640, 3}, {480, 640, 3}}), largeInfo));

    EXPECT_EQ(smallInfo.blockNum, 1U);
    EXPECT_EQ(largeInfo.blockNum, 20U);
    ASSERT_EQ(smallInfo.workspaceSizes.size(), 1U);
    ASSERT_EQ(largeInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(smallInfo.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + 17U * 19U * 3U * sizeof(float));
    EXPECT_EQ(largeInfo.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + 480U * 640U * 3U * sizeof(float));
}

TEST(RetinexEnhanceTiling, AllocatesVerticalIndexForTallDynamicShape)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext({{5000, 3, 3}, {5000, 3, 3}}, {{5000, 3, 3}, {5000, 3, 3}}), info));

    const auto* tiling = reinterpret_cast<const RetinexEnhanceTilingData*>(info.tilingData.get());
    EXPECT_EQ(tiling->tilePixels, 3U);
    EXPECT_EQ(tiling->verticalTileRows, 32U);
    EXPECT_EQ(tiling->verticalTileElementsAligned, 2048U);
    EXPECT_EQ(tiling->maxHaloElementsAligned, 4512U);
    EXPECT_EQ(tiling->verticalIndexElementsAligned, 6504U);
}

TEST(RetinexEnhanceTiling, RejectsInvalidShape)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext({{17, 19, 2}, {17, 19, 2}}, {{17, 19, 2}, {17, 19, 2}}), info));
    EXPECT_FALSE(ExecuteTiling(MakeContext({{17, 19, 5}, {17, 19, 5}}, {{17, 19, 5}, {17, 19, 5}}), info));
}

TEST(RetinexEnhanceTiling, RejectsMismatchedOutputShape)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext({{17, 19, 3}, {17, 19, 3}}, {{17, 20, 3}, {17, 20, 3}}), info));
}

TEST(RetinexEnhanceTiling, RejectsInsufficientUb)
{
    TilingInfo info;
    EXPECT_FALSE(
        ExecuteTiling(MakeContext({{63, 65, 3}, {63, 65, 3}}, {{63, 65, 3}, {63, 65, 3}}, 64, 32U * 1024U), info));
}

TEST(RetinexEnhanceTiling, RejectsUbAlignmentOverflow)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(
        MakeContext({{1, 1431655765, 3}, {1, 1431655765, 3}}, {{1, 1431655765, 3}, {1, 1431655765, 3}}), info));
}

TEST(RetinexEnhanceTiling, RejectsShapesAboveGatherByteOffsetLimit)
{
    // Synthetic UB removes the normal UB rejection, isolating the element-count check.
    constexpr uint64_t syntheticUb = 64ULL * 1024U * 1024U * 1024U;
    for (const int64_t channels : {1, 3, 4}) {
        const int64_t width = (1LL << 30) / channels + 1;
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(MakeContext({{1, width, channels}, {1, width, channels}},
                                               {{1, width, channels}, {1, width, channels}}, 64, syntheticUb),
                                   info));
    }
}

TEST(RetinexEnhanceTiling, AcceptsGatherElementLimitWithSyntheticUb)
{
    constexpr uint64_t syntheticUb = 64ULL * 1024U * 1024U * 1024U;
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(
        MakeContext({{32768, 32768, 1}, {32768, 32768, 1}}, {{32768, 32768, 1}, {32768, 32768, 1}}, 64, syntheticUb),
        info));
    EXPECT_EQ(info.workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + (1ULL << 32));
}

TEST(RetinexEnhanceTiling, RejectsWideImageExceedingUbBudget)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext({{1, 65536, 4}, {1, 65536, 4}}, {{1, 65536, 4}, {1, 65536, 4}}), info));
}
