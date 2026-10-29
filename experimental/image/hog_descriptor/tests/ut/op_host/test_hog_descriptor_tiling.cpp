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

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/hog_descriptor_tiling.h"
#include "../../../op_host/hog_descriptor_ub.h"
#include "hog_descriptor_test_attrs.h"

namespace {
using TilingAttr = gert::TilingContextPara::OpAttr;

void ExpectHogTilingFailure(const HogTestAttrs& attrs, ge::DataType dtype = ge::DT_FLOAT,
                            const gert::StorageShape& shape = {{128, 128}, {128, 128}})
{
    optiling::HogDescriptorCompileInfo compileInfo{48, 196608};
    gert::TilingContextPara para("HOGDescriptor", {{shape, dtype, ge::FORMAT_ND}},
                                 {{{{8100}, {8100}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<TilingAttr>(attrs),
                                 &compileInfo, "Ascend910b", 48, 196608);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

void ExpectHogTilingSuccess(const gert::StorageShape& shape, const gert::StorageShape& outputShape,
                            const HogTestAttrs& attrs = {})
{
    optiling::HogDescriptorCompileInfo compileInfo{48, 196608};
    gert::TilingContextPara para("HOGDescriptor", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<TilingAttr>(attrs),
                                 &compileInfo, "Ascend910b", 48, 196608);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(para, result));
    EXPECT_GT(result.blockNum, 0U);
}
} // namespace

TEST(HogDescriptorTiling, InputDtypesSelectNumericKeys)
{
    for (const auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        optiling::HogDescriptorCompileInfo compileInfo{48, 196608};
        gert::TilingContextPara para("HOGDescriptor", {{{{128, 128}, {128, 128}}, dtype, ge::FORMAT_ND}},
                                     {{{{8100}, {8100}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<TilingAttr>(),
                                     &compileInfo, "Ascend910b", 48, 196608);
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(para, result));
        EXPECT_EQ(result.tilingKey, dtype == ge::DT_FLOAT16 ? 0U : 1U);
        EXPECT_GT(result.blockNum, 0U);
    }
}

TEST(HogDescriptorTiling, LargeRgbBlockFitsByReducingBandLength)
{
    HogTestAttrs p;
    p.cpb = {8, 8};
    p.bins = 36;
    optiling::HogDescriptorCompileInfo compileInfo{48, 196608};
    gert::TilingContextPara para("HOGDescriptor", {{{{128, 512, 3}, {128, 512, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                 {{{{1181952}, {1181952}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<TilingAttr>(p),
                                 &compileInfo, "Ascend910b", 48, 196608);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(para, result));
    EXPECT_GT(result.blockNum, 0U);
}

TEST(HogDescriptorTiling, UbBudgetIncludesRgbQueuesAndBlockStaging)
{
    const uint64_t fixed = optiling::HogFixedUbBytes(512, 3, 4, 64, 40, 8, 8);
    // Removing unused buffers lets a 9-row plane fit; a 10-row plane still exceeds 192 KiB.
    EXPECT_GT(fixed + 10U * 64U * 40U * 4U, 196608U);
    EXPECT_LE(fixed + 9U * 64U * 40U * 4U, 196608U);
    EXPECT_LT(optiling::HogFixedUbBytes(512, 3, 2, 64, 40, 8, 8), fixed);
}

TEST(HogDescriptorTiling, InvalidListLengthsAndDivisors)
{
    HogTestAttrs p;
    p.ppc = {8};
    ExpectHogTilingFailure(p);
    p = HogTestAttrs{};
    p.ppc = {0, 8};
    ExpectHogTilingFailure(p);
    p = HogTestAttrs{};
    p.stride = {1, 0};
    ExpectHogTilingFailure(p);
    p = HogTestAttrs{};
    p.cpb = {0, 2};
    ExpectHogTilingFailure(p);
}

TEST(HogDescriptorTiling, InvalidNormEpsBinsAndDtype)
{
    HogTestAttrs p;
    p.norm = "unknown";
    ExpectHogTilingFailure(p);
    p = HogTestAttrs{};
    p.eps = std::numeric_limits<float>::infinity();
    ExpectHogTilingFailure(p);
    p = HogTestAttrs{};
    p.bins = 37;
    ExpectHogTilingFailure(p);
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_UINT8);
}

TEST(HogDescriptorTiling, RanksBelowTwoAreRejected)
{
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{}, {}});
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{128}, {128}});
}

TEST(HogDescriptorTiling, RanksAboveFourAreRejected)
{
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{1, 16, 16, 1, 1}, {1, 16, 16, 1, 1}});
}

TEST(HogDescriptorTiling, BatchLimitIsEnforced)
{
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{201, 16, 16}, {201, 16, 16}});
    ExpectHogTilingSuccess({{200, 16, 16}, {200, 16, 16}}, {{200, 36}, {200, 36}});
}

TEST(HogDescriptorTiling, HeightLimitIsEnforced)
{
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{10001, 16}, {10001, 16}});
    ExpectHogTilingSuccess({{10000, 16}, {10000, 16}}, {{44964}, {44964}});
}

TEST(HogDescriptorTiling, WidthLimitIsEnforced)
{
    // This must fail the H/W limit before the separate padded-width limit.
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{16, 10001}, {16, 10001}});
}

TEST(HogDescriptorTiling, CellGridMustContainACompleteBlock)
{
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{8, 16}, {8, 16}});
    ExpectHogTilingFailure(HogTestAttrs{}, ge::DT_FLOAT, {{16, 8}, {16, 8}});
    ExpectHogTilingSuccess({{16, 16}, {16, 16}}, {{36}, {36}});
}

TEST(HogDescriptorTiling, DescriptorByteLimitIsEnforced)
{
    HogTestAttrs p;
    p.ppc = {1, 8};
    p.cpb = {8, 8};
    p.bins = 36;
    // Width stays below 1024 and one histogram band fits UB, isolating the size guard.
    ExpectHogTilingFailure(p, ge::DT_FLOAT, {{10000, 512}, {10000, 512}});
    ExpectHogTilingSuccess({{1000, 512}, {1000, 512}}, {{130408704}, {130408704}}, p);
}

TEST(HogDescriptorTiling, PaddedWidthLimitIsEnforced)
{
    HogTestAttrs p;
    p.ppc = {8, 5};
    // The logical width is legal, but ceil(1024/5)*5 is 1025.
    ExpectHogTilingFailure(p, ge::DT_FLOAT, {{16, 1024}, {16, 1024}});
    ExpectHogTilingSuccess({{16, 1020}, {16, 1020}}, {{7308}, {7308}}, p);
    ExpectHogTilingSuccess({{16, 1024}, {16, 1024}}, {{4572}, {4572}});
}
