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
#include <string>
#include <vector>

#include "../../../op_kernel/hit_or_miss_tiling_data.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {

struct HitOrMissCompileInfoForUt {};

gert::TilingContextPara MakeTilingPara(const gert::StorageShape& inputShape, const gert::StorageShape& kernelShape,
                                       const std::vector<int64_t>& axes, const std::vector<int64_t>& anchor,
                                       bool borderValue, void* compileInfo)
{
    gert::StorageShape outputShape = inputShape;

    return gert::TilingContextPara(
        "HitOrMiss", {{inputShape, ge::DT_UINT8, ge::FORMAT_ND}, {kernelShape, ge::DT_INT8, ge::FORMAT_ND}},
        {{outputShape, ge::DT_BOOL, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axes", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(axes)),
         gert::TilingContextPara::OpAttr("anchor", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(anchor)),
         gert::TilingContextPara::OpAttr("border_value", Ops::Cv::AnyValue::CreateFrom<bool>(borderValue))},
        compileInfo, "Ascend910b", 64, 262144, sizeof(HitOrMissTilingData));
}

} // namespace

class HitOrMissTiling : public testing::Test {};

TEST_F(HitOrMissTiling, hit_or_miss_basic)
{
    HitOrMissCompileInfoForUt compileInfo{};

    gert::StorageShape inputShape = {{64, 64}, {64, 64}};

    gert::StorageShape kernelShape = {{3, 3}, {3, 3}};

    auto para = MakeTilingPara(inputShape, kernelShape, {-2, -1}, {-1, -1}, true, &compileInfo);

    TilingInfo tilingInfo;

    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, 0);

    EXPECT_EQ(tilingInfo.blockNum, 8U);

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);

    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
}

TEST_F(HitOrMissTiling, hit_or_miss_non_aligned_explicit_attrs)
{
    HitOrMissCompileInfoForUt compileInfo{};

    gert::StorageShape inputShape = {{65, 67}, {65, 67}};

    gert::StorageShape kernelShape = {{5, 5}, {5, 5}};

    auto para = MakeTilingPara(inputShape, kernelShape, {-2, -1}, {1, 3}, false, &compileInfo);

    TilingInfo tilingInfo;

    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, 0);

    EXPECT_EQ(tilingInfo.blockNum, 8U);

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);

    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
}

TEST_F(HitOrMissTiling, hit_or_miss_duplicate_axes)
{
    HitOrMissCompileInfoForUt compileInfo{};

    gert::StorageShape inputShape = {{64, 64}, {64, 64}};

    gert::StorageShape kernelShape = {{3, 3}, {3, 3}};

    auto para = MakeTilingPara(inputShape, kernelShape, {-1, -1}, {-1, -1}, true, &compileInfo);

    TilingInfo tilingInfo;

    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

namespace {
struct HitOrMissInvalidTilingCase {
    const char* name;
    gert::StorageShape inputShape;
    gert::StorageShape kernelShape;
    std::vector<int64_t> axes;
    std::vector<int64_t> anchor;
};
} // namespace

class HitOrMissInvalidTiling : public testing::TestWithParam<HitOrMissInvalidTilingCase> {};

TEST_P(HitOrMissInvalidTiling, RejectsInvalidParams)
{
    const auto& testCase = GetParam();
    HitOrMissCompileInfoForUt compileInfo{};
    auto para = MakeTilingPara(testCase.inputShape, testCase.kernelShape, testCase.axes, testCase.anchor, true,
                               &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

INSTANTIATE_TEST_SUITE_P(
    RejectParams, HitOrMissInvalidTiling,
    testing::Values(
        HitOrMissInvalidTilingCase{"InputRankLow", {{64}, {64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"InputRankHigh",
                                   {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}},
                                   {{3, 3}, {3, 3}},
                                   {-2, -1},
                                   {-1, -1}},
        HitOrMissInvalidTilingCase{"KernelRankLow", {{64, 64}, {64, 64}}, {{3}, {3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"KernelRankHigh", {{64, 64}, {64, 64}}, {{3, 3, 1}, {3, 3, 1}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"KernelHeightZero", {{64, 64}, {64, 64}}, {{0, 3}, {0, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"KernelWidthZero", {{64, 64}, {64, 64}}, {{3, 0}, {3, 0}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"KernelHeightLarge", {{64, 64}, {64, 64}}, {{64, 3}, {64, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"KernelWidthLarge", {{64, 64}, {64, 64}}, {{3, 64}, {3, 64}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"AxesCountLow", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"AxesCountHigh", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {0, 1, 0}, {-1, -1}},
        HitOrMissInvalidTilingCase{"AxesPositiveRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {0, 2}, {-1, -1}},
        HitOrMissInvalidTilingCase{"AxesNegativeRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-3, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{
            "AxesNormalizedDuplicate", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-1, 1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"AnchorCountLow", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1}},
        HitOrMissInvalidTilingCase{"AnchorCountHigh", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1, -1}},
        HitOrMissInvalidTilingCase{"AnchorNegative", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-2, 0}},
        HitOrMissInvalidTilingCase{"AnchorHeightRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {3, 0}},
        HitOrMissInvalidTilingCase{"AnchorWidthRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {0, 3}},
        HitOrMissInvalidTilingCase{"InputZeroDim", {{0, 64}, {0, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{
            "InputZeroBatchDim", {{0, 64, 64}, {0, 64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidTilingCase{"InputUnknownDim", {{-1, 64}, {-1, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}}),
    [](const testing::TestParamInfo<HitOrMissInvalidTilingCase>& info) { return std::string(info.param.name); });
