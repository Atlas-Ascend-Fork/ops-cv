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
#include <iostream>
#include <string>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class HitOrMissInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HitOrMissInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "HitOrMissInfershape TearDown" << std::endl; }
};

TEST_F(HitOrMissInfershape, hit_or_miss_uint8_2d)
{
    gert::InfershapeContextPara para("HitOrMiss",
                                     {{{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},

                                      {{{3, 3}, {3, 3}}, ge::DT_INT8, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                     std::vector<gert::InfershapeContextPara::OpAttr>{});

    std::vector<std::vector<int64_t>> expectShape = {{64, 64}};

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

TEST_F(HitOrMissInfershape, hit_or_miss_rank4)
{
    gert::InfershapeContextPara para("HitOrMiss",
                                     {{{{1, 3, 32, 32}, {1, 3, 32, 32}}, ge::DT_BOOL, ge::FORMAT_ND},

                                      {{{5, 5}, {5, 5}}, ge::DT_INT8, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                     std::vector<gert::InfershapeContextPara::OpAttr>{});

    std::vector<std::vector<int64_t>> expectShape = {{1, 3, 32, 32}};

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

namespace {
using HitOrMissShapeAttr = gert::InfershapeContextPara::OpAttr;
using HitOrMissShapeTensor = gert::InfershapeContextPara::TensorDescription;

gert::InfershapeContextPara MakeHitOrMissShapePara(const gert::StorageShape& inputShape,
                                                   const gert::StorageShape& kernelShape,
                                                   const std::vector<int64_t>& axes = {-2, -1},
                                                   const std::vector<int64_t>& anchor = {-1, -1})
{
    const std::vector<HitOrMissShapeTensor> inputs = {HitOrMissShapeTensor(inputShape, ge::DT_UINT8, ge::FORMAT_ND),
                                                      HitOrMissShapeTensor(kernelShape, ge::DT_INT8, ge::FORMAT_ND)};
    const std::vector<HitOrMissShapeTensor> outputs = {
        HitOrMissShapeTensor(gert::StorageShape{}, ge::DT_BOOL, ge::FORMAT_ND)};
    const std::vector<HitOrMissShapeAttr> attrs = {
        HitOrMissShapeAttr("axes", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(axes)),
        HitOrMissShapeAttr("anchor", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(anchor)),
        HitOrMissShapeAttr("border_value", Ops::Cv::AnyValue::CreateFrom<bool>(true))};
    return gert::InfershapeContextPara("HitOrMiss", inputs, outputs, attrs);
}

struct HitOrMissInvalidShapeCase {
    const char* name;
    gert::StorageShape inputShape;
    gert::StorageShape kernelShape;
    std::vector<int64_t> axes;
    std::vector<int64_t> anchor;
};
} // namespace

class HitOrMissInvalidInfershape : public testing::TestWithParam<HitOrMissInvalidShapeCase> {};

TEST_P(HitOrMissInvalidInfershape, RejectsInvalidParams)
{
    const auto& testCase = GetParam();
    auto para = MakeHitOrMissShapePara(testCase.inputShape, testCase.kernelShape, testCase.axes, testCase.anchor);
    ExecuteTestCase(para, ge::GRAPH_FAILED, {{}});
}

INSTANTIATE_TEST_SUITE_P(
    RejectParams, HitOrMissInvalidInfershape,
    testing::Values(
        HitOrMissInvalidShapeCase{"InputRankLow", {{64}, {64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"InputRankHigh",
                                  {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}},
                                  {{3, 3}, {3, 3}},
                                  {-2, -1},
                                  {-1, -1}},
        HitOrMissInvalidShapeCase{"KernelRankLow", {{64, 64}, {64, 64}}, {{3}, {3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"KernelRankHigh", {{64, 64}, {64, 64}}, {{3, 3, 1}, {3, 3, 1}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"KernelHeightZero", {{64, 64}, {64, 64}}, {{0, 3}, {0, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"KernelWidthZero", {{64, 64}, {64, 64}}, {{3, 0}, {3, 0}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"KernelHeightLarge", {{64, 64}, {64, 64}}, {{64, 3}, {64, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"KernelWidthLarge", {{64, 64}, {64, 64}}, {{3, 64}, {3, 64}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"AxesCountLow", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"AxesCountHigh", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {0, 1, 0}, {-1, -1}},
        HitOrMissInvalidShapeCase{"AxesPositiveRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {0, 2}, {-1, -1}},
        HitOrMissInvalidShapeCase{"AxesNegativeRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-3, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"AxesNormalizedDuplicate", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-1, 1}, {-1, -1}},
        HitOrMissInvalidShapeCase{"AnchorCountLow", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1}},
        HitOrMissInvalidShapeCase{"AnchorCountHigh", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1, -1}},
        HitOrMissInvalidShapeCase{"AnchorNegative", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-2, 0}},
        HitOrMissInvalidShapeCase{"AnchorHeightRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {3, 0}},
        HitOrMissInvalidShapeCase{"AnchorWidthRange", {{64, 64}, {64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {0, 3}},
        HitOrMissInvalidShapeCase{"InputZeroDim", {{0, 64}, {0, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}},
        HitOrMissInvalidShapeCase{
            "InputZeroBatchDim", {{0, 64, 64}, {0, 64, 64}}, {{3, 3}, {3, 3}}, {-2, -1}, {-1, -1}}),
    [](const testing::TestParamInfo<HitOrMissInvalidShapeCase>& info) { return std::string(info.param.name); });

TEST_F(HitOrMissInfershape, hit_or_miss_unknown_input_dimension)
{
    auto para = MakeHitOrMissShapePara({{-1, 64}, {-1, 64}}, {{3, 3}, {3, 3}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-1, 64}});
}

TEST_F(HitOrMissInfershape, hit_or_miss_unknown_input_rank)
{
    auto para = MakeHitOrMissShapePara({{-2}, {-2}}, {{3, 3}, {3, 3}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-2}});
}

TEST_F(HitOrMissInfershape, hit_or_miss_unknown_kernel_shape)
{
    auto unknownDim = MakeHitOrMissShapePara({{32, 64}, {32, 64}}, {{-1, 3}, {-1, 3}});
    ExecuteTestCase(unknownDim, ge::GRAPH_SUCCESS, {{32, 64}});
    auto unknownRank = MakeHitOrMissShapePara({{32, 64}, {32, 64}}, {{-2}, {-2}});
    ExecuteTestCase(unknownRank, ge::GRAPH_SUCCESS, {{32, 64}});
}

TEST_F(HitOrMissInfershape, hit_or_miss_supported_limits)
{
    auto para = MakeHitOrMissShapePara({{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}}, {{63, 63}, {63, 63}},
                                       {0, 7}, {62, 0});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 1, 1, 1, 1, 1, 1, 1}});
}

TEST_F(HitOrMissInfershape, hit_or_miss_invalid_dynamic_markers)
{
    auto invalidDim = MakeHitOrMissShapePara({{-3, 64}, {-3, 64}}, {{3, 3}, {3, 3}});
    ExecuteTestCase(invalidDim, ge::GRAPH_FAILED, {{}});
    auto invalidRankMarker = MakeHitOrMissShapePara({{-2, 64}, {-2, 64}}, {{3, 3}, {3, 3}});
    ExecuteTestCase(invalidRankMarker, ge::GRAPH_FAILED, {{}});
}
