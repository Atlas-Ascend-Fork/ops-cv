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

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
void RunInfershapeCase(const gert::StorageShape& inputShape, ge::graphStatus expectedStatus,
                       const std::vector<int64_t>& expectedOutputDims = {})
{
    gert::StorageShape outputShape = {{-2}, {-2}};
    gert::InfershapeContextPara context("RetinexEnhance", {{inputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}});
    const std::vector<std::vector<int64_t>> expectedOutputs = {expectedOutputDims};
    ExecuteTestCase(context, expectedStatus, expectedOutputs);
}
} // namespace

TEST(RetinexEnhanceInfershape, CopiesValidShape)
{
    RunInfershapeCase({{17, 19, 3}, {17, 19, 3}}, ge::GRAPH_SUCCESS, {17, 19, 3});
}

TEST(RetinexEnhanceInfershape, AcceptsGrayscaleAndRgba)
{
    RunInfershapeCase({{17, 19, 1}, {17, 19, 1}}, ge::GRAPH_SUCCESS, {17, 19, 1});
    RunInfershapeCase({{17, 19, 4}, {17, 19, 4}}, ge::GRAPH_SUCCESS, {17, 19, 4});
}

TEST(RetinexEnhanceInfershape, PreservesDynamicSpatialDimensions)
{
    RunInfershapeCase({{-1, -1, 3}, {-1, -1, 3}}, ge::GRAPH_SUCCESS, {-1, -1, 3});
}

TEST(RetinexEnhanceInfershape, PreservesDynamicRankThreeShape)
{
    RunInfershapeCase({{-1, -1, -1}, {-1, -1, -1}}, ge::GRAPH_SUCCESS, {-1, -1, -1});
}

TEST(RetinexEnhanceInfershape, RejectsInvalidRank) { RunInfershapeCase({{17, 19}, {17, 19}}, ge::GRAPH_FAILED); }

TEST(RetinexEnhanceInfershape, RejectsInvalidChannelCount)
{
    RunInfershapeCase({{17, 19, 2}, {17, 19, 2}}, ge::GRAPH_FAILED);
    RunInfershapeCase({{17, 19, 5}, {17, 19, 5}}, ge::GRAPH_FAILED);
}

TEST(RetinexEnhanceInfershape, RejectsZeroSpatialDimension)
{
    RunInfershapeCase({{0, 19, 3}, {0, 19, 3}}, ge::GRAPH_FAILED);
}

TEST(RetinexEnhanceInfershape, RejectsInvalidNegativeSpatialDimension)
{
    RunInfershapeCase({{-2, 19, 3}, {-2, 19, 3}}, ge::GRAPH_FAILED);
}
