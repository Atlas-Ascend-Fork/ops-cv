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
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GammaCorrectionInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GammaCorrectionInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GammaCorrectionInfershape TearDown" << std::endl; }
};

TEST_F(GammaCorrectionInfershape, RankFourImage)
{
    gert::InfershapeContextPara para("GammaCorrection",
                                     {{{{1, 3, 721, 1281}, {1, 3, 721, 1281}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 3, 721, 1281}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GammaCorrectionInfershape, RankTwo)
{
    gert::InfershapeContextPara para("GammaCorrection", {{{{17, 19}, {17, 19}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{17, 19}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GammaCorrectionInfershape, RankThree)
{
    gert::InfershapeContextPara para("GammaCorrection", {{{{3, 17, 19}, {3, 17, 19}}, ge::DT_BF16, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 17, 19}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GammaCorrectionInfershape, DynamicShapePassesThrough)
{
    gert::InfershapeContextPara para("GammaCorrection",
                                     {{{{1, 3, -1, -1}, {1, 3, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 3, -1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GammaCorrectionInfershape, InvalidRankFails)
{
    gert::InfershapeContextPara para("GammaCorrection", {{{{17}, {17}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
