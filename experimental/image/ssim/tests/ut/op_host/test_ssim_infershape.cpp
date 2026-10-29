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
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SsimInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(SsimInfershapeTest, ssim_infershape_rank2_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Ssim",
        {{{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SsimInfershapeTest, ssim_infershape_hw_too_small_fail)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Ssim",
        {{{{10, 64}, {10, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{10, 64}, {10, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(SsimInfershapeTest, ssim_infershape_dynamic_hw_success)
{
    // [-1, -1] means H/W are only known at runtime: infershape must not reject them,
    // Tiling4Ssim performs the H/W >= 11 check once the real shape is available.
    gert::InfershapeContextPara infershapeContextPara(
        "Ssim",
        {{{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SsimInfershapeTest, ssim_infershape_mismatch_fail)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Ssim",
        {{{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{64, 32}, {64, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
