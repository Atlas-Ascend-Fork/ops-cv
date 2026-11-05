/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/arch22/stack_group_points_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;

class StackGroupPointsTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "test TestStackGroupPointsTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "test TestStackGroupPointsTiling TearDown" << std::endl; }
};

TEST_F(StackGroupPointsTiling, case_float)
{
    optiling::StackGroupPointsCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{
                                                     4,
                                                 },
                                                 {
                                                     4,
                                                 }},
                                                ge::DT_INT32,
                                                ge::FORMAT_ND},
                                               {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{
                                                     4,
                                                 },
                                                 {
                                                     4,
                                                 }},
                                                ge::DT_INT32,
                                                ge::FORMAT_ND}},
                                              {{{{20, 64, 3}, {20, 64, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "4 20 64 3 5 8192 256 32 32 0 15360 32 3840 48 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StackGroupPointsTiling, case_half)
{
    optiling::StackGroupPointsCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {{{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{
                                                     4,
                                                 },
                                                 {
                                                     4,
                                                 }},
                                                ge::DT_INT32,
                                                ge::FORMAT_ND},
                                               {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{
                                                     4,
                                                 },
                                                 {
                                                     4,
                                                 }},
                                                ge::DT_INT32,
                                                ge::FORMAT_ND}},
                                              {{{{20, 64, 3}, {20, 64, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4 20 64 3 5 4096 256 32 32 0 7680 32 3840 48 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StackGroupPointsTiling, case_invalid_core_num)
{
    optiling::StackGroupPointsCompileInfo compileInfo = {0, 196608};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{20, 64, 3}, {20, 64, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}
