/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class GuidedFilterInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GuidedFilterInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GuidedFilterInfershape TearDown" << std::endl; }
};

// rank3 [H,W,C]：dst 与 src 完全同 shape
TEST_F(GuidedFilterInfershape, guided_filter_infershape_hwc)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {9, 16, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank2 [H,W]：dst 与 src 同 shape
TEST_F(GuidedFilterInfershape, guided_filter_infershape_hw)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{17, 23}, {17, 23}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{17, 23}, {17, 23}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {17, 23},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank4 [N,H,W,C]：dst 与 src 同 shape（通道数可与 guide 不同）
TEST_F(GuidedFilterInfershape, guided_filter_infershape_nhwc)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{2, 9, 13, 3}, {2, 9, 13, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                          {{{2, 9, 13, 4}, {2, 9, 13, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 9, 13, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 shape：未知维透传
TEST_F(GuidedFilterInfershape, guided_filter_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{-1, -1, 3}, {-1, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, -1, 3}, {-1, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank 不一致 -> 失败
TEST_F(GuidedFilterInfershape, guided_filter_infershape_rank_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{9, 16}, {9, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// 空间尺寸不一致 -> 失败
TEST_F(GuidedFilterInfershape, guided_filter_infershape_spatial_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{9, 17, 1}, {9, 17, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// rank4 的 batch 不一致 -> 失败（dst 继承 src 的 shape，而 block 数按 guide 的 N 算）
TEST_F(GuidedFilterInfershape, guided_filter_infershape_batch_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara("GuidedFilter",
                                                      {
                                                          {{{2, 9, 16, 1}, {2, 9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1, 9, 16, 1}, {1, 9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>());
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
