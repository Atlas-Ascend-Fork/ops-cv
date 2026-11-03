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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "any_value.h"

class GridUnnormalInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GridUnnormalInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GridUnnormalInfershape TearDown" << std::endl; }
};

// diff/position 的 shape 与 grid 一致；position dtype = int32
TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_same_shape_fp32)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // diff (inferred)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // position (inferred)
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 6, 5, 2},
        {1, 6, 5, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// fp16 + align_corners=true：shape 透传不变
TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_same_shape_fp16)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(true)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{3, 88, 15, 2}, {3, 88, 15, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // grid
            {{{3, 88, 15, 2}, {3, 88, 15, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // diff
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // position
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 88, 15, 2},
        {3, 88, 15, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维 -1：shape 透传不变
TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_unknown_dim)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{-1, -1, -1, 2}, {-1, -1, -1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
            {{{-1, -1, -1, 2}, {-1, -1, -1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // diff
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // position
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, 2},
        {-1, -1, -1, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_empty_tensor)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{0, 6, 5, 2}, {0, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
            {{{0, 6, 5, 2}, {0, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // diff
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // position
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 6, 5, 2},
        {0, 6, 5, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_does_not_add_dimension_value_checks)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GridUnnormal",
        {
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
            {{{3, 4, 7, 2}, {3, 4, 7, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // diff
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // position
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 6, 5, 2}, {1, 6, 5, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_rejects_non_4d_grid)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara("GridUnnormal",
                                                      {
                                                          {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
                                                          {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // diff
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // position
                                                      },
                                                      attrs);
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_rejects_last_dim_not_2)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::InfershapeContextPara infershapeContextPara("GridUnnormal",
                                                      {
                                                          {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      attrs);
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 动态 rank -2：shape 透传不变
TEST_F(GridUnnormalInfershape, grid_unnormal_infershape_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(true)),
    };
    gert::InfershapeContextPara infershapeContextPara("GridUnnormal",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // grid
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // assist
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // diff
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // position
                                                      },
                                                      attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
