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
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_graph/stack_group_points_proto.h"

class StackGroupPoints : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StackGroupPoints Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StackGroupPoints Proto Test TearDown" << std::endl; }
};

TEST_F(StackGroupPoints, StackGroupPoints_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
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
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20, 64, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ===== 维度校验失败用例 =====

// Negative: features 非 2D（1D）
TEST_F(StackGroupPoints, StackGroupPoints_infershape_neg_features_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32}, {32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Negative: indices 非 2D（3D）
TEST_F(StackGroupPoints, StackGroupPoints_infershape_neg_indices_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3, 1}, {20, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Negative: features_batch_cnt 非 1D（2D）
TEST_F(StackGroupPoints, StackGroupPoints_infershape_neg_fbc_not_1d)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Negative: features_batch_cnt 非 1D（0D 标量）
TEST_F(StackGroupPoints, StackGroupPoints_infershape_neg_fbc_0d)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Negative: indices_batch_cnt 非 1D（0D 标量）
TEST_F(StackGroupPoints, StackGroupPoints_infershape_neg_ibc_not_1d)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ===== Unknown Rank（-2）正例 =====

// 正例：features 为 unknown rank（-2），输出置 unknown rank，返回 SUCCESS
TEST_F(StackGroupPoints, StackGroupPoints_infershape_unknown_rank_features)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

// 正例：indices 为 unknown rank（-2），输出置 unknown rank，返回 SUCCESS
TEST_F(StackGroupPoints, StackGroupPoints_infershape_unknown_rank_indices)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

// ===== Unknown Shape（-1）正例 =====

// 正例：features 含 -1 维度值（unknown shape），输出置 3D unknown shape {-1,-1,-1}，返回 SUCCESS
TEST_F(StackGroupPoints, StackGroupPoints_infershape_unknown_shape_features)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{-1, 64}, {-1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正例：indices 含 -1 维度值（unknown shape），输出置 3D unknown shape {-1,-1,-1}，返回 SUCCESS
TEST_F(StackGroupPoints, StackGroupPoints_infershape_unknown_shape_indices)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{-1, 3}, {-1, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正例：features 和 indices 均 含 -1（交叉出现），输出置 3D unknown shape {-1,-1,-1}，返回 SUCCESS
TEST_F(StackGroupPoints, StackGroupPoints_infershape_unknown_shape_both)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正例：features/indices 具体而 features_batch_cnt 长度未知（-1，动态图形态），
// 输出按 m/c/nsample 推导
TEST_F(StackGroupPoints, StackGroupPoints_infershape_pos_fbc_unknown_len)
{
    gert::InfershapeContextPara infershapeContextPara("StackGroupPoints",
                                                      {{{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{20, 3}, {20, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20, 64, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
