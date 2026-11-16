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
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
using namespace gert;

class PasteSubImgInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PasteSubImgInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PasteSubImgInfershape TearDown" << std::endl; }
};

// 常规场景：输出shape透传combine_img（第3个输入）
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_test_1)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{4, 16, 3}, {4, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{64, 64, 3}, {64, 64, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 64, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 部分未知shape（-1）场景：输出透传combine_img
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_partial_unknown)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{4, -1, 3}, {4, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{64, -1, 3}, {64, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, -1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 全未知shape（-1）场景
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_all_unknown_dim)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知rank（-2）场景：combine_img为-2时输出-2
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_combine_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 与rt1.0一致：patch_img为-2而combine已知时，跳过patch校验，输出combine的shape
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_patch_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{64, 64, 3}, {64, 64, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 64, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 与rt1.0一致：coord长度未知（-1）时跳过长度校验，输出combine的shape
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_coord_unknown_dim)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{4, 16, 3}, {4, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{64, 64, 3}, {64, 64, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 64, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// patch为未知rank但combine非3D时必须报错（combine的3D校验无条件）
TEST_F(PasteSubImgInfershape, paste_sub_img_infershape_patch_unknown_rank_combine_invalid_rank)
{
    gert::InfershapeContextPara infershapeContextPara("PasteSubImg",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
