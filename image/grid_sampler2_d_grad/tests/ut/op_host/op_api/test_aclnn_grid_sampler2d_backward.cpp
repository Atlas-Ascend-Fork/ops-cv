/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grid_sampler2d_backward.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_grid_sampler2d_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class GridSampler2DApiErrors : public testing::TestWithParam<int> {};

TEST_P(GridSampler2DApiErrors, parameter_diagnostic)
{
    const int scenario = GetParam();
    vector<int64_t> gradShape = {1, 1, 3, 3}, inputShape = {1, 1, 5, 8}, gridShape = {1, 3, 3, 2};
    if (scenario == 5)
        inputShape = {1, 5, 8};
    if (scenario == 6)
        gridShape = {1, 3, 2};
    if (scenario == 7)
        gradShape = {1, 3, 3};
    if (scenario == 10)
        inputShape[0] = 2;
    if (scenario == 11)
        inputShape[1] = 2;
    if (scenario == 12)
        inputShape[2] = 0;
    if (scenario == 13)
        gridShape[1] = 2;
    if (scenario == 14)
        gridShape[3] = 3;
    auto inputGradShape = inputShape, gridGradShape = gridShape;
    if (scenario == 15)
        inputGradShape[2] = 4;
    if (scenario == 16)
        gridGradShape[1] = 2;
    const aclDataType dtype = scenario == 4 ? ACL_INT32 : ACL_FLOAT;
    auto grad = TensorDesc(gradShape, scenario == 0 ? ACL_FLOAT16 : dtype, ACL_FORMAT_ND);
    auto input = TensorDesc(inputShape, dtype, ACL_FORMAT_ND);
    auto grid = TensorDesc(gridShape, scenario == 1 ? ACL_FLOAT16 : dtype, ACL_FORMAT_ND);
    auto dx = TensorDesc(inputGradShape, scenario == 2 ? ACL_FLOAT16 : dtype, ACL_FORMAT_ND);
    auto dg = TensorDesc(gridGradShape, scenario == 3 ? ACL_FLOAT16 : dtype, ACL_FORMAT_ND);
    auto mask = BoolArrayDesc(scenario == 17 ? vector<bool>{true} : vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward,
                        INPUT(grad, input, grid, scenario == 8 ? 3 : 0, scenario == 9 ? 3 : 0, false, mask),
                        OUTPUT(dx, dg));
    uint64_t workspace = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(NegativeParameters, GridSampler2DApiErrors, testing::Range(0, 18));

class l2_grid_sampler2d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "grid_sampler2d_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "grid_sampler2d_backward_test TearDown" << std::endl; }
};

// 普通场景（含精度）
TEST_F(l2_grid_sampler2d_backward_test, case_1)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                        7,  8,  9,  10, 11, 12, 13,
                                                                                        14, 15, 16, 17, 18, 19, 20,
                                                                                        21, 22, 23, 24, 25, 26, 27,
                                                                                        28, 29, 30, 31, 32, 33, 34,
                                                                                        35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// // bicubic goes to aicpu
// TEST_F(l2_grid_sampler2d_backward_test, ascend910B2_case_1_BF16)
// {
//     auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_ND);
//     auto input =
//         TensorDesc({1, 1, 5, 8}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
//                                                                               10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
//                                                                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
//                                                                               30, 31, 32, 33, 34, 35, 36, 37, 38,
//                                                                               39});
//     auto grid = TensorDesc({1, 3, 3, 2}, ACL_BF16, ACL_FORMAT_ND)
//                     .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
//     auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_BF16, ACL_FORMAT_ND);
//     auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_BF16, ACL_FORMAT_ND);
//     bool alignCorners = false;
//     auto outputMask = BoolArrayDesc(vector<bool>{true, true});
//     auto ut = OP_API_UT(
//         aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 2, alignCorners, outputMask),
//         OUTPUT(inputGrad, gridGrad));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// input nullptr
TEST_F(l2_grid_sampler2d_backward_test, case_2)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = nullptr;
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// grid nullptr
TEST_F(l2_grid_sampler2d_backward_test, case_3)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                        7,  8,  9,  10, 11, 12, 13,
                                                                                        14, 15, 16, 17, 18, 19, 20,
                                                                                        21, 22, 23, 24, 25, 26, 27,
                                                                                        28, 29, 30, 31, 32, 33, 34,
                                                                                        35, 36, 37, 38, 39});
    auto grid = nullptr;
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gradOutput nullptr
TEST_F(l2_grid_sampler2d_backward_test, case_4)
{
    auto gradOutput = nullptr;
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                        7,  8,  9,  10, 11, 12, 13,
                                                                                        14, 15, 16, 17, 18, 19, 20,
                                                                                        21, 22, 23, 24, 25, 26, 27,
                                                                                        28, 29, 30, 31, 32, 33, 34,
                                                                                        35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// inputGrad nullptr
TEST_F(l2_grid_sampler2d_backward_test, case_5)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                        7,  8,  9,  10, 11, 12, 13,
                                                                                        14, 15, 16, 17, 18, 19, 20,
                                                                                        21, 22, 23, 24, 25, 26, 27,
                                                                                        28, 29, 30, 31, 32, 33, 34,
                                                                                        35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = nullptr;
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gridGrad nullptr
TEST_F(l2_grid_sampler2d_backward_test, case_6)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                        7,  8,  9,  10, 11, 12, 13,
                                                                                        14, 15, 16, 17, 18, 19, 20,
                                                                                        21, 22, 23, 24, 25, 26, 27,
                                                                                        28, 29, 30, 31, 32, 33, 34,
                                                                                        35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = nullptr;
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// bfloat16
TEST_F(l2_grid_sampler2d_backward_test, case_7)
{
    aclDataType dtype = aclDataType::ACL_BF16;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// // bicubic goes to aicpu
// TEST_F(l2_grid_sampler2d_backward_test, case_7_padding_2)
// {
//     aclDataType dtype = aclDataType::ACL_BF16;
//     auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
//     auto input =
//         TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
//                                                                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
//                                                                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
//                                                                            30, 31, 32, 33, 34, 35, 36, 37, 38, 39});
//     auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
//                     .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
//     auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
//     auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
//     bool alignCorners = false;
//     auto outputMask = BoolArrayDesc(vector<bool>{true, true});
//     auto ut = OP_API_UT(
//         aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 2, alignCorners, outputMask),
//         OUTPUT(inputGrad, gridGrad));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// float16
TEST_F(l2_grid_sampler2d_backward_test, case_8)
{
    aclDataType dtype = aclDataType::ACL_FLOAT16;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_grid_sampler2d_backward_test, case_9_BF16)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    aclDataType dtype = aclDataType::ACL_BF16;
    auto gradOutput = TensorDesc({1, 4096, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 4096, 5, 8}, dtype, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 4096, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

TEST_F(l2_grid_sampler2d_backward_test, case_9_BIGC)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto gradOutput = TensorDesc({1, 4096, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 4096, 5, 8}, dtype, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 4096, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

TEST_F(l2_grid_sampler2d_backward_test, case_9_310P)
{
    SetPlatformSocVersion(SocVersion::ASCEND310P);
    aclDataType dtype = aclDataType::ACL_UINT8;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

TEST_F(l2_grid_sampler2d_backward_test, case_9_950)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    aclDataType dtype = aclDataType::ACL_UINT8;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}
// // double
// TEST_F(l2_grid_sampler2d_backward_test, case_9)
// {
//     aclDataType dtype = aclDataType::ACL_DOUBLE;
//     auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
//     auto input =
//         TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
//                                                                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
//                                                                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
//                                                                            30, 31, 32, 33, 34, 35, 36, 37, 38, 39});
//     auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
//                     .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
//     auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
//     auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
//     bool alignCorners = false;
//     auto outputMask = BoolArrayDesc(vector<bool>{true, true});
//     auto ut = OP_API_UT(
//         aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
//         OUTPUT(inputGrad, gridGrad));

//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// uint8
TEST_F(l2_grid_sampler2d_backward_test, case_10)
{
    aclDataType dtype = aclDataType::ACL_UINT8;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int8
TEST_F(l2_grid_sampler2d_backward_test, case_11)
{
    aclDataType dtype = aclDataType::ACL_INT8;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int16
TEST_F(l2_grid_sampler2d_backward_test, case_12)
{
    aclDataType dtype = aclDataType::ACL_INT16;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int32
TEST_F(l2_grid_sampler2d_backward_test, case_13)
{
    aclDataType dtype = aclDataType::ACL_INT32;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int64
TEST_F(l2_grid_sampler2d_backward_test, case_14)
{
    aclDataType dtype = aclDataType::ACL_INT64;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bool
TEST_F(l2_grid_sampler2d_backward_test, case_15)
{
    aclDataType dtype = aclDataType::ACL_BOOL;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// complex64
TEST_F(l2_grid_sampler2d_backward_test, case_16)
{
    aclDataType dtype = aclDataType::ACL_COMPLEX64;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// complex128
TEST_F(l2_grid_sampler2d_backward_test, case_17)
{
    aclDataType dtype = aclDataType::ACL_COMPLEX128;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// nchw nhwc hwcn
TEST_F(l2_grid_sampler2d_backward_test, case_18)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_NHWC).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                      8,  9,  10, 11, 12, 13, 14, 15,
                                                                                      16, 17, 18, 19, 20, 21, 22, 23,
                                                                                      24, 25, 26, 27, 28, 29, 30, 31,
                                                                                      32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_HWCN)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_NHWC);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_HWCN);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// shape not match 1
TEST_F(l2_grid_sampler2d_backward_test, case_19)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({2, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({2, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape not match 2
TEST_F(l2_grid_sampler2d_backward_test, case_20)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape not match 3
TEST_F(l2_grid_sampler2d_backward_test, case_21)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_grid_sampler2d_backward_test, case_22)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND, {40, 40, 1, 5}, 0, {1, 1, 8, 5});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// mode is not illegal
TEST_F(l2_grid_sampler2d_backward_test, case_23)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 3, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// padding mode is not illegal
TEST_F(l2_grid_sampler2d_backward_test, case_24)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                    8,  9,  10, 11, 12, 13, 14, 15,
                                                                                    16, 17, 18, 19, 20, 21, 22, 23,
                                                                                    24, 25, 26, 27, 28, 29, 30, 31,
                                                                                    32, 33, 34, 35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 3, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// format not same
TEST_F(l2_grid_sampler2d_backward_test, case_25)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto gradOutput = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                         7,  8,  9,  10, 11, 12, 13,
                                                                                         14, 15, 16, 17, 18, 19, 20,
                                                                                         21, 22, 23, 24, 25, 26, 27,
                                                                                         28, 29, 30, 31, 32, 33, 34,
                                                                                         35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, dtype, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, dtype, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// illegal empty
TEST_F(l2_grid_sampler2d_backward_test, case_26)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape not match
TEST_F(l2_grid_sampler2d_backward_test, case_27)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape not match
TEST_F(l2_grid_sampler2d_backward_test, case_28)
{
    auto gradOutput = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape not match
TEST_F(l2_grid_sampler2d_backward_test, case_29)
{
    auto gradOutput = TensorDesc({1, 1, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 普通场景（含精度）
TEST_F(l2_grid_sampler2d_backward_test, case_30)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                        7,  8,  9,  10, 11, 12, 13,
                                                                                        14, 15, 16, 17, 18, 19, 20,
                                                                                        21, 22, 23, 24, 25, 26, 27,
                                                                                        28, 29, 30, 31, 32, 33, 34,
                                                                                        35, 36, 37, 38, 39});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{false, false});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// channel 是8对齐
TEST_F(l2_grid_sampler2d_backward_test, case_31)
{
    auto gradOutput = TensorDesc({1, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                         7,  8,  9,  10, 11, 12, 13,
                                                                                         14, 15, 16, 17, 18, 19, 20,
                                                                                         21, 22, 23, 24, 25, 26, 27,
                                                                                         28, 29, 30, 31});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{false, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// channel 是8对齐且获取两个输出
TEST_F(l2_grid_sampler2d_backward_test, ascend910B2_case_float_bilinear_zeros)
{
    auto gradOutput = TensorDesc({1, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0,  1,  2,  3,  4,  5,  6,
                                                                                         7,  8,  9,  10, 11, 12, 13,
                                                                                         14, 15, 16, 17, 18, 19, 20,
                                                                                         21, 22, 23, 24, 25, 26, 27,
                                                                                         28, 29, 30, 31});
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                    .Value(vector<float>{-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1});
    auto inputGrad = TensorDesc({1, 32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空grid场景：kernel不会写输出，非空输出必须被清零，防止脏内存被当作梯度返回
TEST_F(l2_grid_sampler2d_backward_test, empty_grid_output_should_be_zeroed)
{
    auto gradOutput = TensorDesc({1, 1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 0, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 0, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空grid且只要dx场景：outputMask只开inputGrad，同样需要清零
TEST_F(l2_grid_sampler2d_backward_test, empty_grid_mask0_only_should_be_zeroed)
{
    auto gradOutput = TensorDesc({1, 1, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 0, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 1, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 0, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, false});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入C=0场景：inputGrad为空无需处理，但gridGrad非空，必须清零
TEST_F(l2_grid_sampler2d_backward_test, zero_channel_grid_grad_should_be_zeroed)
{
    auto gradOutput = TensorDesc({1, 0, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input = TensorDesc({1, 0, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grid = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGrad = TensorDesc({1, 0, 5, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gridGrad = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool alignCorners = false;
    auto outputMask = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler2DBackward, INPUT(gradOutput, input, grid, 0, 0, alignCorners, outputMask),
                        OUTPUT(inputGrad, gridGrad));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
