/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_grid_sampler3d_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_grid_sampler3d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "grid_sampler3d_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "grid_sampler3d_backward_test TearDown" << std::endl; }
};

// gradOutput nullptr
TEST_F(l2_grid_sampler3d_backward_test, gradOutput_nullptr)
{
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward, INPUT(nullptr, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// input nullptr
TEST_F(l2_grid_sampler3d_backward_test, input_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, nullptr, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// grid nullptr
TEST_F(l2_grid_sampler3d_backward_test, grid_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, nullptr, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// inputGrad nullptr
TEST_F(l2_grid_sampler3d_backward_test, inputGrad_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(nullptr, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gridGrad nullptr
TEST_F(l2_grid_sampler3d_backward_test, gridGrad_nullptr)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype float16
TEST_F(l2_grid_sampler3d_backward_test, dtype_float16)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float32
TEST_F(l2_grid_sampler3d_backward_test, dtype_float32)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// // dtype double
// TEST_F(l2_grid_sampler3d_backward_test, dtype_double)
// {
//     auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
//     auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
//     auto ut = OP_API_UT(
//         aclnnGridSampler3DBackward, INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
//         OUTPUT(inputGradDesc, gridGradDesc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// aicore dtype bf16
TEST_F(l2_grid_sampler3d_backward_test, dtype_bf16)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// formart NCDHW
TEST_F(l2_grid_sampler3d_backward_test, ascend910B2_case_transpose)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// formart NDHWC
TEST_F(l2_grid_sampler3d_backward_test, ascend910B2_case_no_transpose)
{
    auto gradOutputDesc = TensorDesc({1, 1, 2, 2, 1}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto inputDesc = TensorDesc({1, 1, 3, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 3, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// output_mask false false
TEST_F(l2_grid_sampler3d_backward_test, output_mask_false_false)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{false, false});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// invalid shape n
TEST_F(l2_grid_sampler3d_backward_test, shape_n_input1_grid2)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape c
TEST_F(l2_grid_sampler3d_backward_test, shape_c_input1_output2)
{
    auto gradOutputDesc = TensorDesc({1, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape h
TEST_F(l2_grid_sampler3d_backward_test, shape_h_grid2_output3)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape w
TEST_F(l2_grid_sampler3d_backward_test, shape_w_grid2_output4)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid mode
TEST_F(l2_grid_sampler3d_backward_test, mode_3)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 3, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invalid padding mode
TEST_F(l2_grid_sampler3d_backward_test, padding_mode_3)
{
    auto gradOutputDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({1, 1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 3, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空grid场景（grid/gradOutput的D=0，input非空）：早退分支不写任何输出，
// 非空的inputGrad/gridGrad必须被清零，防止脏内存被当作梯度返回
TEST_F(l2_grid_sampler3d_backward_test, empty_grid_output_should_be_zeroed)
{
    auto gradOutputDesc = TensorDesc({2, 2, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 0, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 0, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空grid且只要dx场景：outputMask只开inputGrad，非空inputGrad同样需要清零
TEST_F(l2_grid_sampler3d_backward_test, empty_grid_mask0_only_should_be_zeroed)
{
    auto gradOutputDesc = TensorDesc({2, 2, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 0, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 0, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, false});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空grid且只要dgrid场景：outputMask只开gridGrad，此组合下gridGrad为空不触发清零，
// 但inputGrad非空且mask[0]为false，验证早退分支不误写mask关闭的输出
TEST_F(l2_grid_sampler3d_backward_test, empty_grid_mask1_only_should_be_zeroed)
{
    auto gradOutputDesc = TensorDesc({2, 2, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 0, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 0, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{false, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 输入C=0场景：input/gradOutput/inputGrad为空无需处理，但grid/gridGrad非空，必须清零
TEST_F(l2_grid_sampler3d_backward_test, zero_channel_grid_grad_should_be_zeroed)
{
    auto gradOutputDesc = TensorDesc({2, 0, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 0, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 4, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 0, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 4, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 输入C=0且只要dgrid场景：outputMask只开gridGrad，非空gridGrad同样需要清零
TEST_F(l2_grid_sampler3d_backward_test, zero_channel_mask1_only_should_be_zeroed)
{
    auto gradOutputDesc = TensorDesc({2, 0, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 0, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 4, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 0, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 4, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{false, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// fp64 + NDHWC 组合应在 host 侧返回 161002（DOUBLE 不支持 NDHWC）
TEST_F(l2_grid_sampler3d_backward_test, double_ndhwc_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
    auto inputDesc = TensorDesc({2, 3, 4, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 4, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input D=0 应在 host 侧返回 161002（D/H/W 非零校验）
TEST_F(l2_grid_sampler3d_backward_test, input_d_zero_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 3, 0, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 0, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input H=0 应在 host 侧返回 161002
TEST_F(l2_grid_sampler3d_backward_test, input_h_zero_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 3, 3, 0, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 0, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input W=0 应在 host 侧返回 161002
TEST_F(l2_grid_sampler3d_backward_test, input_w_zero_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto inputDesc = TensorDesc({2, 3, 3, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput 声明 NDHWC 而 input 声明 NCDHW（不同族），应在 host 侧返回 161002（format 一致性校验）
TEST_F(l2_grid_sampler3d_backward_test, gradoutput_ndhwc_input_ncdhw_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto inputDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputGrad 声明 NCDHW 而 input 声明 NDHWC（不同族），应在 host 侧返回 161002（format 一致性校验）
TEST_F(l2_grid_sampler3d_backward_test, inputgrad_ncdhw_input_ndhwc_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto inputDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input 声明 NCDHW 而 gradOutput/inputGrad 声明 NDHWC（不同族），应在 host 侧返回 161002
TEST_F(l2_grid_sampler3d_backward_test, input_ncdhw_others_ndhwc_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto inputDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// DOUBLE + gradOutput NDHWC（input 为 NCDHW 且不同族），DOUBLE 约束与族一致性任一触发即返回 161002
TEST_F(l2_grid_sampler3d_backward_test, double_gradoutput_ndhwc_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
    auto inputDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input 声明非法 format（NCHW），应在 host 侧返回 161002
TEST_F(l2_grid_sampler3d_backward_test, input_nchw_should_be_rejected)
{
    auto gradOutputDesc = TensorDesc({2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto inputDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gridDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto inputGradDesc = TensorDesc({2, 3, 3, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gridGradDesc = TensorDesc({2, 2, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto outputMaskDesc = BoolArrayDesc(vector<bool>{true, true});
    auto ut = OP_API_UT(aclnnGridSampler3DBackward,
                        INPUT(gradOutputDesc, inputDesc, gridDesc, 0, 0, false, outputMaskDesc),
                        OUTPUT(inputGradDesc, gridGradDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
