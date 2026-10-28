/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "level2/aclnn_three_interpolate_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

namespace {
constexpr uint32_t C0 = 16;
constexpr uint32_t THREE_POINT = 3;
} // namespace

class three_interpolate_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "three_interpolate_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "three_interpolate_backward_test TearDown" << std::endl; }
};

TEST_F(three_interpolate_backward_test, grad_x_null_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT((aclTensor*)nullptr, idx, weight, (int)ms),
                        OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(three_interpolate_backward_test, idx_x_null_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, (aclTensor*)nullptr, weight, (int)ms),
                        OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(three_interpolate_backward_test, weight_null_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, (aclTensor*)nullptr, (int)ms),
                        OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(three_interpolate_backward_test, grad_x_dtype_err_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_INT32, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(three_interpolate_backward_test, idx_dtype_err_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(three_interpolate_backward_test, weight_dtype_err_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(three_interpolate_backward_test, m_zero_invalid_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, 0), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(three_interpolate_backward_test, m_negative_invalid_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, -1), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// n==0 且 b,c,m>0：输入为空、grad_y 非空。第一段应成功；
// Fill 可能不占 workspace，清零由第二段执行，不能用 workspaceSize 判断。
TEST_F(three_interpolate_backward_test, empty_n_nonzero_grad_y_test)
{
    auto bs = 1;
    auto cs = 3;
    auto ms = 6;
    auto ns = 0;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// b==0 时 grad_y 也为空，早退成功且不需要 workspace
TEST_F(three_interpolate_backward_test, empty_batch_empty_grad_y_test)
{
    auto bs = 0;
    auto cs = 3;
    auto ms = 6;
    auto ns = 2;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(three_interpolate_backward_test, hostapi_test_1_2_6_5_fp32)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-100, 100);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// ===== issue #1407：grad_x/idx/weight/grad_y 形状一致性负例，均期望 ACLNN_ERR_PARAM_INVALID =====

// idx/weight 的 n 维与 grad_x 的 n 维不一致：kernel 会按 grad_x 的 n 静默截断并产出错误梯度，须拒绝
TEST_F(three_interpolate_backward_test, shape_check_idx_n_mismatch_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns + 1, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns + 1, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// idx/weight 的 batch 维与 grad_x 的 batch 维不一致，须拒绝
TEST_F(three_interpolate_backward_test, shape_check_idx_batch_mismatch_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs + 1, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs + 1, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// idx/weight 末维不为 3（三点插值契约），须拒绝
TEST_F(three_interpolate_backward_test, shape_check_idx_last_dim_not_3_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// idx 与 weight 形状不一致（n 维不同），须拒绝
TEST_F(three_interpolate_backward_test, shape_check_weight_diff_idx_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns - 1, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// idx/weight rank 不为 3，须拒绝
TEST_F(three_interpolate_backward_test, shape_check_idx_rank_2_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns * THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns * THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// grad_x 4D 但末维不为 1，须拒绝
TEST_F(three_interpolate_backward_test, shape_check_grad_x_last_dim_not_1_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// grad_x rank 与 grad_y rank 不一致（3D 输入配 4D 输出），须拒绝
TEST_F(three_interpolate_backward_test, shape_check_grad_x_rank3_grad_y_rank4_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// grad_y 的 m 维与属性 m 不一致，须拒绝
TEST_F(three_interpolate_backward_test, shape_check_grad_y_m_dim_mismatch_test)
{
    auto bs = 1;
    auto cs = 2;
    auto ms = 6;
    auto ns = 5;

    auto grad_x = TensorDesc({bs, cs, ns, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto idx = TensorDesc({bs, ns, THREE_POINT}, ACL_INT32, ACL_FORMAT_ND);
    auto weight = TensorDesc({bs, ns, THREE_POINT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_y = TensorDesc({bs, cs, ms + 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnThreeInterpolateBackward, INPUT(grad_x, idx, weight, (int)ms), OUTPUT(grad_y));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
