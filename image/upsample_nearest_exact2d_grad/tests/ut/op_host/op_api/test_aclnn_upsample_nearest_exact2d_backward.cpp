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
#include "../../../../op_host/op_api/aclnn_upsample_nearest_exact2d_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsample_nearest_exact2d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsample_nearest_exact2d_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsample_nearest_exact2d_backward_test TearDown" << std::endl; }
};

// out nullptr
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_out_nullptr)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// inputsize nullptr
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_outputsize_nullptr)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, nullptr, scales_h, scales_w), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype float16
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_dtype_float16)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype float32
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_dtype_float32)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// gradOutput empty N
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_gradOutput_empty_n)
{
    auto gradOutput_desc = TensorDesc({0, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{0, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// gradOutput empty C
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_gradOutput_empty_c)
{
    auto gradOutput_desc = TensorDesc({1, 0, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 0, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 0, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// gradOutput empty H
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_gradOutput_empty_h)
{
    auto gradOutput_desc = TensorDesc({1, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputsize value 0
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_inputsize_value_0)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 0});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputsize size 5
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_inputsize_size_5)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput shape 3d
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_shape_3d)
{
    auto gradOutput_desc = TensorDesc({1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput and out dtype different
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_dtype_different)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// NHWC不支持
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_NHWC_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput and out format different
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_format_different)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scale
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_scale)
{
    auto gradOutput_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scale negative
TEST_F(l2_upsample_nearest_exact2d_backward_test, l2_upsample_nearest_exact2d_backward_test_scale_negative)
{
    auto gradOutput_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = -2.0;
    const double_t scales_w = -2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// scalesH 与 outputSize 不一致应返回 161002（inputH=3, scalesH=1.0 期望 outputH=3 而非 5）
TEST_F(l2_upsample_nearest_exact2d_backward_test, scaleH_conflict_with_inputsize_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scales>0 且与 inputSize/outputSize 自洽时应返回 SUCCESS（容差放行侧回归）
TEST_F(l2_upsample_nearest_exact2d_backward_test, scaleH_positive_consistent_success)
{
    auto gradOutput_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// scales 与 outputSize 冲突（outputSize 参数与 gradOutput 形状一致但与 scales 推导不符）应返回 161002
TEST_F(l2_upsample_nearest_exact2d_backward_test, scaleH_conflict_with_outputsize_param_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{7, 7});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非整数乘积时 outputSize 应取 floor 而非 ceil：floor(3*1.5)=4/floor(3*2.5)=7 合法应返回 SUCCESS
TEST_F(l2_upsample_nearest_exact2d_backward_test, scale_noninteger_floor_success)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{4, 7});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 1.5;
    const double_t scales_w = 2.5;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非整数乘积时误传 ceil 值（floor(3*1.5)=4 而非 5）应返回 161002
TEST_F(l2_upsample_nearest_exact2d_backward_test, scale_noninteger_ceil_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 7});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 1.5;
    const double_t scales_w = 2.5;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOutput 的 H 与 outputSize 不符应返回 161002（outputSize 与 scales 自洽但 gradOutput 形状错开，
// kernel 按 outputSize 读输入会 GM 越界，需在 host 侧拦截）
TEST_F(l2_upsample_nearest_exact2d_backward_test, gradoutput_dim_not_match_outputsize_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out 的 H/W 与 inputSize 不符应返回 161002（out 是 gradInput 的承载，形状必须等于 inputSize）
TEST_F(l2_upsample_nearest_exact2d_backward_test, out_dim_not_match_inputsize_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputSize 的 N 与 gradOutput 实际形状矛盾应返回 161002（评审 P2 场景：inputSize 与张量 N/C
// 矛盾时 tiling 的 batch 计算错误，需在 host 侧拦截）
TEST_F(l2_upsample_nearest_exact2d_backward_test, inputsize_nc_not_match_gradoutput_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{2, 1, 3, 3});
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空 Tensor 的非法参数（N=0 但 scales 与 outputSize 冲突）应返回 161002：
// 校验先于空 Tensor 早退，避免空输入静默吞掉参数错误
TEST_F(l2_upsample_nearest_exact2d_backward_test, empty_tensor_invalid_scale_fail)
{
    auto gradOutput_desc = TensorDesc({0, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{6, 6});
    auto input_size_desc = IntArrayDesc(vector<int64_t>{0, 1, 3, 3});
    const double_t scales_h = 3.0;
    const double_t scales_w = 3.0;

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2dBackward,
                        INPUT(gradOutput_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // floor(3*3.0)=9 != 6，N=0 空 Tensor 的非法 scales 也应被校验拦截
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
