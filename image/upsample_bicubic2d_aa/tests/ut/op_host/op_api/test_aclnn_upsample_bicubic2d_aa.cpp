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
#include "../../../../op_host/op_api/aclnn_upsample_bicubic2d_aa.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsample_bicubic2d_aa_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsample_bicubic2d_aa_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsample_bicubic2d_aa_test TearDown" << std::endl; }
};

// self nullptr
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_self_nullptr)
{
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(nullptr, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_out_nullptr)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// outputsize nullptr
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_outputsize_nullptr)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, nullptr, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype float16
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_dtype_float16)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// dtype float32
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_dtype_float32)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// self empty N
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_self_empty_n)
{
    auto self_desc = TensorDesc({0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({0, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// self empty C
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_self_empty_c)
{
    auto self_desc = TensorDesc({1, 0, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self empty H
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_self_empty_h)
{
    auto self_desc = TensorDesc({1, 1, 0, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// outputsize value 0
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_outputsize_value_0)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 0});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// outputsize size 3
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_outputsize_size_3)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self shape 3d
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_shape_3d)
{
    auto self_desc = TensorDesc({1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self and out dtype different
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_dtype_different)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self and out N different
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_out_n_different)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({2, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self and out C different
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_out_c_different)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 2, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out shape 3d
TEST_F(l2_upsample_bicubic2d_aa_test, l2_upsample_bicubic2d_aa_test_out_shape_3d)
{
    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_desc = TensorDesc({1, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(vector<int64_t>{5, 5});
    bool align_corners = false;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAA, INPUT(self_desc, output_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
