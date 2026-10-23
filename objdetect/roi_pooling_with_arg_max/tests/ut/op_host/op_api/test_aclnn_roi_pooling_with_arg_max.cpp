/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_api/aclnn_roi_pooling_with_arg_max.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_roi_pooling_with_arg_max_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_roi_pooling_with_arg_max_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_roi_pooling_with_arg_max_test TearDown" << endl; }
};

TEST_F(l2_roi_pooling_with_arg_max_test, success_case_float)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

TEST_F(l2_roi_pooling_with_arg_max_test, success_case_fp16)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_roi_pooling_with_arg_max_test, success_case_single_roi)
{
    auto x_desc = TensorDesc({1, 8, 14, 14}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 2;
    int64_t pooled_w = 2;
    float spatial_scale_h = 0.5f;
    float spatial_scale_w = 0.5f;
    auto y_desc = TensorDesc({1, 8, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({1, 8, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_roi_pooling_with_arg_max_test, success_case_large_pooled)
{
    auto x_desc = TensorDesc({4, 32, 56, 56}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 7;
    int64_t pooled_w = 7;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({4, 32, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({4, 32, 7, 7}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_nullptr_x)
{
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT((aclTensor*)nullptr, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_nullptr_rois)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, (aclTensor*)nullptr, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_nullptr_y)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT((aclTensor*)nullptr, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_nullptr_argmax)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_pooled_h_zero)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 0;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 0, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_pooled_w_zero)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 0;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 0}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_spatial_scale_h_zero)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 0.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_spatial_scale_w_zero)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 0.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_y_dtype_not_match_x)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_argmax_dtype_not_int32)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_x_dtype_unsupported)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_y_shape_wrong_num_rois)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({3, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_empty_rois_success)
{
    auto x_desc = TensorDesc({1, 8, 14, 14}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 2;
    int64_t pooled_w = 2;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({0, 8, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({0, 8, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_y_shape_wrong_channels)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 8, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_y_shape_wrong_pooled_dims)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_rois_shape_dim1_not_five)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_argmax_shape_mismatch)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_x_rois_dtype_mismatch)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_x_shape_not_4d)
{
    auto x_desc = TensorDesc({2, 16, 25}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_format_x_not_nd)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_with_arg_max_test, case_invalid_format_rois_not_nd)
{
    auto x_desc = TensorDesc({2, 16, 25, 42}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int64_t pooled_h = 3;
    int64_t pooled_w = 3;
    float spatial_scale_h = 1.0f;
    float spatial_scale_w = 1.0f;
    auto y_desc = TensorDesc({2, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({2, 16, 3, 3}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingWithArgMax,
                        INPUT(x_desc, rois_desc, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                        OUTPUT(y_desc, argmax_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
