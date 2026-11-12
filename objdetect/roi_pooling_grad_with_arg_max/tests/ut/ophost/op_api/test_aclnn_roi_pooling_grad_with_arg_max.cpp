/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_roi_pooling_grad_with_arg_max.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_roi_pooling_grad_with_arg_max_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "roi_pooling_grad_with_arg_max_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "roi_pooling_grad_with_arg_max_test TearDown" << endl; }
};

TEST_F(l2_roi_pooling_grad_with_arg_max_test, case_FLOAT)
{
    auto grad_out_desc = TensorDesc({4, 32, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({4, 32, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_ref_desc = TensorDesc({1, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingGradWithArgMax,
                        INPUT(grad_out_desc, grad_input_ref_desc, rois_desc, argmax_desc, 2, 2, 1.0), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

TEST_F(l2_roi_pooling_grad_with_arg_max_test, case_FLOAT16)
{
    auto grad_out_desc = TensorDesc({4, 32, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({4, 32, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_input_ref_desc = TensorDesc({1, 32, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingGradWithArgMax,
                        INPUT(grad_out_desc, grad_input_ref_desc, rois_desc, argmax_desc, 2, 2, 1.0), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_roi_pooling_grad_with_arg_max_test, case_FLOAT_empty_grad_output)
{
    auto grad_out_desc = TensorDesc({0, 32, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({0, 32, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_ref_desc = TensorDesc({4, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingGradWithArgMax,
                        INPUT(grad_out_desc, grad_input_ref_desc, rois_desc, argmax_desc, 2, 2, 1.0), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_roi_pooling_grad_with_arg_max_test, case_FLOAT_rejects_height_mismatch)
{
    auto grad_out_desc = TensorDesc({4, 32, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({4, 32, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_ref_desc = TensorDesc({1, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingGradWithArgMax,
                        INPUT(grad_out_desc, grad_input_ref_desc, rois_desc, argmax_desc, 2, 2, 1.0), OUTPUT());
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_pooling_grad_with_arg_max_test, case_FLOAT_rejects_width_mismatch)
{
    auto grad_out_desc = TensorDesc({4, 32, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({4, 32, 2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_ref_desc = TensorDesc({1, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRoiPoolingGradWithArgMax,
                        INPUT(grad_out_desc, grad_input_ref_desc, rois_desc, argmax_desc, 2, 2, 1.0), OUTPUT());
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// Non-regbase SoC: CheckSocValid must fail with ACLNN_ERR_RUNTIME_ERROR and report EZ1009.
TEST_F(l2_roi_pooling_grad_with_arg_max_test, case_soc_not_supported_910b)
{
    auto grad_out_desc = TensorDesc({4, 32, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto argmax_desc = TensorDesc({4, 32, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto rois_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_ref_desc = TensorDesc({1, 32, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
    auto ut = OP_API_UT(aclnnRoiPoolingGradWithArgMax,
                        INPUT(grad_out_desc, grad_input_ref_desc, rois_desc, argmax_desc, 2, 2, 1.0), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
}
