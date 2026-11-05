/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "opdev/platform.h"

#include "../../../../op_host/op_api/aclnn_upsample_nearest_2d_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_upsample_nearest_2d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "upsample_nearest_2d_backward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "upsample_nearest_2d_backward_test TearDown" << endl; }
};

TEST_F(l2_upsample_nearest_2d_backward_test, case_float_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_float16_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_float_normal_NHWC)
{
    auto grad_out_desc = TensorDesc({2, 6, 20, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 3, 5, 2};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 3, 5, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> ValidList = {ACL_UINT8, ACL_INT8,   ACL_INT16,     ACL_INT32,     ACL_INT64,
                                     ACL_BOOL,  ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    int length = ValidList.size();

    for (int i = 0; i < length; i++) {
        auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ValidList[i], ACL_FORMAT_NCHW);
        auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ValidList[i], ACL_FORMAT_NCHW);
        auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                            INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                            OUTPUT(grad_input_desc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_shape_0_2_6_10_normal)
{
    auto grad_out_desc = TensorDesc({0, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {0, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({0, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_shape_2_0_6_10_normal)
{
    auto grad_out_desc = TensorDesc({2, 0, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 0, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 0, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_shape_2_2_0_10_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 0, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {0, 10};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_shape_2_2_4_0_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 0}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 0};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_1d_abnormal)
{
    auto grad_out_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_2d_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_5d_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_grad_out_nullptr_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT((aclTensor*)nullptr, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_output_size_nullptr_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, (aclIntArray*)nullptr, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_input_size_nullptr_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, (aclIntArray*)nullptr, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_float_float16_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_output_size_3_4_5_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {3, 6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_output_size_0_5_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {0, 10};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_input_size_2_3_5_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_input_size_2_2_3_0_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 0};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 0}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_nc_diff_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {4, 1, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({4, 1, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_hw_diff_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {4, 10};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float16_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 20}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 20};
    vector<int64_t> input_size = {2, 2, 3, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 4.0;
    auto grad_input_desc = TensorDesc({2, 2, 3, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_scales_h)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 10};
    vector<int64_t> input_size = {2, 2, 6, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 0.0;
    const double_t scales_w = 1.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_scales_w)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 10};
    vector<int64_t> input_size = {2, 2, 6, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 1.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_big_n_c)
{
    auto grad_out_desc = TensorDesc({10, 40, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {1, 1};
    vector<int64_t> input_size = {10, 40, 1, 1};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    auto grad_input_desc = TensorDesc({10, 40, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_big_h)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 35}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 35};
    vector<int64_t> input_size = {2, 2, 6, 35};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 35}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_scale_invalid)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 10};
    vector<int64_t> input_size = {2, 2, 6, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, case_float_normal_2)
{
    auto grad_out_desc = TensorDesc({1, 1, 20, 40}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {20, 40};
    vector<int64_t> input_size = {1, 1, 20, 40};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    auto grad_input_desc = TensorDesc({1, 1, 20, 40}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_normal_NHWC_1)
{
    auto grad_out_desc = TensorDesc({1, 20, 40, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> output_size = {20, 40};
    vector<int64_t> input_size = {1, 10, 20, 1};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    auto grad_input_desc = TensorDesc({1, 10, 20, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_normal_NHWC_2)
{
    auto grad_out_desc = TensorDesc({1, 20, 40, 1}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    vector<int64_t> output_size = {20, 40};
    vector<int64_t> input_size = {1, 10, 20, 1};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    auto grad_input_desc = TensorDesc({1, 10, 20, 1}, ACL_FLOAT16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B2_case_float_size)
{
    auto grad_out_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 10};
    vector<int64_t> input_size = {2, 2, 6, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
// 单维超过 INT32_MAX（issue #1536）：regbase(950) 上缺少上限校验，超规格 shape 下沉派发返回 561103，
// 应改为第一段直接返回 161002（参数不合法）
TEST_F(l2_upsample_nearest_2d_backward_test, ascend950_case_dim_exceed_int32_max)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    const int64_t overLimit = 2147483648L; // 2^31 > INT32_MAX
    auto grad_out_desc = TensorDesc({1, 1, 2, overLimit}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {2, overLimit};
    vector<int64_t> input_size = {1, 1, 2, overLimit};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    auto grad_input_desc = TensorDesc({1, 1, 2, overLimit}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

// 单维超过 INT32_MAX（issue #1536）：910B 上同样缺少上限校验，第一段应返回 161002
TEST_F(l2_upsample_nearest_2d_backward_test, ascend910B_case_dim_exceed_int32_max)
{
    const int64_t overLimit = 2147483648L; // 2^31 > INT32_MAX
    auto grad_out_desc = TensorDesc({1, 1, 2, overLimit}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {2, overLimit};
    vector<int64_t> input_size = {1, 1, 2, overLimit};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    auto grad_input_desc = TensorDesc({1, 1, 2, overLimit}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest2dBackward,
                        INPUT(grad_out_desc, output_size_desc, input_size_desc, scales_h, scales_w),
                        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
