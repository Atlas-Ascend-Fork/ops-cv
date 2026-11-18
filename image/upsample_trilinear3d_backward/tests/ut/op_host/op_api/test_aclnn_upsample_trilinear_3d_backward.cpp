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
#include <limits>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_upsample_trilinear_3d_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_upsample_trilinear_3d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_upsample_trilinear_3d_backward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_upsample_trilinear_3d_backward_test TearDown" << endl; }
};

// Double类型调用Aicpu不开源
// TEST_F(l2_upsample_trilinear_3d_backward_test, case_double_normal)
// {
//     auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     vector<int64_t> output_size = {3, 4, 5};
//     vector<int64_t> input_size = {2, 2, 6, 8, 10};
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto input_size_desc = IntArrayDesc(input_size);
//     bool align_corners = false;
//     const double_t scales_d = 0.0;
//     const double_t scales_h = 0.0;
//     const double_t scales_w = 0.0;
//     auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_DOUBLE, ACL_FORMAT_NCDHW);

//     auto ut = OP_API_UT(
//         aclnnUpsampleTrilinear3dBackward,
//         INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
//         OUTPUT(grad_input_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // SAMPLE: precision simulate
//     // ut.TestPrecision();
// }

TEST_F(l2_upsample_trilinear_3d_backward_test, case_float_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_float_with_scale_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = true;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_float16_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> ValidList = {ACL_UINT8, ACL_INT8,   ACL_INT16,     ACL_INT32,     ACL_INT64,
                                     ACL_BOOL,  ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    int length = ValidList.size();

    for (int i = 0; i < length; i++) {
        auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ValidList[i], ACL_FORMAT_NCDHW);
        auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ValidList[i], ACL_FORMAT_NCDHW);
        auto ut = OP_API_UT(
            aclnnUpsampleTrilinear3dBackward,
            INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
            OUTPUT(grad_input_desc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_shape_0_2_3_4_5_normal)
{
    auto grad_out_desc = TensorDesc({0, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {0, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({0, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_shape_2_0_3_4_5_normal)
{
    auto grad_out_desc = TensorDesc({2, 0, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 0, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 0, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_shape_2_2_0_4_5_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 0, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_shape_2_2_3_0_5_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 0, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_shape_2_2_3_4_0_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_2d_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_6d_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_grad_out_nullptr_abnormal)
{
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT((aclTensor*)nullptr, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_output_size_nullptr_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, (aclIntArray*)nullptr, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_input_size_nullptr_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    auto output_size_desc = IntArrayDesc(output_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, (aclIntArray*)nullptr, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_grad_input_nullptr_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_float_double_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_DOUBLE, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_output_size_3_4_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_output_size_0_4_5_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {0, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_input_size_2_2_6_8_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_input_size_2_2_0_8_10_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 0, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 0, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_nc_diff_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 3, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 3, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 3, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_dhw_diff_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 8, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 8, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_ndhwc_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    vector<int64_t> output_size = {2, 3, 4};      // D H W
    vector<int64_t> input_size = {2, 5, 4, 6, 8}; // N C D H W
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 4, 6, 8, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_scales_normal)
{
    auto grad_out_desc = TensorDesc({2, 2, 4, 8, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {4, 8, 12};
    vector<int64_t> input_size = {2, 2, 2, 4, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 2.0;
    const double_t scales_h = 2.0;
    const double_t scales_w = 1.5;
    auto grad_input_desc = TensorDesc({2, 2, 2, 4, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_scales_negative_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 4, 8, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {4, 8, 12};
    vector<int64_t> input_size = {2, 2, 2, 4, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = -1.0;
    const double_t scales_h = 2.0;
    const double_t scales_w = 1.5;
    auto grad_input_desc = TensorDesc({2, 2, 2, 4, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_scales_nan_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 4, 8, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {4, 8, 12};
    vector<int64_t> input_size = {2, 2, 2, 4, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = std::numeric_limits<double_t>::quiet_NaN();
    const double_t scales_h = 2.0;
    const double_t scales_w = 1.5;
    auto grad_input_desc = TensorDesc({2, 2, 2, 4, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_scales_inf_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 4, 8, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {4, 8, 12};
    vector<int64_t> input_size = {2, 2, 2, 4, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = std::numeric_limits<double_t>::infinity();
    const double_t scales_h = 2.0;
    const double_t scales_w = 1.5;
    auto grad_input_desc = TensorDesc({2, 2, 2, 4, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_scales_partial_zero_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 4, 8, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {4, 8, 12};
    vector<int64_t> input_size = {2, 2, 2, 4, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 2.0;
    const double_t scales_w = 1.5;
    auto grad_input_desc = TensorDesc({2, 2, 2, 4, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_trilinear_3d_backward_test, case_grad_input_format_diff_abnormal)
{
    auto grad_out_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {3, 4, 5};
    vector<int64_t> input_size = {2, 2, 6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    bool align_corners = false;
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto grad_input_desc = TensorDesc({2, 6, 8, 10, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(
        aclnnUpsampleTrilinear3dBackward,
        INPUT(grad_out_desc, output_size_desc, input_size_desc, align_corners, scales_d, scales_h, scales_w),
        OUTPUT(grad_input_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
