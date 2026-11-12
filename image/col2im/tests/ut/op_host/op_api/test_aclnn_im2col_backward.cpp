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

#include "../../../../op_api/aclnn_im2col_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_im2col_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "im2col_backward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "im2col_backward_test TearDown" << endl; }
};

TEST_F(l2_im2col_backward_test, case_null_tensor)
{
    auto tensor_desc = TensorDesc({0, 60, 56}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({0, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_empty_grad_output_non_empty_out)
{
    auto tensor_desc = TensorDesc({1, 9, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> input = {2, 2};
    vector<int64_t> kernel = {3, 3};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_im2col_backward_test, case_FLOAT)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_NOT_CONTIGUOUS)
{
    auto tensor_desc = TensorDesc({60, 56}, ACL_FLOAT, ACL_FORMAT_ND, {1, 60}, 0, {56, 60});
    auto out_desc = TensorDesc({3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_range_value)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_FLOAT16)
{
    auto tensor_desc = TensorDesc({1, 12, 12}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> input = {4, 5};
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_im2col_backward_test, ascend910B2_case_BF16)
{
    auto tensor_desc = TensorDesc({1, 12, 12}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
    vector<int64_t> input = {4, 5};
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_FLOAT16_FLOAT)
{
    auto tensor_desc = TensorDesc({1, 12, 12}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> input = {4, 5};
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_im2col_backward_test, ascend910B2_case_BF16_FLOAT)
{
    auto tensor_desc = TensorDesc({1, 12, 12}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> input = {4, 5};
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_dim2_Float)
{
    auto tensor_desc = TensorDesc({60, 56}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_NCHW)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_NHWC)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_HWCN)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_HWCN);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // // ut.TestPrecision();
}

TEST_F(l2_im2col_backward_test, case_error_output_shape)
{
    auto tensor_desc = TensorDesc({7, 9, 11}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto out_desc = TensorDesc({1, 3, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_backward_test, case_error_array_size)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> size = {10, 12};
    vector<int64_t> incorrect_size = {4, 5, 1};
    auto array_dec = IntArrayDesc(size);
    auto incorrect_array_dec = IntArrayDesc(incorrect_size);

    auto ut1 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, incorrect_array_dec, array_dec, array_dec, array_dec, array_dec),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut2 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, array_dec, incorrect_array_dec, array_dec, array_dec, array_dec),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut3 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, array_dec, array_dec, incorrect_array_dec, array_dec, array_dec),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut4 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, array_dec, array_dec, array_dec, incorrect_array_dec, array_dec),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut5 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, array_dec, array_dec, array_dec, array_dec, incorrect_array_dec),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_backward_test, case_LARGE_SHAPE)
{
    auto tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_backward_test, case_NULLPTR)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(1.0f);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(nullptr, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    auto ut1 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, nullptr, kernel_desc, dilation_desc, padding_desc, stride_desc),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    auto ut2 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, nullptr, dilation_desc, padding_desc, stride_desc),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    auto ut3 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, kernel_desc, nullptr, padding_desc, stride_desc),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    auto ut4 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, nullptr, stride_desc),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    auto ut5 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, nullptr),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    auto ut6 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                         OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut6.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_im2col_backward_test, case_DTYPE)
{
    auto tensor_desc = TensorDesc({1, 60, 56}, ACL_UINT32, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 3, 10, 12}, ACL_UINT32, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(1.0f);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_backward_test, case_error_array_value)
{
    auto tensor_desc = TensorDesc({1, 12, 12}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> input = {4, 5};
    vector<int64_t> kernel = {2, 2};
    vector<int64_t> error_kernel = {-2, 2};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> error_dilation = {-1, 1};
    vector<int64_t> padding = {0, 0};
    vector<int64_t> stride = {1, 1};
    vector<int64_t> error_stride = {1, -1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto error_kernel_desc = IntArrayDesc(error_kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto error_dilation_desc = IntArrayDesc(error_dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto error_stride_desc = IntArrayDesc(error_stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, error_kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    auto ut1 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, kernel_desc, error_dilation_desc, padding_desc, stride_desc),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    auto ut2 = OP_API_UT(aclnnIm2colBackward,
                         INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, error_stride_desc),
                         OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_im2col_backward_test, case_padding_negative)
{
    auto tensor_desc = TensorDesc({60, 56}, ACL_FLOAT, ACL_FORMAT_ND, {1, 60}, 0, {56, 60});
    auto out_desc = TensorDesc({3, 10, 12}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> input = {10, 12};
    vector<int64_t> kernel = {4, 5};
    vector<int64_t> dilation = {1, 1};
    vector<int64_t> padding = {0, -1};
    vector<int64_t> stride = {1, 1};
    auto input_desc = IntArrayDesc(input);
    auto kernel_desc = IntArrayDesc(kernel);
    auto dilation_desc = IntArrayDesc(dilation);
    auto padding_desc = IntArrayDesc(padding);
    auto stride_desc = IntArrayDesc(stride);
    auto ut = OP_API_UT(aclnnIm2colBackward,
                        INPUT(tensor_desc, input_desc, kernel_desc, dilation_desc, padding_desc, stride_desc),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
