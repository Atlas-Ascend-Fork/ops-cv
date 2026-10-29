/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_upsample_nearest_3d.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_upsample_nearest_3d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "upsample_nearest_3d_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "upsample_nearest_3d_test TearDown" << endl; }
};

TEST_F(l2_upsample_nearest_3d_test, case_float_normal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_nearest_3d_test, case_float_out_shape_fail)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    // out shape expect (2,2,6,8,10)
    auto out_desc = TensorDesc({2, 6, 8, 10, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_float16_normal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// TEST_F(l2_upsample_nearest_3d_test, case_double_normal)
// {
//     auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
//     vector<int64_t> output_size = {6, 8, 10};
//     auto output_size_desc = IntArrayDesc(output_size);
//     const double_t scales_d = 0.0;
//     const double_t scales_h = 0.0;
//     const double_t scales_w = 0.0;
//     auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_DOUBLE, ACL_FORMAT_NCDHW);

//     auto ut = OP_API_UT(
//         aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w), OUTPUT(out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // SAMPLE: precision simulate
//     ut.TestPrecision();
// }

TEST_F(l2_upsample_nearest_3d_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> ValidList = {ACL_INT8, ACL_INT16,  ACL_INT32,     ACL_INT64,
                                     ACL_BOOL, ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    int length = ValidList.size();

    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ValidList[i], ACL_FORMAT_NCDHW);
        auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ValidList[i], ACL_FORMAT_NCDHW);
        auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                            OUTPUT(out_desc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_upsample_nearest_3d_test, case_shape_0_2_3_4_5_normal)
{
    auto self_desc = TensorDesc({0, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({0, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_3d_test, case_shape_2_0_3_4_5_abnormal)
{
    auto self_desc = TensorDesc({2, 0, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 0, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_shape_2_2_0_4_5_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 0, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_shape_3d_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_shape_6d_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_self_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d,
                        INPUT((aclTensor*)nullptr, output_size_desc, scales_d, scales_h, scales_w), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_3d_test, case_output_size_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, (aclIntArray*)nullptr, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_3d_test, case_output_size_empty_array_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_out_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_3d_test, case_float_float16_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_output_size_6_8_10_6_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_output_size_element_0_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 0};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_3d_test, case_NDHWC_normal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    vector<int64_t> output_size = {6, 8, 10}; // D H W
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 6, 8, 10, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_nearest_3d_test, case_scales_normal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {4, 8, 12};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 1.5;
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.5;
    auto out_desc = TensorDesc({2, 2, 4, 8, 12}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_upsample_nearest_3d_test, ascend310P_case_NCDHW_normal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales_d = 0.0;
    const double_t scales_h = 0.0;
    const double_t scales_w = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnUpsampleNearest3d, INPUT(self_desc, output_size_desc, scales_d, scales_h, scales_w),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
