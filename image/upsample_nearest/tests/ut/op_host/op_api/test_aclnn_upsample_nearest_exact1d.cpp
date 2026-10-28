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

#include "../../../../op_host/op_api/aclnn_upsample_nearest_exact1d.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_upsample_nearest_exact1d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "upsample_nearest_exact1d_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "upsample_nearest_exact1d_test TearDown" << endl; }
};

TEST_F(l2_upsample_nearest_exact1d_test, case_float_normal)
{
    auto self_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 2, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_float_out_shape_fail)
{
    auto self_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {0};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_float16_normal)
{
    auto self_desc = TensorDesc({1, 6, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_bfloat16_normal)
{
    auto self_desc = TensorDesc({1, 6, 3}, ACL_BF16, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 6, 6}, ACL_BF16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_shape_6d_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_self_nullptr_abnormal)
{
    auto self_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 2, 6}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT((aclTensor*)nullptr, output_size_desc, scales),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_uint8_ncl_normal)
{
    auto self_desc = TensorDesc({1, 4, 3}, ACL_UINT8, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 4, 6}, ACL_UINT8, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_uint8_nd_normal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_UINT8, ACL_FORMAT_ND);
    vector<int64_t> output_size = {8};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({2, 3, 8}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_uint8_minimal_shape)
{
    auto self_desc = TensorDesc({1, 1, 1}, ACL_UINT8, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {4};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 1, 4}, ACL_UINT8, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_uint8_downscale)
{
    auto self_desc = TensorDesc({1, 8, 16}, ACL_UINT8, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {8};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 8, 8}, ACL_UINT8, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_uint8_dtype_mismatch)
{
    auto self_desc = TensorDesc({1, 2, 3}, ACL_UINT8, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 2, 6}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact1d_test, case_uint8_output_size_invalid)
{
    auto self_desc = TensorDesc({1, 2, 3}, ACL_UINT8, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 2, 6}, ACL_UINT8, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// scales 与 outputSize 不一致应返回 161002（floor(4*1.0)=4 != 7）
TEST_F(l2_upsample_nearest_exact1d_test, case_scale_conflict_with_outputsize_fail)
{
    auto self_desc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {7};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 1.0;
    auto out_desc = TensorDesc({1, 1, 7}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out 的 L 维与 outputSize 不一致应返回 161002（不再落到设备侧 561103）
TEST_F(l2_upsample_nearest_exact1d_test, case_out_last_dim_not_match_fail)
{
    auto self_desc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {8};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 0.0;
    auto out_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scales>0 且与 outputSize 自洽时应返回 SUCCESS（容差放行侧回归：floor(4*2.0)=8）
TEST_F(l2_upsample_nearest_exact1d_test, case_scale_positive_consistent_success)
{
    auto self_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {8};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 2.0;
    auto out_desc = TensorDesc({1, 2, 8}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// outputSize 与 floor(inputL*scales) 恰差 1 应返回 161002（off-by-one 不被容差窗口放过）
TEST_F(l2_upsample_nearest_exact1d_test, case_scale_off_by_one_fail)
{
    auto self_desc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {7};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 2.0;
    auto out_desc = TensorDesc({1, 1, 7}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非整数乘积时 outputSize 应取 floor 而非 ceil：floor(4*1.5)=6 合法应返回 SUCCESS
TEST_F(l2_upsample_nearest_exact1d_test, case_scale_noninteger_floor_success)
{
    auto self_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 1.5;
    auto out_desc = TensorDesc({1, 2, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非整数乘积时误传 ceil 值（floor(4*1.5)=6 而非 7）应返回 161002
TEST_F(l2_upsample_nearest_exact1d_test, case_scale_noninteger_ceil_fail)
{
    auto self_desc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    vector<int64_t> output_size = {7};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scales = 1.5;
    auto out_desc = TensorDesc({1, 1, 7}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact1d, INPUT(self_desc, output_size_desc, scales), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
