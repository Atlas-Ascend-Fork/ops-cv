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

#include "../../../../op_host/op_api/aclnn_upsample_nearest_exact2d.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_upsample_nearest_exact2d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "upsample_nearest_exact2d_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "upsample_nearest_exact2d_test TearDown" << endl; }
};

TEST_F(l2_upsample_nearest_exact2d_test, case_float_normal)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_float_out_shape_success)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 0};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 0, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_float16_normal)
{
    auto self_desc = TensorDesc({1, 6, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_bfloat16_normal)
{
    auto self_desc = TensorDesc({1, 6, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 6}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_shape_6d_abnormal)
{
    auto self_desc = TensorDesc({2, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 8, 10};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({2, 2, 6, 8, 10}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_self_nullptr_abnormal)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT((aclTensor*)nullptr, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_float_nhwc)
{
    auto self_desc = TensorDesc({1, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_nchw_normal)
{
    auto self_desc = TensorDesc({1, 4, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 4, 6, 6}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_nhwc_normal)
{
    auto self_desc = TensorDesc({1, 3, 3, 4}, ACL_UINT8, ACL_FORMAT_NHWC);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 4}, ACL_UINT8, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_downscale)
{
    auto self_desc = TensorDesc({1, 4, 8, 8}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {4, 4};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 4, 4, 4}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_1x1_to_large)
{
    auto self_desc = TensorDesc({1, 4, 1, 1}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {8, 8};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 4, 8, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_empty_tensor)
{
    auto self_desc = TensorDesc({0, 4, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({0, 4, 6, 6}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // N=0 的空 Tensor：参数校验通过后走快速返回 → SUCCESS + workspace_size == 0
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspace_size, 0u);
}

// 空 Tensor 的非法参数（N=0 但 scales 与 outputSize 冲突）应返回 161002：
// 校验先于空 Tensor 早退，避免空输入静默吞掉参数错误
TEST_F(l2_upsample_nearest_exact2d_test, case_empty_tensor_invalid_scale_fail)
{
    auto self_desc = TensorDesc({0, 4, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 3.0;
    const double_t scalesW = 3.0;
    auto out_desc = TensorDesc({0, 4, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // floor(3*3.0)=9 != 6，N=0 空 Tensor 的非法 scales 也应被校验拦截
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_dtype_mismatch)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_format_mismatch)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 6, 6, 2}, ACL_UINT8, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsample_nearest_exact2d_test, case_uint8_output_size_invalid)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scalesH 与 outputSize 不一致应返回 161002（floor(3*1.0)=3 != 6）
TEST_F(l2_upsample_nearest_exact2d_test, case_scaleH_conflict_with_outputsize_fail)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 1.0;
    const double_t scalesW = 1.0;
    auto out_desc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out 的 H 维与 outputSize 不一致应返回 161002（不再落到设备侧 561103）
TEST_F(l2_upsample_nearest_exact2d_test, case_out_h_dim_not_match_fail)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {6, 6};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 0.0;
    const double_t scalesW = 0.0;
    auto out_desc = TensorDesc({1, 2, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非整数乘积时 outputSize 应取 floor 而非 ceil：floor(3*1.5)=4/floor(3*2.5)=7 合法应返回 SUCCESS
TEST_F(l2_upsample_nearest_exact2d_test, case_scale_noninteger_floor_success)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {4, 7};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 1.5;
    const double_t scalesW = 2.5;
    auto out_desc = TensorDesc({1, 2, 4, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非整数乘积时误传 ceil 值（floor(3*1.5)=4 而非 5）应返回 161002
TEST_F(l2_upsample_nearest_exact2d_test, case_scale_noninteger_ceil_fail)
{
    auto self_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    vector<int64_t> output_size = {5, 7};
    auto output_size_desc = IntArrayDesc(output_size);
    const double_t scalesH = 1.5;
    const double_t scalesW = 2.5;
    auto out_desc = TensorDesc({1, 2, 5, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleNearestExact2d, INPUT(self_desc, output_size_desc, scalesH, scalesW),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
