/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_upsample_nearest_2d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsamplenearest2d_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplenearest2d_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplenearest2d_test TearDown" << std::endl; }
};

// 正常路径，float32 NCHW
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_normal_float32)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float16 NHWC（shape 为 NHWC 语义：N=1,H=3,W=2,C=2 -> N=1,H=3,W=4,C=2）
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_normal_float16_nhwc)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).Value(vector<float>(24));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{3, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// N=0 空输入（issue #1401 实测 self=fp32[0,1,2,2] 曾静默返回成功），第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_self_dim_n_zero)
{
    auto selfDesc = TensorDesc({0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(0));
    auto outDesc = TensorDesc({0, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(0));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// C=0 空输入，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_self_dim_c_zero)
{
    auto selfDesc = TensorDesc({1, 0, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(0));
    auto outDesc = TensorDesc({1, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(0));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// H=0 空输入，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_self_dim_h_zero)
{
    auto selfDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(0));
    auto outDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// W=0 空输入，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_self_dim_w_zero)
{
    auto selfDesc = TensorDesc({1, 1, 2, 0}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(0));
    auto outDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self 不是 4 维，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_self_dimnum_invalid)
{
    auto selfDesc = TensorDesc({1, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(6, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// outputSize 中存在 0 元素，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_output_size_zero)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 0});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// outputSize 不是 2 个元素，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_output_size_multi_elements)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out 的 H/W 与 outputSize 不一致（NCHW），第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_out_dim_not_match_output_size)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{3, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out 的 N/C 与 self 不一致，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_out_nc_dim_not_match)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({2, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(96));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self 与 out 数据类型不同，第一段应返回 161002
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_dtype_not_match)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self nullptr，第一段应返回 161001
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_self_nullptr)
{
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(nullptr, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// outputSize nullptr，第一段应返回 161001
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_output_size_nullptr)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(48));

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, nullptr), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr，第一段应返回 161001
TEST_F(l2_upsamplenearest2d_test, l2_upsamplenearest2d_test_out_nullptr)
{
    auto selfDesc = TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(12, 1.0));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{4, 4});

    auto ut = OP_API_UT(aclnnUpsampleNearest2d, INPUT(selfDesc, sizeDesc), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}
