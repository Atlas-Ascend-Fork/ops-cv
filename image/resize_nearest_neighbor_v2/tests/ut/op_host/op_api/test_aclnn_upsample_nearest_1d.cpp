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
#include "../../../../op_api/aclnn_upsample_nearest_1d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_upsamplenearest1d_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplenearest1d_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplenearest1d_test TearDown" << std::endl; }
};

// 正常路径，float32 NCL
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_normal_float32)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(24));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{8});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float16 NCL
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_normal_float16)
{
    auto selfDesc = TensorDesc({2, 2, 5}, ACL_FLOAT16, ACL_FORMAT_NCL).Value(vector<float>(20, 1.0));
    auto outDesc = TensorDesc({2, 2, 10}, ACL_FLOAT16, ACL_FORMAT_NCL).Value(vector<float>(40));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{10});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// out 最后一维与 outputSize[0] 不一致，第一段应返回 161002（issue #1398）
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_out_dim_not_match_output_size)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(24));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{7});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// error-scenario 变异：outputSize 与 out 一致但 self 尺寸不同，仍应被末维校验拦住
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_output_size_not_match_out)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 7}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(21));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{8});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out 不是 3 维，第一段应返回 161002
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_out_dimnum_invalid)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(8));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{8});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 多元素 outputSize，第一段应返回 161002（issue #1400 现象一）
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_output_size_multi_elements)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 7}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(21));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{7, 9});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空 outputSize，第一段应返回 161002
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_output_size_empty)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(24));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// outputSize[0] 为 0，第一段应返回 161002
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_output_size_zero)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(24));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{0});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self nullptr
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_self_nullptr)
{
    auto outDesc = TensorDesc({1, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(24));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{8});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(nullptr, sizeDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// outputSize nullptr
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_output_size_nullptr)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({1, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(24));

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, nullptr), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_out_nullptr)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto sizeDesc = IntArrayDesc(vector<int64_t>{8});

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, sizeDesc), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out 的 N 维与 self 不一致应返回 161002（N/C 维一致性校验，与 2D CheckNCDimValid 对齐）
TEST_F(l2_upsamplenearest1d_test, l2_upsamplenearest1d_test_out_nc_dim_not_match)
{
    auto selfDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(12, 1.0));
    auto outDesc = TensorDesc({2, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCL).Value(vector<float>(48));
    vector<int64_t> outputSize = {8};
    auto outputSizeDesc = IntArrayDesc(outputSize);

    auto ut = OP_API_UT(aclnnUpsampleNearest1d, INPUT(selfDesc, outputSizeDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
