/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_upsample_nearest_1d_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

class l2_upsamplenearest1dv2_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplenearest1dv2_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplenearest1dv2_test TearDown" << std::endl; }
};

// input nullptr
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_input_nullptr)
{
    auto outDesc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(nullptr, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// output nullptr
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_output_nullptr)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// output size nullptr
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_output_size_nullptr)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, nullptr, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// input empty
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_input_empty_0)
{
    auto selfDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({0, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// input empty
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_input_empty_1)
{
    auto selfDesc = TensorDesc({1, 0, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 0, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// size 0
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_size_0)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 0}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({0});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid size
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_size_valid)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// outputSize must contain exactly one element
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_output_size_extra_element)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc(std::vector<int64_t>{6, 99});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid dtype int32
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_dtype_int32)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_INT32, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_INT32, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid dtype int64
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_dtype_int64)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_INT64, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_INT64, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid shape 2
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_shape_2)
{
    auto selfDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// size 256
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_size_256)
{
    auto selfDesc = TensorDesc({1, 3, 64}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 3, 256}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({256});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype float16, aicore
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_dtype_float16)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype bfloat16, aicore
TEST_F(l2_upsamplenearest1dv2_test, ascend910B2_l2_upsamplenearest1dv2_test_dtype_float16)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_BF16, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_BF16, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype float32, aicore
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_dtype_float32)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype double, aicpu
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_dtype_double)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_DOUBLE, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_DOUBLE, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

// dtype uint8, aicpu
// TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_dtype_uint8)
// {
//     auto selfDesc = TensorDesc({1, 1, 2}, ACL_UINT8, ACL_FORMAT_NCL);
//     auto outDesc = TensorDesc({1, 1, 4}, ACL_UINT8, ACL_FORMAT_NCL);
//     auto sizeDesc = IntArrayDesc({4});
//     float scaleL = -99.0;
//     auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
// }

// different format
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_different_format)
{
    auto selfDesc = TensorDesc({1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 3, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// invalid format
TEST_F(l2_upsamplenearest1dv2_test, l2_upsamplenearest1dv2_test_invalid_format)
{
    auto selfDesc = TensorDesc({1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sizeDesc = IntArrayDesc({6});
    float scaleL = -99.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplenearest1dv2_test, ascend910B4_case_float_scale_01)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = 2.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest1dv2_test, ascend910B4_case_float_scale_02)
{
    auto selfDesc = TensorDesc({1, 1, 36}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 72}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({72});
    float scaleL = 2.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest1dv2_test, ascend910B4_case_float_scale_03)
{
    auto selfDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({2});
    float scaleL = 0.5;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest1dv2_test, Ascend950DT_9591_case_float_scale_01)
{
    auto selfDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({4});
    float scaleL = 2.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest1dv2_test, Ascend950DT_9591_case_float_scale_02)
{
    auto selfDesc = TensorDesc({1, 1, 36}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 72}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({72});
    float scaleL = 2.0;
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplenearest1dv2_test, Ascend950DT_9591_case_float_scale_03)
{
    auto selfDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto outDesc = TensorDesc({1, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto sizeDesc = IntArrayDesc({2});
    float scaleL = 0.5;
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleNearest1dV2, INPUT(selfDesc, sizeDesc, scaleL), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}
