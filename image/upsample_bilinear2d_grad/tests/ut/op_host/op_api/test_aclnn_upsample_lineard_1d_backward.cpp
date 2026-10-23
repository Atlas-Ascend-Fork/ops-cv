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
#include "../../../../op_host/op_api/aclnn_upsample_linear_1d_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

class l2_upsamplelinear1d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplelinear1d_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplelinear1d_backward_test TearDown" << std::endl; }
};

// 支持float32
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_001)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 支持float16
// TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_002)
// {
//     const double_t scales_l = 2.0;
//     bool align_corners = false;
//     vector<int64_t> output_size = {3};
//     vector<int64_t> input_size = {1, 1, 6};

//     auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto input_size_desc = IntArrayDesc(input_size);
//     auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT16, ACL_FORMAT_NCL);

//     auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
//         INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
//         OUTPUT(output_desc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
// }

// 不支持double
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_003)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_DOUBLE, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_DOUBLE, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int8
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_004)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_INT8, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_INT8, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持uint8
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_005)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_UINT8, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_UINT8, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int32
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_006)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_INT32, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_INT32, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int64
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_007)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_INT64, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_INT64, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int16
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_008)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_INT16, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_INT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 支持bfloat16
// TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_009)
// {
//     const double_t scales_l = 2.0;
//     bool align_corners = false;
//     vector<int64_t> output_size = {3};
//     vector<int64_t> input_size = {1, 1, 6};

//     auto self_desc = TensorDesc({1, 1, 3}, ACL_BF16, ACL_FORMAT_NCL);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto input_size_desc = IntArrayDesc(input_size);
//     auto output_desc = TensorDesc({1, 1, 6}, ACL_BF16, ACL_FORMAT_NCL);

//     auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
//         INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
//         OUTPUT(output_desc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
// }

// 不支持bool
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_010)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_BOOL, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_BOOL, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex64
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_011)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_COMPLEX64, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_COMPLEX64, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex128
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_012)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_COMPLEX128, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_COMPLEX128, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 支持空tensor
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_013)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {0, 1, 6};

    auto self_desc = TensorDesc({0, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({0, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 不支持shape为2的输入
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_014)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持shape为5的输入
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_015)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持空指针
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_016)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(nullptr, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持输入和输出的dtype不一致
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_017)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output_size大小为2
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_018)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output_size元素值小于1
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_019)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {0};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input_size大小为5
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_020)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 3, 3, 3};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input_size元素值小于1
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_021)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {0, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out的L维度与inputSize[2]不一致时返回161002，禁止下发kernel后设备侧挂死
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_out_L_not_match_inputSize)
{
    const double_t scales_l = 0.0;
    bool align_corners = false;
    vector<int64_t> output_size = {13};
    vector<int64_t> input_size = {161, 161, 161};

    auto self_desc = TensorDesc({161, 161, 13}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({161, 161, 13}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out的L维度大于inputSize[2]时返回161002
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_out_L_greater_than_inputSize)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 8}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out的N维度与inputSize[0]不一致时返回161002
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_out_N_not_match_inputSize)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {2, 1, 6};

    auto self_desc = TensorDesc({2, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// align_corners为True
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_022)
{
    const double_t scales_l = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {0, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和inputsize在nc维度上不一样
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_023)
{
    const double_t scales_l = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {2, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和outputsize在L维度上不一样
TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_024)
{
    const double_t scales_l = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {2};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 支持scale与size不一致
// TEST_F(l2_upsamplelinear1d_backward_test, l2_upsamplelinear1d_backward_test_025)
// {
//     const double_t scales_l = 3.0;
//     bool align_corners = false;
//     vector<int64_t> output_size = {3};
//     vector<int64_t> input_size = {1, 1, 6};

//     auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto input_size_desc = IntArrayDesc(input_size);
//     auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

//     auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
//         INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
//         OUTPUT(output_desc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
// }

// 支持float32
TEST_F(l2_upsamplelinear1d_backward_test, ascend910B3_normal_0)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 不支持500以上缩放倍数
TEST_F(l2_upsamplelinear1d_backward_test, ascend910B3_scale501)
{
    const double_t scales_l = 550.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3300};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3300}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
// 不支持500以上缩放倍数
TEST_F(l2_upsamplelinear1d_backward_test, ascend910B3_scale400)
{
    const double_t scales_l = 400.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2400};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 2400}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持500以上缩放倍数
TEST_F(l2_upsamplelinear1d_backward_test, ascend910B3_checkNC)
{
    const double_t scales_l = 400.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2400};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 2, 2400}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplelinear1d_backward_test, Ascend95089_l2_upsamplelinear1d_backward_test_001)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCL);

    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

TEST_F(l2_upsamplelinear1d_backward_test, Ascend95089_l2_upsamplelinear1d_backward_test_002)
{
    const double_t scales_l = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3};
    vector<int64_t> input_size = {1, 1, 6};

    auto self_desc = TensorDesc({1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleLinear1dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_l),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}
