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
#include "../../../../op_api/aclnn_upsample_bilinear_2d_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

class l2_upsamplebilinear2d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplebilinear2d_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplebilinear2d_backward_test TearDown" << std::endl; }
};

// 支持float32
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_001)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 支持float16
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_002)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持double
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_003)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int8
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_004)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT8, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持uint8
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_005)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_UINT8, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int32
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_006)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT32, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int64
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_007)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int16
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_008)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_INT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持bfloat16
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_009)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持bool
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_010)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BOOL, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex64
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_011)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex128
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_012)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 支持空tensor
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_013)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {0, 1, 3, 3};

    auto self_desc = TensorDesc({0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    // ut.TestPrecision();
}

// 不支持shape为2的输入
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_014)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持shape为5的输入
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_015)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持空指针
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_016)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(nullptr, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持输入和输出的dtype不一致
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_017)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的size大小为3
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_018)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的元素值小于1
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_019)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {0, 2};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的size大小为5
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_020)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 3, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的元素值小于1
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_021)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {0, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// align_corners为True
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_022)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {0, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和inputsize在nc维度上不一样
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_023)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {2, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和outputsize在hw维度上不一样
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_024)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {2, 4};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_025)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = true;
    vector<int64_t> output_size = {2, 4};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// ====== 950(RegBase)平台：同尺寸 + gradOut为FLOAT32 + out为FLOAT16/BFLOAT16 ======
// 资料约定（npu="950"）：gradOut为FLOAT时out可为FLOAT32/FLOAT16/BFLOAT16，该规则与是否同尺寸无关

// 950平台：同尺寸，fp32 gradOut + fp16 out 应放行（issue #1531 修复目标）
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_026_regbase_same_size_fp32_to_fp16)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：同尺寸，fp32 gradOut + bf16 out 应放行（issue #1531 修复目标）
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_027_regbase_same_size_fp32_to_bf16)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：同尺寸，fp32 gradOut + fp32 out 对照应放行
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_028_regbase_same_size_fp32_to_fp32)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：非同尺寸，fp32 gradOut + fp16 out 应回归放行
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_029_regbase_diff_size_fp32_to_fp16)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {8, 8};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：非同尺寸，fp32 gradOut + bf16 out 应回归放行
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_030_regbase_diff_size_fp32_to_bf16)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {8, 8};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：同尺寸，fp16 gradOut + fp16 out 应回归放行
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_031_regbase_same_size_fp16_to_fp16)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：同尺寸，bf16 gradOut + bf16 out 应回归放行
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_032_regbase_same_size_bf16_to_bf16)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：同尺寸，fp16 gradOut + fp32 out 仍应拒绝（gradOut非FLOAT时out须与gradOut一致）
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_033_regbase_same_size_fp16_to_fp32)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 950平台：同尺寸，int32 gradOut + int32 out 仍应拒绝（gradOut不在支持列表内）
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_034_regbase_same_size_int32_to_int32)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_INT32, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_INT32, ACL_FORMAT_NCHW);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// 910B平台：同尺寸，fp32 gradOut + fp16 out 仍应拒绝（非RegBase平台out须与gradOut一致）
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_035_910b_same_size_fp32_to_fp16)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 1, 4, 4};

    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 950平台：同尺寸NHWC，fp32 gradOut + fp16 out 应放行
TEST_F(l2_upsamplebilinear2d_backward_test, l2_upsamplebilinear2d_backward_test_036_regbase_same_size_nhwc_fp32_to_fp16)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {4, 4};
    vector<int64_t> input_size = {1, 4, 4, 2};

    auto self_desc = TensorDesc({1, 4, 4, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 4, 4, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC);

    SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnUpsampleBilinear2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}
