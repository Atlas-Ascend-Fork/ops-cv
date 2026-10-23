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
#include "../../../../op_host/op_api/aclnn_upsample_bicubic_2d_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

class l2_upsamplebicubic2d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_upsamplebicubic2d_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_upsamplebicubic2d_backward_test TearDown" << std::endl; }
};

// 支持float32
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_001)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 支持float16
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_002)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持double
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_003)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int8
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_004)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持uint8
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_005)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int32
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_006)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int64
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_007)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持int16
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_008)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// // 不支持bfloat16
// TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_009)
// {
//     const double_t scales_h = 2.0;
//     const double_t scales_w = 2.0;
//     bool align_corners = false;
//     vector<int64_t> output_size = {2, 2};
//     vector<int64_t> input_size = {1, 1, 3, 3};

//     auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto input_size_desc = IntArrayDesc(input_size);
//     auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);

//     auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
//         INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
//         OUTPUT(output_desc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
// }

// 不支持bool
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_010)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex64
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_011)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持complex128
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_012)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持inputSize的N为0
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_013)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持shape为2的输入
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_014)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持shape为5的输入
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_015)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持空指针
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_016)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(nullptr, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持输入和输出的dtype不一致
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_017)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的size大小为3
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_018)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持output的元素值小于1
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_019)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的size大小为5
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_020)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持input的元素值小于1
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_021)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// align_corners为True
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_022)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和inputsize在nc维度上不一样
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_023)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入和outputsize在hw维度上不一样
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_024)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 910A不支持bfloat16
// TEST_F(l2_upsamplebicubic2d_backward_test, ascend910A_l2_upsamplebicubic2d_backward_test_025)
// {
//     const double_t scales_h = 2.0;
//     const double_t scales_w = 2.0;
//     bool align_corners = false;
//     vector<int64_t> output_size = {2, 2};
//     vector<int64_t> input_size = {1, 1, 3, 3};

//     auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);
//     auto output_size_desc = IntArrayDesc(output_size);
//     auto input_size_desc = IntArrayDesc(input_size);
//     auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);

//     auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
//         INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
//         OUTPUT(output_desc));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
// }

// 910B支持bfloat16
TEST_F(l2_upsamplebicubic2d_backward_test, ascend910B_l2_upsamplebicubic2d_backward_test_026)
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

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_upsamplebicubic2d_backward_test, ascend910B_l2_upsamplebicubic2d_backward_test_027)
{
    const double_t scales_h = 1.0;
    const double_t scales_w = 1.0;
    bool align_corners = false;
    vector<int64_t> output_size = {3, 3};
    vector<int64_t> input_size = {1, 1, 3, 3};

    auto self_desc = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 支持float32 950
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_028)
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
    SetPlatformSocVersion(SocVersion::ASCEND950);

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 支持float32 scale>50
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_029)
{
    const double_t scales_h = 0.01;
    const double_t scales_w = 0.01;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 1, 200, 200};

    auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 1, 200, 200}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 不支持inputSize的C为0
TEST_F(l2_upsamplebicubic2d_backward_test, l2_upsamplebicubic2d_backward_test_030)
{
    const double_t scales_h = 2.0;
    const double_t scales_w = 2.0;
    bool align_corners = false;
    vector<int64_t> output_size = {2, 2};
    vector<int64_t> input_size = {1, 0, 3, 3};

    auto self_desc = TensorDesc({1, 0, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_size_desc = IntArrayDesc(output_size);
    auto input_size_desc = IntArrayDesc(input_size);
    auto output_desc = TensorDesc({1, 0, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnUpsampleBicubic2dBackward,
                        INPUT(self_desc, output_size_desc, input_size_desc, align_corners, scales_h, scales_w),
                        OUTPUT(output_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
