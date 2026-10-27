/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_roi_align_v2_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

class l2_roi_align_v2_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "roi_align_v2_backward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "roi_align_v2_backward_test TearDown" << endl; }
};

TEST_F(l2_roi_align_v2_backward_test, temp_case)
{
    auto gradOutput = TensorDesc({524, 256, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({524, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 256, 40, 40};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 256, 40, 40}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 7, 7, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针
TEST_F(l2_roi_align_v2_backward_test, case_nullptr)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut0 = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(nullptr, boxes, inputShape, 3, 3, 0.25f, 2, false),
                         OUTPUT(gradInput));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, nullptr, inputShape, 3, 3, 0.25f, 2, false),
                         OUTPUT(gradInput));
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRoiAlignV2Backward,
                         INPUT(gradOutput, boxes, (aclIntArray*)nullptr, 3, 3, 0.25f, 2, false), OUTPUT(gradInput));
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                         OUTPUT(nullptr));
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非法dtype场景0
TEST_F(l2_roi_align_v2_backward_test, case_dtype_invalid_0)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_BF16, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景1
TEST_F(l2_roi_align_v2_backward_test, case_dtype_invalid_1)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景2
TEST_F(l2_roi_align_v2_backward_test, case_dtype_invalid_2)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景0
TEST_F(l2_roi_align_v2_backward_test, case_format_invalid_0)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景1
TEST_F(l2_roi_align_v2_backward_test, case_format_invalid_1)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景2
TEST_F(l2_roi_align_v2_backward_test, case_format_invalid_2)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景0
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_0)
{
    auto gradOutput = TensorDesc({1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景1
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_1)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景2
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_2)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景3
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_3)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景4
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_4)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景5
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_5)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景6
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_6)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 7};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景7
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_7)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 4, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景8
TEST_F(l2_roi_align_v2_backward_test, case_shape_invalid_8)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 4, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景0
TEST_F(l2_roi_align_v2_backward_test, case_attr_invalid_0)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, -2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景1
TEST_F(l2_roi_align_v2_backward_test, case_attr_invalid_1)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.0f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_roi_align_v2_backward_test, case_empty_tensor)
{
    auto gradOutput = TensorDesc({1, 1, 0, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.0, -2.0, -2.0, 22.0, 22.0});
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 0, 3, 0.25f, 2, true),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float场景（含精度）
TEST_F(l2_roi_align_v2_backward_test, case_float)
{
    auto gradOutput = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW)
                          .Value(vector<float>{4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5});
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.0, -2.0, -2.0, 22.0, 22.0});
    vector<int64_t> inputShape_dims = {1, 1, 6, 6};
    auto inputShape = IntArrayDesc(inputShape_dims);
    auto gradInput = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2Backward, INPUT(gradOutput, boxes, inputShape, 3, 3, 0.25f, 2, false),
                        OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}
