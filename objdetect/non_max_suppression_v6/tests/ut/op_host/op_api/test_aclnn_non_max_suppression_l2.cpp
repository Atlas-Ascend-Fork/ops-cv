/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_host/op_api/aclnn_non_max_suppression.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_non_max_suppression_v6_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_non_max_suppression_v6_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_non_max_suppression_v6_test TearDown" << std::endl; }
};

// 检查入参是否为nullptr
TEST_F(l2_non_max_suppression_v6_test, case_boxes_nullptr_1)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut0 = OP_API_UT(aclnnNonMaxSuppression,
                         INPUT(nullptr, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                         OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_non_max_suppression_v6_test, case_boxes_nullptr_2)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut1 = OP_API_UT(aclnnNonMaxSuppression,
                         INPUT(boxes, nullptr, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                         OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_non_max_suppression_v6_test, case_boxes_nullptr_3)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut2 = OP_API_UT(aclnnNonMaxSuppression,
                         INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                         OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_non_max_suppression_v6_test, case_iou_threshold_nullptr)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, nullptr, scoreThreshold, centerPointBox),
                        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_non_max_suppression_v6_test, case_score_threshold_nullptr)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, nullptr, centerPointBox),
                        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_non_max_suppression_v6_test, case_iou_threshold_empty)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_score_threshold_empty)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 检查输入和输出的 type是否在算子的支持列表内
TEST_F(l2_non_max_suppression_v6_test, case_invalid_type_1)
{
    auto boxes_invalid = TensorDesc({1, 6, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut0 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes_invalid, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_invalid_type_2)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores_invalid = TensorDesc({1, 1, 6}, ACL_INT64, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut1 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes, scores_invalid, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 检查输入的数据类型是否满足约束
TEST_F(l2_non_max_suppression_v6_test, case_invalid_type_3)
{
    auto boxes_invalid_fp16 = TensorDesc({1, 6, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scores_invalid_fp32 = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut0 = OP_API_UT(aclnnNonMaxSuppression,
                         INPUT(boxes_invalid_fp16, scores_invalid_fp32, maxOutputBoxesPerClass, iouThreshold,
                               scoreThreshold, centerPointBox),
                         OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 检查输入和输出的 format是否在算子的支持列表内
TEST_F(l2_non_max_suppression_v6_test, case_invalid_format_1)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores_invalid = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut1 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes, scores_invalid, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 检查输入和输出的 shape是否在算子的支持列表内
TEST_F(l2_non_max_suppression_v6_test, case_invalid_shape_1)
{
    auto boxes_invalid = TensorDesc({1, 6, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut0 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes_invalid, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_invalid_shape_2)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores_invalid = TensorDesc({1, 1, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut1 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes, scores_invalid, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_invalid_shape_3)
{
    auto boxes_invalid = TensorDesc({2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores_invalid = TensorDesc({1, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut2 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes_invalid, scores_invalid, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_invalid_shape_4)
{
    auto boxes_invalid = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores_invalid = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut3 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes_invalid, scores_invalid, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_invalid_shape_5)
{
    auto boxes_invalid = TensorDesc({1, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut4 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes_invalid, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 检查非法attr
TEST_F(l2_non_max_suppression_v6_test, case_invalid_attr_1)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox_invalid = 2;

    auto ut0 = OP_API_UT(
        aclnnNonMaxSuppression,
        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox_invalid),
        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_non_max_suppression_v6_test, case_max_output_list_size_invalid)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{1, 1});
    auto iouThreshold = FloatArrayDesc(vector<float>{1});
    auto scoreThreshold = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, 0), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// maxOutputBoxesPerClass 大于700 拦截
TEST_F(l2_non_max_suppression_v6_test, case_maxoutput_invalid)
{
    auto boxes = TensorDesc({1, 6000, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6000}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{800});
    auto iouThreshold = FloatArrayDesc(vector<float>{0.5});
    auto scoreThreshold = FloatArrayDesc(vector<float>{0.0});
    auto out = TensorDesc({800, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut0 = OP_API_UT(aclnnNonMaxSuppression,
                         INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                         OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// maxOutputBoxesPerClass 必须为单元素数组
TEST_F(l2_non_max_suppression_v6_test, case_maxoutput_size_invalid)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{3, 999});
    auto iouThreshold = FloatArrayDesc(vector<float>{0.5});
    auto scoreThreshold = FloatArrayDesc(vector<float>{0.0});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 检查精度
TEST_F(l2_non_max_suppression_v6_test, case_normal_1)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.5, 0.5,  1.0, 1.0,   0.5, 0.6,
                                                                                     1.0, 1.0,  0.5, 0.4,   1.0, 1.0,
                                                                                     0.5, 10.5, 1.0, 1.0,   0.5, 10.6,
                                                                                     1.0, 1.0,  0.5, 100.5, 1.0, 1.0});
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{3});
    auto iouThreshold = FloatArrayDesc(vector<float>{0.5});
    auto scoreThreshold = FloatArrayDesc(vector<float>{0.0});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;

    auto ut0 = OP_API_UT(aclnnNonMaxSuppression,
                         INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                         OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
        EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
    } else {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_non_max_suppression_v6_test, ascend910b_case_soc_not_supported)
{
    auto boxes = TensorDesc({1, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scores = TensorDesc({1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOutputBoxesPerClass = IntArrayDesc(vector<int64_t>{3});
    auto iouThreshold = FloatArrayDesc(vector<float>{0.5});
    auto scoreThreshold = FloatArrayDesc(vector<float>{0.0});
    auto out = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto centerPointBox = 0;
    SetPlatformSocVersion(SocVersion::ASCEND910B);
    auto ut = OP_API_UT(aclnnNonMaxSuppression,
                        INPUT(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox),
                        OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
}
