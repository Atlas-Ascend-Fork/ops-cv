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
#include <float.h>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../../op_api/aclnn_iou.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_iou_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_iou_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_iou_test TearDown" << endl; }
};

TEST_F(l2_iou_test, success_case_1)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_2)
{
    auto bBoxes = TensorDesc({30, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({30, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({30, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_3)
{
    auto bBoxes = TensorDesc({1024, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({60, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1024, 60}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_4)
{
    auto bBoxes = TensorDesc({1025, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({61, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1025, 61}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_5)
{
    auto bBoxes = TensorDesc({16, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({16, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({16, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_6)
{
    auto bBoxes = TensorDesc({10, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({10, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({10, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_7)
{
    auto bBoxes = TensorDesc({101, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({201, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({101, 201}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.01, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_iou_test, success_case_8)
{
    auto bBoxes = TensorDesc({100, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({200, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({100, 200}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空指针
TEST_F(l2_iou_test, case_nullptr_bboxes)
{
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT((aclTensor*)nullptr, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_iou_test, case_nullptr_gtboxes)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, (aclTensor*)nullptr, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_iou_test, case_nullptr_overlap)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非法dtype场景0
TEST_F(l2_iou_test, case_dtype_invalid_0)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景1
TEST_F(l2_iou_test, case_dtype_invalid_1)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景2
TEST_F(l2_iou_test, case_dtype_invalid_2)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景3
TEST_F(l2_iou_test, case_dtype_invalid_3)
{
    auto bBoxes = TensorDesc({50, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({30, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({50, 30}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景0
TEST_F(l2_iou_test, case_format_invalid_0)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景1
TEST_F(l2_iou_test, case_format_invalid_1)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景2
TEST_F(l2_iou_test, case_format_invalid_2)
{
    auto bBoxes = TensorDesc({40, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({60, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({40, 60}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景3
TEST_F(l2_iou_test, case_format_invalid_3)
{
    auto bBoxes = TensorDesc({40, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gtBoxes = TensorDesc({60, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto overlap = TensorDesc({40, 60}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景0
TEST_F(l2_iou_test, case_shape_invalid_0)
{
    auto bBoxes = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景1
TEST_F(l2_iou_test, case_shape_invalid_1)
{
    auto bBoxes = TensorDesc({20, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景2
TEST_F(l2_iou_test, case_shape_invalid_2)
{
    auto bBoxes = TensorDesc({21, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景3
TEST_F(l2_iou_test, case_shape_invalid_3)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_iou_test, case_aligned_output_first_dim_invalid)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({21, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_iou_test, case_aligned_output_shape_valid)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非法shape场景4
TEST_F(l2_iou_test, case_shape_invalid_4)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景5
TEST_F(l2_iou_test, case_shape_invalid_5)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 20}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景6
TEST_F(l2_iou_test, case_shape_invalid_6)
{
    auto bBoxes = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({40, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 40}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景7
TEST_F(l2_iou_test, case_shape_invalid_7)
{
    auto bBoxes = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景8
TEST_F(l2_iou_test, case_shape_invalid_8)
{
    auto bBoxes = TensorDesc({4, 100}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 100}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({100, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", 0.0, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景0
TEST_F(l2_iou_test, case_attr_invalid_0)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", -0.1, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景1
TEST_F(l2_iou_test, case_attr_invalid_1)
{
    auto bBoxes = TensorDesc({40, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({50, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({40, 50}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iof", -1.0, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景2
TEST_F(l2_iou_test, case_attr_invalid_2)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "i", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景3
TEST_F(l2_iou_test, case_attr_invalid_3)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({30, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 30}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iouf", 0.01, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_iou_test, case_valid_0)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({30, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 30}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, false), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_iou_test, case_valid_1)
{
    auto bBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({20, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIou, INPUT(bBoxes, gtBoxes, "iou", 0.01, true), OUTPUT(overlap));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ;
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
