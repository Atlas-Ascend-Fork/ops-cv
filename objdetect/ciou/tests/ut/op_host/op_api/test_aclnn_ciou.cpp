/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_api/aclnn_ciou.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_ciou_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_ciou_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_ciou_test TearDown" << endl; }
};

TEST_F(l2_ciou_test, ascend910_95_success_case_0)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(l2_ciou_test, ascend910_95_success_case_1)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空指针
TEST_F(l2_ciou_test, ascend910_95_case_nullptr_bboxes)
{
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT((aclTensor*)nullptr, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_ciou_test, ascend910_95_case_nullptr_gtboxes)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, (aclTensor*)nullptr, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_ciou_test, ascend910_95_case_nullptr_overlap)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT((aclTensor*)nullptr, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_ciou_test, ascend910_95_case_nullptr_atansub)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非法dtype场景0
TEST_F(l2_ciou_test, ascend910_95_case_dtype_invalid_0)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_INT64, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_INT64, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景1
TEST_F(l2_ciou_test, ascend910_95_case_dtype_invalid_1)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景2
TEST_F(l2_ciou_test, ascend910_95_case_dtype_invalid_2)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景3
TEST_F(l2_ciou_test, ascend910_95_case_dtype_invalid_3)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景0
TEST_F(l2_ciou_test, ascend910_95_case_format_invalid_0)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景1
TEST_F(l2_ciou_test, ascend910_95_case_format_invalid_1)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景2
TEST_F(l2_ciou_test, ascend910_95_case_format_invalid_2)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景3
TEST_F(l2_ciou_test, ascend910_95_case_format_invalid_3)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_NCHW);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景0
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_0)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景1
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_1)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景2
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_2)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({5, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景3
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_3)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景4
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_4)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景5
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_5)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({2, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景6
TEST_F(l2_ciou_test, ascend910_95_case_shape_invalid_6)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景0
TEST_F(l2_ciou_test, ascend910_95_case_attr_invalid_0)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, true, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景1
TEST_F(l2_ciou_test, ascend910_95_case_attr_invalid_1)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, true, "iou"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景3
TEST_F(l2_ciou_test, ascend910_95_case_attr_invalid_2)
{
    auto bBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gtBoxes = TensorDesc({4, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto overlap = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto atanSub = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    SetPlatformSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnCIoU, INPUT(bBoxes, gtBoxes, false, false, "iouf"), OUTPUT(overlap, atanSub));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
