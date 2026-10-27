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
#include <limits>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_roi_align_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

class l2_roi_align_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "roi_align_v2_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "roi_align_v2_test TearDown" << endl; }
};

TEST_F(l2_roi_align_v2_test, temp_case)
{
    auto self = TensorDesc({1, 256, 40, 40}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({524, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({524, 256, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 7, 7, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针
TEST_F(l2_roi_align_v2_test, case_nullptr)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut0 = OP_API_UT(aclnnRoiAlignV2, INPUT(nullptr, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnRoiAlignV2, INPUT(self, nullptr, 3, 3, 0.25f, 2, false), OUTPUT(out));
    aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(nullptr));
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非法dtype场景0
TEST_F(l2_roi_align_v2_test, case_dtype_invalid_0)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_BF16, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景1
TEST_F(l2_roi_align_v2_test, case_dtype_invalid_1)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype场景2
TEST_F(l2_roi_align_v2_test, case_dtype_invalid_2)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景0
TEST_F(l2_roi_align_v2_test, case_format_invalid_0)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景1
TEST_F(l2_roi_align_v2_test, case_format_invalid_1)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format场景2
TEST_F(l2_roi_align_v2_test, case_format_invalid_2)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景0
TEST_F(l2_roi_align_v2_test, case_shape_invalid_0)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景1
TEST_F(l2_roi_align_v2_test, case_shape_invalid_1)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5, 4}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景2
TEST_F(l2_roi_align_v2_test, case_shape_invalid_2)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景3
TEST_F(l2_roi_align_v2_test, case_shape_invalid_3)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 4}, dtype, ACL_FORMAT_ND);
    auto batchIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景4
TEST_F(l2_roi_align_v2_test, case_shape_invalid_4)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景5
TEST_F(l2_roi_align_v2_test, case_shape_invalid_5)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 2, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景6
TEST_F(l2_roi_align_v2_test, case_shape_invalid_6)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 2, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape场景7
TEST_F(l2_roi_align_v2_test, case_shape_invalid_7)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 2}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景0
TEST_F(l2_roi_align_v2_test, case_attr_invalid_0)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, -2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法attr场景1
TEST_F(l2_roi_align_v2_test, case_attr_invalid_1)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 6, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_roi_align_v2_test, case_empty_tensor)
{
    aclDataType dtype = aclDataType::ACL_FLOAT;
    auto self = TensorDesc({1, 1, 0, 6}, dtype, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, dtype, ACL_FORMAT_ND).Value(vector<float>{0.0, -2.0, -2.0, 22.0, 22.0});
    auto out = TensorDesc({1, 1, 3, 3}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// float场景（含精度）
TEST_F(l2_roi_align_v2_test, case_float)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                                         9,  10, 11, 12, 13, 14, 15, 16,
                                                                                         17, 18, 19, 20, 21, 22, 23, 24,
                                                                                         25, 26, 27, 28, 29, 30, 31, 32,
                                                                                         33, 34, 35, 36});
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.0, -2.0, -2.0, 22.0, 22.0});
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// float16场景（含精度）
TEST_F(l2_roi_align_v2_test, case_float16)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW)
                    .Value(vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0.0, -2.0, -2.0, 22.0, 22.0});
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

namespace {

aclnnStatus RunSpatialScaleCase(float spatialScale)
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, spatialScale, 2, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

} // namespace

TEST_F(l2_roi_align_v2_test, case_spatial_scale_positive_finite) { EXPECT_EQ(RunSpatialScaleCase(1.0f), ACL_SUCCESS); }

TEST_F(l2_roi_align_v2_test, case_spatial_scale_smallest_positive_normal)
{
    EXPECT_EQ(RunSpatialScaleCase(std::numeric_limits<float>::min()), ACL_SUCCESS);
}

TEST_F(l2_roi_align_v2_test, case_spatial_scale_nan)
{
    EXPECT_EQ(RunSpatialScaleCase(std::numeric_limits<float>::quiet_NaN()), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_align_v2_test, case_spatial_scale_positive_infinity)
{
    EXPECT_EQ(RunSpatialScaleCase(std::numeric_limits<float>::infinity()), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roi_align_v2_test, case_spatial_scale_negative_infinity)
{
    EXPECT_EQ(RunSpatialScaleCase(-std::numeric_limits<float>::infinity()), ACLNN_ERR_PARAM_INVALID);
}
