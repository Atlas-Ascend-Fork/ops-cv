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

#include "../../../../op_api/aclnn_resize.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_resize_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Resize Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Resize Test TearDown" << endl; }
};

// float nearest
TEST_F(l2_resize_test, case_1)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16 nearest
TEST_F(l2_resize_test, case_2)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float bilinear
TEST_F(l2_resize_test, case_3)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_resize_test, Ascend950PR_9599_case_3)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16 bilinear
TEST_F(l2_resize_test, case_4)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self and scales shape not match
TEST_F(l2_resize_test, case_5)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self nullptr
TEST_F(l2_resize_test, case_6)
{
    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(nullptr, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// scales nullptr
TEST_F(l2_resize_test, case_7)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, nullptr, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_resize_test, case_8)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = (aclTensor*)nullptr;
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// unsupport mode
TEST_F(l2_resize_test, case_9)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "trilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// unmatch type
TEST_F(l2_resize_test, case_10)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// unmatch shape
TEST_F(l2_resize_test, case_11)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// unmatch format
TEST_F(l2_resize_test, case_12)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_resize_test, case_13)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2.4, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// large shape
TEST_F(l2_resize_test, case_14)
{
    auto self = TensorDesc({1, 1, 1, 23}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(23));

    auto scales = FloatArrayDesc(vector<float>{1, 1, 1, 970155});
    auto out = TensorDesc({1, 1, 1, 22313565}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(22313565));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// large shape
TEST_F(l2_resize_test, case_15)
{
    auto self = TensorDesc({1, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4, 5, 6});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 1.666666});
    auto out = TensorDesc({1, 1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>(20));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_resize_test, case_16)
{
    auto self = TensorDesc({1, 1, 2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4, 5, 6});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 1.666666, 1});
    auto out = TensorDesc({1, 1, 4, 5, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>(20));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/*
// unmatch shape
TEST_F(l2_resize_test, case_17)
{
    auto self = TensorDesc({1,1,2,2}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>{1,2,3,4});

    auto scales = FloatArrayDesc(vector<float>{1,1,2,2});
    auto out = TensorDesc({1,1,4,8}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>(16));
    const char *mode="nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// unmatch shape
TEST_F(l2_resize_test, case_18)
{
    auto self = TensorDesc({1,1,2,2}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>{1,2,3,4});

    auto scales = FloatArrayDesc(vector<float>{1,1,2,2});
    auto out = TensorDesc({1,1,4,8}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>(16));
    const char *mode="nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_resize_test, case_19)
{
    auto self = TensorDesc({1,1,2,2}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>{1,2,3,4});

    auto scales = FloatArrayDesc(vector<float>{1,1,2.6,2});
    auto out = TensorDesc({1,1,4,4}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>(16));
    const char *mode="nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_resize_test, case_20)
{
    auto self = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>{1,2});

    auto scales = FloatArrayDesc(vector<float>{1,1,2.6,2});
    auto out = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_NHWC)
        .Value(vector<float>(16));
    const char *mode="nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
*/

TEST_F(l2_resize_test, case_21)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2.6, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_resize_test, case_22)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2.6, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/*
TEST_F(l2_resize_test, case_23)
{
    auto self = TensorDesc({1,1,2,2}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>{1,2,3,4});

    auto scales = FloatArrayDesc(vector<float>{1,1,2,2});
    auto out = TensorDesc({1,1,4,4}, ACL_FLOAT, ACL_FORMAT_NCHW)
        .Value(vector<float>(16));
    const char *mode="nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
*/

TEST_F(l2_resize_test, case_24)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_resize_test, case_20)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = NULL;
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// scales empty array, should return ACLNN_ERR_PARAM_INVALID instead of crash
TEST_F(l2_resize_test, case_25)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scales size less than 4 (size 1/2/3), element access must be guarded by size check
TEST_F(l2_resize_test, case_26)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_resize_test, case_27)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_resize_test, case_28)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scales size larger than 4
TEST_F(l2_resize_test, case_29)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2, 1});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearest";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mode with valid prefix but invalid suffix, must be rejected by exact match
TEST_F(l2_resize_test, case_30)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "bilinearXYZ";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_resize_test, case_31)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "nearestX";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mode with wrong case, must be rejected by exact match
TEST_F(l2_resize_test, case_32)
{
    auto self = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4});

    auto scales = FloatArrayDesc(vector<float>{1, 1, 2, 2});
    auto out = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>(16));
    const char* mode = "Bilinear";
    auto ut = OP_API_UT(aclnnResize, INPUT(self, scales, mode, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
