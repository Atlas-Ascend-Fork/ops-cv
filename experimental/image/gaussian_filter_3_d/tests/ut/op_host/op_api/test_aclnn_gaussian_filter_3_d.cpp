/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_gaussian_filter_3_d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_gaussian_filter_3_d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_gaussian_filter_3_d_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_gaussian_filter_3_d_test TearDown" << endl; }

public:
    void CommonSuccess(const vector<int64_t>& xShape, aclDataType dtype, double sigma, const char* mode)
    {
        auto x = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto y = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnGaussianFilter3D, INPUT(x, sigma, mode), OUTPUT(y));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }

    void CommonFail(const vector<int64_t>& xShape, aclDataType dtype, double sigma, const char* mode,
                    aclDataType outDtype = ACL_DT_UNDEFINED)
    {
        auto x = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto y = TensorDesc(xShape, outDtype == ACL_DT_UNDEFINED ? dtype : outDtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnGaussianFilter3D, INPUT(x, sigma, mode), OUTPUT(y));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
};

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_rank_dtype_success)
{
    CommonSuccess({8, 16, 67}, ACL_FLOAT, 1.0, "REFLECT");
    CommonSuccess({2, 8, 16, 67}, ACL_FLOAT16, 1.0, "REFLECT");
    CommonSuccess({2, 3, 8, 16, 67}, ACL_BF16, 1.0, "REFLECT");
    CommonSuccess({1, 1, 1, 1, 1}, ACL_FLOAT, 1.0, "REFLECT");
}

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_sigma_success)
{
    CommonSuccess({8, 16, 16}, ACL_FLOAT, 0.0, "REFLECT");
    CommonSuccess({8, 16, 16}, ACL_FLOAT, 0.5, "CONSTANT");
    CommonSuccess({8, 16, 16}, ACL_FLOAT, 2.0, "NEAREST");
    CommonSuccess({8, 16, 16}, ACL_FLOAT, 4.0, "MIRROR");
    CommonSuccess({8, 16, 16}, ACL_FLOAT, 100.0, "WRAP");
}

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_mode_success)
{
    CommonSuccess({4, 8, 8}, ACL_FLOAT, 1.0, "REFLECT");
    CommonSuccess({4, 8, 8}, ACL_FLOAT, 1.0, "CONSTANT");
    CommonSuccess({4, 8, 8}, ACL_FLOAT, 1.0, "NEAREST");
    CommonSuccess({4, 8, 8}, ACL_FLOAT, 1.0, "MIRROR");
    CommonSuccess({4, 8, 8}, ACL_FLOAT, 1.0, "WRAP");
}

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_rank_invalid)
{
    CommonFail({16, 67}, ACL_FLOAT, 1.0, "REFLECT");
    CommonFail({1, 2, 8, 16, 67, 3}, ACL_FLOAT, 1.0, "REFLECT");
}

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_sigma_invalid)
{
    CommonFail({8, 16, 67}, ACL_FLOAT, -1.0, "REFLECT");
    CommonFail({8, 16, 67}, ACL_FLOAT, 100.5, "REFLECT");
}

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_mode_invalid) { CommonFail({8, 16, 67}, ACL_FLOAT, 1.0, "reflect2"); }

TEST_F(l2_gaussian_filter_3_d_test, ascend910B3_dtype_invalid)
{
    CommonFail({8, 16, 67}, ACL_INT32, 1.0, "REFLECT");
    CommonFail({8, 16, 67}, ACL_DOUBLE, 1.0, "REFLECT");
    CommonFail({8, 16, 67}, ACL_FLOAT, 1.0, "REFLECT", ACL_FLOAT16);
}
