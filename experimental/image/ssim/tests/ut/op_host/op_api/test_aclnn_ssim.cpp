/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_ssim.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_ssim_test : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

public:
    void CommonTest(const vector<int64_t>& xShape, aclDataType dtype, aclnnStatus expectRet)
    {
        auto x1 = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto x2 = TensorDesc(xShape, dtype, ACL_FORMAT_ND);
        auto yOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
        uint64_t workspaceSize = 0;
        auto ut = OP_API_UT(aclnnSsim, INPUT(x1, x2), OUTPUT(yOut));
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expectRet);
    }
};

TEST_F(l2_ssim_test, ssim_rank234_success)
{
    CommonTest({64, 64}, ACL_FLOAT, ACLNN_SUCCESS);
    CommonTest({64, 64}, ACL_UINT8, ACLNN_SUCCESS);
    CommonTest({64, 64, 3}, ACL_FLOAT, ACLNN_SUCCESS);
    CommonTest({2, 64, 64, 3}, ACL_FLOAT, ACLNN_SUCCESS);
}

TEST_F(l2_ssim_test, ssim_dtype_invalid) { CommonTest({64, 64}, ACL_DOUBLE, ACLNN_ERR_PARAM_INVALID); }
