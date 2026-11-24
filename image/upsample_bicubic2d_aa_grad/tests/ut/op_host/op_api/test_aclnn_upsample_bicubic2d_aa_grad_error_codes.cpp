/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <climits>
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_upsample_bicubic2d_aa_grad.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class BicubicAAGradApiErrors : public testing::TestWithParam<int> {};

TEST_P(BicubicAAGradApiErrors, parameter_diagnostic)
{
    const int scenario = GetParam();
    uint64_t workspace = 0;
    aclOpExecutor* executor = nullptr;
    if (scenario >= 16) {
        EXPECT_EQ(aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(nullptr, nullptr, nullptr, false, 0, 0, nullptr,
                                                               scenario == 16 ? nullptr : &workspace,
                                                               scenario == 17 ? nullptr : &executor),
                  ACLNN_ERR_PARAM_NULLPTR);
        return;
    }
    vector<int64_t> gradShape = {1, 1, 8, 8}, outShape = {1, 1, 4, 4};
    vector<int64_t> outputSize = {8, 8}, inputSize = {1, 1, 4, 4};
    aclFormat gradFormat = ACL_FORMAT_ND, outFormat = ACL_FORMAT_ND;
    if (scenario == 2)
        gradFormat = ACL_FORMAT_NCHW;
    if (scenario == 3)
        gradShape = {1, 8, 8};
    if (scenario == 4)
        outShape = {1, 4, 4};
    if (scenario == 5)
        inputSize.pop_back();
    if (scenario == 6)
        outputSize.pop_back();
    if (scenario == 9)
        gradShape[2] = 0;
    if (scenario == 10)
        gradFormat = outFormat = ACL_FORMAT_NHWC;
    if (scenario == 13)
        gradShape[0] = static_cast<int64_t>(INT32_MAX) + 1;
    if (scenario == 14)
        inputSize[0] = static_cast<int64_t>(INT32_MAX) + 1;
    if (scenario == 15) {
        gradFormat = outFormat = ACL_FORMAT_NCHW;
        outShape[0] = 2;
    }
    auto grad = TensorDesc(gradShape, scenario == 0 ? ACL_INT32 : ACL_FLOAT, gradFormat);
    auto out = TensorDesc(outShape, scenario == 1 ? ACL_FLOAT16 : scenario == 0 ? ACL_INT32 : ACL_FLOAT, outFormat);
    auto input = IntArrayDesc(inputSize), output = IntArrayDesc(outputSize);
    const double scalesH = scenario == 8 ? -1.0 : scenario == 11 ? 100.0 : 0.0;
    const double scalesW = scenario == 7 ? -1.0 : scenario == 12 ? 100.0 : 0.0;
    auto ut = OP_API_UT(aclnnUpsampleBicubic2dAAGrad, INPUT(grad, output, input, false, scalesH, scalesW), OUTPUT(out));
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(NegativeParameters, BicubicAAGradApiErrors, testing::Range(0, 18));
