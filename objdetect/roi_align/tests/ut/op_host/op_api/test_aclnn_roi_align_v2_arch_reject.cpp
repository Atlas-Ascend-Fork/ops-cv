/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "opdev/platform.h"

#include "../../../../op_api/aclnn_roi_align_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

namespace {

aclnnStatus RunRoiAlignV2OnCurrentSoc()
{
    auto self = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto boxes = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoiAlignV2, INPUT(self, boxes, 3, 3, 0.25f, 2, false), OUTPUT(out));

    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

} // namespace

TEST(RoiAlignV2ArchSupportApi, ascend910b_valid_input_succeeds)
{
    SetPlatformSocVersion(SocVersion::ASCEND910B);
    EXPECT_EQ(RunRoiAlignV2OnCurrentSoc(), ACL_SUCCESS);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}

// Regbase SoC (npuArch DAV_3510/DAV_9201/DAV_5162): CheckSocValid must fail with ACLNN_ERR_RUNTIME_ERROR and
// report EZ1009 (not the internal EZ9903 path from OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, ...)).
TEST(RoiAlignV2ArchSupportApi, ascend950_regbase_rejected_with_diagnosable_error)
{
    SetPlatformSocVersion(SocVersion::ASCEND950);
    EXPECT_EQ(RunRoiAlignV2OnCurrentSoc(), ACLNN_ERR_RUNTIME_ERROR);
    SetPlatformSocVersion(SocVersion::ASCEND910B);
}
