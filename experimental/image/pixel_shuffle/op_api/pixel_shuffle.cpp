/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pixel_shuffle.h"
#include "op_api/aclnn_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(PixelShuffle);

// scale_factor 上界：空间维数 S<=3，故 r^S <= r^3 <= 2^60 < INT64_MAX，
// 保证下方 rPow 累乘不会溢出；否则溢出为 0 时 "GetDim(1) % rPow" 会整数除零。
static constexpr int64_t MAX_SCALE_FACTOR = 1LL << 20;

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                                     DataType::DT_FLOAT};

static inline const std::initializer_list<DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_LOGI("PixelShuffle", "curArch is %u", static_cast<uint32_t>(curArch));
    return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
}

static inline bool IsAiCoreSupport(const aclTensor* x)
{
    return CheckType(x->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

static const aclTensor* PixelShuffleAiCore(const aclTensor* x, int64_t scaleFactor, aclTensor* y,
                                           aclOpExecutor* executor)
{
    L0_DFX(PixelShuffleAiCore, x, y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(PixelShuffle, OP_INPUT(x), OP_ATTR(scaleFactor), OP_OUTPUT(y));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PixelShuffleAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return y;
}

const aclTensor* PixelShuffle(const aclTensor* x, int64_t scaleFactor, aclOpExecutor* executor)
{
    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PixelShuffle does not support dtype %s on current platform.",
                op::ToString(x->GetDataType()).GetString());
        return nullptr;
    }

    // 计算输出 shape：4D [N,C,H,W] -> [N,C/r^2,H*r,W*r]；5D [N,C,D,H,W] -> [N,C/r^3,D*r,H*r,W*r]
    op::Shape inShape = x->GetViewShape();
    int64_t dimNum = inShape.GetDimNum();
    if (dimNum != 4 && dimNum != 5) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PixelShuffle only supports 4D or 5D input, got %ld dims.", dimNum);
        return nullptr;
    }
    int64_t r = scaleFactor;
    if (r < 2 || r > MAX_SCALE_FACTOR) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PixelShuffle scaleFactor must be in [2, %ld], but got %ld.", MAX_SCALE_FACTOR,
                r);
        return nullptr;
    }
    int64_t s = (dimNum == 5) ? 3 : 2;
    int64_t rPow = 1;
    for (int64_t i = 0; i < s; ++i) {
        rPow *= r;
    }
    if (inShape.GetDim(1) % rPow != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "PixelShuffle input channel %ld must be divisible by r^%ld (%ld), but it is not.", inShape.GetDim(1), s,
                rPow);
        return nullptr;
    }
    op::Shape yShape;
    yShape.SetDimNum(dimNum);
    yShape.SetDim(0, inShape.GetDim(0));
    yShape.SetDim(1, inShape.GetDim(1) / rPow);
    if (dimNum == 5) {
        yShape.SetDim(2, inShape.GetDim(2) * r);
        yShape.SetDim(3, inShape.GetDim(3) * r);
        yShape.SetDim(4, inShape.GetDim(4) * r);
    } else {
        yShape.SetDim(2, inShape.GetDim(2) * r);
        yShape.SetDim(3, inShape.GetDim(3) * r);
    }

    auto y = executor->AllocTensor(yShape, x->GetDataType());
    return PixelShuffleAiCore(x, scaleFactor, y, executor);
}

} // namespace l0op
