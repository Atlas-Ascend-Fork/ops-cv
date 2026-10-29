/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"

#include "opdev/make_op_executor.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/format_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_util.h"

#include "ssim.h"
#include "aclnn_ssim.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace SsimACLNN {
constexpr size_t MIN_SUPPORT_DIMS_NUMS = 2;
constexpr size_t MAX_SUPPORT_DIMS_NUMS = 8;
constexpr int64_t MIN_WIN_SIZE = 11;

struct SsimInputTensor {
    const aclTensor* x1;
    const aclTensor* x2;
};

struct SsimOutputTensor {
    aclTensor* yOut;
};

static bool CheckNotNull(SsimInputTensor& inputTensor, SsimOutputTensor& outputTensor)
{
    OP_CHECK_NULL(inputTensor.x1, return false);
    OP_CHECK_NULL(inputTensor.x2, return false);
    OP_CHECK_NULL(outputTensor.yOut, return false);
    return true;
}

static bool CheckDtypeValid(SsimInputTensor& inputTensor, SsimOutputTensor& outputTensor)
{
    const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT8, op::DataType::DT_FLOAT};
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.x1, inputTensor.x2, return false);
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.yOut, op::DataType::DT_FLOAT, return false);
    return true;
}

static bool CheckShapeValid(SsimInputTensor& inputTensor, SsimOutputTensor& outputTensor)
{
    OP_CHECK_MAX_DIM(inputTensor.x1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.x2, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(inputTensor.x1, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(inputTensor.x2, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, inputTensor.x2, return false);

    const op::Shape x1Shape = inputTensor.x1->GetViewShape();
    size_t dimNum = x1Shape.GetDimNum();
    // rank-2: [H, W]; rank >= 3: [..., H, W, C] (channels last, same as tiling)
    int64_t height = x1Shape.GetDim(dimNum - 2);
    int64_t width = x1Shape.GetDim(dimNum - 1);
    if (dimNum >= 3) {
        height = x1Shape.GetDim(dimNum - 3);
        width = x1Shape.GetDim(dimNum - 2);
    }
    if (height < MIN_WIN_SIZE || width < MIN_WIN_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "height/width must be >= 11, got H=%lld W=%lld.", height, width);
        return false;
    }

    const op::Shape yOutShape = outputTensor.yOut->GetViewShape();
    if (yOutShape.GetDimNum() != 1 || yOutShape.GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut shape must be [1].");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(SsimInputTensor& inputTensor, SsimOutputTensor& outputTensor)
{
    CHECK_RET(CheckNotNull(inputTensor, outputTensor), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus ComputeSsim(SsimInputTensor& inputTensor, SsimOutputTensor& outputTensor, aclOpExecutor* executor)
{
    auto ssimOuts = l0op::Ssim(inputTensor.x1, inputTensor.x2, executor);
    aclTensor* yComputeOut = std::get<0>(ssimOuts);
    CHECK_RET(yComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyYResult = l0op::ViewCopy(yComputeOut, outputTensor.yOut, executor);
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace SsimACLNN

aclnnStatus aclnnSsimGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, aclTensor* yOut,
                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnSsimGetWorkspaceSize.");
    L2_DFX_PHASE_1(aclnnSsim, DFX_IN(x1, x2), DFX_OUT(yOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    SsimACLNN::SsimInputTensor inputTensorOri = {x1, x2};
    SsimACLNN::SsimOutputTensor outputTensor = {yOut};
    auto ret = SsimACLNN::CheckParams(inputTensorOri, outputTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGW("Got empty tensor in aclnnSsim!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    SsimACLNN::SsimInputTensor inputTensor = {x1Cont, x2Cont};
    ret = SsimACLNN::ComputeSsim(inputTensor, outputTensor, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnSsimGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSsim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSsim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
