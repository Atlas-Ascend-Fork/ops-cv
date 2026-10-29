/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_trilinear_3d_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/level2_base_cv.h"
#include "aclnn_upsample_trilinear_3d_backward.h"
#include "op_api/aclnn_check.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static bool CheckFormat(const aclTensor* gradOut)
{
    const op::Format gradOutFormat = gradOut->GetStorageFormat();
    OP_CHECK(gradOutFormat == op::Format::FORMAT_ND || gradOutFormat == op::Format::FORMAT_NCDHW ||
                 gradOutFormat == op::Format::FORMAT_NDHWC,
             OP_LOGE_FOR_INVALID_FORMAT("aclnnUpsampleTrilinear3dBackward", "gradOut",
                                        op::ToString(gradOutFormat).GetString(), "NCDHW or NDHWC"),
             return false);
    return true;
}

static bool CheckInputElement(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                              const aclTensor* gradInput)
{
    int64_t outD = (*outputSize)[DIM_ZERO];
    int64_t outH = (*outputSize)[DIM_ONE];
    int64_t outW = (*outputSize)[DIM_TWO];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputD = (*inputSize)[DIM_TWO];
    int64_t inputH = (*inputSize)[DIM_THREE];
    int64_t inputW = (*inputSize)[DIM_FOUR];
    FVector<int64_t> fullOutputSize = {batch, channels, outD, outH, outW};
    FVector<int64_t> fullInputSize = {batch, channels, inputD, inputH, inputW};
    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        fullOutputSize = {batch, outD, outH, outW, channels};
    }
    if (gradInput->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        fullInputSize = {batch, inputD, inputH, inputW, channels};
    }
    auto gradOutShape = gradOut->GetViewShape();
    auto gradInShape = gradInput->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();

    OP_CHECK(inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3dBackward", "input_h",
                                                      std::to_string(inputH).c_str(),
                                                      "all input and output dims must be greater than 0"),
             return false);

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3dBackward", "gradoutput_dim",
                                                     std::to_string(gradOutShape.GetDim(i)).c_str(),
                                                     "each dim of gradOutput must match the expected output size");
            return false;
        }
    }
    for (size_t i = 0; i < dimNum; ++i) {
        if (gradInShape.GetDim(i) != fullInputSize[i]) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3dBackward", "gradinput_dim",
                                                     std::to_string(gradInShape.GetDim(i)).c_str(),
                                                     "each dim of gradInput must match the expected input size");
            return false;
        }
    }
    return true;
}

static bool CheckUplimit(const aclTensor* gradOut)
{
    if (IsRegBase()) {
        return true;
    }
    int64_t gradOutN = gradOut->GetViewShape().GetDim(DIM_ZERO);
    int64_t gradOutC = gradOut->GetViewShape().GetDim(DIM_ONE);
    int64_t outD = gradOut->GetViewShape().GetDim(DIM_TWO);
    int64_t outH = gradOut->GetViewShape().GetDim(DIM_THREE);
    int64_t outW = gradOut->GetViewShape().GetDim(DIM_FOUR);
    int64_t inputN = gradOut->GetViewShape().GetDim(DIM_ZERO);
    int64_t inputC = gradOut->GetViewShape().GetDim(DIM_ONE);
    int64_t inputD = gradOut->GetViewShape().GetDim(DIM_TWO);
    int64_t inputH = gradOut->GetViewShape().GetDim(DIM_THREE);
    int64_t inputW = gradOut->GetViewShape().GetDim(DIM_FOUR);

    OP_CHECK(
        gradOutN <= INT32_MAX && gradOutC <= INT32_MAX && outD <= INT32_MAX && outH <= INT32_MAX && outW <= INT32_MAX,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3dBackward", "gradOutN",
                                                 std::to_string(gradOutN).c_str(),
                                                 "all dims must not exceed INT32_MAX"),
        return false);
    OP_CHECK(
        inputN <= INT32_MAX && inputC <= INT32_MAX && inputD <= INT32_MAX && inputH <= INT32_MAX && inputW <= INT32_MAX,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleTrilinear3dBackward", "inputN",
                                                 std::to_string(inputN).c_str(), "all dims must not exceed INT32_MAX"),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2In2Out(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid1Out1In(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape, format是否支持
    CHECK_RET(CheckUpsampleShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gradOut), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 5. 校验上边界
    CHECK_RET(CheckUplimit(gradOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* upsampleTrilinear3dBackwardCompute(const aclTensor* gradOutContiguous, const aclIntArray* outputSize,
                                                    const aclIntArray* inputSize, bool alignCorners,
                                                    const aclFloatArray* scales, const aclFloatArray* castScales,
                                                    aclOpExecutor* executor)
{
    if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNCDHWList[] = {0, 4, 1, 2, 3};
        auto permuteNCDHWArray = executor->AllocIntArray(permuteNCDHWList, UPSAMPLE_DIM_LIMIT);
        CHECK_RET(permuteNCDHWArray != nullptr, nullptr);

        auto gradOutTranspose = l0op::Transpose(gradOutContiguous, permuteNCDHWArray, executor);
        CHECK_RET(gradOutTranspose != nullptr, nullptr);

        auto upsampleTrilinearGradOut = l0op::UpsampleTrilinear3dGradNcdhw(gradOutTranspose, outputSize, inputSize,
                                                                           alignCorners, scales, castScales, executor);
        CHECK_RET(upsampleTrilinearGradOut != nullptr, nullptr);

        const int64_t permuteNDHWCList[] = {0, 2, 3, 4, 1};
        auto permuteNDHWCArray = executor->AllocIntArray(permuteNDHWCList, UPSAMPLE_DIM_LIMIT);
        CHECK_RET(permuteNDHWCArray != nullptr, nullptr);

        return l0op::Transpose(upsampleTrilinearGradOut, permuteNDHWCArray, executor);
    }
    return l0op::UpsampleTrilinear3dGradNcdhw(gradOutContiguous, outputSize, inputSize, alignCorners, scales,
                                              castScales, executor);
}
} // namespace

aclnnStatus aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                             const aclIntArray* inputSize, bool alignCorners,
                                                             double scalesD, double scalesH, double scalesW,
                                                             aclTensor* gradInput, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleTrilinear3dBackward,
                   DFX_IN(gradOut, outputSize, inputSize, alignCorners, scalesD, scalesH, scalesW), DFX_OUT(gradInput));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradOut, outputSize, inputSize, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOut->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vector<float> scalesList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scalesList.push_back(scalesD);
        scalesList.push_back(scalesH);
        scalesList.push_back(scalesW);
        outputSize = uniqueExecutor.get()->AllocIntArray({}, 0);
    }
    const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vector<float> scalesCastList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scalesCastList.push_back(static_cast<float>(1.0 / scalesD));
        scalesCastList.push_back(static_cast<float>(1.0 / scalesH));
        scalesCastList.push_back(static_cast<float>(1.0 / scalesW));
    } else {
        scalesCastList.push_back(0.0);
        scalesCastList.push_back(0.0);
        scalesCastList.push_back(0.0);
    }
    const aclFloatArray* castScales = uniqueExecutor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
    CHECK_RET(castScales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = upsampleTrilinear3dBackwardCompute(gradOutContiguous, outputSize, inputSize, alignCorners, scales,
                                                     castScales, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleTrilinear3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleTrilinear3dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
