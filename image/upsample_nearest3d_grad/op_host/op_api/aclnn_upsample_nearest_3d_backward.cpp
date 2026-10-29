/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_nearest_3d_grad.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_upsample_nearest_3d_backward.h"
#include "op_api/level2_base_cv.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckInputElement(const aclTensor* gradOut, const aclTensor* gradInput, const aclIntArray* outputSize,
                              const aclIntArray* inputSize)
{
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputD = (*inputSize)[DIM_TWO];
    int64_t inputH = (*inputSize)[DIM_THREE];
    int64_t inputW = (*inputSize)[DIM_FOUR];
    int64_t outD = (*outputSize)[DIM_ZERO];
    int64_t outH = (*outputSize)[DIM_ONE];
    int64_t outW = (*outputSize)[DIM_TWO];
    FVector<int64_t> fullOutSize = {batch, channels, outD, outH, outW};
    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        fullOutSize = {batch, outD, outH, outW, channels};
    }
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();

    OP_CHECK(inputD > 0 && inputH > 0 && inputW > 0 && outD > 0 && outH > 0 && outW > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleNearest3dBackward", "input_h",
                                                      std::to_string(inputH).c_str(),
                                                      "all input and output dims must be greater than 0"),
             return false);

    auto res = CheckSizeLoop(dimNum, gradOutShape, fullOutSize);
    CHECK_RET(res == true, res);

    op::Shape expectShape = op::Shape{batch, channels, inputD, inputH, inputW};
    if (gradInput->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        expectShape = op::Shape{batch, inputD, inputH, inputW, channels};
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, expectShape, return false);

    if (!IsRegBase()) {
        OP_CHECK(
            batch <= INT32_MAX && channels <= INT32_MAX && outD <= INT32_MAX && outH <= INT32_MAX && outW <= INT32_MAX,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleNearest3dBackward", "batch",
                                                     std::to_string(batch).c_str(),
                                                     "all dims must not exceed INT32_MAX"),
            return false);
        OP_CHECK(batch <= INT32_MAX && channels <= INT32_MAX && inputD <= INT32_MAX && inputH <= INT32_MAX &&
                     inputW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleNearest3dBackward", "batch",
                                                          std::to_string(batch).c_str(),
                                                          "all dims of inputSize must not exceed INT32_MAX"),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2In2Out(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid1Out1In(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckUpsampleShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(gradOut, gradInput, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest3dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                           const aclIntArray* inputSize, double scalesD, double scalesH,
                                                           double scalesW, aclTensor* gradInput,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearest3dBackward, DFX_IN(gradOut, outputSize, inputSize, scalesD, scalesH, scalesW),
                   DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto checkNearest3dBackwardRet = CheckParams(gradOut, outputSize, inputSize, gradInput);
    CHECK_RET(checkNearest3dBackwardRet == ACLNN_SUCCESS, checkNearest3dBackwardRet);

    // 空tensor支持
    if (gradOut->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOut转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    vector<float> scaleList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scaleList.insert(scaleList.end(),
                         {static_cast<float>(scalesD), static_cast<float>(scalesH), static_cast<float>(scalesW)});
        outputSize = uniqueExecutor.get()->AllocIntArray({}, 0);
    }
    const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scaleList.data(), scaleList.size());
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutTranspose = gradOutContiguous;
    if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNCDHWList[] = {DIM_ZERO, DIM_FOUR, DIM_ONE, DIM_TWO, DIM_THREE};
        auto permuteNCDHWArray = uniqueExecutor->AllocIntArray(permuteNCDHWList, UPSAMPLE_DIM_LIMIT);
        CHECK_RET(permuteNCDHWArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutTranspose = l0op::Transpose(gradOutContiguous, permuteNCDHWArray, uniqueExecutor.get());
        CHECK_RET(gradOutTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    vector<float> scalesCastList{};
    if (scalesD > 0 && scalesH > 0 && scalesW > 0) {
        scalesCastList.push_back(static_cast<float>(scalesD));
        scalesCastList.push_back(static_cast<float>(scalesH));
        scalesCastList.push_back(static_cast<float>(scalesW));
    } else {
        scalesCastList.push_back(0.0);
        scalesCastList.push_back(0.0);
        scalesCastList.push_back(0.0);
    }
    const aclFloatArray* castScales = uniqueExecutor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
    CHECK_RET(castScales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用UpsampleNearest3dGradNcdhw算子kernel, inputSize对应[N, C, D, H, W]
    auto result = l0op::UpsampleNearest3dGradNcdhw(gradOutTranspose, outputSize, inputSize, scales, castScales,
                                                   uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resultTranspose = result;
    if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NDHWC) {
        const int64_t permuteNDHWCList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_FOUR, DIM_ONE};
        auto permuteNDHWCArray = uniqueExecutor->AllocIntArray(permuteNDHWCList, UPSAMPLE_DIM_LIMIT);
        CHECK_RET(permuteNDHWCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        resultTranspose = l0op::Transpose(result, permuteNDHWCArray, uniqueExecutor.get());
    }

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(resultTranspose, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest3dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
