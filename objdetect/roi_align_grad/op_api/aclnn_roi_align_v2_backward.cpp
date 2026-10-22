/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// check
#include "aclnn_roi_align_v2_backward.h"
#include "roi_align_grad.h"
#include <cmath>
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "level0/concat.h"
#include "level0/fill.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr size_t DIM_FIVE = 5;

static const std::initializer_list<DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* boxes, const aclIntArray* inputShape,
                         const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(boxes, return false);
    OP_CHECK_NULL(inputShape, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* boxes, const aclTensor* gradInput)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, boxes->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, gradInput->GetDataType(), return false);

    return true;
}

static bool CheckFormatValid(const aclTensor* gradOutput, const aclTensor* boxes, const aclTensor* gradInput)
{
    bool formatValid = gradOutput->GetStorageFormat() == op::Format::FORMAT_NCHW &&
                       boxes->GetStorageFormat() == op::Format::FORMAT_ND &&
                       gradInput->GetStorageFormat() == op::Format::FORMAT_NCHW;
    if (formatValid == false) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "gradOutput's format should be NCHW, boxes's format should be ND, gradInput's format should be NCHW");
    }
    return formatValid;
}

static bool CheckInputShape(const aclTensor* gradOutput, const aclTensor* boxes, const aclIntArray* inputShape,
                            int64_t pooledHeight, int64_t pooledWidth)
{
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(boxes, DIM_TWO, return false);

    auto const& gradOutputShape = gradOutput->GetViewShape();
    auto const& boxesShape = boxes->GetViewShape();
    if (boxesShape.GetDim(1) != DIM_FIVE) { // 5: boxes dim 1
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim1 [%ld] should be 5", boxesShape.GetDim(1));
        return false;
    }
    if (inputShape->Size() != DIM_FOUR) { // 4:inputShape ints number
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputShape should contain 4 ints, but got %lu", inputShape->Size());
        return false;
    }
    for (size_t i = 0; i < inputShape->Size(); ++i) {
        if ((*inputShape)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputShape[%zu] [%ld] should be greater than 0", i, (*inputShape)[i]);
            return false;
        }
    }
    if (boxesShape.GetDim(DIM_ZERO) != gradOutputShape.GetDim(DIM_ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim0 [%ld] and gradOutput shape dim0 [%ld] should be equal",
                boxesShape.GetDim(DIM_ZERO), gradOutputShape.GetDim(DIM_ZERO));
        return false;
    }
    if (gradOutputShape.GetDim(DIM_TWO) != pooledHeight) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput shape dim2 [%ld] and pooledHeight [%ld] should be equal",
                gradOutputShape.GetDim(DIM_TWO), pooledHeight);
        return false;
    }
    if (gradOutputShape.GetDim(DIM_THREE) != pooledWidth) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput shape dim3 [%ld] and pooledWidth [%ld] should be equal",
                gradOutputShape.GetDim(DIM_THREE), pooledWidth);
        return false;
    }

    return true;
}

static bool CheckOutputShape(const aclTensor* gradOutput, const aclIntArray* inputShape, const aclTensor* gradInput)
{
    OP_CHECK_WRONG_DIMENSION(gradInput, DIM_FOUR, return false);

    auto const& gradOutputShape = gradOutput->GetViewShape();
    auto const& gradInputShape = gradInput->GetViewShape();

    if (gradInputShape.GetDim(DIM_ONE) != gradOutputShape.GetDim(DIM_ONE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradInput shape dim1 [%ld] and gradOutput shape dim1 [%ld] should be equal",
                gradInputShape.GetDim(DIM_ONE), gradOutputShape.GetDim(DIM_ONE));
        return false;
    }
    for (size_t i = 0; i < inputShape->Size(); ++i) {
        if (gradInputShape.GetDim(i) != (*inputShape)[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradInput shape [%ld] and inputShape [%ld] should be equal",
                    gradInputShape.GetDim(i), (*inputShape)[i]);
            return false;
        }
    }

    return true;
}

static bool CheckAttr(int64_t samplingRatio, float spatialScale)
{
    if (samplingRatio < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "samplingRatio [%ld] should be greater than or equal to 0", samplingRatio);
        return false;
    }
    if (!std::isfinite(spatialScale) || spatialScale <= 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "spatialScale [%f] should be finite and greater than 0", spatialScale);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* boxes, const aclIntArray* inputShape,
                               int64_t pooledHeight, int64_t pooledWidth, float spatialScale, int64_t samplingRatio,
                               const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, boxes, inputShape, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, boxes, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入、输出的数据格式是否支持
    CHECK_RET(CheckFormatValid(gradOutput, boxes, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入tensor的shape
    CHECK_RET(CheckInputShape(gradOutput, boxes, inputShape, pooledHeight, pooledWidth), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输出tensor的shape
    CHECK_RET(CheckOutputShape(gradOutput, inputShape, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查属性
    CHECK_RET(CheckAttr(samplingRatio, spatialScale), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetOutTensorWithValueZero(aclTensor* gradInput, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, gradInput->GetDataType());
    auto gradInputDims = op::ToShapeVector(gradInput->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(gradInputDims.data(), gradInputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* boxes,
                                                    const aclIntArray* inputShape, int64_t pooledHeight,
                                                    int64_t pooledWidth, float spatialScale, int64_t samplingRatio,
                                                    bool aligned, aclTensor* gradInput, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnRoiAlignV2Backward,
        DFX_IN(gradOutput, boxes, inputShape, pooledHeight, pooledWidth, spatialScale, samplingRatio, aligned),
        DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, boxes, inputShape, pooledHeight, pooledWidth, spatialScale, samplingRatio,
                           gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // roialigngrad算子的空tensor在kernel中支持
    if (gradOutput->IsEmpty() || boxes->IsEmpty()) {
        // 根据实际支持情况补充
        auto fillOut = GetOutTensorWithValueZero(gradInput, uniqueExecutor.get());
        CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(fillOut, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入boxes转换成连续的tensor
    auto boxesContiguous = l0op::Contiguous(boxes, uniqueExecutor.get());
    CHECK_RET(boxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // A5不需要转为私有格式，直接按原格式传入；其他芯片需要转为私有格式NC1HWC0
    bool is950 = (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510);
    const aclTensor* gradOutputForCompute = gradOutputContiguous;
    const aclTensor* roiAlignBackwardOut = nullptr;
    if (!is950) {
        // 将gradOutput转为私有格式NC1HWC0
        gradOutputForCompute = l0op::TransDataSpecial(gradOutputContiguous, op::Format::FORMAT_NC1HWC0, 0,
                                                      uniqueExecutor.get());
        CHECK_RET(gradOutputForCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 基于aligned判断roiEndMode取值
    int64_t roiEndMode = aligned ? DIM_TWO : DIM_ZERO; // roiEndMode = 2 for torch aligned = true

    const aclTensor* roisN = nullptr;

    // 进行计算
    roiAlignBackwardOut = l0op::ROIAlignGrad(gradOutputForCompute, boxesContiguous, roisN, inputShape, pooledHeight,
                                             pooledWidth, spatialScale, samplingRatio, roiEndMode,
                                             uniqueExecutor.get());
    CHECK_RET(roiAlignBackwardOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // A5不需要TransData转回NCHW，直接ViewCopy；其他芯片需要将私有格式数据转为NCHW
    const aclTensor* gradInputForCopy = roiAlignBackwardOut;
    if (!is950) {
        // 将roiAlignBackwardOut的私有格式数据转为NCHW
        gradInputForCopy = l0op::TransDataSpecial(roiAlignBackwardOut, gradInput->GetOriginalFormat(), 0,
                                                  uniqueExecutor.get());
        CHECK_RET(gradInputForCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(gradInputForCopy, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiAlignV2Backward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoiAlignV2Backward);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
