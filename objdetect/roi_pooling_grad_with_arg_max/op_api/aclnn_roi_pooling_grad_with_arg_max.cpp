/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_roi_pooling_grad_with_arg_max.cpp
 * \brief 声明ROI池化梯度计算（带argmax）的aclnn接口
 */
#include "aclnn_roi_pooling_grad_with_arg_max.h"
#include "roi_pooling_grad_with_arg_max.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"
#include "level0/fill.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t ROIS_DIM = 2;
static constexpr size_t DIM_FOUR = 4;
static constexpr size_t ROIS_ONE_SIZE = 5;
static constexpr uint64_t DIM0 = 0;
static constexpr uint64_t DIM1 = 1;
static constexpr uint64_t DIM2 = 2;
static constexpr uint64_t DIM3 = 3;
static constexpr uint64_t BATCH_MAX = 1024;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {op::DataType::DT_FLOAT,
                                                                               op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_ARGMAX_REGBASE = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* rois, const aclTensor* argmax,
                                const aclTensor* gradInputRef)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(rois, return false);
    OP_CHECK_NULL(argmax, return false);
    OP_CHECK_NULL(gradInputRef, return false);
    return true;
}

static bool CheckDtype(const aclTensor* gradOutput, const aclTensor* rois, const aclTensor* argmax,
                       const aclTensor* gradInputRef)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST_REGBASE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(rois, DTYPE_SUPPORT_LIST_REGBASE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(argmax, DTYPE_SUPPORT_LIST_ARGMAX_REGBASE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInputRef, DTYPE_SUPPORT_LIST_REGBASE, return false);

    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* rois, const aclTensor* argmax,
                        const aclTensor* gradInputRef)
{
    if (IsRegBase()) {
        // 如果输入格式是私有格式，记录日志，直接报错
        if (op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(rois->GetStorageFormat()) ||
            op::IsPrivateFormat(argmax->GetStorageFormat()) || op::IsPrivateFormat(gradInputRef->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND.");
            return false;
        }

        OP_CHECK(
            gradOutput->GetViewFormat() == rois->GetViewFormat() &&
                gradOutput->GetViewFormat() == argmax->GetViewFormat() &&
                gradOutput->GetViewFormat() == gradInputRef->GetViewFormat(),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Format of input and output should be equal, gradOutput [%s], rois [%s], argmax [%s], gradInputRef "
                    "[%s].",
                    op::ToString(gradOutput->GetViewFormat()).GetString(),
                    op::ToString(rois->GetViewFormat()).GetString(), op::ToString(argmax->GetViewFormat()).GetString(),
                    op::ToString(gradInputRef->GetViewFormat()).GetString()),
            return false);
    }
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* rois, const aclTensor* argmax,
                       const aclTensor* gradInputRef, int64_t pooledH, int64_t pooledW)
{
    OP_LOGD("CheckShape start");
    if (gradOutput->GetViewShape().GetDimNum() != DIM_FOUR || rois->GetViewShape().GetDimNum() != ROIS_DIM ||
        argmax->GetViewShape().GetDimNum() != DIM_FOUR || gradInputRef->GetViewShape().GetDimNum() != DIM_FOUR) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected gradOutput dim [%zu] to be 4, Expected rois dim [%zu] to be 2,Expected argmax dim [%zu] to "
                "be 4, Expected out dim [%zu] to be 4 but check failed.",
                gradOutput->GetViewShape().GetDimNum(), rois->GetViewShape().GetDimNum(),
                argmax->GetViewShape().GetDimNum(), gradInputRef->GetViewShape().GetDimNum());
        return false;
    }

    if (rois->GetViewShape().GetDim(DIM0) != gradOutput->GetViewShape().GetDim(DIM0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected rois dim 0 to be equal to gradOutput.shape[0] but check failed.");
        return false;
    }
    if (rois->GetViewShape().GetDim(DIM1) != ROIS_ONE_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected rois dim 1 to be 5 but check failed.");
        return false;
    }
    if (rois->GetViewShape().GetDim(DIM0) > BATCH_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected rois dim 0 should be less than 1024.");
        return false;
    }
    if (gradInputRef->GetViewShape().GetDim(DIM0) > BATCH_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected gradInputRef dim 0 should be less than 1024.");
        return false;
    }
    if (argmax->GetViewShape().GetDim(DIM0) != gradOutput->GetViewShape().GetDim(DIM0) ||
        argmax->GetViewShape().GetDim(DIM1) != gradOutput->GetViewShape().GetDim(DIM1) ||
        argmax->GetViewShape().GetDim(DIM2) != gradOutput->GetViewShape().GetDim(DIM2) ||
        argmax->GetViewShape().GetDim(DIM3) != gradOutput->GetViewShape().GetDim(DIM3)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected argmax.shape to be equal to gradOutput.shape but check failed.");
        return false;
    }
    if (argmax->GetViewShape().GetDim(DIM2) != pooledH || argmax->GetViewShape().GetDim(DIM3) != pooledW) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected argmax dim 2 to be equal to pooledH and dim 3 to be equal to pooledW but check failed.");
        return false;
    }
    if (argmax->GetViewShape().GetDim(DIM1) != gradInputRef->GetViewShape().GetDim(DIM1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected argmax dim 1 to be equal to x.shape[3] but check failed.");
        return false;
    }

    OP_LOGD("CheckShape end!");
    return true;
}

static bool CheckAttrValue(int64_t pooledH, int64_t pooledW)
{
    OP_CHECK(pooledH > 0 && pooledW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "pooledH and pooledW must be greater than zero, but got stride=(%ld,%ld).", pooledH, pooledW),
             return false);
    return true;
}

static bool CheckAttr(double spatialScale, int64_t pooledH, int64_t pooledW)
{
    OP_CHECK(CheckAttrValue(pooledH, pooledW), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckAttrValue failed."),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* rois, const aclTensor* argmax,
                               const aclTensor* gradInputRef, int64_t pooledH, int64_t pooledW, double spatialScale)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, rois, argmax, gradInputRef), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtype(gradOutput, rois, argmax, gradInputRef), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据格式是否在API支持的数据格式范围之内，需要根据api定义校验
    CHECK_RET(CheckFormat(gradOutput, rois, argmax, gradInputRef), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckShape(gradOutput, rois, argmax, gradInputRef, pooledH, pooledW), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数组是否满足要求
    CHECK_RET(CheckAttr(spatialScale, pooledH, pooledW), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetOutTensorWithValueZero(const aclTensor* gradInput, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, gradInput->GetDataType());
    auto gradInputDims = op::ToShapeVector(gradInput->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(gradInputDims.data(), gradInputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

// gradOutput 为空时无梯度可累加；若输出非空则写回全 0，与 kernel InitOutput(0) 语义一致。
static aclnnStatus FillEmptyGradOutput(const aclTensor* gradInputRef, uint64_t* workspaceSize, aclOpExecutor* executor)
{
    if (gradInputRef->IsEmpty()) {
        *workspaceSize = 0;
        return ACLNN_SUCCESS;
    }
    auto fillOut = GetOutTensorWithValueZero(gradInputRef, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(fillOut, gradInputRef, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executor->GetWorkspaceSize();
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* gradInputRef,
                                                          const aclTensor* rois, const aclTensor* argmax,
                                                          int64_t pooledH, int64_t pooledW, double spatialScale,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRoiPoolingGradWithArgMax,
                   DFX_IN(gradOutput, gradInputRef, rois, argmax, pooledH, pooledW, spatialScale),
                   DFX_OUT(gradInputRef));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, rois, argmax, gradInputRef, pooledH, pooledW, spatialScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty()) {
        ret = FillEmptyGradOutput(gradInputRef, workspaceSize, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputRefContiguous = l0op::Contiguous(gradInputRef, uniqueExecutor.get());
    CHECK_RET(gradInputRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto roisContiguous = l0op::Contiguous(rois, uniqueExecutor.get());
    CHECK_RET(roisContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto argmaxContiguous = l0op::Contiguous(argmax, uniqueExecutor.get());
    CHECK_RET(argmaxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t numRois = static_cast<int64_t>(rois->GetViewShape().GetDim(DIM0));
    int64_t roiNumArr[] = {numRois, 5};
    aclIntArray* roiNumArray = uniqueExecutor.get()->AllocIntArray(roiNumArr, 2);
    CHECK_RET(roiNumArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto roiActualNumTensor = uniqueExecutor.get()->ConvertToTensor(roiNumArray, op::DataType::DT_INT32);
    CHECK_RET(roiActualNumTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto roiActualNumContiguous = l0op::Contiguous(roiActualNumTensor, uniqueExecutor.get());
    CHECK_RET(roiActualNumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t poolChannel = gradOutput->GetViewShape().GetDim(DIM1);
    auto outOpt = l0op::RoiPoolingGradWithArgMax(gradOutputContiguous, gradInputRefContiguous, roisContiguous,
                                                 roiActualNumContiguous, argmaxContiguous, pooledH, pooledW,
                                                 spatialScale, spatialScale, poolChannel, uniqueExecutor.get());
    CHECK_RET(outOpt != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outCast = l0op::Cast(outOpt, gradInputRef->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(outCast, gradInputRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiPoolingGradWithArgMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoiPoolingGradWithArgMax);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
