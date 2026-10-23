/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_roi_pooling_with_arg_max.h"
#include "roi_pooling_with_arg_max.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/fill.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"
#include "opdev/platform.h"

using namespace op;

static aclnnStatus CheckSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!IsRegBase()) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_FLOAT16};
static constexpr size_t NCHW_DIMS = 4;
static constexpr size_t ROIS_DIM1 = 5;

static bool CheckNotNull(const aclTensor* x, const aclTensor* rois, const aclTensor* y, const aclTensor* argmax)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(rois, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(argmax, return false);
    return true;
}

static bool CheckDtype(const aclTensor* x, const aclTensor* rois, const aclTensor* y, const aclTensor* argmax)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(rois, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, rois->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, y->GetDataType(), return false);
    if (argmax->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "argmax dtype must be DT_INT32.");
        return false;
    }
    return true;
}

static bool CheckFormatValid(const aclTensor* x, const aclTensor* rois, const aclTensor* y, const aclTensor* argmax)
{
    bool formatValid = x->GetStorageFormat() == op::Format::FORMAT_ND &&
                       rois->GetStorageFormat() == op::Format::FORMAT_ND &&
                       y->GetStorageFormat() == op::Format::FORMAT_ND &&
                       argmax->GetStorageFormat() == op::Format::FORMAT_ND;
    if (!formatValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's format should be ND, rois's format should be ND, y's format should be "
                                         "ND, argmax's format should be ND.");
    }
    return formatValid;
}

static bool CheckShape(const aclTensor* x, const aclTensor* rois, int64_t pooled_h, int64_t pooled_w,
                       const aclTensor* y, const aclTensor* argmax)
{
    OP_CHECK_WRONG_DIMENSION(x, NCHW_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(rois, 2, return false);
    if (rois->GetViewShape().GetDim(1) != ROIS_DIM1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rois shape dim1 must be 5, got %zu.", rois->GetViewShape().GetDim(1));
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(y, NCHW_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(argmax, NCHW_DIMS, return false);
    int64_t numRois = static_cast<int64_t>(rois->GetViewShape().GetDim(0));
    int64_t channels = static_cast<int64_t>(x->GetViewShape().GetDim(1));
    if (y->GetViewShape().GetDim(0) != numRois || y->GetViewShape().GetDim(1) != channels ||
        y->GetViewShape().GetDim(2) != pooled_h || y->GetViewShape().GetDim(3) != pooled_w) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y shape must be [%ld, %ld, %ld, %ld].", numRois, channels, pooled_h,
                pooled_w);
        return false;
    }
    if (argmax->GetViewShape().GetDim(0) != numRois || argmax->GetViewShape().GetDim(1) != channels ||
        argmax->GetViewShape().GetDim(2) != pooled_h || argmax->GetViewShape().GetDim(3) != pooled_w) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "argmax shape must be [%ld, %ld, %ld, %ld].", numRois, channels, pooled_h,
                pooled_w);
        return false;
    }
    return true;
}

static bool CheckAttr(int64_t pooled_h, int64_t pooled_w, float spatial_scale_h, float spatial_scale_w)
{
    if (pooled_h <= 0 || pooled_w <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pooled_h and pooled_w must be positive.");
        return false;
    }
    if (spatial_scale_h <= 0.f || spatial_scale_w <= 0.f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "spatial_scale_h and spatial_scale_w must be positive.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* rois, int64_t pooled_h, int64_t pooled_w,
                               float spatial_scale_h, float spatial_scale_w, const aclTensor* y,
                               const aclTensor* argmax)
{
    CHECK_RET(CheckNotNull(x, rois, y, argmax), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(x, rois, y, argmax), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(x, rois, y, argmax), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, rois, pooled_h, pooled_w, y, argmax), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttr(pooled_h, pooled_w, spatial_scale_h, spatial_scale_w), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* GetOutTensorWithValueZero(aclTensor* out, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, out->GetDataType());
    auto outputDims = op::ToShapeVector(out->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outputDims.data(), outputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

aclnnStatus aclnnRoiPoolingWithArgMaxGetWorkspaceSize(const aclTensor* x, const aclTensor* rois, int64_t pooled_h,
                                                      int64_t pooled_w, float spatial_scale_h, float spatial_scale_w,
                                                      aclTensor* y, aclTensor* argmax, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    auto ret = CheckSocValid();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    L2_DFX_PHASE_1(aclnnRoiPoolingWithArgMax, DFX_IN(x, rois, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w),
                   DFX_OUT(y, argmax));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ret = CheckParams(x, rois, pooled_h, pooled_w, spatial_scale_h, spatial_scale_w, y, argmax);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Empty x or empty rois: zero-fill outputs and return success.
    if (x->IsEmpty() || rois->IsEmpty()) {
        auto fillY = GetOutTensorWithValueZero(y, uniqueExecutor.get());
        CHECK_RET(fillY != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyY = l0op::ViewCopy(fillY, y, uniqueExecutor.get());
        CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto fillArgmax = GetOutTensorWithValueZero(argmax, uniqueExecutor.get());
        CHECK_RET(fillArgmax != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyArgmax = l0op::ViewCopy(fillArgmax, argmax, uniqueExecutor.get());
        CHECK_RET(viewCopyArgmax != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto roisContiguous = l0op::Contiguous(rois, uniqueExecutor.get());
    CHECK_RET(roisContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t numRois = static_cast<int64_t>(rois->GetViewShape().GetDim(0));
    int64_t poolChannel = static_cast<int64_t>(x->GetViewShape().GetDim(1));

    int64_t roiNumArr[] = {numRois};
    aclIntArray* roiNumArray = uniqueExecutor.get()->AllocIntArray(roiNumArr, 1);
    CHECK_RET(roiNumArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto roiActualNumTensor = uniqueExecutor.get()->ConvertToTensor(roiNumArray, op::DataType::DT_INT32);
    CHECK_RET(roiActualNumTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* outY = nullptr;
    const aclTensor* outArgmax = nullptr;
    auto outFirst = l0op::RoiPoolingWithArgMax(xContiguous, roisContiguous, roiActualNumTensor, pooled_h, pooled_w,
                                               spatial_scale_h, spatial_scale_w, poolChannel, uniqueExecutor.get(),
                                               &outY, &outArgmax);
    CHECK_RET(outFirst != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outY != nullptr && outArgmax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY = l0op::ViewCopy(outY, y, uniqueExecutor.get());
    CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyArgmax = l0op::ViewCopy(outArgmax, argmax, uniqueExecutor.get());
    CHECK_RET(viewCopyArgmax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiPoolingWithArgMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoiPoolingWithArgMax);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
