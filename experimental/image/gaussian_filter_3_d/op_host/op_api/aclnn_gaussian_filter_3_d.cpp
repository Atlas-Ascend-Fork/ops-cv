/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

#include "aclnn_gaussian_filter_3_d.h"
#include "gaussian_filter_3_d.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def_cv.h"
#include "aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                   DataType::DT_BF16};

static const int64_t MAX_SUPPORT_RANK = 5;
static const int64_t MIN_SUPPORT_RANK = 3;
static constexpr double MAX_SIGMA = 100.0;

static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    int64_t dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (dimNum < MIN_SUPPORT_RANK || dimNum > MAX_SUPPORT_RANK) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x rank should be in [3, 5], but got %ld.", dimNum);
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
    return true;
}

static bool CheckSigmaValid(double sigma)
{
    if (sigma < 0.0 || sigma > MAX_SIGMA) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sigma should be in [0, %f], but got %f.", MAX_SIGMA, sigma);
        return false;
    }
    return true;
}

static bool CheckModeValid(const char* mode)
{
    if (mode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mode should not be null.");
        return false;
    }
    std::string modeStr(mode);
    if (modeStr != "REFLECT" && modeStr != "CONSTANT" && modeStr != "NEAREST" && modeStr != "MIRROR" &&
        modeStr != "WRAP") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode only support REFLECT/CONSTANT/NEAREST/MIRROR/WRAP, but got %s.", mode);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, double sigma, const char* mode, const aclTensor* y)
{
    CHECK_RET(CheckNotNull(x, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSigmaValid(sigma), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckModeValid(mode), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGaussianFilter3DGetWorkspaceSize(const aclTensor* x, double sigma, const char* mode,
                                                  const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGaussianFilter3D, DFX_IN(x, sigma, mode), DFX_OUT(y));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, sigma, mode, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x如果非连续，需要转连续
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子GaussianFilter3D进行计算
    auto result = l0op::GaussianFilter3D(xContiguous, sigma, mode, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(result, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGaussianFilter3D(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGaussianFilter3D);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
