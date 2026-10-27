/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ciou.h"
#include "ciou.h"
#include <dlfcn.h>
#include <acl/acl.h>
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t TENSOR_DIM_NUM = 2;
static const int64_t INPUT_FIRST_DIM = 4;
static const int64_t CIoU_RESULT_SIZE = 2;
static const int64_t INPUT_SECOND_DIM_CONSTRAINT = 1024;

static const std::initializer_list<op::DataType> TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor* bBoxes, const aclTensor* gtBoxes, const aclTensor* overlap,
                         const aclTensor* atanSub)
{
    OP_CHECK_NULL(bBoxes, return false);
    OP_CHECK_NULL(gtBoxes, return false);
    OP_CHECK_NULL(overlap, return false);
    OP_CHECK_NULL(atanSub, return false);
    return true;
}

static aclnnStatus CheckSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!IsRegBase()) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

static bool CheckTupleNullptr(std::tuple<aclTensor*, aclTensor*> tensorTuple)
{
    if (std::tuple_size<decltype(tensorTuple)>::value != CIoU_RESULT_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the length of tuple returned by CIoU is not %ld.", CIoU_RESULT_SIZE);
        return false;
    }
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

static bool CheckDtypeValid(const aclTensor* bBoxes, const aclTensor* gtBoxes, const aclTensor* overlap,
                            const aclTensor* atanSub)
{
    OP_CHECK_DTYPE_NOT_MATCH(bBoxes, gtBoxes->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(bBoxes, overlap->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(bBoxes, atanSub->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(bBoxes, TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gtBoxes, TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(overlap, TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(atanSub, TYPE_SUPPORT_LIST, return false);

    OP_CHECK(bBoxes->GetStorageFormat() == Format::FORMAT_ND && gtBoxes->GetStorageFormat() == Format::FORMAT_ND &&
                 overlap->GetStorageFormat() == Format::FORMAT_ND && atanSub->GetStorageFormat() == Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input and output format should be ND."), return false);
    return true;
}

static bool CheckAttr(bool isCross, const char* mode)
{
    if (mode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode is null, please check input arguments");
        return false;
    }
    // 检查self和other能否做数据类型推导
    OP_CHECK(isCross == false,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "isCross should currently only support false, but got %d.", isCross),
             return false);
    OP_CHECK(strcmp(mode, "iou") == 0 || strcmp(mode, "iof") == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode should be iou or iof, but got %s.", mode), return false);
    return true;
}

static bool CheckShape(const aclTensor* bBoxes, const aclTensor* gtBoxes, bool isCross, const aclTensor* overlap,
                       const aclTensor* atanSub)
{
    OP_CHECK(bBoxes->GetViewShape().GetDimNum() == TENSOR_DIM_NUM &&
                 gtBoxes->GetViewShape().GetDimNum() == TENSOR_DIM_NUM &&
                 overlap->GetViewShape().GetDimNum() == TENSOR_DIM_NUM &&
                 atanSub->GetViewShape().GetDimNum() == TENSOR_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputs and outputs should be 2D."), return false);
    OP_CHECK(
        bBoxes->GetViewShape().GetDim(0) == INPUT_FIRST_DIM && gtBoxes->GetViewShape().GetDim(0) == INPUT_FIRST_DIM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the first dim of inputs should be 4."), return false);
    OP_CHECK(bBoxes->GetViewShape().GetDim(1) % INPUT_SECOND_DIM_CONSTRAINT == 0 &&
                 gtBoxes->GetViewShape().GetDim(1) % INPUT_SECOND_DIM_CONSTRAINT == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the second dim of inputs should be a multiple of 1024."), return false);
    if (isCross == false) {
        OP_CHECK(
            bBoxes->GetViewShape().GetDim(1) == gtBoxes->GetViewShape().GetDim(1) &&
                bBoxes->GetViewShape().GetDim(1) == overlap->GetViewShape().GetDim(1) &&
                bBoxes->GetViewShape().GetDim(1) == atanSub->GetViewShape().GetDim(1),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "the second dim of bBoxes, gtBoxes, overlap and atanSub should be equal, when isCross is false."),
            return false);
        OP_CHECK(overlap->GetViewShape().GetDim(0) == 1 && atanSub->GetViewShape().GetDim(0) == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "the first dim of overlap and atanSub should be 1, when isCross is false."),
                 return false);
    } else {
        OP_CHECK(
            gtBoxes->GetViewShape().GetDim(1) == overlap->GetViewShape().GetDim(0) &&
                bBoxes->GetViewShape().GetDim(1) == overlap->GetViewShape().GetDim(1),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "the shape of overlap should be [M, N], where M and N are equal to the second dim of gtBoxes and "
                    "bBoxes respectively, when isCross is true."),
            return false);
        OP_CHECK(
            gtBoxes->GetViewShape().GetDim(1) == atanSub->GetViewShape().GetDim(0) &&
                bBoxes->GetViewShape().GetDim(1) == atanSub->GetViewShape().GetDim(1),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "the shape of atanSub should be [M, N], where M and N are equal to the second dim of gtBoxes and "
                    "bBoxes respectively, when isCross is true."),
            return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* bBoxes, const aclTensor* gtBoxes, bool isCross, const char* mode,
                               aclTensor* overlap, aclTensor* atanSub)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(bBoxes, gtBoxes, overlap, atanSub), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(bBoxes, gtBoxes, overlap, atanSub), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性
    CHECK_RET(CheckAttr(isCross, mode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出输出shape
    CHECK_RET(CheckShape(bBoxes, gtBoxes, isCross, overlap, atanSub), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCIoUGetWorkspaceSize(const aclTensor* bBoxes, const aclTensor* gtBoxes, bool trans, bool isCross,
                                      const char* mode, aclTensor* overlap, aclTensor* atanSub, uint64_t* workspaceSize,
                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCIoU, DFX_IN(bBoxes, gtBoxes, trans, isCross, mode), DFX_OUT(overlap, atanSub));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckSocValid();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckParams(bBoxes, gtBoxes, isCross, mode, overlap, atanSub);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (bBoxes->IsEmpty() || gtBoxes->IsEmpty()) {
        OP_LOGD("aclnnCIoU, dealing with empty tensor.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入bBoxes转换成连续的tensor
    auto bBoxesContiguous = l0op::Contiguous(bBoxes, uniqueExecutor.get());
    CHECK_RET(bBoxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入gtBoxes转换成连续的tensor
    auto gtBoxesContiguous = l0op::Contiguous(gtBoxes, uniqueExecutor.get());
    CHECK_RET(gtBoxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行计算
    std::tuple<aclTensor*, aclTensor*> CIoUOut = l0op::CIoU(bBoxesContiguous, gtBoxesContiguous, trans, isCross, mode,
                                                            true, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(CIoUOut), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出overlap上，overlap可能是非连续的tensor
    auto overlapResult = std::get<0>(CIoUOut);
    CHECK_RET(overlapResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto atansubResult = std::get<1>(CIoUOut);
    CHECK_RET(atansubResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(overlapResult, overlap, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    viewCopyResult = l0op::ViewCopy(atansubResult, atanSub, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCIoU(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCIoU);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
