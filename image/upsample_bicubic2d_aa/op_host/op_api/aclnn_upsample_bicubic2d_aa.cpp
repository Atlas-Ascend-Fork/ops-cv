/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_upsample_bicubic2d_aa.h"
#include "upsample_bicubic2d_aa.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_BF16};

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 2;
static const int64_t DIM_LIMIT = 4;
static constexpr float MAX_SUPPORT_SCALE = 50.0;

static bool CheckNotNull(const aclTensor* x, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(x, DIM_LIMIT, return false);
    // Validate out rank before reading its dimensions so a malformed output shape cannot be indexed out of bounds.
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBicubic2dAa", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "2"),
             return false);
    auto inputShape = x->GetViewShape();
    auto outputShape = out->GetViewShape();
    int64_t inputN = inputShape.GetDim(DIM_ZERO);
    int64_t inputC = inputShape.GetDim(DIM_ONE);
    int64_t inputH = inputShape.GetDim(DIM_TWO);
    int64_t inputW = inputShape.GetDim(DIM_THREE);
    int64_t outputN = outputShape.GetDim(DIM_ZERO);
    int64_t outputC = outputShape.GetDim(DIM_ONE);
    int64_t outputH = outputShape.GetDim(DIM_TWO);
    int64_t outputW = outputShape.GetDim(DIM_THREE);
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];
    OP_CHECK(
        inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2dAa", "input_h", std::to_string(inputH).c_str(),
                                                 "all input and output dims must be greater than 0"),
        return false);
    OP_CHECK(
        inputC > 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnUpsampleBicubic2dAa", "self", op::ToString(inputShape).GetString(),
                                              "a non-empty 4D data tensor is required"),
        return false);
    OP_CHECK(inputN == outputN && inputC == outputC,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnUpsampleBicubic2dAa", "input,output",
                                                    (std::string(op::ToString(x->GetViewShape()).GetString()) + ", " +
                                                     op::ToString(out->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "N and C of input and output must be equal"),
             return false);
    OP_CHECK(outputH == outH && outputW == outW,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnUpsampleBicubic2dAa", "output,out",
                                                    (std::string(op::ToString(out->GetViewShape()).GetString()) + ", " +
                                                     op::ToString(out->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "H and W of output tensor must match outputSize"),
             return false);
    OP_CHECK(
        (x->GetStorageFormat() == op::Format::FORMAT_ND || x->GetStorageFormat() == op::Format::FORMAT_NCHW) &&
            (out->GetStorageFormat() == op::Format::FORMAT_ND || out->GetStorageFormat() == op::Format::FORMAT_NCHW),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleBicubic2dAa", "input,output",
                                                (std::string(op::ToString(x->GetStorageFormat()).GetString()) + ", " +
                                                 op::ToString(out->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "only NCHW is supported for input and output storage format"),
        return false);

    // 检查上边界
    if (!IsRegBase()) {
        OP_CHECK(inputN <= INT32_MAX && inputC <= INT32_MAX && inputH <= INT32_MAX && inputW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2dAa", "inputN",
                                                          std::to_string(inputN).c_str(),
                                                          "all dims must not exceed INT32_MAX"),
                 return false);
        OP_CHECK(
            outH <= INT32_MAX && outW <= INT32_MAX,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2dAa", "outH", std::to_string(outH).c_str(),
                                                     "all dims must not exceed INT32_MAX"),
            return false);
    }
    return true;
}

static bool CheckMaxScaleSupport(const aclTensor* x, const aclIntArray* outputSize, const double scalesH,
                                 const double scalesW)
{
    auto selfShape = x->GetViewShape();
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : static_cast<float>(inputH) / outputH;
    const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : static_cast<float>(inputW) / outputW;

    OP_CHECK(
        realScalesH <= MAX_SUPPORT_SCALE && realScalesW <= MAX_SUPPORT_SCALE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnUpsampleBicubic2dAa", "scales_h,scales_w",
            (std::to_string(realScalesH).c_str() + std::string(", ") + std::to_string(realScalesW).c_str()).c_str(),
            "scales must be less than 50"),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclIntArray* outputSize, const aclTensor* out,
                               const double scalesH, const double scalesW)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(x, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查scale是否支持
    if (!(IsRegBase())) {
        CHECK_RET(CheckMaxScaleSupport(x, outputSize, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(const aclTensor* x, const aclIntArray* outputSize,
                                                     const bool alignCorners, const double scalesH,
                                                     const double scalesW, aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleBicubic2dAA, DFX_IN(x, outputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckNotNull(x, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);
    // 固定写法，参数检查
    auto ret = CheckParams(x, outputSize, out, scalesH, scalesW);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空tensor在kernel中支持
    if (x->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    float realScalesH = scalesH > 0 ? static_cast<float>(scalesH) : 0;
    float realScalesW = scalesW > 0 ? static_cast<float>(scalesW) : 0;
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (!(IsRegBase(curArch))) {
        auto dtype = x->GetDataType();
        // 将fp16/bf16类型cast成fp32处理，保证精度
        if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
            selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 先用double算好1/scale再转float，减少精度损失
        realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;
    }

    // 调用UpsampleBicubic2dAA算子kernel
    const aclTensor* upsampleOut = l0op::UpsampleBicubic2dAA(selfContiguous, outputSize, alignCorners, out, realScalesH,
                                                             realScalesW, uniqueExecutor.get());
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (!(IsRegBase(curArch))) {
        auto dtype = x->GetDataType();
        if (dtype == op::DataType::DT_BF16) {
            // CAST回bf16
            upsampleOut = l0op::Cast(upsampleOut, op::DataType::DT_BF16, uniqueExecutor.get());
        } else if (dtype == op::DataType::DT_FLOAT16) {
            // CAST回fp16
            upsampleOut = l0op::Cast(upsampleOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        }
    }
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dAA(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBicubic2dAA);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
