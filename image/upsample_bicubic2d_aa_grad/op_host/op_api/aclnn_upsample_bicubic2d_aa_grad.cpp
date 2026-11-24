/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_bicubic2d_aa_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_upsample_bicubic2d_aa_grad.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 4;
static const double MIN_SUPPORT_SCALE = 0.02;
static constexpr size_t EXPECT_OUTPUTSIZE = 2;

static bool CheckRequiredPointer(const void* ptr, const char* name)
{
    if (ptr != nullptr) {
        return true;
    }
    const std::vector<const char*> keys = {"functionName", "paraName"};
    const std::vector<const char*> values = {"aclnnUpsampleBicubic2dAAGradGetWorkspaceSize", name};
    REPORT_PREDEFINED_ERR_MSG("EZ1011", keys, values);
    return false;
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize,
                         const aclTensor* out)
{
    CHECK_RET(CheckRequiredPointer(gradOutput, "gradOutput"), false);
    CHECK_RET(CheckRequiredPointer(outputSize, "outputSize"), false);
    CHECK_RET(CheckRequiredPointer(inputSize, "inputSize"), false);
    CHECK_RET(CheckRequiredPointer(out, "out"), false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* out)
{
    OP_CHECK(CheckType(gradOut->GetDataType(), DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE("aclnnUpsampleBicubic2dAAGrad", "gradOutput",
                                       op::ToString(gradOut->GetDataType()).GetString(),
                                       op::ToString(DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(gradOut->GetDataType() == out->GetDataType(),
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "gradOutput,out",
                                                    (std::string(op::ToString(gradOut->GetDataType()).GetString()) +
                                                     "," + op::ToString(out->GetDataType()).GetString())
                                                        .c_str(),
                                                    "dtypes must match"),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* out, const aclIntArray* outputSize,
                       const aclIntArray* inputSize)
{
    const op::Format gradOutputFormat = gradOutput->GetStorageFormat();
    if (gradOutputFormat != out->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "gradOutput,out",
                                                (std::string(op::ToString(gradOutputFormat).GetString()) + "," +
                                                 op::ToString(out->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "storage formats must match");
        return false;
    }
    OP_CHECK(gradOutput->GetViewShape().GetDimNum() == DIM_LIMIT,
             OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnUpsampleBicubic2dAAGrad", "gradOutput",
                                          std::to_string(gradOutput->GetViewShape().GetDimNum()).c_str(), "4"),
             return false);
    OP_CHECK(out->GetViewShape().GetDimNum() == DIM_LIMIT,
             OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnUpsampleBicubic2dAAGrad", "out",
                                          std::to_string(out->GetViewShape().GetDimNum()).c_str(), "4"),
             return false);
    size_t inputSizeNum = inputSize->Size();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK(inputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBicubic2dAAGrad", "inputSize",
                                          std::to_string(inputSizeNum).c_str(), "4"),
             return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUTSIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBicubic2dAAGrad", "outputSize",
                                          std::to_string(outputSizeNum).c_str(), "2"),
             return false);
    return true;
}

static bool CheckScalesValid(const double width, const double high)
{
    if (width < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "scalesW", std::to_string(width).c_str(),
                                              "must be nonnegative");
        return false;
    }
    if (high < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "scalesH", std::to_string(high).c_str(),
                                              "must be nonnegative");
        return false;
    }
    return true;
}

static double ComputeBicubic2dAABackwardScales(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0) {
        return 1.0 / scale;
    } else {
        return output_size != 0 ? (static_cast<double>(input_size) / output_size) : 0.0;
    }
}

static bool CheckInputElement(const aclTensor* gradOutput, const aclIntArray* inputSize, const aclTensor* out,
                              double scalesH, double scalesW)
{
    auto gradOutputShape = gradOutput->GetViewShape();
    int64_t outN = 0;
    int64_t outC = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t inputN = (*inputSize)[DIM_ZERO];
    int64_t inputC = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];

    outN = gradOutputShape.GetDim(DIM_ZERO);
    outC = gradOutputShape.GetDim(DIM_ONE);
    outH = gradOutputShape.GetDim(DIM_TWO);
    outW = gradOutputShape.GetDim(DIM_THREE);

    OP_CHECK(outN > 0 && inputH > 0 && inputW > 0 && outC > 0 && outH > 0 && outW > 0,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnUpsampleBicubic2dAAGrad", "gradOutput,inputSize",
                 (std::string(op::ToString(gradOutputShape).GetString()) + "," + std::to_string(inputN) + "," +
                  std::to_string(inputC) + "," + std::to_string(inputH) + "," + std::to_string(inputW))
                     .c_str(),
                 "gradOutput N,C,H,W and inputSize H,W must be positive"),
             return false);
    OP_CHECK(
        (gradOutput->GetStorageFormat() == op::Format::FORMAT_ND ||
         gradOutput->GetStorageFormat() == op::Format::FORMAT_NCHW) &&
            (out->GetStorageFormat() == op::Format::FORMAT_ND || out->GetStorageFormat() == op::Format::FORMAT_NCHW),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "gradOutput,out",
                                                (std::string(op::ToString(gradOutput->GetStorageFormat()).GetString()) +
                                                 "," + op::ToString(out->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "only ND and NCHW are supported"),
        return false);
    if (!IsRegBase()) {
        double realScalesH = ComputeBicubic2dAABackwardScales(inputH, outH, scalesH);
        double realScalesW = ComputeBicubic2dAABackwardScales(inputW, outW, scalesW);
        OP_CHECK(realScalesH >= MIN_SUPPORT_SCALE && realScalesW >= MIN_SUPPORT_SCALE,
                 OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                     "aclnnUpsampleBicubic2dAAGrad", "scalesH,scalesW",
                     (std::to_string(realScalesH) + "," + std::to_string(realScalesW)).c_str(),
                     "the effective backward scales must be at least 0.02"),
                 return false);
        OP_CHECK(outN <= INT32_MAX && outC <= INT32_MAX && outH <= INT32_MAX && outW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "gradOutput",
                                                       op::ToString(gradOutputShape).GetString(),
                                                       "dimensions must not exceed INT32_MAX"),
                 return false);
        OP_CHECK(inputN <= INT32_MAX && inputC <= INT32_MAX && inputH <= INT32_MAX && inputW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "inputSize",
                                                       (std::to_string(inputN) + "," + std::to_string(inputC) + "," +
                                                        std::to_string(inputH) + "," + std::to_string(inputW))
                                                           .c_str(),
                                                       "elements must not exceed INT32_MAX"),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               double scalesH, double scalesW, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, outputSize, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 3.检验scales_w/scales_h，与资料保持一致
    CHECK_RET(CheckScalesValid(scalesW, scalesH), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(gradOutput, out, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 5.检查gradOutput和out N/C轴的大小是否一致
    if (gradOutput->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        OP_CHECK(
            gradOutput->GetViewShape().GetDim(0) == out->GetViewShape().GetDim(0) &&
                gradOutput->GetViewShape().GetDim(1) == out->GetViewShape().GetDim(1),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnUpsampleBicubic2dAAGrad", "gradOutput,out",
                                                   (std::string(op::ToString(gradOutput->GetViewShape()).GetString()) +
                                                    "," + op::ToString(out->GetViewShape()).GetString())
                                                       .c_str(),
                                                   "N and C dimensions must match"),
            return ACLNN_ERR_PARAM_INVALID);
    }

    // 6. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(gradOutput, inputSize, out, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(const aclTensor* gradOutput, const aclIntArray* outputSize,
                                                         const aclIntArray* inputSize, bool alignCorners,
                                                         double scalesH, double scalesW, aclTensor* out,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(CheckRequiredPointer(workspaceSize, "workspaceSize"), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckRequiredPointer(executor, "executor"), ACLNN_ERR_PARAM_NULLPTR);

    L2_DFX_PHASE_1(aclnnUpsampleBicubic2dAAGrad,
                   DFX_IN(gradOutput, outputSize, inputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto checkRet = CheckParams(gradOutput, outputSize, inputSize, scalesH, scalesW, out);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    if (gradOutput->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入out转换成连续的tensor
    auto out_contiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(out_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
    const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
    const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

    // 调用算子计算
    const aclTensor* upsampleOut = l0op::UpsampleBicubic2dAAGrad(
        selfContiguous, outputSize, inputSize, out, alignCorners, realScalesH, realScalesW, uniqueExecutor.get());
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dAAGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBicubic2dAAGrad);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
