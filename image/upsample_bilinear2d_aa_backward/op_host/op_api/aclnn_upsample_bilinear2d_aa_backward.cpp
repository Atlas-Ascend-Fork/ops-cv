/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include "upsample_bilinear2d_aa_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "level0/fill.h"
#include "aclnn_upsample_bilinear2d_aa_backward.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 4;
static const double MIN_SUPPORT_SCALE = 0.02;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_INPUT_SIZE = 4;
static constexpr size_t EXPECT_OUTPUT_SIZE = 2;

static bool CheckNotNull(const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize,
                         const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* out, const aclIntArray* inputSize,
                       const aclIntArray* outputSize)
{
    const op::Format gradOutputFormat = gradOutput->GetStorageFormat();
    if (gradOutputFormat != out->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleBilinear2dAaBackward", "gradoutput,out",
                                                (std::string(op::ToString(gradOutputFormat).GetString()) + ", " +
                                                 op::ToString(out->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "format of gradoutput and out must be the same");
        return false;
    }
    size_t inputSizeNum = inputSize->Size();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == EXPECT_INPUT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBilinear2dAaBackward", "input_size",
                                          std::to_string(inputSizeNum).c_str(), "4"),
             return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBilinear2dAaBackward", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "2"),
             return false);
    OP_CHECK(gradOutputFormat == op::Format::FORMAT_ND || gradOutputFormat == op::Format::FORMAT_NCHW,
             OP_LOGE_FOR_INVALID_FORMAT("aclnnUpsampleBilinear2dAaBackward", "self",
                                        op::ToString(gradOutputFormat).GetString(), "NCHW"),
             return false);
    return true;
}

static bool CheckScalesValid(const double weight, const double high)
{
    if (!std::isfinite(weight) || !std::isfinite(high)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnUpsampleBilinear2dAaBackward", "scales_w,scales_h",
            (std::string(std::to_string(weight).c_str()) + ", " + std::to_string(high).c_str()).c_str(),
            "scales must be finite and non-negative");
        return false;
    }
    if ((high < 0) || (weight < 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnUpsampleBilinear2dAaBackward", "scales_w,scales_h",
            (std::string(std::to_string(weight).c_str()) + ", " + std::to_string(high).c_str()).c_str(),
            "scales_w and scales_h cannot be negative");
        return false;
    }
    return true;
}

static double ComputeBilinear2dAABackwardScales(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0) {
        return 1.0 / scale;
    } else {
        return output_size != 0 ? (static_cast<double>(input_size) / output_size) : 0.0;
    }
}

static bool CheckInputElement(const aclIntArray* outputSize, const aclIntArray* inputSize, double scalesH,
                              double scalesW)
{
    int64_t inputN = (*inputSize)[DIM_ZERO];
    int64_t inputC = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];

    OP_CHECK(inputN > 0 && inputC > 0 && inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dAaBackward", "input_n",
                                                      std::to_string(inputN).c_str(),
                                                      "all input and output dims must be greater than 0"),
             return false);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        OP_CHECK(inputN <= INT32_MAX && inputC <= INT32_MAX && inputH <= INT32_MAX && inputW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dAaBackward", "inputN",
                                                          std::to_string(inputN).c_str(),
                                                          "all dims must not exceed INT32_MAX"),
                 return false);
        OP_CHECK(outH <= INT32_MAX && outW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dAaBackward", "outH",
                                                          std::to_string(outH).c_str(),
                                                          "all dims must not exceed INT32_MAX"),
                 return false);

        int64_t M = inputN * inputC * outH;
        OP_CHECK(M <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dAaBackward", "M",
                                                          std::to_string(M).c_str(),
                                                          "N * C * outputSize_H must not exceed INT32_MAX"),
                 return false);

        double realScalesH = ComputeBilinear2dAABackwardScales(inputH, outH, scalesH);
        double realScalesW = ComputeBilinear2dAABackwardScales(inputW, outW, scalesW);
        OP_CHECK(
            realScalesH >= MIN_SUPPORT_SCALE && realScalesW >= MIN_SUPPORT_SCALE,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                "aclnnUpsampleBilinear2dAaBackward", "scales_h,scales_w",
                (std::string(std::to_string(realScalesH).c_str()) + ", " + std::to_string(realScalesW).c_str()).c_str(),
                "scales_h and scales_w are too large; the supported scale range is exceeded"),
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

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOutput, out, inputSize, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 4.检查gradOutput和out N/C轴的大小是否一致
    if (gradOutput->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        CHECK_RET(CheckNCDimValid(gradOutput, out), ACLNN_ERR_PARAM_INVALID);
    }

    // 6.检验scales_w/scales_h，与资料保持一致
    CHECK_RET(CheckScalesValid(scalesW, scalesH), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入元素是否合法
    CHECK_RET(CheckInputElement(outputSize, inputSize, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);

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

aclnnStatus aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize(const aclTensor* gradOutput,
                                                              const aclIntArray* outputSize,
                                                              const aclIntArray* inputSize, bool alignCorners,
                                                              double scalesH, double scalesW, aclTensor* out,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleBilinear2dAABackward,
                   DFX_IN(gradOutput, outputSize, inputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, outputSize, inputSize, scalesH, scalesW, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // An empty out has no elements to write; return success directly.
    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Empty gradOutput with non-empty out must produce zeros instead of leaving out uninitialized.
    if (gradOutput->IsEmpty()) {
        auto fillOut = GetOutTensorWithValueZero(out, uniqueExecutor.get());
        CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(fillOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* upsampleOut;
    if (IsRegBase()) {
        const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

        upsampleOut = l0op::UpsampleBilinear2dAABackward(selfContiguous, outputSize, inputSize, out, alignCorners,
                                                         realScalesH, realScalesW, uniqueExecutor.get());
        CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if ((*inputSize)[DIM_TWO] == (*outputSize)[DIM_ZERO] && (*inputSize)[DIM_THREE] == (*outputSize)[DIM_ONE]) {
        upsampleOut = selfContiguous;
    } else {
        auto dtype = gradOutput->GetDataType();
        if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
            selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
        const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

        // 调用算子计算
        upsampleOut = l0op::UpsampleBilinear2dAABackward(selfContiguous, outputSize, inputSize, out, alignCorners,
                                                         realScalesH, realScalesW, uniqueExecutor.get());
        CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (dtype == op::DataType::DT_BF16) {
            upsampleOut = l0op::Cast(upsampleOut, op::DataType::DT_BF16, uniqueExecutor.get());
        } else if (dtype == op::DataType::DT_FLOAT16) {
            upsampleOut = l0op::Cast(upsampleOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        }
        CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2dAABackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBilinear2dAABackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
