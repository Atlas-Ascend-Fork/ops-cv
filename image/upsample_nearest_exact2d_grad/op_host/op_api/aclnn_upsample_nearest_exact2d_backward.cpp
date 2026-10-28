/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "upsample_nearest_exact2d_grad.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include <cmath>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"
#include "aclnn_upsample_nearest_exact2d_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DIM_LIMIT = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 4;
static const double MIN_SUPPORT_SCALE = 0.02;
static constexpr size_t EXPECT_OUTPUTSIZE = 2;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* out, const aclIntArray* outputSize,
                       const aclIntArray* inputSize)
{
    const op::Format gradOutputFormat = gradOutput->GetStorageFormat();
    if (gradOutputFormat != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, gradOutput [%s], out [%s].",
                op::ToString(gradOutputFormat).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    size_t inputSizeNum = inputSize->Size();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == EXPECT_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected input_size to be 4, but got %zu", inputSizeNum), return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUTSIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected output_size to be 2, but got %zu", outputSizeNum),
             return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, out->GetDataType(), return false);
    return true;
}

static double ComputeNearestExact2dBackwardScales(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0) {
        return 1.0 / scale;
    } else {
        return output_size != 0 ? (static_cast<double>(input_size) / output_size) : 0.0;
    }
}

static bool CheckScalesValid(const double weight, const double high)
{
    if ((weight < 0) || (high < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scales_w and scales_h cannot be negative , scales_w [%f], scales_h [%f].",
                weight, high);
        return false;
    }
    return true;
}

static bool CheckInputElement(const aclTensor* gradOutput, const aclTensor* out, const aclIntArray* outputSize,
                              const aclIntArray* inputSize, double scalesH, double scalesW)
{
    auto gradOutputShape = gradOutput->GetViewShape();
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    int64_t outN = 0;
    int64_t outC = 0;
    int64_t outH = 0;
    int64_t outW = 0;

    outN = gradOutputShape.GetDim(DIM_ZERO);
    outC = gradOutputShape.GetDim(DIM_ONE);
    outH = gradOutputShape.GetDim(DIM_TWO);
    outW = gradOutputShape.GetDim(DIM_THREE);

    double realScalesH = ComputeNearestExact2dBackwardScales(inputH, outH, scalesH);
    double realScalesW = ComputeNearestExact2dBackwardScales(inputW, outW, scalesW);

    OP_CHECK(realScalesH >= MIN_SUPPORT_SCALE && realScalesW >= MIN_SUPPORT_SCALE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scalesH and scalesW are too large, scalesH [%f], scalesW [%f].",
                     realScalesH, realScalesW),
             return false);

    OP_CHECK(inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Input and output sizes should be greater than 0 but got input (N: %ld, C: %ld,"
                     " H: %ld, W: %ld) output (H: %ld, W: %ld)",
                     outN, outC, inputH, inputW, outH, outW),
             return false);

    // gradOutput 的 shape 应等于 (inputSize[0], inputSize[1], outputSize[0], outputSize[1])，out 的 shape
    // 应等于 (inputSize[0], inputSize[1], inputSize[2], inputSize[3])，N/C 取自 inputSize，
    // 防止 kernel 按 outputSize/inputSize 读写与实际形状不符的内存
    op::Shape expectGradOutputShape = op::Shape{(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], (*outputSize)[DIM_ZERO],
                                                (*outputSize)[DIM_ONE]};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradOutput, expectGradOutputShape, return false);
    op::Shape expectOutShape = op::Shape{(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], inputH, inputW};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectOutShape, return false);

    OP_CHECK((*inputSize)[DIM_ZERO] >= 0 && (*inputSize)[DIM_ONE] >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input N/C sizes should be non-negative, but got N: %ld, C: %ld.",
                     (*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE]),
             return false);

    static constexpr double EPSILON = 1e-6;
    // 资料约束：outputSize = floor(inputSize * scales)，非整数乘积时合法值是 floor 而非 ceil；
    // 绝对容差仅覆盖乘积为整数时的浮点表示噪声，不随乘积放大，不放过 off-by-one 的非法组合
    if (scalesH > 0) {
        int64_t outSizeH = (*outputSize)[DIM_ZERO];
        double expectDimH = static_cast<double>(inputH) * scalesH;
        OP_CHECK(static_cast<double>(outSizeH) == std::floor(expectDimH) ||
                     fabs(static_cast<double>(outSizeH) - expectDimH) <= EPSILON,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Scale conflicts with outputSize. outputSize_H should be equal to inputSize_H * scalesH"),
                 return false);
    }
    if (scalesW > 0) {
        int64_t outSizeW = (*outputSize)[DIM_ONE];
        double expectDimW = static_cast<double>(inputW) * scalesW;
        OP_CHECK(static_cast<double>(outSizeW) == std::floor(expectDimW) ||
                     fabs(static_cast<double>(outSizeW) - expectDimW) <= EPSILON,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Scale conflicts with outputSize. outputSize_W should be equal to inputSize_W * scalesW"),
                 return false);
    }
    return true;
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclIntArray* inputSize, const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckNCDimEqual(const aclTensor* self, const aclTensor* out)
{
    auto format = self->GetStorageFormat();
    int64_t selfDimN = self->GetViewShape().GetDim(DIM_ZERO);
    int64_t selfDimC = self->GetViewShape().GetDim(DIM_ONE);
    int64_t outDimN = out->GetViewShape().GetDim(DIM_ZERO);
    int64_t outDimC = out->GetViewShape().GetDim(DIM_ONE);
    if (format == op::Format::FORMAT_NHWC) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input and output format only supports [NCHW, ND] format.");
        return false;
    }
    if ((outDimC != selfDimC) || (outDimN != selfDimN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "selfDimC[%ld]/outDimC[%ld] or selfDimN[%ld]/outDimN[%ld] are not equal.",
                selfDimC, outDimC, selfDimN, outDimN);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               double scalesH, double scalesW, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOutput, out, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4.检查gradOutput和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimEqual(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    // 5.检验scales_w/scales_h，与资料保持一致
    CHECK_RET(CheckScalesValid(scalesW, scalesH), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入元素是否合法（含 gradOutput/out 形状与 outputSize/inputSize 的交叉校验）
    CHECK_RET(CheckInputElement(gradOutput, out, outputSize, inputSize, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact2dBackwardGetWorkspaceSize(const aclTensor* gradOutput,
                                                                const aclIntArray* outputSize,
                                                                const aclIntArray* inputSize, double scalesH,
                                                                double scalesW, aclTensor* out, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact2dBackward, DFX_IN(gradOutput, outputSize, inputSize, scalesH, scalesW),
                   DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, outputSize, inputSize, scalesH, scalesW, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

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
    const float realScalesH = scalesH > 0 ? static_cast<float>(scalesH) : 0;
    const float realScalesW = scalesW > 0 ? static_cast<float>(scalesW) : 0;

    // 升精度计算
    auto dataType = gradOutput->GetDataType();
    bool isRegBase = IsRegBase();
    if (!isRegBase && (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType)) {
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用算子计算
    const aclTensor* upsampleOut = l0op::UpsampleNearestExact2dGrad(
        selfContiguous, outputSize, inputSize, out, realScalesH, realScalesW, true, uniqueExecutor.get());
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 转回初始精度
    if (!isRegBase && (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType)) {
        upsampleOut = l0op::Cast(upsampleOut, dataType, uniqueExecutor.get());
        CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
