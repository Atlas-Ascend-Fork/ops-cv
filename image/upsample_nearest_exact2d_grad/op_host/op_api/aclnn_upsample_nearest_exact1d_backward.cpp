/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn/aclnn_base.h"

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

#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "upsample_nearest_exact2d_grad.h"
#include "aclnn_upsample_nearest_exact1d_backward.h"
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

static const int64_t DIM_LIMIT = 3;
static const double MIN_SUPPORT_SCALE = 0.02;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 3;
static constexpr size_t EXPECT_OUTPUTSIZE = 1;

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, out->GetDataType(), return false);
    return true;
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclIntArray* inputSize, const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* out, const aclIntArray* outputSize,
                       const aclIntArray* inputSize)
{
    const op::Format gradOutputFormat1D = gradOutput->GetStorageFormat();
    if (gradOutputFormat1D != out->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleNearestExact1dBackward", "gradoutput,out",
                                                (std::string(op::ToString(gradOutputFormat1D).GetString()) + ", " +
                                                 op::ToString(out->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "format of gradoutput and out must be the same");
        return false;
    }
    size_t inputSizeNum = inputSize->Size();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleNearestExact1dBackward", "input_size",
                                          std::to_string(inputSizeNum).c_str(), "3"),
             return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUTSIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleNearestExact1dBackward", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "1"),
             return false);
    return true;
}

static bool CheckScalesValid(const double weight)
{
    if (weight < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnUpsampleNearestExact1dBackward", "scales",
                                              std::to_string(weight).c_str(), "scales cannot be negative");
        return false;
    }
    return true;
}

static double ComputeNearestExact1dBackwardScales(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0) {
        return 1.0 / scale;
    } else {
        return output_size != 0 ? (static_cast<double>(input_size) / output_size) : 0.0;
    }
}

static constexpr double EPSILON = 1e-6;

static bool CheckInputElement(const aclTensor* gradOutput, const aclTensor* out, const aclIntArray* outputSize,
                              const aclIntArray* inputSize, double scales)
{
    auto gradOutputShape = gradOutput->GetViewShape();
    int64_t outN = 0;
    int64_t outC = 0;
    int64_t outL = 0;
    int64_t inputL = (*inputSize)[DIM_TWO];

    outN = gradOutputShape.GetDim(DIM_ZERO);
    outC = gradOutputShape.GetDim(DIM_ONE);
    outL = gradOutputShape.GetDim(DIM_TWO);

    double realScales = ComputeNearestExact1dBackwardScales(inputL, outL, scales);
    OP_CHECK(realScales >= MIN_SUPPORT_SCALE,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnUpsampleNearestExact1dBackward", "scales",
                                                   std::to_string(realScales).c_str(),
                                                   "scales is too large; the supported scale range is exceeded"),
             return false);

    OP_CHECK(outN > 0 && inputL > 0 && outC > 0 && outL > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleNearestExact1dBackward", "input_l",
                                                      std::to_string(inputL).c_str(),
                                                      "all input and output dims must be greater than 0"),
             return false);

    // gradOutput 的 shape 应等于 (inputSize[0], inputSize[1], outputSize[0])，out 的 shape 应等于
    // (inputSize[0], inputSize[1], inputSize[2])，N/C 取自 inputSize（对齐 3d 反向），
    // 防止 kernel 按 outputSize/inputSize 读写与实际形状不符的内存
    op::Shape expectGradOutputShape = op::Shape{(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], (*outputSize)[DIM_ZERO]};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradOutput, expectGradOutputShape, return false);
    op::Shape expectOutShape = op::Shape{(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], inputL};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectOutShape, return false);

    if (scales > 0) {
        // 资料约束：outputSize = floor(inputSize_L * scales)，非整数乘积时合法值是 floor 而非 ceil；
        // 绝对容差仅覆盖乘积为整数时的浮点表示噪声，不随乘积放大，不放过 off-by-one 的非法组合
        int64_t outSizeL = (*outputSize)[DIM_ZERO];
        double expectDimL = static_cast<double>(inputL) * scales;
        OP_CHECK(static_cast<double>(outSizeL) == std::floor(expectDimL) ||
                     fabs(static_cast<double>(outSizeL) - expectDimL) <= EPSILON,
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                     "aclnnUpsampleNearestExact1dBackward", "scales", std::to_string(scales).c_str(),
                     "scale conflicts with outputSize; outputSize_L should be equal to inputSize_L * scales"),
                 return false);
    }
    return true;
}

static bool CheckNCDimEqual(const aclTensor* self, const aclTensor* out)
{
    int64_t selfDimN = self->GetViewShape().GetDim(DIM_ZERO);
    int64_t selfDimC = self->GetViewShape().GetDim(DIM_ONE);
    int64_t outDimN = out->GetViewShape().GetDim(DIM_ZERO);
    int64_t outDimC = out->GetViewShape().GetDim(DIM_ONE);
    if ((selfDimC != outDimC) || (selfDimN != outDimN)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleNearestExact1dBackward", "self_dim_c",
                                                 std::to_string(selfDimC).c_str(),
                                                 "dim C of self must match dim C of out (same for dim N)");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               double scales, const aclTensor* out)
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
    CHECK_RET(CheckScalesValid(scales), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入元素是否合法（含 gradOutput/out 形状与 outputSize/inputSize 的交叉校验）
    CHECK_RET(CheckInputElement(gradOutput, out, outputSize, inputSize, scales), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* View4dAs3d(const aclTensor* input, const aclTensor* out, aclOpExecutor* executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t removeDim[] = {2};
    aclIntArray* dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    auto reformatInput = l0op::ReFormat(squeezedInput, out->GetStorageFormat());
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

static const aclTensor* View3dAs4d(const aclTensor* input, aclOpExecutor* executor)
{
    // NCL -> contigious -> unsqueeze(2) -> reformat -> NCHW
    // contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    const int64_t appendDim[] = {2};
    aclIntArray* dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

aclnnStatus aclnnUpsampleNearestExact1dBackwardGetWorkspaceSize(const aclTensor* gradOutput,
                                                                const aclIntArray* outputSize,
                                                                const aclIntArray* inputSize, double scales,
                                                                aclTensor* out, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact1dBackward, DFX_IN(gradOutput, outputSize, inputSize, scales),
                   DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, outputSize, inputSize, scales, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor，3维视作4维
    auto selfContiguous = View3dAs4d(gradOutput, uniqueExecutor.get());

    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outContiguous = View3dAs4d(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const int64_t inputSizeList[] = {(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], 1, (*inputSize)[DIM_TWO]};
    auto inputSizeArray = uniqueExecutor.get()->AllocIntArray(inputSizeList, 4);
    CHECK_RET(inputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t outputSizeList[] = {1, (*outputSize)[DIM_ZERO]};
    auto outputSizeArray = uniqueExecutor.get()->AllocIntArray(outputSizeList, 2);
    CHECK_RET(outputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
    const float realScales_w = scales > 0 ? static_cast<float>(scales) : 0;
    const float realScales_h = static_cast<float>(1.0);

    // 升精度计算
    auto dataType = gradOutput->GetDataType();
    bool isRegBase = IsRegBase();
    if (!isRegBase && (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType)) {
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用算子计算
    const aclTensor* upsampleOut = l0op::UpsampleNearestExact2dGrad(
        selfContiguous, outputSizeArray, inputSizeArray, out, realScales_h, realScales_w, true, uniqueExecutor.get());
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 转回初始精度
    if (!isRegBase && (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType)) {
        upsampleOut = l0op::Cast(upsampleOut, dataType, uniqueExecutor.get());
        CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* resizeNearestOut = nullptr;
    resizeNearestOut = l0op::TransData(upsampleOut, outContiguous->GetStorageFormat(), 0, uniqueExecutor.get());
    CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* out3d = nullptr;
    out3d = View4dAs3d(resizeNearestOut, out, uniqueExecutor.get());
    auto viewCopyResult = l0op::ViewCopy(out3d, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact1dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact1dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
