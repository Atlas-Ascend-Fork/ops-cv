/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "resize_bilinear_v2_grad.h"
#include "aclnn_upsample_bilinear_2d_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr size_t DIM_LIMIT = 4;
static constexpr int64_t EXPECT_SIZE = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                         const aclTensor* out)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckIOSizesIsSame(const aclTensor* gradOut, const aclIntArray* inputSize)
{
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != (*inputSize)[i]) {
            return false;
        }
    }
    return true;
}

static bool ChecksInputElement(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();
    FVector<int64_t> fullOutputSize = {batch, channels, outH, outW};

    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputH = (*inputSize)[DIM_ONE];
        inputW = (*inputSize)[DIM_TWO];
        channels = (*inputSize)[DIM_THREE];
        fullOutputSize[DIM_ONE] = outH;
        fullOutputSize[DIM_TWO] = outW;
        fullOutputSize[DIM_THREE] = channels;
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Input and output sizes should be greater than 0, but got input (H: %ld,"
                     " W: %ld) output (H: %ld, W: %ld)",
                     inputH, inputW, outH, outW),
             return false);

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected grad_output to have the same shape as output;"
                    " output.size(%zu) = %ld but got grad_output.size(%zu) = %ld",
                    i, fullOutputSize[i], i, gradOutShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* out, const aclIntArray* inputSize)
{
    if (IsRegBase()) {
        // 资料约定（npu="950"）：gradOut为FLOAT时out可为FLOAT32/FLOAT16/BFLOAT16，否则out与gradOut保持一致，
        // 该规则与gradOut和inputSize是否同尺寸无关，需覆盖同尺寸与非同尺寸两条路径
        OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST_REGBASE, return false);
        if (gradOut->GetDataType() != op::DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_MATCH(out, gradOut->GetDataType(), return false);
        } else {
            OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_REGBASE, return false);
        }
        return true;
    }
    if (!CheckIOSizesIsSame(gradOut, inputSize)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, gradOut->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* gradOut, const aclTensor* out)
{
    OP_CHECK(gradOut->GetStorageFormat() == out->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gradOut and out must have the same format"), return false);
    OP_CHECK((out->GetStorageFormat() == op::Format::FORMAT_NCHW || out->GetStorageFormat() == op::Format::FORMAT_NHWC),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The format must be NCHW or NHWC"), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                       const aclTensor* out)
{
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected output_size to be 2, but got %zu", outputSizeNum),
             return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected input_size to be 4, but got %zu", inputSizeNum), return false);

    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    int64_t N = 0;
    int64_t C = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    FVector<int64_t> fullOutputSize = {0, 0, 0, 0};
    FVector<int64_t> fullInputSize = {0, 0, 0, 0};
    if (out->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        N = (*inputSize)[DIM_ZERO];
        C = (*inputSize)[DIM_ONE];
        inputH = (*inputSize)[DIM_TWO];
        inputW = (*inputSize)[DIM_THREE];
        fullOutputSize = {N, C, outputH, outputW};
        fullInputSize = {N, C, inputH, inputW};
    } else {
        N = (*inputSize)[DIM_ZERO];
        inputH = (*inputSize)[DIM_ONE];
        inputW = (*inputSize)[DIM_TWO];
        C = (*inputSize)[DIM_THREE];
        fullOutputSize = {N, outputH, outputW, C};
        fullInputSize = {N, inputH, inputW, C};
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outputH > 0 && outputW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Size of H and W must be greater than 0, but got input (H: %ld, W: %ld), "
                     "output (H: %ld, W: %ld)",
                     inputH, inputW, outputH, outputW),
             return false);

    auto gradOutShape = gradOut->GetViewShape();
    auto outShape = out->GetViewShape();
    for (size_t i = 0; i < DIM_FOUR; ++i) {
        OP_CHECK(gradOutShape.GetDim(i) == fullOutputSize[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim %zu of gradOut should be %ld, but got %ld", i,
                         fullOutputSize[i], gradOutShape.GetDim(i)),
                 return false);
        OP_CHECK(outShape.GetDim(i) == fullInputSize[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim %zu of out should be %ld, but got %ld", i, fullInputSize[i],
                         outShape.GetDim(i)),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查shape
    CHECK_RET(CheckShape(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOut, out, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 校验gradOut的shape是否与输出的output的shape一致
    CHECK_RET(ChecksInputElement(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 5.检查gradOut和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(gradOut, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查format
    if (IsRegBase()) {
        CHECK_RET(CheckFormat(gradOut, out), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                            const aclIntArray* inputSize, bool alignCorners,
                                                            double scalesH, double scalesW, aclTensor* out,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleBilinear2dBackward,
                   DFX_IN(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, outputSize, inputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOut->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* outTransdata = gradOutContiguous;
    if (!CheckIOSizesIsSame(gradOut, inputSize)) {
        bool halfPixelCenters = !alignCorners;
        op::Shape imageShape;
        imageShape.SetDimNum(DIM_LIMIT);
        for (size_t i = 0; i < DIM_LIMIT; ++i) {
            imageShape.SetDim(i, (*inputSize)[i]);
        }

        if (IsRegBase()) {
            auto originalImage = uniqueExecutor.get()->AllocTensor(imageShape, out->GetDataType(),
                                                                   out->GetViewFormat());
            CHECK_RET(originalImage != nullptr, ACLNN_ERR_INNER_NULLPTR);

            const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
            auto scales = uniqueExecutor->AllocFloatArray(scalesList, DIM_TWO);
            CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

            outTransdata = l0op::ResizeBilinearV2Grad(gradOutContiguous, originalImage, alignCorners, halfPixelCenters,
                                                      scales, uniqueExecutor.get());
            CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto gradOutTransdata = l0op::TransDataSpecial(gradOutContiguous, op::Format::FORMAT_NC1HWC0, 0,
                                                           uniqueExecutor.get());
            CHECK_RET(gradOutTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto gradOutCast = l0op::CastOnlyForConvBackward(gradOutTransdata, op::DataType::DT_FLOAT,
                                                             uniqueExecutor.get());
            CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto image = uniqueExecutor.get()->AllocTensor(imageShape, gradOutCast->GetDataType(),
                                                           gradOutCast->GetViewFormat());
            CHECK_RET(image != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto imageTransdata = l0op::TransDataSpecial(image, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
            CHECK_RET(imageTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto v2GradOut = l0op::ResizeBilinearV2Grad5Hd(gradOutCast, imageTransdata, alignCorners, halfPixelCenters,
                                                           uniqueExecutor.get());
            CHECK_RET(v2GradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto outCast = l0op::CastOnlyForConvBackward(v2GradOut, out->GetDataType(), uniqueExecutor.get());
            CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

            outTransdata = l0op::TransData(outCast, out->GetStorageFormat(), 0, uniqueExecutor.get());
            CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else if (gradOutContiguous->GetDataType() != out->GetDataType()) {
        // 同尺寸且gradOut与out数据类型不一致（gradOut为FLOAT、out为FLOAT16/BFLOAT16时资料允许）：
        // 恒等反向仅做数据类型转换，与非同尺寸路径kernel直接产出out数据类型的行为保持一致
        outTransdata = l0op::Cast(gradOutContiguous, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(CheckReduceOutShape(outTransdata, out), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(outTransdata, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2dBackward(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBilinear2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

#ifdef __cplusplus
}
#endif
