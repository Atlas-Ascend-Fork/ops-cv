/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "resize_grad_d.h"
#include "upsample_bicubic2d_grad.h"
#include "image/resize_bicubic_v2_grad/op_api/resize_bicubic_v2_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_upsample_bicubic_2d_backward.h"

#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "op_api/aclnn_check.h"
#include "op_api/level2_base_cv.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
// According to the API definition, all supported dtypes need to be listed
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 4;
static const int64_t EXPECT_SIZE = 2;
static const int64_t FOURDIMS = 4;
static const int64_t FP16_DETERMINSTIC_CAST = 1;
static const int64_t BF16_CAST = 2;
static const float MAX_SCALE = 50;
static const float HALF_ONE = 0.5;
static const float MAX_W_RADIO = 10000;
static const float MAX_W = 130000;

static const std::string CUBIC_MODE = "cubic";

static bool CheckNotNull(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                         const aclTensor* gradInput)
{
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(gradInput, return false);
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(outputSize, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* gradInput)
{
    // Check if the data type of gradOut is in the supported list of
    // the ResizeBicubicV2Grad operator
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST_REGBASE, return false);
    } else {
        bool is910BSocVersion = (curArch == NpuArch::DAV_2201 || IsRegBase(curArch));
        const std::initializer_list<op::DataType> dtypeSupportList = is910BSocVersion ? DTYPE_SUPPORT_LIST_910B :
                                                                                        DTYPE_SUPPORT_LIST_910;

        OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, dtypeSupportList, return false);
    }

    // Check that the data type of gradInput is consistent with gradOut
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, gradInput->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclTensor* gradIn, const aclIntArray* outputSize,
                       const aclIntArray* inputSize)
{
    // NCHW NHWC和ND格式判断
    auto srcOriginFormat = gradOut->GetOriginalFormat();
    bool supportFormat = false;
    if (IsRegBase()) {
        supportFormat = srcOriginFormat == op::Format::FORMAT_ND || srcOriginFormat == op::Format::FORMAT_NCHW ||
                        srcOriginFormat == op::Format::FORMAT_NHWC;
        OP_CHECK(supportFormat, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input format only supports NCHW, NHWC or ND"),
                 return false);
        OP_CHECK(gradOut->GetStorageFormat() == gradIn->GetStorageFormat(),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gradOut and gradInput must have the same format"), return false);
    } else {
        supportFormat = srcOriginFormat == op::Format::FORMAT_ND || srcOriginFormat == op::Format::FORMAT_NCHW;
        OP_CHECK(supportFormat, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input format only supports NCHW or ND"),
                 return false);
    }
    size_t inputSizeNum = inputSize->Size();
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected input_size to be 4, but got %zu", inputSizeNum), return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected output_size to be 2, but got %zu", outputSizeNum),
             return false);

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

static bool CheckInputElement(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                              const aclTensor* gradIn)
{
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(gradIn, DIM_FOUR, return false);
    OP_CHECK(outputSize->Size() == DIM_TWO,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The outputSize must have two elements, but got %zu", outputSize->Size()),
             return false);
    OP_CHECK(inputSize->Size() == DIM_FOUR,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The inputSize must have four elements, but got %zu", inputSize->Size()),
             return false);

    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    int64_t N = 0;
    int64_t C = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    FVector<int64_t> fullOutputSize = {0, 0, 0, 0};
    FVector<int64_t> fullInputSize = {0, 0, 0, 0};
    if (gradIn->GetStorageFormat() == op::Format::FORMAT_NCHW || gradIn->GetStorageFormat() == op::Format::FORMAT_ND) {
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

    OP_CHECK(N > 0 && C > 0 && inputH > 0 && inputW > 0 && outputH > 0 && outputW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Size of N, C, H and W must be greater than 0, but got input (N: %ld, C: %ld, H: %ld, W: %ld), "
                     "output (H: %ld, W: %ld)",
                     N, C, inputH, inputW, outputH, outputW),
             return false);

    auto gradOutShape = gradOut->GetViewShape();
    auto gradInShape = gradIn->GetViewShape();
    for (size_t i = 0; i < DIM_FOUR; ++i) {
        OP_CHECK(gradOutShape.GetDim(i) == fullOutputSize[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim %zu of gradOut should be %ld, but got %ld", i,
                         fullOutputSize[i], gradOutShape.GetDim(i)),
                 return false);
        OP_CHECK(gradInShape.GetDim(i) == fullInputSize[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim %zu of gradIn should be %ld, but got %ld", i,
                         fullInputSize[i], gradInShape.GetDim(i)),
                 return false);
    }

    return true;
}

static bool CheckUplimit(const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_2201) {
        return true;
    }

    int64_t N = (*inputSize)[DIM_ZERO];
    int64_t C = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    OP_CHECK(N <= INT32_MAX && C <= INT32_MAX && inputH <= INT32_MAX && inputW <= INT32_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "GradInput sizes should not be greater than %d, but got gradInput(%ld, %ld, %ld, %ld)", INT32_MAX,
                     N, C, inputH, inputW),
             return false);

    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    OP_CHECK(outputH <= INT32_MAX && outputW <= INT32_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "GradOut sizes should not be greater than %d, but got gradOut(%ld, %ld, %ld, %ld)", INT32_MAX, N,
                     C, outputH, outputW),
             return false);

    int64_t M = N * C * outputH;
    OP_CHECK(M <= INT32_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N * C * outputSize_H should not be greater than %d, but got %ld",
                     INT32_MAX, M),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* gradInput)
{
    // check input args
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // check input element
    CHECK_RET(CheckInputElement(gradOut, outputSize, inputSize, gradInput), ACLNN_ERR_PARAM_INVALID);

    // check input shape
    CHECK_RET(CheckShape(gradOut, gradInput, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // check input dtype
    CHECK_RET(CheckDtypeValid(gradOut, gradInput), ACLNN_ERR_PARAM_INVALID);

    // check uplimit
    CHECK_RET(CheckUplimit(outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckCanCalc(const aclIntArray* inputSize, float scalesHeight, float scalesWidth)
{
    if (scalesWidth < MAX_SCALE && scalesWidth >= (1 / MAX_SCALE) && scalesHeight < MAX_SCALE &&
        scalesHeight >= (1 / MAX_SCALE)) {
        return true;
    }
    int64_t N = (*inputSize)[DIM_ZERO];
    int64_t C = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];

    if (inputW > MAX_W) {
        return false;
    }
    if (scalesHeight >= MAX_SCALE) {
        return false;
    }

    if (scalesWidth >= MAX_SCALE && N * C * inputH > inputW * HALF_ONE) {
        return false;
    }
    if (scalesHeight < 1 / MAX_SCALE && scalesWidth < 1 / MAX_SCALE && N * C * inputH > inputW * MAX_W_RADIO) {
        return false;
    }
    return true;
}

static const aclTensor* ContiguousCast(const aclTensor* input, op::DataType dstDtype, aclOpExecutor* executor)
{
    auto inputContiguous = l0op::Contiguous(input, executor);
    auto outputContiguous = l0op::Cast(inputContiguous, dstDtype, executor);
    return outputContiguous;
}

static const aclTensor* adaptOutput(const aclTensor* ResizeGradDOut, aclIntArray* permuteHWNCArray,
                                    aclOpExecutor* executor, const int64_t* newReverseShape, int64_t castFp32Condition)
{
    const aclTensor* resizeGradDOutContiguous = nullptr;
    if (castFp32Condition == FP16_DETERMINSTIC_CAST) {
        resizeGradDOutContiguous = ContiguousCast(ResizeGradDOut, op::DataType::DT_FLOAT16, executor);
    } else if (castFp32Condition == BF16_CAST) {
        resizeGradDOutContiguous = ContiguousCast(ResizeGradDOut, op::DataType::DT_BF16, executor);
    } else {
        resizeGradDOutContiguous = l0op::Contiguous(ResizeGradDOut, executor);
    }
    CHECK_RET(resizeGradDOutContiguous != nullptr, nullptr);

    aclIntArray* shapeArrayReverse = executor->AllocIntArray(newReverseShape, 4);
    resizeGradDOutContiguous = l0op::Reshape(resizeGradDOutContiguous, shapeArrayReverse, executor);
    CHECK_RET(resizeGradDOutContiguous != nullptr, nullptr);

    // ResizeGradDOut transpose
    auto resizeGradDOutTranspose = l0op::Transpose(resizeGradDOutContiguous, permuteHWNCArray, executor);
    CHECK_RET(resizeGradDOutTranspose != nullptr, nullptr);

    return resizeGradDOutTranspose;
}
} // namespace

aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                           const aclIntArray* inputSize, const bool alignCorners,
                                                           double scalesH, double scalesW, aclTensor* gradInput,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleBicubic2dBackward,
                   DFX_IN(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW), DFX_OUT(gradInput));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(gradOut, outputSize, inputSize, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto computeScale = [](double scale, int64_t inputSizes, int64_t outputSizes, bool alignCorner) -> float {
        if (alignCorner) {
            if (outputSizes > 1) {
                return static_cast<double>(inputSizes - 1) / static_cast<double>(outputSizes - 1);
            } else {
                return 0;
            }
        }
        if (scale > 0) {
            return static_cast<double>(1.0) / scale;
        } else {
            return static_cast<double>(inputSizes) / static_cast<double>(outputSizes);
        }
    };
    const float scalesWidth = computeScale(scalesW, (*inputSize)[DIM_THREE], (*outputSize)[DIM_ONE], alignCorners);
    const float scalesHeight = computeScale(scalesH, (*inputSize)[DIM_TWO], (*outputSize)[DIM_ZERO], alignCorners);
    const aclTensor* gradOutContiguous = nullptr;
    const aclTensor* gradInputContiguous = nullptr;
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        // constuct aclFloatArray
        const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
        const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList, EXPECT_SIZE);
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradInputContiguous = l0op::Contiguous(gradInput, uniqueExecutor.get());
        CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // const aclTensor* kernelOut;
        op::Shape imageShape = gradOut->GetViewShape();
        if (!CheckIOSizesIsSame(gradOutContiguous, inputSize)) {
            imageShape.SetDimNum(DIM_FOUR);
            for (size_t i = 0; i < DIM_FOUR; ++i) {
                imageShape.SetDim(i, (*inputSize)[i]);
            }
        }

        auto originalImage = uniqueExecutor.get()->AllocTensor(imageShape, gradOutContiguous->GetDataType(),
                                                               gradOutContiguous->GetStorageFormat());
        CHECK_RET(originalImage != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* kernelOut = l0op::ResizeBicubicV2Grad(gradOutContiguous, originalImage, alignCorners, scales,
                                                               gradInputContiguous, uniqueExecutor.get());
        CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // view copy out
        auto viewCopyResult = l0op::ViewCopy(kernelOut, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if ((curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) &&
               CheckType(gradOut->GetDataType(), DTYPE_SUPPORT_LIST_910B) &&
               CheckCanCalc(inputSize, scalesHeight, scalesWidth)) {
        gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if ((*inputSize)[DIM_TWO] == (*outputSize)[DIM_ZERO] && (*inputSize)[DIM_THREE] == (*outputSize)[DIM_ONE]) {
            auto viewCopyResult = l0op::ViewCopy(gradOutContiguous, gradInput, uniqueExecutor.get());
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            // 将fp16/bf16类型cast成fp32处理，保证精度
            const aclTensor* gradInputFp32 = l0op::Contiguous(gradInput, uniqueExecutor.get());
            CHECK_RET(gradInputFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto dtype = gradOut->GetDataType();
            if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
                gradOutContiguous = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
                gradInputFp32 = l0op::Cast(gradInputFp32, op::DataType::DT_FLOAT, uniqueExecutor.get());
            }
            CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            const aclTensor* upsampleBicubic2dGradOut = l0op::UpsampleBicubic2dGrad(
                gradOutContiguous, alignCorners, scalesHeight, scalesWidth, gradInputFp32, uniqueExecutor.get());
            CHECK_RET(upsampleBicubic2dGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // CAST回bf16
            if (dtype == op::DataType::DT_BF16) {
                upsampleBicubic2dGradOut = l0op::Cast(upsampleBicubic2dGradOut, op::DataType::DT_BF16,
                                                      uniqueExecutor.get());
            } else if (dtype == op::DataType::DT_FLOAT16) {
                upsampleBicubic2dGradOut = l0op::Cast(upsampleBicubic2dGradOut, op::DataType::DT_FLOAT16,
                                                      uniqueExecutor.get());
            }
            CHECK_RET(upsampleBicubic2dGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto viewCopyResult = l0op::ViewCopy(upsampleBicubic2dGradOut, gradInput, uniqueExecutor.get());
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        // judge deterministic
        auto dtype = gradOut->GetDataType();
        int64_t castFp32Condition = 0;
        int64_t deterministicValue = 1;
        aclError retRts = aclrtCtxGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicValue);
        if (retRts != ACL_ERROR_NONE) {
            deterministicValue = 0;
            OP_LOGD("Unable to get system param deterministic = %ld.", deterministicValue);
        }
        OP_LOGD("deterministic is = %ld.", deterministicValue);

        if (deterministicValue == 1) {
            OP_LOGW("deterministic mode is not supported");
        }

        // gradout transpose
        bool fp16Cast = (deterministicValue == 1) || (curArch == NpuArch::DAV_1001);
        if (fp16Cast && dtype == op::DataType::DT_FLOAT16) {
            castFp32Condition = FP16_DETERMINSTIC_CAST;
            gradOutContiguous = ContiguousCast(gradOut, op::DataType::DT_FLOAT, uniqueExecutor.get());
        } else if (dtype == op::DataType::DT_BF16) {
            castFp32Condition = BF16_CAST;
            gradOutContiguous = ContiguousCast(gradOut, op::DataType::DT_FLOAT, uniqueExecutor.get());
        } else {
            gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
        }
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t permuteHWNCList[] = {2, 3, 0, 1};
        auto permuteHWNCArray = uniqueExecutor.get()->AllocIntArray(permuteHWNCList, FOURDIMS);
        CHECK_RET(permuteHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gradOutTranspose = l0op::Transpose(gradOutContiguous, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(gradOutTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        int64_t batch = (*inputSize)[DIM_ZERO];
        int64_t channels = (*inputSize)[DIM_ONE];
        int64_t inputH = (*inputSize)[DIM_TWO];
        int64_t inputW = (*inputSize)[DIM_THREE];

        // constuct aclFloatArray
        const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
        const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList, EXPECT_SIZE);
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // gradinput transpose
        if ((fp16Cast && dtype == op::DataType::DT_FLOAT16) || (dtype == op::DataType::DT_BF16)) {
            gradInputContiguous = ContiguousCast(gradInput, op::DataType::DT_FLOAT, uniqueExecutor.get());
        } else {
            gradInputContiguous = l0op::Contiguous(gradInput, uniqueExecutor.get());
        }
        CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t newShape[4] = {batch, channels, inputH, inputW};
        aclIntArray* shapeArray = uniqueExecutor.get()->AllocIntArray(newShape, 4);
        gradInputContiguous = l0op::Reshape(gradInputContiguous, shapeArray, uniqueExecutor.get());
        CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gradInputTranspose = l0op::Transpose(gradInputContiguous, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(gradInputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // call l0op resizeGradD
        const aclTensor* ResizeGradDOut = l0op::ResizeGradD(gradOutTranspose, inputSize, scales, alignCorners,
                                                            gradInputTranspose, CUBIC_MODE, uniqueExecutor.get());
        CHECK_RET(ResizeGradDOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const int64_t newReverseShape[4] = {inputH, inputW, batch, channels};
        auto adaptResult = adaptOutput(ResizeGradDOut, permuteHWNCArray, uniqueExecutor.get(), newReverseShape,
                                       castFp32Condition);

        // view copy out
        auto viewCopyResult = l0op::ViewCopy(adaptResult, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBicubic2dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
