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
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"

#include <cmath>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

#include "image/resize_bilinear_v2_grad/op_api/resize_bilinear_v2_grad.h"
#include "upsample_bilinear2d_grad.h"
#include "aclnn_upsample_bilinear_2d_backward_v2.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_LIMIT = 4;
static constexpr int64_t EXPECT_SIZE = 2;
static const double MIN_SUPPORT_SCALE = 0.01;

namespace {
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_310P = {op::DataType::DT_FLOAT,
                                                                            op::DataType::DT_FLOAT16};
} // namespace

static bool CheckIOSizesIsSame(const aclTensor* gradOutput, const aclIntArray* inputSize)
{
    auto gradOutShape = gradOutput->GetViewShape();
    size_t dimCount = gradOutShape.GetDimNum();

    for (size_t i = 0; i < dimCount; ++i) {
        if (gradOutShape.GetDim(i) != (*inputSize)[i]) {
            return false;
        }
    }
    return true;
}

static bool CheckNotNull(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                         const aclTensor* out)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

bool CheckInputsElement(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t outW = (*outputSize)[DIM_ONE];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channel = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimCount = gradOutShape.GetDimNum();
    FVector<int64_t> fullOutputSize = {batch, channel, outH, outW};

    if (gradOut->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputH = (*inputSize)[DIM_ONE];
        inputW = (*inputSize)[DIM_TWO];
        channel = (*inputSize)[DIM_THREE];
        fullOutputSize[DIM_ONE] = outH;
        fullOutputSize[DIM_TWO] = outW;
        fullOutputSize[DIM_THREE] = channel;
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outH > 0 && outW > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dBackwardV2", "input_h",
                                                      std::to_string(inputH).c_str(),
                                                      "all input and output dims must be greater than 0"),
             return false);

    for (size_t i = 0; i < dimCount; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dBackwardV2", "gradoutput_dim",
                                                     std::to_string(gradOutShape.GetDim(i)).c_str(),
                                                     "each dim of gradOutput must match the expected output size");
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* out, const aclIntArray* inputSize)
{
    if (!CheckIOSizesIsSame(gradOut, inputSize)) {
        auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
        bool bf16Support = curArch == NpuArch::DAV_2201 || IsRegBase();
        bool bf16NoSupport = (curArch == NpuArch::DAV_2002 || curArch == NpuArch::DAV_1001);
        if (bf16Support) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
        } else if (bf16NoSupport) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST_310P, return false);
        } else {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "the current SoC is not supported");
            {
                const std::vector<const char*> ez3002Key = {"optype", "opskernel", "reason"};
                const std::vector<const char*> ez3002Value = {
                    "aclnnUpsampleBilinear2dBackwardV2", "AiCore",
                    "the current SoC is not supported by this implementation"};
                REPORT_PREDEFINED_ERR_MSG("EZ3002", ez3002Key, ez3002Value);
            }
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, gradOut->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* out)
{
    auto format_out = out->GetStorageFormat();
    if (format_out != op::Format::FORMAT_NCHW && format_out != op::Format::FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnUpsampleBilinear2dBackwardV2", "out", op::ToString(format_out).GetString(),
                                   "NCHW or NHWC");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == DIM_LIMIT,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBilinear2dBackwardV2", "input_size",
                                          std::to_string(inputSizeNum).c_str(), "4"),
             return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBilinear2dBackwardV2", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "2"),
             return false);
    return true;
}

static bool CheckUplimit(const aclTensor* gradOut)
{
    if (IsRegBase()) {
        return true;
    }
    int64_t outN = gradOut->GetViewShape().GetDim(DIM_ZERO);
    int64_t outC = gradOut->GetViewShape().GetDim(DIM_ONE);
    int64_t outH = gradOut->GetViewShape().GetDim(DIM_TWO);
    int64_t outW = gradOut->GetViewShape().GetDim(DIM_THREE);
    int64_t gradOutN = gradOut->GetViewShape().GetDim(DIM_ZERO);
    int64_t gradOutC = gradOut->GetViewShape().GetDim(DIM_ONE);
    int64_t gradOutH = gradOut->GetViewShape().GetDim(DIM_TWO);
    int64_t gradOutW = gradOut->GetViewShape().GetDim(DIM_THREE);

    OP_CHECK(gradOutN <= INT32_MAX && gradOutC <= INT32_MAX && gradOutH <= INT32_MAX && gradOutW <= INT32_MAX,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dBackwardV2", "gradOutN",
                                                      std::to_string(gradOutN).c_str(),
                                                      "all dims must not exceed INT32_MAX"),
             return false);
    OP_CHECK(
        outN <= INT32_MAX && outC <= INT32_MAX && outH <= INT32_MAX && outW <= INT32_MAX,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2dBackwardV2", "outN",
                                                 std::to_string(outN).c_str(), "all dims must not exceed INT32_MAX"),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* out)
{
    // 1. 检查参数是否是空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查shape是否合法
    CHECK_RET(CheckShape(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 3. 输入输出格式校验
    CHECK_RET(CheckFormat(out), ACLNN_ERR_PARAM_INVALID);

    // 4. 校验gradOut的shape是否与输出的output的shape一致
    CHECK_RET(CheckInputsElement(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 5.检查gradOut和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(gradOut, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验 输入输出格式校验
    CHECK_RET(CheckDtypeValid(gradOut, out, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 7. 校验上边界
    CHECK_RET(CheckUplimit(gradOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static double ComputeBilinear2dBackwardScales(const int64_t input_size, const int64_t output_size, const double scale)
{
    if (scale > 0) {
        return 1.0 / scale;
    } else {
        return output_size != 0 ? (static_cast<double>(input_size) / output_size) : 0.0;
    }
}

static bool CheckMinScale(const aclTensor* gradOutput, const aclIntArray* inputSize, const double scalesH,
                          const double scalesW)
{
    auto gradOutputShape = gradOutput->GetViewShape();
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t inputH = (*inputSize)[DIM_TWO];
    int64_t inputW = (*inputSize)[DIM_THREE];

    outH = gradOutputShape.GetDim(DIM_TWO);
    outW = gradOutputShape.GetDim(DIM_THREE);

    double realScalesH = ComputeBilinear2dBackwardScales(inputH, outH, scalesH);
    double realScalesW = ComputeBilinear2dBackwardScales(inputW, outW, scalesW);
    if (realScalesH < MIN_SUPPORT_SCALE || realScalesW < MIN_SUPPORT_SCALE) {
        return false;
    }
    return true;
}

static bool Check_scales(const int64_t inputSize, const int64_t outputSize, const double scaleVal)
{
    if (outputSize != static_cast<int64_t>(floor(inputSize * scaleVal))) {
        return false;
    }
    return true;
}

static aclnnStatus DoResizeBilinearV2Grad(const aclIntArray* inputSize, bool alignCorners,
                                          const aclTensor* gradOutContiguous, const aclTensor* outTransdata,
                                          aclTensor* out, aclOpExecutor* executor)
{
    bool halfPixelCenters = !alignCorners;
    op::Shape imageShape;
    imageShape.SetDimNum(DIM_LIMIT);
    for (size_t i = 0; i < DIM_LIMIT; ++i) {
        imageShape.SetDim(i, (*inputSize)[i]);
    }

    auto gradOutTransdata = l0op::TransDataSpecial(gradOutContiguous, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(gradOutTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutCast = l0op::CastOnlyForConvBackward(gradOutTransdata, op::DataType::DT_FLOAT, executor);
    CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto image = executor->AllocTensor(imageShape, gradOutCast->GetDataType(), gradOutCast->GetViewFormat());
    CHECK_RET(image != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto imageTransdata = l0op::TransDataSpecial(image, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(imageTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto v2GradOut = l0op::ResizeBilinearV2Grad5Hd(gradOutCast, imageTransdata, alignCorners, halfPixelCenters,
                                                   executor);
    CHECK_RET(v2GradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outCast = l0op::CastOnlyForConvBackward(v2GradOut, out->GetDataType(), executor);
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTransdata = l0op::TransData(outCast, out->GetStorageFormat(), 0, executor);
    CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CheckReduceOutShape(outTransdata, out), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(outTransdata, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus DoBilinearGradNHWC(const aclTensor* gradOut, const aclIntArray* outputSize,
                                      const aclIntArray* inputSize, bool alignCorners, aclTensor* out,
                                      const aclTensor* gradOutContiguous, const float realScalesH,
                                      const float realScalesW, aclOpExecutor* executor)
{
    op::Shape nchwShape;
    nchwShape.SetDimNum(DIM_LIMIT);
    nchwShape.SetDim(DIM_ZERO, (*inputSize)[DIM_ZERO]);
    nchwShape.SetDim(DIM_ONE, (*inputSize)[DIM_THREE]);
    nchwShape.SetDim(DIM_TWO, (*inputSize)[DIM_ONE]);
    nchwShape.SetDim(DIM_THREE, (*inputSize)[DIM_TWO]);
    auto outNchw = executor->AllocTensor(nchwShape, gradOut->GetDataType(), op::Format::FORMAT_NHWC);
    const int64_t permuteNCHWList[] = {0, 3, 1, 2};
    auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, DIM_LIMIT);
    auto selfTranspose = l0op::Transpose(gradOutContiguous, permuteNCHWArray, executor);
    CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* outCast = gradOutContiguous;
    const int64_t inputSizeList[] = {(*inputSize)[DIM_ZERO], (*inputSize)[DIM_THREE], (*inputSize)[DIM_ONE],
                                     (*inputSize)[DIM_TWO]};
    auto inputSizeArray = executor->AllocIntArray(inputSizeList, 4);

    auto gradOutCast = l0op::Cast(selfTranspose, op::DataType::DT_FLOAT, executor);
    CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto upsampleOut = l0op::UpsampleBilinear2dGrad(gradOutCast, outputSize, inputSizeArray, outNchw, alignCorners,
                                                    realScalesH, realScalesW, executor);
    CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const int64_t permuteNHWCList[] = {0, 2, 3, 1};
    auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, DIM_LIMIT);
    CHECK_RET(permuteNHWCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outTransposedata = l0op::Transpose(upsampleOut, permuteNHWCArray, executor);
    CHECK_RET(outTransposedata != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(outTransposedata, out), ACLNN_ERR_PARAM_INVALID);
    outCast = l0op::Cast(outTransposedata, out->GetDataType(), executor);
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(outCast, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

bool ComputeCheckScale(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                       double scalesH, double scalesW)
{
    bool check_scale = true;
    if ((std::abs(scalesH) > 1e-9) || (std::abs(scalesW) > 1e-9)) {
        auto gradout_formt = gradOut->GetStorageFormat();
        int64_t output_H = (*outputSize)[DIM_ZERO];
        int64_t output_W = (*outputSize)[DIM_ONE];
        if (gradout_formt == op::Format::FORMAT_NCHW) {
            int64_t input_H = (*inputSize)[DIM_TWO];
            int64_t input_W = (*inputSize)[DIM_THREE];
            check_scale = Check_scales(input_H, output_H, scalesH) && Check_scales(input_W, output_W, scalesW);
        }
        if (gradout_formt == op::Format::FORMAT_NHWC) {
            int64_t input_H = (*inputSize)[DIM_ONE];
            int64_t input_W = (*inputSize)[DIM_TWO];
            check_scale = Check_scales(input_H, output_H, scalesH) && Check_scales(input_W, output_W, scalesW);
        }
    }
    return check_scale;
}
} // namespace

aclnnStatus aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                              const aclIntArray* inputSize, bool alignCorners,
                                                              double scalesH, double scalesW, aclTensor* out,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleBilinear2dBackwardV2,
                   DFX_IN(gradOut, outputSize, inputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, outputSize, inputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();

    if (gradOut->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* outTransdata = gradOutContiguous;
    const aclTensor* outCast = gradOutContiguous;
    auto out_contiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(out_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    bool check_scale = true;
    check_scale = ComputeCheckScale(gradOut, outputSize, inputSize, scalesH, scalesW);
    if (IsRegBase()) {
        bool halfPixelCenters = !alignCorners;
        op::Shape imageShape;
        imageShape.SetDimNum(DIM_LIMIT);
        for (size_t i = 0; i < DIM_LIMIT; ++i) {
            imageShape.SetDim(i, (*inputSize)[i]);
        }

        auto originalImage = uniqueExecutor.get()->AllocTensor(imageShape, out->GetDataType(), out->GetViewFormat());
        CHECK_RET(originalImage != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
        auto scales = uniqueExecutor->AllocFloatArray(scalesList, DIM_TWO);
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        outTransdata = l0op::ResizeBilinearV2Grad(gradOutContiguous, originalImage, alignCorners, halfPixelCenters,
                                                  scales, uniqueExecutor.get());
        CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

        CHECK_RET(CheckReduceOutShape(outTransdata, out), ACLNN_ERR_PARAM_INVALID);
        auto viewCopyResult = l0op::ViewCopy(outTransdata, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (!CheckIOSizesIsSame(gradOut, inputSize) && (!(curArch == NpuArch::DAV_2201) || !check_scale ||
                                                           !CheckMinScale(gradOut, inputSize, scalesH, scalesW))) {
        aclnnStatus status = DoResizeBilinearV2Grad(inputSize, alignCorners, gradOutContiguous, outTransdata, out,
                                                    uniqueExecutor.get());
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    } else if (curArch == NpuArch::DAV_2201) {
        const float realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        const float realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;
        if (gradOutContiguous->GetStorageFormat() == op::Format::FORMAT_NHWC) {
            aclnnStatus status = DoBilinearGradNHWC(gradOut, outputSize, inputSize, alignCorners, out,
                                                    gradOutContiguous, realScalesH, realScalesW, uniqueExecutor.get());
            if (status != ACLNN_SUCCESS) {
                return status;
            }
        } else {
            auto gradOutCast = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto upsampleOut = l0op::UpsampleBilinear2dGrad(gradOutCast, outputSize, inputSize, out, alignCorners,
                                                            realScalesH, realScalesW, uniqueExecutor.get());
            CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            outCast = l0op::Cast(upsampleOut, out->GetDataType(), uniqueExecutor.get());
            CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(CheckReduceOutShape(upsampleOut, out), ACLNN_ERR_PARAM_INVALID);
            auto viewCopyResult = l0op::ViewCopy(outCast, out, uniqueExecutor.get());
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2dBackwardV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBilinear2dBackwardV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
