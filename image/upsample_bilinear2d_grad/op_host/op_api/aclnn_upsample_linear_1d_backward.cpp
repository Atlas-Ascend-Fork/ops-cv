/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_upsample_linear_1d_backward.h"
#include <cmath>
#include "image/resize_bilinear_v2_grad/op_api/resize_bilinear_v2_grad.h"
#include "image/upsample_bilinear2d_grad/op_host/op_api/upsample_bilinear2d_grad.h"
#include "image/resize_linear_grad/op_api/resize_linear_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr size_t DIM_LIMIT = 3;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr int64_t EXPECT_SIZE = 1;
static const double MAX_SUPPORT_SCALE = 500.0;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

struct Liear1dBackWardData {
    const aclTensor* gradOut = nullptr;
    const aclIntArray* inputSize = nullptr;
    bool alignCorners = false;
    double scales = 0.0f;
};

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
    auto outShape = gradOut->GetViewShape();
    size_t dimNum = outShape.GetDimNum();

    for (size_t i = 0; i < dimNum; ++i) {
        if (outShape.GetDim(i) != (*inputSize)[i]) {
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* out, const aclIntArray* inputSize)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool isAscend910SocVersion = (curArch == NpuArch::DAV_2201 || curArch == NpuArch::DAV_1001 || IsRegBase(curArch));
    if (!isAscend910SocVersion) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "the current SoC is not supported");
        {
            const std::vector<const char*> ez3002Key = {"optype", "opskernel", "reason"};
            const std::vector<const char*> ez3002Value = {"aclnnUpsampleLinear1dBackward", "AiCore",
                                                          "the current SoC is not supported by this operator"};
            REPORT_PREDEFINED_ERR_MSG("EZ3002", ez3002Key, ez3002Value);
        }
        return false;
    }
    if (IsRegBase(curArch)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST_REGBASE, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, gradOut->GetDataType(), return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                            const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_THREE, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_THREE, return false);
    OP_CHECK(outputSize->Size() == DIM_ONE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleLinear1dBackward", "outputSize",
                                          std::to_string(outputSize->Size()).c_str(), "1"),
             return false);
    OP_CHECK(inputSize->Size() == DIM_THREE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleLinear1dBackward", "inputSize",
                                          std::to_string(inputSize->Size()).c_str(), "3"),
             return false);

    int64_t outputL = (*outputSize)[DIM_ZERO];
    int64_t N = (*inputSize)[DIM_ZERO];
    int64_t C = (*inputSize)[DIM_ONE];
    int64_t inputL = (*inputSize)[DIM_TWO];
    FVector<int64_t> fullOutputSize = {N, C, outputL};
    FVector<int64_t> fullInputSize = {N, C, inputL};

    OP_CHECK(N > 0 && C > 0 && inputL > 0 && outputL > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "input_l",
                                                      std::to_string(inputL).c_str(),
                                                      "N, C and L dimensions must be greater than 0"),
             return false);

    auto gradOutShape = gradOut->GetViewShape();
    auto outShape = out->GetViewShape();
    for (size_t i = 0; i < DIM_THREE; ++i) {
        OP_CHECK(gradOutShape.GetDim(i) == fullOutputSize[i],
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "gradOut",
                                                          std::to_string(gradOutShape.GetDim(i)).c_str(),
                                                          "each dim of gradOut must match the expected size"),
                 return false);
        OP_CHECK(outShape.GetDim(i) == fullInputSize[i],
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "out",
                                                          std::to_string(outShape.GetDim(i)).c_str(),
                                                          "each dim of out must match the expected size"),
                 return false);
    }

    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                       const aclTensor* out)
{
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleLinear1dBackward", "output_size",
                                          std::to_string(outputSizeNum).c_str(), "1"),
             return false);

    OP_CHECK(inputSizeNum == DIM_LIMIT,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleLinear1dBackward", "input_size",
                                          std::to_string(inputSizeNum).c_str(), "3"),
             return false);

    // out的shape必须与inputSize(N/C/L)一致，否则下发kernel后会因out与inputSize不匹配导致设备侧异常
    auto outShape = out->GetViewShape();
    for (size_t i = 0; i < DIM_LIMIT; ++i) {
        OP_CHECK(outShape.GetDim(i) == (*inputSize)[i],
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "out_dim",
                                                          std::to_string(outShape.GetDim(i)).c_str(),
                                                          "each dim of out must match the expected size"),
                 return false);
    }
    return true;
}

static bool CheckInputElement(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    int64_t outH = (*outputSize)[DIM_ZERO];
    int64_t batch = (*inputSize)[DIM_ZERO];
    int64_t channels = (*inputSize)[DIM_ONE];
    int64_t inputH = (*inputSize)[DIM_TWO];
    auto gradOutShape = gradOut->GetViewShape();
    size_t dimNum = gradOutShape.GetDimNum();
    FVector<int64_t> fullOutputSize = {batch, channels, outH};

    OP_CHECK(inputH > 0 && outH > 0,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "input_h",
                                                      std::to_string(inputH).c_str(),
                                                      "all input and output dims must be greater than 0"),
             return false);

    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "gradoutput_dim",
                                                     std::to_string(gradOutShape.GetDim(i)).c_str(),
                                                     "each dim of gradOutput must match the expected output size");
            return false;
        }
    }
    return true;
}

static bool CheckNCValid(const aclTensor* gradOut, const aclTensor* out)
{
    int64_t selfDimN = 0;
    int64_t selfDimC = 0;
    int64_t outDimN = 0;
    int64_t outDimC = 0;
    selfDimN = gradOut->GetViewShape().GetDim(DIM_ZERO);
    selfDimC = gradOut->GetViewShape().GetDim(DIM_ONE);
    outDimN = out->GetViewShape().GetDim(DIM_ZERO);
    outDimC = out->GetViewShape().GetDim(DIM_ONE);
    if ((selfDimN != outDimN) || (selfDimC != outDimC)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "self_dim_n",
                                                 std::to_string(selfDimN).c_str(),
                                                 "dim N of self must match dim N of out (same for dim C)");
        return false;
    }
    return true;
}

static bool CheckUplimit(const aclTensor* gradOut, const aclTensor* out)
{
    if (IsRegBase()) {
        return true;
    }
    int64_t gradOutN = gradOut->GetViewShape().GetDim(DIM_ZERO);
    int64_t gradOutC = gradOut->GetViewShape().GetDim(DIM_ONE);
    int64_t gradOutH = gradOut->GetViewShape().GetDim(DIM_TWO);
    int64_t outN = out->GetViewShape().GetDim(DIM_ZERO);
    int64_t outC = out->GetViewShape().GetDim(DIM_ONE);
    int64_t outH = out->GetViewShape().GetDim(DIM_TWO);

    OP_CHECK(gradOutN <= INT32_MAX && gradOutC <= INT32_MAX && gradOutH <= INT32_MAX,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "gradOutN",
                                                      std::to_string(gradOutN).c_str(),
                                                      "all dims must not exceed INT32_MAX"),
             return false);
    OP_CHECK(
        outN <= INT32_MAX && outC <= INT32_MAX && outH <= INT32_MAX,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleLinear1dBackward", "outN", std::to_string(outN).c_str(),
                                                 "all dims must not exceed INT32_MAX"),
        return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize,
                               const aclTensor* out)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查shape
    if (IsRegBase(curArch)) {
        CHECK_RET(CheckShapeValid(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckShape(gradOut, outputSize, inputSize, out), ACLNN_ERR_PARAM_INVALID);
    }

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOut, out, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 4. 校验gradOut的shape是否与输出的output的shape一致
    CHECK_RET(CheckInputElement(gradOut, outputSize, inputSize), ACLNN_ERR_PARAM_INVALID);

    // 5. 校验NC是否一致
    CHECK_RET(CheckNCValid(gradOut, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 校验上边界
    CHECK_RET(CheckUplimit(gradOut, out), ACLNN_ERR_PARAM_INVALID);

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

static double ComputeLinear1dBackwardScales(int64_t input_size, int64_t output_size, double scales)
{
    if (scales > 0) {
        return scales;
    } else {
        return input_size != 0 ? (static_cast<double>(output_size) / input_size) : 0.0;
    }
}

static bool CheckScales(const aclIntArray* inputSize, const aclIntArray* outputSize, double scales)
{
    int64_t inputL = (*inputSize)[DIM_TWO];
    int64_t outputL = (*outputSize)[DIM_ZERO];
    double realScales = ComputeLinear1dBackwardScales(inputL, outputL, scales);

    OP_CHECK(realScales <= MAX_SUPPORT_SCALE,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnUpsampleLinear1dBackward", "scales",
                                                   std::to_string(realScales).c_str(),
                                                   "scales is too large; the supported scale range is exceeded"),
             return false);
    return true;
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
    auto unsqueezedResult = l0op::UnsqueezeNd(contiguousInput, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedResult != nullptr, nullptr);

    // reformat
    auto reformatInput = l0op::ReFormat(unsqueezedResult, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static bool Check_scales(const int64_t input_size, const int64_t output_size, const double scale)
{
    if (output_size != static_cast<int64_t>(floor(input_size * scale))) {
        return false;
    }
    return true;
}

static bool ComputeCheckScale(const aclIntArray* outputSize, const aclIntArray* inputSize, double scales)
{
    bool check_scale = true;
    if (std::abs(scales) > 1e-9) {
        int64_t output_L = (*outputSize)[DIM_ZERO];
        int64_t input_L = (*inputSize)[DIM_TWO];
        check_scale = Check_scales(input_L, output_L, scales);
    }
    return check_scale;
}
} // namespace

aclnnStatus aclnnUpsampleLinear1dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize,
                                                          const aclIntArray* inputSize, bool alignCorners,
                                                          double scales, aclTensor* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleLinear1dBackward, DFX_IN(gradOut, outputSize, inputSize, alignCorners, scales),
                   DFX_OUT(out));
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
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        op::Shape imageShape = gradOut->GetViewShape();
        const aclTensor* kernelOut = gradOut;
        const aclTensor* originalImage;
        const aclTensor* viewCopyResult;

        // 固定写法，将输入gradOut转换成连续的tensor
        auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (!CheckIOSizesIsSame(gradOut, inputSize)) {
            imageShape.SetDimNum(DIM_THREE);
            for (size_t i = 0; i < DIM_THREE; ++i) {
                imageShape.SetDim(i, (*inputSize)[i]);
            }
            originalImage = uniqueExecutor.get()->AllocTensor(imageShape, out->GetDataType(), out->GetStorageFormat());
            CHECK_RET(originalImage != nullptr, ACLNN_ERR_INNER_NULLPTR);
            kernelOut = l0op::ResizeLinearGrad(gradOutContiguous, originalImage, alignCorners,
                                               static_cast<float>(scales), out, uniqueExecutor.get());
            CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            viewCopyResult = l0op::ViewCopy(kernelOut, out, uniqueExecutor.get());
        } else {
            viewCopyResult = l0op::ViewCopy(gradOutContiguous, out, uniqueExecutor.get());
        }
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // gradOut 补第二维
        auto gradOutContiguous = View3dAs4d(gradOut, uniqueExecutor.get());
        CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto outContiguous = View3dAs4d(out, uniqueExecutor.get());
        CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // inputSize 补第二维 (outputSize 后面没有用到，暂不补维)
        const int64_t originSizeList[] = {(*inputSize)[DIM_ZERO], (*inputSize)[DIM_ONE], 1, (*inputSize)[DIM_TWO]};
        auto originSizeArray = uniqueExecutor.get()->AllocIntArray(originSizeList, 4);
        const int64_t outputSizeList[] = {1, (*outputSize)[DIM_ZERO]};
        CHECK_RET(originSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto outputSizeArray = uniqueExecutor.get()->AllocIntArray(outputSizeList, 2);
        CHECK_RET(outputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor* outCast = gradOutContiguous;
        bool checkSize = CheckIOSizesIsSame(gradOutContiguous, originSizeArray);
        bool checkSocVer = curArch == NpuArch::DAV_2201;
        bool check_scale = true;
        check_scale = ComputeCheckScale(outputSize, inputSize, scales);
        const int64_t MAX_GRAD_SIZE = 1000000;
        bool isBigSize = (*outputSize)[DIM_ZERO] >= MAX_GRAD_SIZE ? true : false;
        if (isBigSize || (!checkSize && (!checkSocVer || !check_scale))) {
            bool halfPixelCenters = !alignCorners;
            op::Shape imageShape;
            imageShape.SetDimNum(DIM_LIMIT + 1);
            for (size_t i = 0; i < DIM_LIMIT + 1; ++i) {
                imageShape.SetDim(i, (*originSizeArray)[i]);
            }
            auto gradOutCast = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto gradOutTransdata = l0op::TransDataSpecial(gradOutCast, op::Format::FORMAT_NC1HWC0, 0,
                                                           uniqueExecutor.get());
            CHECK_RET(gradOutTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto image = uniqueExecutor.get()->AllocTensor(imageShape, gradOutCast->GetDataType(),
                                                           gradOutCast->GetViewFormat());
            CHECK_RET(image != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto imageTransdata = l0op::TransDataSpecial(image, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
            CHECK_RET(imageTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto v2GradOut = l0op::ResizeBilinearV2Grad5Hd(gradOutTransdata, imageTransdata, alignCorners,
                                                           halfPixelCenters, uniqueExecutor.get());
            CHECK_RET(v2GradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto outTransdata = l0op::TransData(v2GradOut, op::Format::FORMAT_NCHW, 0, uniqueExecutor.get());
            CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

            outCast = l0op::Cast(outTransdata, out->GetDataType(), uniqueExecutor.get());
            CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else if (curArch == NpuArch::DAV_2201) {
            CHECK_RET(CheckScales(inputSize, outputSize, scales), ACLNN_ERR_PARAM_INVALID);
            const float realScales_w = scales > 0 ? static_cast<float>(1.0 / scales) : 0;
            const float realScales_h = static_cast<float>(1.0);
            auto gradOutCast = l0op::Cast(gradOutContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            const aclTensor* upsampleOut = l0op::UpsampleBilinear2dGrad(gradOutCast, outputSizeArray, originSizeArray,
                                                                        outContiguous, alignCorners, realScales_h,
                                                                        realScales_w, uniqueExecutor.get());
            CHECK_RET(upsampleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto outTransdata = l0op::TransData(upsampleOut, op::Format::FORMAT_NCHW, 0, uniqueExecutor.get());
            CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);
            outCast = l0op::Cast(outTransdata, out->GetDataType(), uniqueExecutor.get());
            CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // out 降维 (squeeze第二维)
        const aclTensor* out3d = nullptr;
        out3d = View4dAs3d(outCast, out, uniqueExecutor.get());
        CHECK_RET(CheckReduceOutShape(out3d, out), ACLNN_ERR_PARAM_INVALID);
        auto viewCopyResult = l0op::ViewCopy(out3d, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleLinear1dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleLinear1dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
