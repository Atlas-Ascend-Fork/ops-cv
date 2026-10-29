/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_upsample_bicubic_2d.cpp
 * \brief aclnn_upsample_bicubic_2d
 */

#include "aclnn_upsample_bicubic_2d.h"
#include "image/upsample_linear1d/op_host/op_api/resize_d.h"
#include "upsample_bicubic2d.h"
#include "image/resize_bicubic_v2/op_api/resize_bicubic_v2.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/level2_base_cv.h"
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
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static constexpr size_t NCHW_DIM_ZERO = 0;
static constexpr size_t NCHW_DIM_ONE = 1;
static constexpr size_t NCHW_DIM_TWO = 2;
static constexpr size_t NCHW_DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 2;
static constexpr float MAX_SUPPORT_SCALE = 40.0;
static constexpr float MAX_SUPPORT_SCALE_DOUBLE = 30.0;
static constexpr double EPSILON = 1e-5;
static const int64_t DIM_LIMIT = 4;
static const int64_t FOURDIMS = 4;
static const std::string MODE = "cubic";

struct BicubicV2Data {
    const aclTensor* self = nullptr;
    const aclTensor* out = nullptr;
    const aclIntArray* outputSize = nullptr;
    const aclFloatArray* scales = nullptr;
    bool alignCorners = false;
};

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckFormatValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(self->GetStorageFormat() == out->GetStorageFormat(),
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleBicubic2d", "self,out",
                                                     (std::string(op::ToString(self->GetStorageFormat()).GetString()) +
                                                      ", " + op::ToString(out->GetStorageFormat()).GetString())
                                                         .c_str(),
                                                     "format of self and out must be the same"),
             return false);

    OP_CHECK((out->GetStorageFormat() == op::Format::FORMAT_NCHW ||
              out->GetStorageFormat() == op::Format::FORMAT_NHWC || out->GetStorageFormat() == op::Format::FORMAT_ND),
             OP_LOGE_FOR_INVALID_FORMAT("aclnnUpsampleBicubic2d", "out",
                                        op::ToString(out->GetStorageFormat()).GetString(), "NCHW, NHWC or ND"),
             return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBicubic2d", "outputSize", std::to_string(outputSizeNum).c_str(),
                                          "2"),
             return false);
    auto selfShape = self->GetViewShape();
    int64_t n = selfShape.GetDim(NCHW_DIM_ZERO);
    int64_t c = selfShape.GetDim(NCHW_DIM_ONE);
    int64_t inputH = selfShape.GetDim(NCHW_DIM_TWO);
    int64_t inputW = selfShape.GetDim(NCHW_DIM_THREE);
    int64_t outputH = (*outputSize)[NCHW_DIM_ZERO];
    int64_t outputW = (*outputSize)[NCHW_DIM_ONE];
    FVector<int64_t> fullOutputSize = {n, c, outputH, outputW};
    if (self->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputH = selfShape.GetDim(NCHW_DIM_ONE);
        inputW = selfShape.GetDim(NCHW_DIM_TWO);
        c = selfShape.GetDim(NCHW_DIM_THREE);
        fullOutputSize = {n, outputH, outputW, c};
    }
    OP_CHECK(
        c > 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnUpsampleBicubic2d", "self", op::ToString(selfShape).GetString(),
                                              "a non-empty 4D data tensor is required"),
        return false);
    OP_CHECK(
        inputH > 0 && inputW > 0 && outputH > 0 && outputW > 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2d", "inputH", std::to_string(inputH).c_str(),
                                                 "all input and output dims must be greater than 0"),
        return false);
    auto outShape = out->GetViewShape();
    for (size_t i = 0; i < DIM_LIMIT; ++i) {
        OP_CHECK(outShape.GetDim(i) == fullOutputSize[i],
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2d", "out",
                                                          std::to_string(outShape.GetDim(i)).c_str(),
                                                          "each dim of out must match the expected size"),
                 return false);
    }
    // 校验上边界
    if (!IsRegBase()) {
        OP_CHECK(n <= INT32_MAX && c <= INT32_MAX && inputH <= INT32_MAX && inputW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2d", "n", std::to_string(n).c_str(),
                                                          "all dims must not exceed INT32_MAX"),
                 return false);
        OP_CHECK(outputH <= INT32_MAX && outputW <= INT32_MAX,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBicubic2d", "outputH",
                                                          std::to_string(outputH).c_str(),
                                                          "all dims must not exceed INT32_MAX"),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查format
    CHECK_RET(CheckFormatValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShapeValid(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckDstSize(const int64_t inputSize, const int64_t outputSize, const double scales)
{
    if (scales <= 0) {
        return true;
    }

    int64_t dstDim = static_cast<int64_t>(static_cast<double>(inputSize) * scales);
    int64_t dstDimMax = static_cast<int64_t>(static_cast<double>(inputSize) * (scales + EPSILON));
    int64_t dstDimMin = static_cast<int64_t>(static_cast<double>(inputSize) * (scales - EPSILON));

    return (outputSize == dstDim) || (outputSize >= dstDimMin && outputSize <= dstDimMax);
}

static bool CheckScales(const aclTensor* self, const aclIntArray* outputSize, const double scalesH,
                        const double scalesW)
{
    auto selfShape = self->GetViewShape();
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t outputH = (*outputSize)[NCHW_DIM_ZERO];
    int64_t outputW = (*outputSize)[NCHW_DIM_ONE];
    if (self->GetStorageFormat() == op::Format::FORMAT_NCHW || self->GetStorageFormat() == op::Format::FORMAT_ND) {
        inputH = selfShape.GetDim(NCHW_DIM_TWO);
        inputW = selfShape.GetDim(NCHW_DIM_THREE);
    } else {
        inputH = selfShape.GetDim(NCHW_DIM_ONE);
        inputW = selfShape.GetDim(NCHW_DIM_TWO);
    }

    OP_CHECK(CheckDstSize(inputH, outputH, scalesH),
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnUpsampleBicubic2d", "scales_h",
                                                    (std::to_string(inputH) + ", " + std::to_string(outputH)).c_str(),
                                                    "scales_h conflicts with outputSize; scales_h * input_h should be "
                                                    "equal to outputSize_h"),
             return false);

    OP_CHECK(CheckDstSize(inputW, outputW, scalesW),
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnUpsampleBicubic2d", "scales_w",
                                                    (std::to_string(inputW) + ", " + std::to_string(outputW)).c_str(),
                                                    "scales_w conflicts with outputSize; scales_w * input_w should be "
                                                    "equal to outputSize_w"),
             return false);
    return true;
}

static bool CheckMaxScaleSupport(const aclTensor* self, const aclIntArray* outputSize, const double scalesH,
                                 const double scalesW, const bool alignCorners)
{
    auto selfShape = self->GetViewShape();
    int64_t inputH = selfShape.GetDim(NCHW_DIM_TWO);
    int64_t inputW = selfShape.GetDim(NCHW_DIM_THREE);
    int64_t outputH = (*outputSize)[NCHW_DIM_ZERO];
    int64_t outputW = (*outputSize)[NCHW_DIM_ONE];
    bool align_corner = alignCorners;
    float scale_h = 0.0;
    float scale_w = 0.0;
    if (align_corner) {
        scale_h = outputH > 1 ? static_cast<float>(inputH - 1) / static_cast<float>(outputH - 1) :
                                static_cast<float>(0.0);
        scale_w = outputW > 1 ? static_cast<float>(inputW - 1) / static_cast<float>(outputW - 1) :
                                static_cast<float>(0.0);
    } else {
        scale_h = scalesH > 0 ? static_cast<float>(1.0 / scalesH) :
                                static_cast<float>(inputH) / static_cast<float>(outputH);
        scale_w = scalesW > 0 ? static_cast<float>(1.0 / scalesW) :
                                static_cast<float>(inputW) / static_cast<float>(outputW);
    }
    if (scale_h > MAX_SUPPORT_SCALE || scale_w > MAX_SUPPORT_SCALE) {
        return false;
    }
    if (scale_h > MAX_SUPPORT_SCALE_DOUBLE && scale_w > MAX_SUPPORT_SCALE_DOUBLE) {
        return false;
    }
    return true;
}

static bool CheckIsBicubic2dPlatform(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    } else {
        return false;
    }
    return true;
}

static bool CheckIsBicubic2dPlatform310p(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2002 || curArch == NpuArch::DAV_3002) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    } else {
        return false;
    }
    return true;
}
} // namespace

aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSizeRegBase(const aclTensor* self, const aclIntArray* outputSize,
                                                          const bool alignCorners, const double scalesH,
                                                          const double scalesW, aclTensor* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto checkBicubic2dRet = CheckParams(self, outputSize, out);
    CHECK_RET(checkBicubic2dRet == ACLNN_SUCCESS, checkBicubic2dRet);
    CHECK_RET(CheckScales(self, outputSize, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);

    // 空tensor在kernel中支持
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
    const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList, NCHW_DIM_TWO);
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resizeOut = l0op::ResizeBicubicV2(selfContiguous, outputSize, alignCorners, scales, outContiguous,
                                           uniqueExecutor.get());
    CHECK_RET(resizeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(resizeOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                   const bool alignCorners, const double scalesH, const double scalesW,
                                                   aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnUpsampleBicubic2d, DFX_IN(self, outputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    if (IsRegBase()) {
        return aclnnUpsampleBicubic2dGetWorkspaceSizeRegBase(self, outputSize, alignCorners, scalesH, scalesW, out,
                                                             workspaceSize, &(*executor));
    }
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查scale与outputSize是否一致
    CHECK_RET(CheckScales(self, outputSize, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);

    // 空tensor在kernel中支持
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (CheckIsBicubic2dPlatform(self) && CheckMaxScaleSupport(self, outputSize, scalesH, scalesW, alignCorners)) {
        auto dtype = self->GetDataType();
        // 将fp16/bf16类型cast成fp32处理，保证精度
        if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
            selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 先用double算好1/scale再转float，减少精度损失
        const float realScales_h = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        const float realScales_w = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;

        // 调用Bicubic2d算子kernel
        const aclTensor* Bicubic2dOut = l0op::UpsampleBicubic2d(selfContiguous, outputSize, alignCorners, realScales_h,
                                                                realScales_w, out, uniqueExecutor.get());
        CHECK_RET(Bicubic2dOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (dtype == op::DataType::DT_BF16) {
            // CAST回bf16
            Bicubic2dOut = l0op::Cast(Bicubic2dOut, op::DataType::DT_BF16, uniqueExecutor.get());
        } else if (dtype == op::DataType::DT_FLOAT16) {
            // CAST回fp16
            Bicubic2dOut = l0op::Cast(Bicubic2dOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        }
        CHECK_RET(Bicubic2dOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(Bicubic2dOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto selfshape = self->GetViewShape();
        int64_t batch = selfshape.GetDim(NCHW_DIM_ZERO);
        int64_t channels = selfshape.GetDim(NCHW_DIM_ONE);
        int64_t outH = (*outputSize)[NCHW_DIM_ZERO];
        int64_t outW = (*outputSize)[NCHW_DIM_ONE];

        const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
        const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList, NCHW_DIM_TWO);
        CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self进行transpose，shape：NCHW-->HWNC
        const int64_t permuteHWNCList[] = {2, 3, 0, 1};
        auto permuteHWNCArray = uniqueExecutor.get()->AllocIntArray(permuteHWNCList, FOURDIMS);
        CHECK_RET(permuteHWNCArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfTranspose = l0op::Transpose(selfContiguous, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将cast移至transpose后，转换成连续的tensor
        auto dtype = selfTranspose->GetDataType();
        if (dtype == op::DataType::DT_BF16) {
            selfTranspose = l0op::Cast(selfTranspose, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // out reshape
        const int64_t new_reshape[4] = {batch, channels, outH, outW};
        aclIntArray* shapeArray = uniqueExecutor.get()->AllocIntArray(new_reshape, FOURDIMS);
        auto OutReshape = l0op::Reshape(out, shapeArray, uniqueExecutor.get());
        CHECK_RET(OutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // out Transpose
        auto outTranspose = l0op::Transpose(OutReshape, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(outTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // CAST
        if (dtype == op::DataType::DT_BF16) {
            outTranspose = l0op::Cast(outTranspose, op::DataType::DT_FLOAT, uniqueExecutor.get());
        }
        CHECK_RET(outTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor* resizeOut = nullptr;
        if (CheckIsBicubic2dPlatform310p(self)) {
            // 调用Bicubic2d算子kernel
            // 先用double算好1/scale再转float，可以减少精度损失
            const float realScales_h = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
            const float realScales_w = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;
            resizeOut = l0op::UpsampleBicubic2d(selfTranspose, outputSize, alignCorners, realScales_h, realScales_w,
                                                outTranspose, uniqueExecutor.get());
        } else {
            // 调用ResizeD算子kernel
            resizeOut = l0op::ResizeD(selfTranspose, outputSize, alignCorners, outTranspose, scales, MODE,
                                      uniqueExecutor.get());
        }
        CHECK_RET(resizeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // resizeDOut reshape
        const int64_t new_reshape_reverse[4] = {outH, outW, batch, channels};
        aclIntArray* shapeArrayReverse = uniqueExecutor.get()->AllocIntArray(new_reshape_reverse, FOURDIMS);
        auto resizeDOutReshape = l0op::Reshape(resizeOut, shapeArrayReverse, uniqueExecutor.get());
        CHECK_RET(resizeDOutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // CAST回bf16
        if (dtype == op::DataType::DT_BF16) {
            resizeDOutReshape = l0op::Cast(resizeDOutReshape, op::DataType::DT_BF16, uniqueExecutor.get());
        }
        CHECK_RET(resizeDOutReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // resizeDOut Transpose, shape：HWNC-->NCHW
        auto resizeDOutTranspose = l0op::Transpose(resizeDOutReshape, permuteHWNCArray, uniqueExecutor.get());
        CHECK_RET(resizeDOutTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(resizeDOutTranspose, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBicubic2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBicubic2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
