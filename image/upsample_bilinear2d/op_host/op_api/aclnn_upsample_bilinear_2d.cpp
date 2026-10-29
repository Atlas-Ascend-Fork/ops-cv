/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_upsample_bilinear_2d.h"
#include <cmath>
#include "upsample_bilinear2d.h"
#include "image/resize_bilinear_v2/op_api/resize_bilinear_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "level0/fill.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "error_report_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
const int64_t AICPU_SHAPE = 2L;
const int64_t AICPU_OFFSET_NHWC = 1L;

const double MAX_SUPPORT_SCALE = 50;

const int64_t DIM_ZERO = 0;
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
const float MAX_SUPPORT_ZOOM_SCALE_REV = 0.02f;
constexpr double UNSUPPORT_SCALES_TWO = 2.0;
constexpr double UNSUPPORT_SCALES_ZERO = 0.0;
const int64_t FOURDIMS = 4;
} // namespace
constexpr uint32_t ZERO = 0;
constexpr uint32_t ONE = 1;
constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_ALL = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_FOR_AICORE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const int64_t DIMLIMIT = 4;

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_ALL, return false);
    return true;
}

static bool CheckDtypeEqual(const aclTensor* selfRef, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(selfRef, out->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    // 需要根据算子实际情况添加校验
    const op::Format selfFormat = self->GetStorageFormat();
    if (selfFormat != out->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnUpsampleBilinear2d", "self,out",
                                                (std::string(op::ToString(selfFormat).GetString()) + ", " +
                                                 op::ToString(out->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "format of self and out must be the same");
        return false;
    }
    // 如果输入格式不支持
    if (selfFormat != op::Format::FORMAT_NCHW && selfFormat != op::Format::FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnUpsampleBilinear2d", "self", op::ToString(selfFormat).GetString(),
                                   "NCHW or NHWC");
        return false;
    }
    // 如果输入格式是不为4D，记录日志，直接报错
    OP_CHECK_WRONG_DIMENSION(self, DIMLIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIMLIMIT, return false);

    const op::DataType selfType = self->GetDataType();
    if ((selfFormat == op::Format::FORMAT_NCHW) && (selfType == op::DataType::DT_DOUBLE)) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnUpsampleBilinear2d", "self",
                                   op::ToString(self->GetStorageFormat()).GetString(),
                                   "NHWC is the only supported format for the double dtype");
        return false;
    }
    return true;
}

static bool CheckScalesAndShapeValid(const aclTensor* self, const aclTensor* out, const double scaleH,
                                     const double scaleW)
{
    auto inputShape = self->GetViewShape();
    auto outputShape = out->GetViewShape();
    int64_t input_h = inputShape.GetDim(DIM_TWO);
    int64_t input_w = inputShape.GetDim(DIM_THREE);
    int64_t output_h = outputShape.GetDim(DIM_TWO);
    int64_t output_w = outputShape.GetDim(DIM_THREE);
    int64_t scaleSizeH = static_cast<int64_t>(input_h * scaleH);
    int64_t scaleSizeW = static_cast<int64_t>(input_w * scaleW);
    return scaleSizeH == output_h && scaleSizeW == output_w;
}

static bool CheckScalesValid(const double weight, const double high)
{
    if ((weight < 0) || (high < 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnUpsampleBilinear2d", "scales_w,scales_h",
            (std::to_string(weight).c_str() + std::string(", ") + std::to_string(high).c_str()).c_str(),
            "scales must be finite and non-negative");
        return false;
    }
    return true;
}

static bool CheckOutputSize(const aclTensor* out, const aclIntArray* outputSize)
{
    if (outputSize == nullptr) {
        error_report::ReportInvalidArgument("aclnnUpsampleBilinear2d", "outputSize", "outputSize must not be null");
        return false;
    }
    if (outputSize->Size() != TWO) {
        OP_LOGE_FOR_INVALID_LISTSIZE("aclnnUpsampleBilinear2d", "outputSize",
                                     std::to_string(outputSize->Size()).c_str(), "2");
        return false;
    }
    auto outShape = out->GetViewShape();
    const op::Format outFormat = out->GetStorageFormat();
    if (outFormat == op::Format::FORMAT_NCHW) {
        if (outputSize->GetData()[ZERO] != outShape.GetDim(TWO) ||
            outputSize->GetData()[ONE] != outShape.GetDim(THREE)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2d", "out_h",
                                                     std::to_string(outShape.GetDim(TWO)).c_str(),
                                                     "output tensor H/W dims must match outputSize");
            return false;
        }
    } else if (outFormat == op::Format::FORMAT_NHWC) {
        if (outputSize->GetData()[ZERO] != outShape.GetDim(ONE) || outputSize->GetData()[ONE] != outShape.GetDim(TWO)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnUpsampleBilinear2d", "out_h",
                                                     std::to_string(outShape.GetDim(TWO)).c_str(),
                                                     "output tensor H/W dims must match outputSize");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* selfRef, const aclTensor* out, const double scalesW,
                               const double scalesH, const aclIntArray* outputSize)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查selfRef和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckDtypeEqual(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持、输入输出格式是否相等
    CHECK_RET(CheckFormat(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 5.检查self和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(selfRef, out), ACLNN_ERR_PARAM_INVALID);

    // 6.检验scalesW/scalesH，与资料保持一致
    CHECK_RET(CheckScalesValid(scalesW, scalesH), ACLNN_ERR_PARAM_INVALID);

    // 7.校验outputSize与out的HW轴是否一致
    CHECK_RET(CheckOutputSize(out, outputSize), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static double GetBilinearScales(int64_t input_size, int64_t output_size, double scale, const bool alignCorners)
{
    double real_scale = 0.0;

    if (output_size == input_size) {
        return static_cast<double>(1);
    }

    if (alignCorners) {
        if (output_size > 1) {
            real_scale = static_cast<double>(input_size - 1) / (output_size - 1);
        } else {
            real_scale = static_cast<double>(0);
        }
    } else {
        real_scale = (scale > 0) ? static_cast<double>(1.0 / scale) : (static_cast<double>(input_size) / output_size);
    }

    return real_scale;
}

static bool CheckBilinear2dScales(const aclTensor* x, const aclTensor* y, const double scaleH, const double scaleW,
                                  const bool alignCorners)
{
    auto dataType = x->GetDataType();
    auto inputShape = x->GetViewShape();
    auto outputShape = y->GetViewShape();
    int64_t input_h = inputShape.GetDim(DIM_TWO);
    int64_t input_w = inputShape.GetDim(DIM_THREE);
    int64_t output_h = outputShape.GetDim(DIM_TWO);
    int64_t output_w = outputShape.GetDim(DIM_THREE);
    double scale_h1 = scaleH;
    double scale_w1 = scaleW;

    if ((std::abs(scaleW - UNSUPPORT_SCALES_TWO) < 1e-9) && (std::abs(scaleH - UNSUPPORT_SCALES_TWO) < 1e-9) &&
        dataType != op::DataType::DT_BF16) {
        return false;
    }

    if ((std::abs(scaleH) < 1e-12) || (std::abs(scaleW) < 1e-12)) {
        return false;
    }

    // size(fp16&fp32) go resizeBilinearV2, size(bf16)go upsampleBilinear2d
    if ((std::abs(scaleH - 1.0) < 1e-9) && (std::abs(scaleW - 1.0) < 1e-9) &&
        (output_h != input_h || output_w != input_w)) {
        if (dataType != op::DataType::DT_BF16) {
            return false;
        } else {
            scale_h1 = 0.0;
            scale_w1 = 0.0;
        }
    }

    float scales_h = GetBilinearScales(input_h, output_h, scale_h1, alignCorners);
    float scales_w = GetBilinearScales(input_w, output_w, scale_w1, alignCorners);

    return scales_h <= MAX_SUPPORT_SHRINK_SCALE && scales_w <= MAX_SUPPORT_SHRINK_SCALE;
}

static const aclTensor* GoResizeBilinearV2AICORE(const aclTensor* selfRefContiguous, const aclIntArray* outputSize,
                                                 const bool alignCorners, const aclTensor* outContiguous,
                                                 const aclTensor* out, aclOpExecutor* executor)
{
    auto dstFormat = out->GetStorageFormat();
    auto size = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT64));
    auto castSize = l0op::Cast(size, op::DataType::DT_INT32, executor);

    auto dataType = selfRefContiguous->GetDataType();
    if (op::DataType::DT_BF16 == dataType) {
        selfRefContiguous = l0op::Cast(selfRefContiguous, op::DataType::DT_FLOAT, executor);
        outContiguous = l0op::Cast(outContiguous, op::DataType::DT_FLOAT, executor);
    }

    auto selfTransdata = l0op::TransDataSpecial(selfRefContiguous, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(selfTransdata != nullptr, nullptr);

    auto outTransdata = l0op::TransDataSpecial(outContiguous, op::Format::FORMAT_NC1HWC0, 0, executor);
    CHECK_RET(outTransdata != nullptr, nullptr);

    // 调用UpsampleBilinear算子kernel
    const aclTensor* upsampleBilinearout = l0op::ResizeBilinearV2(selfTransdata, castSize, alignCorners, outTransdata,
                                                                  executor);
    CHECK_RET(upsampleBilinearout != nullptr, nullptr);

    auto upsampleBilinearoutTransdata = l0op::TransData(upsampleBilinearout, dstFormat, 0, executor);
    CHECK_RET(upsampleBilinearoutTransdata != nullptr, nullptr);

    if (op::DataType::DT_BF16 == dataType) {
        upsampleBilinearoutTransdata = l0op::Cast(upsampleBilinearoutTransdata, op::DataType::DT_BF16, executor);
        return upsampleBilinearoutTransdata;
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    const aclTensor* castOut = l0op::Cast(upsampleBilinearoutTransdata, selfRefContiguous->GetDataType(), executor);

    return castOut;
}

static const aclTensor* GoResizeBilinearV2AiCoreWith4d(const aclTensor* selfRefContiguous,
                                                       const aclIntArray* outputSize, const bool alignCorners,
                                                       const aclFloatArray* scales, const aclTensor* outContiguous,
                                                       aclOpExecutor* executor)
{
    auto size = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT64));
    auto castSize = l0op::Cast(size, op::DataType::DT_INT32, executor);

    // 调用UpsampleBilinear算子kernel
    const aclTensor* upsampleBilinearout = l0op::ResizeBilinearV2With4d(selfRefContiguous, castSize, alignCorners,
                                                                        scales, outContiguous, executor);
    CHECK_RET(upsampleBilinearout != nullptr, nullptr);

    // 固定写法，将计算结果转换成输出out的数据类型
    const aclTensor* castOut = l0op::Cast(upsampleBilinearout, selfRefContiguous->GetDataType(), executor);

    return castOut;
}

static const aclTensor* GoUpsampleBilinear2DAICORE(const aclTensor* selfRefContiguous, const aclIntArray* outputSize,
                                                   const bool alignCorners, const double scalesH, const double scalesW,
                                                   const aclTensor* outContiguous, aclOpExecutor* executor)
{
    auto dataType = selfRefContiguous->GetDataType();
    auto size = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT64));
    auto castSize = l0op::Cast(size, op::DataType::DT_INT32, executor);
    auto inputShape = selfRefContiguous->GetViewShape();
    auto outputShape = outContiguous->GetViewShape();

    int64_t batch = inputShape.GetDim(DIM_ZERO);
    int64_t channels = inputShape.GetDim(DIM_ONE);
    int64_t input_h = inputShape.GetDim(DIM_TWO);
    int64_t input_w = inputShape.GetDim(DIM_THREE);
    int64_t output_h = outputShape.GetDim(DIM_TWO);
    int64_t output_w = outputShape.GetDim(DIM_THREE);
    if (input_h == output_h && input_w == output_w) {
        return selfRefContiguous;
    }

    const int64_t permuteHWNCList[] = {0, 1, 3, 2};
    auto permuteHWNCArray = executor->AllocIntArray(permuteHWNCList, FOURDIMS);

    const aclTensor* upsampleBilinearout;
    if (input_w == 1 && input_h != 1) {
        auto selfTranspose = l0op::Transpose(selfRefContiguous, permuteHWNCArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        const int64_t new_out_reshape[4] = {batch, channels, output_h, output_w};
        aclIntArray* out_shape_array = executor->AllocIntArray(new_out_reshape, FOURDIMS);
        auto outReshape = l0op::Reshape(outContiguous, out_shape_array, executor);
        CHECK_RET(outReshape != nullptr, nullptr);

        auto outTranspose = l0op::Transpose(outReshape, permuteHWNCArray, executor);
        CHECK_RET(outTranspose != nullptr, nullptr);

        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            selfTranspose = l0op::Cast(selfTranspose, op::DataType::DT_FLOAT, executor);
            outTranspose = l0op::Cast(outTranspose, op::DataType::DT_FLOAT, executor);
        }
        auto outRes = l0op::UpsampleBilinear2dNcdhw(selfTranspose, castSize, alignCorners, scalesW, scalesH,
                                                    outTranspose, executor);

        const int64_t out_reshape2[4] = {batch, channels, output_w, output_h};
        aclIntArray* out_shape2 = executor->AllocIntArray(out_reshape2, FOURDIMS);
        upsampleBilinearout = l0op::Reshape(outRes, out_shape2, executor);
    } else {
        if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
            selfRefContiguous = l0op::Cast(selfRefContiguous, op::DataType::DT_FLOAT, executor);
            outContiguous = l0op::Cast(outContiguous, op::DataType::DT_FLOAT, executor);
        }
        upsampleBilinearout = l0op::UpsampleBilinear2dNcdhw(selfRefContiguous, castSize, alignCorners, scalesH, scalesW,
                                                            outContiguous, executor);
    }
    CHECK_RET(upsampleBilinearout != nullptr, nullptr);

    const aclTensor* castOut;
    if (op::DataType::DT_BF16 == dataType || op::DataType::DT_FLOAT16 == dataType) {
        castOut = l0op::Cast(upsampleBilinearout, dataType, executor);
    } else {
        castOut = l0op::Cast(upsampleBilinearout, selfRefContiguous->GetDataType(), executor);
    }

    if (input_w == 1 && input_h != 1) {
        castOut = l0op::Transpose(castOut, permuteHWNCArray, executor);
    }
    return castOut;
}

// self 为空（H==0 或 W==0）时没有有效输入可插值。aclnn 早退不写输出会把未初始化内存返回给用户。
static const aclTensor* GetOutTensorWithValueZero(aclTensor* out, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, out->GetDataType());
    auto outDims = op::ToShapeVector(out->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outDims.data(), outDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                    const bool alignCorners, const double scalesH, const double scalesW,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleBilinear2d, DFX_IN(self, outputSize, alignCorners, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out, scalesW, scalesH, outputSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // IndexPut算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || out->IsEmpty()) {
        // 输出为空（如 N==0 或 C==0）时没有元素可写，直接成功。
        // 输出非空（self 的 H==0 或 W==0，但 out 各维均大于 0）时必须写成全 0，避免把未初始化内存返回给用户。
        if (out->IsEmpty()) {
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
        auto fillOut = GetOutTensorWithValueZero(out, uniqueExecutor.get());
        CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(fillOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfRefContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入out转换成连续的tensor
    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* castOut;
    const float scalesList[] = {static_cast<float>(scalesH), static_cast<float>(scalesW)};
    const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList, DIM_TWO);
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((curArch == NpuArch::DAV_2201) && CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_FOR_AICORE) &&
        CheckBilinear2dScales(self, out, scalesH, scalesW, alignCorners) &&
        CheckScalesAndShapeValid(self, out, scalesH, scalesW)) {
        castOut = GoUpsampleBilinear2DAICORE(selfRefContiguous, outputSize, alignCorners, scalesH, scalesW,
                                             outContiguous, uniqueExecutor.get());
    } else if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_FOR_AICORE)) {
        if (IsRegBase(curArch)) {
            castOut = GoResizeBilinearV2AiCoreWith4d(selfRefContiguous, outputSize, alignCorners, scales, outContiguous,
                                                     uniqueExecutor.get());
        } else {
            castOut = GoResizeBilinearV2AICORE(selfRefContiguous, outputSize, alignCorners, outContiguous, out,
                                               uniqueExecutor.get());
        }
    } else {
        auto outShape = op::ToShapeVector(outContiguous->GetViewShape());
        aclIntArray* newOutputSize = uniqueExecutor.get()->AllocIntArray(outShape.data(), outShape.size());
        auto size = uniqueExecutor.get()->ConvertToTensor(newOutputSize, op::ToOpDataType(ACL_INT32));
        const aclTensor* newSize = uniqueExecutor.get()->CreateView(size, op::Shape({AICPU_SHAPE}), AICPU_OFFSET_NHWC);
        const aclTensor* upsampleBilinearout = nullptr;
        if (IsRegBase(curArch)) {
            upsampleBilinearout = l0op::ResizeBilinearV2With4d(selfRefContiguous, newSize, alignCorners, scales,
                                                               outContiguous, uniqueExecutor.get());
        } else {
            upsampleBilinearout = l0op::ResizeBilinearV2(selfRefContiguous, newSize, alignCorners, outContiguous,
                                                         uniqueExecutor.get());
        }
        CHECK_RET(upsampleBilinearout != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果转换成输出out的数据类型
        castOut = l0op::Cast(upsampleBilinearout, self->GetDataType(), uniqueExecutor.get());
    }
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleBilinear2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleBilinear2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
