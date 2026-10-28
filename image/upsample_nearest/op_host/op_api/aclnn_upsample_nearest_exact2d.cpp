/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include <cmath>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "upsample_nearest_exact2d.h"
#include "aclnn_upsample_nearest_exact2d.h"
#include "image/resize_nearest_neighbor_v2/op_api/resize_nearest_neighbor_v2.h"
#include "op_api/aclnn_check.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/transdata.h"
#include "image/upsample_nearest_exact3d/op_host/op_api/upsample_nearest_exact3d.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
// 根据API定义，需要列出所能支持的所有dtype
// A2/A3（非 RegBase）支持 uint8
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16, op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const int64_t DIM_LIMIT = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t EXPECT_SIZE = 2;
static constexpr size_t UP_BATCH = 17;

// CaseNearestExact3d阈值常量定义
static constexpr int64_t CASE_NEAREST_EXACT3D_INPUT_H_MIN = 740;
static constexpr int64_t CASE_NEAREST_EXACT3D_INPUT_H_MAX = 750;
static constexpr float CASE_NEAREST_EXACT3D_SCALE_H_MIN = 7.5f;
static constexpr float CASE_NEAREST_EXACT3D_SCALE_H_MAX = 8.0f;
static constexpr int64_t CASE_NEAREST_EXACT3D_INPUT_W_MIN = 740;
static constexpr int64_t CASE_NEAREST_EXACT3D_INPUT_W_MAX = 750;
static constexpr float CASE_NEAREST_EXACT3D_SCALE_W_MIN = 6.5f;
static constexpr float CASE_NEAREST_EXACT3D_SCALE_W_MAX = 7.0f;

static const aclTensor* View4dAs5d(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW or ND -> contigious -> unsqueeze(2) -> reformat -> NCDHW
    // contiguous input to 5D tensor
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze( 2 )
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, static_cast<int64_t>(DIM_TWO), executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);

    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* View5dAs4d(const aclTensor* input, op::Format format, aclOpExecutor* executor)
{
    // NCHW -> squeeze -> reformat -> NCHW or ND
    // squeeze out into 4D
    const int64_t removeDim[] = {DIM_TWO};
    aclIntArray* dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);

    // reformat to NCHW or ND
    auto reformatInput = l0op::ReFormat(squeezedInput, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在UpsampleNearestExact2d算子的支持列表内（按 SoC 区分：950 不支持 uint8）
    if (IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_REGBASE, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    // 检查self的数据类型是否与out一致
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    size_t outputSizeNum = outputSize->Size();
    OP_CHECK_WRONG_DIMENSION(self, DIM_LIMIT, return false);
    OP_CHECK(outputSizeNum == EXPECT_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected output_size to be 2, but got %zu", outputSizeNum),
             return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_LIMIT, return false);

    // out 的 shape 应等于 (N, C, outputSize[0], outputSize[1])（NHWC 时 C 在末维）
    op::Shape inputShape = self->GetViewShape();
    op::Shape expectShape;
    if (self->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        expectShape.AppendDim(inputShape.GetDim(DIM_ZERO));
        expectShape.AppendDim((*outputSize)[DIM_ZERO]);
        expectShape.AppendDim((*outputSize)[DIM_ONE]);
        expectShape.AppendDim(inputShape.GetDim(DIM_THREE));
    } else {
        expectShape.AppendDim(inputShape.GetDim(DIM_ZERO));
        expectShape.AppendDim(inputShape.GetDim(DIM_ONE));
        expectShape.AppendDim((*outputSize)[DIM_ZERO]);
        expectShape.AppendDim((*outputSize)[DIM_ONE]);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);

    return true;
}

static constexpr double EPSILON = 1e-6;

static bool CheckInputElement(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out,
                              double scalesH, double scalesW)
{
    auto selfShape = self->GetViewShape();
    auto outShape = out->GetViewShape();
    int64_t inputN = selfShape.GetDim(DIM_ZERO);
    int64_t inputC = selfShape.GetDim(DIM_ONE);
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outputN = outShape.GetDim(DIM_ZERO);
    int64_t outputC = outShape.GetDim(DIM_ONE);
    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    if (self->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputC = selfShape.GetDim(DIM_THREE);
        outputC = outShape.GetDim(DIM_THREE);
        inputH = selfShape.GetDim(DIM_ONE);
        inputW = selfShape.GetDim(DIM_TWO);
    }

    OP_CHECK(inputH > 0 && inputW > 0 && outputH > 0 && outputW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Input and output sizes should be greater than 0, but got input ("
                     "H: %ld, W: %ld) output (H: %ld, W: %ld)",
                     inputH, inputW, outputH, outputW),
             return false);
    if ((inputN != outputN) || (inputC != outputC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputN[%ld]/outputN[%ld] or inputC[%ld]/outputC[%ld] are not equal.", inputN,
                outputN, inputC, outputC);
        return false;
    }

    // 输入输出一致性校验锁定与输入同值，此处只需校验输入侧
    OP_CHECK(inputN >= 0 && inputC >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input N/C sizes should be non-negative, but got N: %ld, C: %ld.", inputN,
                     inputC),
             return false);
    // 资料约束：outputSize = floor(inputSize * scales)，非整数乘积时合法值是 floor 而非 ceil；
    // 绝对容差仅覆盖乘积为整数时的浮点表示噪声，不随乘积放大，不放过 off-by-one 的非法组合
    if (scalesH > 0) {
        double expectDimH = static_cast<double>(inputH) * scalesH;
        OP_CHECK(static_cast<double>(outputH) == std::floor(expectDimH) ||
                     fabs(static_cast<double>(outputH) - expectDimH) <= EPSILON,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Scale conflicts with outputSize. scale_H * input_H should be equal to outputSize_H"),
                 return false);
    }
    if (scalesW > 0) {
        double expectDimW = static_cast<double>(inputW) * scalesW;
        OP_CHECK(static_cast<double>(outputW) == std::floor(expectDimW) ||
                     fabs(static_cast<double>(outputW) - expectDimW) <= EPSILON,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Scale conflicts with outputSize. scale_W * input_W should be equal to outputSize_W"),
                 return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    auto selfFormat = self->GetStorageFormat();
    auto outFormat = out->GetStorageFormat();
    if (selfFormat != op::Format::FORMAT_NCHW && selfFormat != op::Format::FORMAT_NHWC &&
        selfFormat != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self only supports [NCHW, NHWC, ND], but format is [%s]",
                op::ToString(selfFormat).GetString());
        return false;
    }

    if (selfFormat != outFormat) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self and out should be equal, self [%s], out [%s].",
                op::ToString(selfFormat).GetString(), op::ToString(outFormat).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, double scalesH, double scalesW,
                               const aclTensor* out)
{
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckInputElement(self, outputSize, out, scalesH, scalesW), ACLNN_ERR_PARAM_INVALID);
    // 检查输入输出的Format
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsCaseNearestExact3d(const aclTensor* self, const aclIntArray* outputSize)
{
    if (self->GetStorageFormat() != op::Format::FORMAT_NCHW && self->GetStorageFormat() != op::Format::FORMAT_NHWC) {
        return false;
    }
    auto selfShape = self->GetViewShape();
    int64_t inputN = selfShape.GetDim(DIM_ZERO);
    int64_t inputC = selfShape.GetDim(DIM_ONE);
    int64_t inputH = selfShape.GetDim(DIM_TWO);
    int64_t inputW = selfShape.GetDim(DIM_THREE);
    int64_t outputH = (*outputSize)[DIM_ZERO];
    int64_t outputW = (*outputSize)[DIM_ONE];
    float scaleH = inputH > 0 ? static_cast<float>(outputH) / inputH : 0.0f;
    float scaleW = inputW > 0 ? static_cast<float>(outputW) / inputW : 0.0f;
    if (inputN * inputC != UP_BATCH) {
        return false;
    }
    if (inputH < CASE_NEAREST_EXACT3D_INPUT_H_MIN || inputH > CASE_NEAREST_EXACT3D_INPUT_H_MAX ||
        scaleH < CASE_NEAREST_EXACT3D_SCALE_H_MIN || scaleH > CASE_NEAREST_EXACT3D_SCALE_H_MAX) {
        return false;
    }
    if (inputW < CASE_NEAREST_EXACT3D_INPUT_W_MIN || inputW > CASE_NEAREST_EXACT3D_INPUT_W_MAX ||
        scaleW < CASE_NEAREST_EXACT3D_SCALE_W_MIN || scaleW > CASE_NEAREST_EXACT3D_SCALE_W_MAX) {
        return false;
    }
    return true;
}

static const aclTensor* upsampleNearestExact2dCompute(const aclTensor* selfContiguous, const aclIntArray* outputSize,
                                                      const aclFloatArray* scales, const aclTensor* outContiguous,
                                                      aclOpExecutor* executor)
{
    float scalesH = (*scales)[DIM_ZERO];
    float scalesW = (*scales)[DIM_ONE];
    if (IsRegBase()) {
        if (IsCaseNearestExact3d(selfContiguous, outputSize)) {
            auto self5d = View4dAs5d(selfContiguous, executor);
            CHECK_RET(self5d != nullptr, nullptr);

            FVector<int64_t> outputSizeVector{1, outputSize->GetData()[0], outputSize->GetData()[1]};
            aclIntArray* outputSizeArray = executor->AllocIntArray(outputSizeVector.data(), 3);
            CHECK_RET(outputSizeArray != nullptr, nullptr);
            FVector<float> scaleVector{1.0f, scalesH, scalesW};

            auto scalesArr = executor->AllocFloatArray(scaleVector.data(), 3);

            auto result = l0op::UpsampleNearestExact3dNcdhw(self5d, outputSizeArray, scalesArr, executor);
            CHECK_RET(result != nullptr, nullptr);
            return View5dAs4d(result, selfContiguous->GetStorageFormat(), executor);
        } else {
            auto size = executor->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT32));
            return l0op::ResizeNearestNeighborV2(selfContiguous, size, scales, false, true, outContiguous, executor);
        }
    } else {
        if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NCHW ||
            selfContiguous->GetStorageFormat() == op::Format::FORMAT_ND) {
            const int64_t permuteNCHWList[] = {DIM_ZERO, DIM_TWO, DIM_THREE, DIM_ONE};
            auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, DIM_LIMIT);
            CHECK_RET(permuteNCHWArray != nullptr, nullptr);

            auto selfTranspose = l0op::Transpose(selfContiguous, permuteNCHWArray, executor);
            CHECK_RET(selfTranspose != nullptr, nullptr);

            auto self = l0op::ReFormat(selfTranspose, op::Format::FORMAT_NHWC);
            CHECK_RET(self != nullptr, nullptr);

            auto selfUpsampleNearestExact = l0op::UpsampleNearestExact2d(self, outputSize, scalesH, scalesW, true,
                                                                         executor);
            CHECK_RET(selfUpsampleNearestExact != nullptr, nullptr);

            const int64_t permuteNHWCList[] = {DIM_ZERO, DIM_THREE, DIM_ONE, DIM_TWO};
            auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, DIM_LIMIT);
            CHECK_RET(permuteNHWCArray != nullptr, nullptr);

            auto outReformat = l0op::ReFormat(selfUpsampleNearestExact, selfContiguous->GetStorageFormat());
            CHECK_RET(outReformat != nullptr, nullptr);

            return l0op::Transpose(outReformat, permuteNHWCArray, executor);
        } else {
            return l0op::UpsampleNearestExact2d(selfContiguous, outputSize, scalesH, scalesW, true, executor);
        }
    }
    return nullptr;
}
} // namespace

aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize,
                                                        double scalesH, double scalesW, aclTensor* out,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnUpsampleNearestExact2d, DFX_IN(self, outputSize, scalesH, scalesW), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, outputSize, scalesH, scalesW, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor支持
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self、输出out转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    float realScalesH = scalesH > 0 ? static_cast<float>(scalesH) : 0;
    float realScalesW = scalesW > 0 ? static_cast<float>(scalesW) : 0;
    if (!IsRegBase()) {
        // 使用double类型计算1/scale，避免tiling中用float计算造成精度损失
        realScalesH = scalesH > 0 ? static_cast<float>(1.0 / scalesH) : 0;
        realScalesW = scalesW > 0 ? static_cast<float>(1.0 / scalesW) : 0;
    }

    // 调用upsampleNearestExact2dCompute计算
    vector<float> scalesList{};
    scalesList.push_back(realScalesH);
    scalesList.push_back(realScalesW);
    const aclFloatArray* scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto result = upsampleNearestExact2dCompute(selfContiguous, outputSize, scales, outContiguous,
                                                uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearestExact2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearestExact2d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
