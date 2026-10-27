/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_upsample_nearest_1d_v2.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/contiguous.h"
#include "image/resize_nearest_neighbor_v2/op_api/resize_nearest_neighbor_v2.h"
#include "image/upsample_nearest/op_host/op_api/upsample_nearest_exact2d.h"
#include "upsample_nearest_3d.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include <iomanip>
#include <sstream>
#include <string>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static const size_t THREEDIMS = 3;
static const size_t DIM_IDX_2 = 2;
static const int64_t ZERO = 0;
static const uint64_t ONE = 1;
static const double DOUBLE_ONE = 1.0;
static const int64_t SCALE_PRECISION_LEN = 6; // 清洗 aclnn float 入参二进制噪声保留的十进制小数位

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_UINT8,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self)
{
    auto supportDtypeList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportDtypeList, return false);
    return true;
}

static bool CheckDtypeEqual(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out, const aclIntArray* outputSize)
{
    uint64_t size = outputSize->Size();
    OP_CHECK(size == ONE, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of outputSize should be 1, but got %zu", size),
             return false);
    auto outShape = out->GetViewShape();
    if (self->GetViewShape().GetDimNum() != THREEDIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected input size to be %zu, but got %zu.", THREEDIMS,
                self->GetViewShape().GetDimNum());
        return false;
    }
    if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(DIM_IDX_2) == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Non-empty 3D data tensor expected but got a tensor with sizes %s.",
                op::ToString(self->GetViewShape()).GetString());
        return false;
    }
    if (outputSize->GetData()[ZERO] != outShape.GetDim(DIM_IDX_2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected output size (L: %ld), but got output tensor size (L: %ld).",
                outputSize->GetData()[ZERO], outShape.GetDim(DIM_IDX_2));
        return false;
    }
    if (outputSize->GetData()[ZERO] == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input and output sizes should be greater than %ld, "
                "but got output (W: %ld).",
                ZERO, outputSize->GetData()[ZERO]);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], out [%s].",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    if (self->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input format should be NCL. Actual: self [%s].",
                op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out数据类型是否相同
    CHECK_RET(CheckDtypeEqual(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(self, out, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查format是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* View4dAs3d(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t removeDim[] = {2};
    aclIntArray* dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);

    // reformat to NCL
    auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* View3dAs4d(const aclTensor* input, aclOpExecutor* executor)
{
    // NCL -> contigious -> unsqueeze(2) -> reformat -> NCHW
    // 转换成连续的tensor
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    // 新增一维
    const int64_t appendDim[] = {2};
    aclIntArray* dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

static const aclTensor* View4dAs5d(const aclTensor* input, aclOpExecutor* executor)
{
    const int64_t appendDim[] = {2};
    aclIntArray* dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInputTensor = l0op::UnsqueezeNd(input, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInputTensor != nullptr, nullptr);

    auto reformatInput = l0op::ReFormat(unsqueezedInputTensor, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

static const aclTensor* View5dAs4d(const aclTensor* input, aclOpExecutor* executor)
{
    const int64_t removeDim[] = {2};
    aclIntArray* dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInputTensor = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInputTensor != nullptr, nullptr);

    auto reformatInput = l0op::ReFormat(squeezedInputTensor, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

static const aclTensor* upsampleNearest1dV2AiCpuCompute(const aclTensor* selfContiguous, const aclTensor* outContiguous,
                                                        const aclTensor* size, aclOpExecutor* executor)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        const int64_t permuteNHWCList[] = {0, 2, 3, 1};
        auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, 4);
        CHECK_RET(permuteNHWCArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNHWCArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);
        auto outTranspose = l0op::Transpose(outContiguous, permuteNHWCArray, executor);
        CHECK_RET(outTranspose != nullptr, nullptr);

        const aclTensor* resizeNearestOutAiCpu = l0op::ResizeNearestNeighborV2(selfTranspose, size, nullptr, false,
                                                                               false, outTranspose, executor);
        CHECK_RET(resizeNearestOutAiCpu != nullptr, nullptr);

        const int64_t permuteNCHWList[] = {0, 3, 1, 2};
        auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, 4);
        CHECK_RET(permuteNCHWArray != nullptr, nullptr);

        return l0op::Transpose(resizeNearestOutAiCpu, permuteNCHWArray, executor);
    } else {
        return l0op::ResizeNearestNeighborV2(selfContiguous, size, nullptr, false, false, outContiguous, executor);
    }
}

static const aclTensor* upsampleNearest1dAiCoreCompute(const aclTensor* selfContiguous, const aclIntArray* outputSize,
                                                       float scaleL, aclOpExecutor* executor)
{
    auto self = View4dAs5d(selfContiguous, executor);
    CHECK_RET(self != nullptr, nullptr);
    // 通过十进制字符串清洗 aclnn float 入参的二进制噪声后再取倒数，
    // 对常规量级 scale 的结果与参考实现以 double 精度计算 1/scale 的语义对齐，避免取整边界差 1
    std::stringstream scaleLStream;
    scaleLStream << std::fixed << std::setprecision(SCALE_PRECISION_LEN) << scaleL;
    // 清洗结果可能被量化为 0（scaleL < 0.5e-6 时），此时回退原始 scale 取倒数，避免 1/0 得到 inf
    const double cleanedScaleL = std::stod(scaleLStream.str());
    float scalesW = scaleL > 0 ? static_cast<float>(DOUBLE_ONE /
                                                    (cleanedScaleL > 0 ? cleanedScaleL : static_cast<double>(scaleL))) :
                                 0;

    vector<float> scalesList{};
    scalesList.push_back(1.0);
    scalesList.push_back(1.0);
    scalesList.push_back(scaleL);
    const aclFloatArray* scales = executor->AllocFloatArray(scalesList.data(), scalesList.size());
    CHECK_RET(scales != nullptr, nullptr);

    vector<float> scalesCastList{};
    scalesCastList.push_back(1.0);
    scalesCastList.push_back(1.0);
    scalesCastList.push_back(scalesW);
    const aclFloatArray* castScales = executor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
    CHECK_RET(castScales != nullptr, nullptr);

    vector<int64_t> sizeList{};
    sizeList.push_back(1);
    sizeList.push_back(1);
    sizeList.push_back((*outputSize)[ONE]);
    const aclIntArray* size = executor->AllocIntArray(sizeList.data(), sizeList.size());
    CHECK_RET(size != nullptr, nullptr);
    auto outUpsampleNearest = l0op::UpsampleNearest3dNcdhw(self, size, scales, castScales, executor);
    CHECK_RET(outUpsampleNearest != nullptr, nullptr);

    return View5dAs4d(outUpsampleNearest, executor);
}
} // namespace

aclnnStatus aclnnUpsampleNearest1dV2GetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, float scaleL,
                                                     aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleNearest1dV2, DFX_IN(self, outputSize, scaleL), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = View3dAs4d(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outContiguous = View3dAs4d(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    FVector<int64_t> outputSizeVector{1, outputSize->GetData()[0]};
    aclIntArray* outputSizeArray = uniqueExecutor.get()->AllocIntArray(outputSizeVector.data(), 2);
    CHECK_RET(outputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto size = uniqueExecutor.get()->ConvertToTensor(outputSizeArray, op::ToOpDataType(ACL_INT32));
    CHECK_RET(size != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* resizeNearestOut = nullptr;
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        bool isAscendCSupport = curArch == NpuArch::DAV_2201 || (curArch == NpuArch::DAV_2002 && scaleL > 0);
        if (isAscendCSupport) {
            resizeNearestOut = upsampleNearest1dAiCoreCompute(selfContiguous, outputSizeArray, scaleL,
                                                              uniqueExecutor.get());
        } else {
            if (IsRegBase(curArch)) {
                vector<float> scalesList{};
                scalesList.push_back(1.0f);
                scalesList.push_back(scaleL);
                const aclFloatArray* scaleArray = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
                CHECK_RET(scaleArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

                resizeNearestOut = l0op::ResizeNearestNeighborV2(selfContiguous, size, scaleArray, false, false,
                                                                 outContiguous, uniqueExecutor.get());
            } else {
                auto selfTransdata = l0op::TransDataSpecial(selfContiguous, op::Format::FORMAT_NC1HWC0, 0,
                                                            uniqueExecutor.get());
                CHECK_RET(selfTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

                auto outTransdata = l0op::TransDataSpecial(outContiguous, op::Format::FORMAT_NC1HWC0, 0,
                                                           uniqueExecutor.get());
                CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

                const aclTensor* resizeNearestOutAiCore = l0op::ResizeNearestNeighborV2(
                    selfTransdata, size, nullptr, false, false, outTransdata, uniqueExecutor.get());
                CHECK_RET(resizeNearestOutAiCore != nullptr, ACLNN_ERR_INNER_NULLPTR);

                resizeNearestOut = l0op::TransData(resizeNearestOutAiCore, selfContiguous->GetStorageFormat(), 0,
                                                   uniqueExecutor.get());
            }
        }
    } else {
        resizeNearestOut = upsampleNearest1dV2AiCpuCompute(selfContiguous, outContiguous, size, uniqueExecutor.get());
    }
    CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* out3d = nullptr;
    out3d = View4dAs3d(resizeNearestOut, uniqueExecutor.get());
    CHECK_RET(out3d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(out3d, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest1dV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest1dV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
