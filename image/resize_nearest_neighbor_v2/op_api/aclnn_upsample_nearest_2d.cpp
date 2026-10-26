/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "resize_nearest_neighbor_v2.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_upsample_nearest_2d.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int64_t FOURDIMS = 4;
static const int64_t ZERO = 0;
static const int64_t ONE = 1;
static const uint64_t TWO = 2;
static const int64_t THREE = 3;
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
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
    if (curArch == NpuArch::DAV_2201) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase(curArch)) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}
static bool CheckDtypeValid(const aclTensor* self)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    return true;
}

static bool CheckDtypeEqual(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* outputSize)
{
    uint64_t size = outputSize->Size();
    if (size != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output size equals to %lu, but got size %lu.", TWO, size);
        return false;
    }

    auto selfShape = self->GetViewShape();
    if (selfShape.GetDimNum() != FOURDIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input size equals to %ld, but got size %zu.", FOURDIMS,
                selfShape.GetDimNum());
        return false;
    }

    if (selfShape.GetDim(TWO) == ZERO || selfShape.GetDim(THREE) == ZERO || outputSize->GetData()[ZERO] == ZERO ||
        outputSize->GetData()[ONE] == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input and output sizes should be greater than %ld, "
                "but got input (H: %ld, W: %ld) output (H: %ld, W: %ld).",
                ZERO, selfShape.GetDim(TWO), selfShape.GetDim(THREE), outputSize->GetData()[ZERO],
                outputSize->GetData()[ONE]);
        return false;
    }

    if (selfShape.GetDim(ONE) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Non-empty 4D data tensor expected but got a tensor with size %s.",
                op::ToString(selfShape).GetString());
        return false;
    }

    return true;
}

static bool CheckFormatValid(const aclTensor* self)
{
    if (self->GetStorageFormat() == op::Format::FORMAT_NCHW || self->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        return true;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support this format [%s] of input.",
            op::ToString(self->GetStorageFormat()).GetString());
    return false;
}

static bool CheckFormatEqual(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], out [%s].",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckOutputSize(const aclTensor* out, const aclIntArray* outputSize)
{
    auto outShape = out->GetViewShape();
    if (out->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        if (outputSize->GetData()[ZERO] != outShape.GetDim(TWO) ||
            outputSize->GetData()[ONE] != outShape.GetDim(THREE)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected output size (H: %ld, W: %ld), but got output tensor size (H: %ld, W: %ld).",
                    outputSize->GetData()[ZERO], outputSize->GetData()[ONE], outShape.GetDim(TWO),
                    outShape.GetDim(THREE));
            return false;
        }
    } else if (out->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        if (outputSize->GetData()[ZERO] != outShape.GetDim(ONE) || outputSize->GetData()[ONE] != outShape.GetDim(TWO)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected output size (H: %ld, W: %ld), but got output tensor size (H: %ld, W: %ld).",
                    outputSize->GetData()[ZERO], outputSize->GetData()[ONE], outShape.GetDim(ONE),
                    outShape.GetDim(TWO));
            return false;
        }
    }
    return true;
}

static bool CheckUplimit(const aclTensor* self, const aclTensor* out)
{
    if (IsRegBase()) {
        return true;
    }
    int64_t inN = self->GetViewShape().GetDim(ZERO);
    int64_t inC = self->GetViewShape().GetDim(ONE);
    int64_t inH = self->GetViewShape().GetDim(TWO);
    int64_t inW = self->GetViewShape().GetDim(THREE);
    int64_t outN = out->GetViewShape().GetDim(ZERO);
    int64_t outC = out->GetViewShape().GetDim(ONE);
    int64_t outH = out->GetViewShape().GetDim(TWO);
    int64_t outW = out->GetViewShape().GetDim(THREE);

    OP_CHECK(
        inN <= INT32_MAX && inC <= INT32_MAX && inH <= INT32_MAX && inW <= INT32_MAX,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self sizes should not be greater than %d, but got self(%ld, %ld, %ld, %ld)",
                INT32_MAX, inN, inC, inH, inW),
        return false);
    OP_CHECK(
        outN <= INT32_MAX && outC <= INT32_MAX && outH <= INT32_MAX && outW <= INT32_MAX,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out sizes should not be greater than %d, but got out(%ld, %ld, %ld, %ld)",
                INT32_MAX, outN, outC, outH, outW),
        return false);
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
    CHECK_RET(CheckShape(self, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入的数据格式是否在API支持的数据格式范围之内
    CHECK_RET(CheckFormatValid(self), ACLNN_ERR_PARAM_INVALID);

    // 7. 检查self和out数据格式是否相同
    CHECK_RET(CheckFormatEqual(self, out), ACLNN_ERR_PARAM_INVALID);

    // 8.检查self和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 9.检查out和outputSize的H和W的大小是否一致
    CHECK_RET(CheckOutputSize(out, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 10. 校验上边界
    CHECK_RET(CheckUplimit(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* upsampleNearest2dAiCpuCompute(const aclTensor* selfContiguous, const aclTensor* outContiguous,
                                               const aclTensor* size, aclOpExecutor* executor)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        const int64_t permuteNHWCList[] = {0, 2, 3, 1};
        auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, FOURDIMS);
        CHECK_RET(permuteNHWCArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNHWCArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        auto outTranspose = l0op::Transpose(outContiguous, permuteNHWCArray, executor);
        CHECK_RET(outTranspose != nullptr, nullptr);

        const aclTensor* resizeNearestOutAiCpu = l0op::ResizeNearestNeighborV2(selfTranspose, size, nullptr, false,
                                                                               false, outTranspose, executor);
        CHECK_RET(resizeNearestOutAiCpu != nullptr, nullptr);

        const int64_t permuteNCHWList[] = {0, 3, 1, 2};
        auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, FOURDIMS);
        CHECK_RET(permuteNCHWArray != nullptr, nullptr);

        return l0op::Transpose(resizeNearestOutAiCpu, permuteNCHWArray, executor);
    } else {
        return l0op::ResizeNearestNeighborV2(selfContiguous, size, nullptr, false, false, outContiguous, executor);
    }
}
} // namespace

aclnnStatus aclnnUpsampleNearest2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleNearest2d, DFX_IN(self, outputSize), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto size = uniqueExecutor.get()->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT32));
    const aclTensor* resizeNearestOut = nullptr;

    if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        if (IsRegBase()) {
            resizeNearestOut = l0op::ResizeNearestNeighborV2(selfContiguous, size, nullptr, false, false, outContiguous,
                                                             uniqueExecutor.get());
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

            resizeNearestOut = l0op::TransData(resizeNearestOutAiCore, self->GetStorageFormat(), 0,
                                               uniqueExecutor.get());
        }
    } else {
        resizeNearestOut = upsampleNearest2dAiCpuCompute(selfContiguous, outContiguous, size, uniqueExecutor.get());
    }
    CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(resizeNearestOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest2d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
