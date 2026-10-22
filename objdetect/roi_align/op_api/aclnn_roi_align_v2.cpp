/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>

#include "aclnn_roi_align_v2.h"
#include "roi_align.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "level0/concat.h"
#include "level0/fill.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr size_t DIM_FIVE = 5;
static constexpr int64_t MAX_POOLED_SIZE = std::numeric_limits<int32_t>::max();

static const std::initializer_list<DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* boxes, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(boxes, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* boxes, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, boxes->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static bool CheckFormatValid(const aclTensor* self, const aclTensor* boxes, const aclTensor* out)
{
    if (self->GetStorageFormat() != op::Format::FORMAT_NCHW || boxes->GetStorageFormat() != op::Format::FORMAT_ND ||
        out->GetStorageFormat() != op::Format::FORMAT_NCHW) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid format. self and out only support NCHW, boxes only support ND.");
        return false;
    }

    return true;
}

static bool CheckPooledSizeValid(int64_t pooledHeight, int64_t pooledWidth)
{
    if (pooledHeight > MAX_POOLED_SIZE || pooledWidth > MAX_POOLED_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pooledHeight [%ld] and pooledWidth [%ld] must not exceed INT32_MAX [%ld]",
                pooledHeight, pooledWidth, MAX_POOLED_SIZE);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* boxes, const aclTensor* out, int64_t pooledHeight,
                       int64_t pooledWidth)
{
    OP_CHECK_WRONG_DIMENSION(self, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(boxes, DIM_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_FOUR, return false);

    auto const& selfShape = self->GetViewShape();
    auto const& boxesShape = boxes->GetViewShape();
    auto const& outShape = out->GetViewShape();
    if (selfShape.GetDim(DIM_ZERO) == DIM_ZERO || selfShape.GetDim(DIM_ONE) == DIM_ZERO ||
        selfShape.GetDim(DIM_TWO) == DIM_ZERO || selfShape.GetDim(DIM_THREE) == DIM_ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self shape [%ld,%ld,%ld,%ld] contains 0 in N/C/H/W",
                selfShape.GetDim(DIM_ZERO), selfShape.GetDim(DIM_ONE), selfShape.GetDim(DIM_TWO),
                selfShape.GetDim(DIM_THREE));
        return false;
    }
    if (boxesShape.GetDim(DIM_ONE) != DIM_FIVE) { // 5: boxes dim 1
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "boxes shape dim1 [%ld] should be 5", boxesShape.GetDim(DIM_ONE));
        return false;
    }
    if (outShape.GetDim(DIM_ZERO) != boxesShape.GetDim(DIM_ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim0 [%ld] and boxes shape dim0 [%ld] should be equal",
                outShape.GetDim(DIM_ZERO), boxesShape.GetDim(DIM_ZERO));
        return false;
    }
    if (outShape.GetDim(DIM_ONE) != selfShape.GetDim(DIM_ONE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim1 [%ld] and self shape dim1 [%ld] should be equal",
                outShape.GetDim(DIM_ONE), selfShape.GetDim(DIM_ONE));
        return false;
    }
    if (outShape.GetDim(DIM_TWO) != pooledHeight) { // 2: dim2
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim2 [%ld] and pooledHeight [%ld] should be equal",
                outShape.GetDim(DIM_TWO), pooledHeight); // 2: dim2
        return false;
    }
    if (outShape.GetDim(DIM_THREE) != pooledWidth) { // 3: dim3
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim3 [%ld] and pooledWidth [%ld] should be equal",
                outShape.GetDim(DIM_THREE), pooledWidth); // 3: dim3
        return false;
    }

    return true;
}

static bool CheckAttr(int64_t samplingRatio, float spatialScale)
{
    if (samplingRatio < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "samplingRatio [%ld] should be greater than or equal to 0", samplingRatio);
        return false;
    }
    if (!std::isfinite(spatialScale) || spatialScale <= 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "spatialScale [%f] should be finite and greater than 0", spatialScale);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* boxes, const aclTensor* out,
                               int64_t pooledHeight, int64_t pooledWidth, int64_t samplingRatio, float spatialScale)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, boxes, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, boxes, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入、输出的数据格式是否支持
    CHECK_RET(CheckFormatValid(self, boxes, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查池化尺寸是否能安全传递给 int32 表示的执行链路
    CHECK_RET(CheckPooledSizeValid(pooledHeight, pooledWidth), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入tensor的shape
    CHECK_RET(CheckShape(self, boxes, out, pooledHeight, pooledWidth), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查属性
    CHECK_RET(CheckAttr(samplingRatio, spatialScale), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetOutTensorWithValueZero(aclTensor* out, aclOpExecutor* executor)
{
    aclScalar* scalar = executor->AllocScalar(0);
    auto valueTensor = executor->ConvertToTensor(scalar, out->GetDataType());
    auto outputDims = op::ToShapeVector(out->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outputDims.data(), outputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    return l0op::Fill(dimTensor, valueTensor, dimArray, executor);
}

aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(const aclTensor* self, const aclTensor* boxes, int64_t pooledHeight,
                                            int64_t pooledWidth, float spatialScale, int64_t samplingRatio,
                                            bool aligned, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRoiAlignV2,
                   DFX_IN(self, boxes, pooledHeight, pooledWidth, spatialScale, samplingRatio, aligned), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, boxes, out, pooledHeight, pooledWidth, samplingRatio, spatialScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // roialign算子的空tensor在kernel中支持
    if (self->IsEmpty() || boxes->IsEmpty()) {
        // 根据实际支持情况补充
        auto fillOut = GetOutTensorWithValueZero(out, uniqueExecutor.get());
        CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(fillOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入boxes转换成连续的tensor
    auto boxesContiguous = l0op::Contiguous(boxes, uniqueExecutor.get());
    CHECK_RET(boxesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将self转为私有格式NC1HWC0
    auto selfTransData = l0op::TransDataSpecial(selfContiguous, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
    CHECK_RET(selfTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 基于aligned判断roiEndMode取值
    int64_t roiEndMode = aligned ? DIM_TWO : DIM_ZERO; // roiEndMode = 2 for torch aligned = true

    // 进行计算
    auto roiAlignOut = l0op::ROIAlignV2(selfTransData, boxesContiguous, spatialScale, pooledHeight, pooledWidth,
                                        samplingRatio, "avg", roiEndMode, uniqueExecutor.get());
    CHECK_RET(roiAlignOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将roiAlignOut的私有格式数据转为NCHW
    auto outTransData = l0op::TransDataSpecial(roiAlignOut, out->GetOriginalFormat(), 0, uniqueExecutor.get());
    CHECK_RET(outTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(outTransData, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiAlignV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoiAlignV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
