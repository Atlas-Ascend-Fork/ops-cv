/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_roi_align.h"
#include "roi_align.h"
#include <cmath>
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
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> INT_DTYPE_SUPPORT_LIST = {DataType::DT_INT32};

static bool CheckNotNull(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                         const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(rois, return false);
    OP_CHECK_NULL(batchIndices, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                            const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batchIndices, INT_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_MATCH(self, rois->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static bool CheckFormatValid(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                             const aclTensor* out)
{
    if (self->GetStorageFormat() != op::Format::FORMAT_NCHW || rois->GetStorageFormat() != op::Format::FORMAT_ND ||
        batchIndices->GetStorageFormat() != op::Format::FORMAT_ND ||
        out->GetStorageFormat() != op::Format::FORMAT_NCHW) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format error. self and out only support NCHW, rois and batchIndices only support ND.");
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                       const aclTensor* out, int outputHeight, int outputWidth)
{
    OP_CHECK_WRONG_DIMENSION(self, 4, return false);
    OP_CHECK_WRONG_DIMENSION(rois, 2, return false);
    OP_CHECK_WRONG_DIMENSION(batchIndices, 1, return false);
    OP_CHECK_WRONG_DIMENSION(out, 4, return false);

    auto const& selfShape = self->GetViewShape();
    auto const& roisShape = rois->GetViewShape();
    auto const& batchIndicesShape = batchIndices->GetViewShape();
    auto const& outShape = out->GetViewShape();
    if (roisShape.GetDim(0) != batchIndicesShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rois shape dim0 [%ld] and batchIndices shape dim0 [%ld] should be equal",
                roisShape.GetDim(0), batchIndicesShape.GetDim(0));
        return false;
    }
    if (roisShape.GetDim(1) != 4) { // 4: rois dim 1
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rois shape dim1 [%ld] should be 4", roisShape.GetDim(1));
        return false;
    }
    if (outShape.GetDim(0) != roisShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim0 [%ld] and rois shape dim0 [%ld] should be equal",
                outShape.GetDim(0), roisShape.GetDim(0));
        return false;
    }
    if (outShape.GetDim(1) != selfShape.GetDim(1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim1 [%ld] and self shape dim1 [%ld] should be equal",
                outShape.GetDim(1), selfShape.GetDim(1));
        return false;
    }
    if (outShape.GetDim(2) != outputHeight) { // 2: dim2
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim2 [%ld] and outputHeight [%d] should be equal",
                outShape.GetDim(2), outputHeight); // 2: dim2
        return false;
    }
    if (outShape.GetDim(3) != outputWidth) { // 3: dim3
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape dim3 [%ld] and outputWidth [%d] should be equal",
                outShape.GetDim(3), outputWidth); // 3: dim3
        return false;
    }

    return true;
}

static bool CheckAttr(const char* mode, int samplingRatio, float spatialScale)
{
    if (mode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode is null, please check input arguments");
        return false;
    }
    if (strcmp(mode, "avg") != 0 && strcmp(mode, "max") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode should be [avg] or [max], but got [%s]", mode);
        return false;
    }
    if (samplingRatio < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "samplingRatio [%d] should be greater than or equal to 0", samplingRatio);
        return false;
    }
    if (!std::isfinite(spatialScale) || spatialScale <= 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "spatialScale [%f] should be finite and greater than 0", spatialScale);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                               const aclTensor* out, const char* mode, int outputHeight, int outputWidth,
                               int samplingRatio, float spatialScale)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, rois, batchIndices, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, rois, batchIndices, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入、输出的数据格式是否支持
    CHECK_RET(CheckFormatValid(self, rois, batchIndices, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入tensor的shape
    CHECK_RET(CheckShape(self, rois, batchIndices, out, outputHeight, outputWidth), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查属性
    CHECK_RET(CheckAttr(mode, samplingRatio, spatialScale), ACLNN_ERR_PARAM_INVALID);

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

aclnnStatus aclnnRoiAlignGetWorkspaceSize(const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
                                          const char* mode, int outputHeight, int outputWidth, int samplingRatio,
                                          float spatialScale, aclTensor* out, uint64_t* workspaceSize,
                                          aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRoiAlign,
                   DFX_IN(self, rois, batchIndices, mode, outputHeight, outputWidth, samplingRatio, spatialScale),
                   DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, rois, batchIndices, out, mode, outputHeight, outputWidth, samplingRatio, spatialScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // roialign算子的空tensor在kernel中支持
    if (self->IsEmpty() || rois->IsEmpty()) {
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

    // 固定写法，将输入rois转换成连续的tensor
    auto roisContiguous = l0op::Contiguous(rois, uniqueExecutor.get());
    CHECK_RET(roisContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入batchIndices转换成连续的tensor
    auto batchIndicesContiguous = l0op::Contiguous(batchIndices, uniqueExecutor.get());
    CHECK_RET(batchIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将batchIndices补充成2维[num_rois, 1]
    int64_t batchIndicesNewShape[2] = {batchIndices->GetViewShape().GetDim(0), 1};        // 2: dim num
    auto batchIndicesSize = uniqueExecutor.get()->AllocIntArray(batchIndicesNewShape, 2); // 2: dim num
    auto batchIndicesReshape = l0op::Reshape(batchIndicesContiguous, batchIndicesSize, uniqueExecutor.get());
    CHECK_RET(batchIndicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将batchIndices转为rois的数据类型
    auto batchIndicesCast = l0op::Cast(batchIndicesReshape, rois->GetDataType(), uniqueExecutor.get());
    CHECK_RET(batchIndicesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将batchIndices与rois拼接成新rois，shape为[num_rois, 5]
    op::FVector<const aclTensor*> tensorListTemp;
    tensorListTemp.emplace_back(batchIndicesCast);
    tensorListTemp.emplace_back(roisContiguous);
    auto tensorList = uniqueExecutor.get()->AllocTensorList(tensorListTemp.data(), tensorListTemp.size());
    auto roisConcat = l0op::ConcatD(tensorList, 1, rois->GetDataType(), uniqueExecutor.get());
    CHECK_RET(roisConcat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Ascend950 SIMT kernel直接处理ND/NCHW格式，无需TransData到NC1HWC0
    const aclTensor* selfForCompute = nullptr;
    const aclTensor* roiAlignOut = nullptr;
    if (IsRegBase()) {
        // Ascend950: 直接使用连续tensor，无需格式转换
        selfForCompute = selfContiguous;

        // 进行计算
        roiAlignOut = l0op::ROIAlign(selfForCompute, roisConcat, batchIndicesContiguous, spatialScale, outputHeight,
                                     outputWidth, samplingRatio, mode, uniqueExecutor.get());
        CHECK_RET(roiAlignOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(roiAlignOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 老平台: 转为NC1HWC0格式
        auto selfTransData = l0op::TransDataSpecial(selfContiguous, op::Format::FORMAT_NC1HWC0, 0,
                                                    uniqueExecutor.get());
        CHECK_RET(selfTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);
        selfForCompute = selfTransData;

        // 进行计算
        roiAlignOut = l0op::ROIAlign(selfForCompute, roisConcat, batchIndicesContiguous, spatialScale, outputHeight,
                                     outputWidth, samplingRatio, mode, uniqueExecutor.get());
        CHECK_RET(roiAlignOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将roiAlignOut的私有格式数据转为NCHW
        auto outTransData = l0op::TransDataSpecial(roiAlignOut, out->GetOriginalFormat(), 0, uniqueExecutor.get());
        CHECK_RET(outTransData != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(outTransData, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiAlign(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoiAlign);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
