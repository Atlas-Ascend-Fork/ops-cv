/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "crop_and_resize_aicpu.h"
#include "aicpu/cv_aicpu_register.h"

namespace {
const char* const kCropAndResize = "CropAndResize";
const std::string kMethodBiliner = "bilinear";
const std::string kMethodBilinerV2 = "bilinear_v2";
const std::string kMethodNearest = "nearest";

const int kInputIndexX = 0;
const int kInputIndexBoxes = 1;
const int kInputIndexBoxIndex = 2;
const int kInputIndexCropSize = 3;
constexpr size_t kBoxesShapeSize = 2;
constexpr size_t kXShapeSize = 4;
} // namespace

namespace aicpu {
uint32_t CropAndResizeMsCpuKernel::GetMethodAndAttr(const CpuKernelContext& ctx)
{
    AttrValue* method = ctx.GetAttr("method");
    KERNEL_CHECK_NULLPTR(method, KERNEL_STATUS_PARAM_INVALID, "Get attr:[method] failed.");

    method_ = method->GetString();
    KERNEL_LOG_INFO("CropAndResize method: [%s]", method_.c_str());
    KERNEL_CHECK_FALSE(((method_ == kMethodBiliner) || (method_ == kMethodBilinerV2) || (method_ == kMethodNearest)),
                       KERNEL_STATUS_PARAM_INVALID, "Invalid attr[method]: [%s], must be in [%s, %s, %s]",
                       method_.c_str(), kMethodBiliner.c_str(), kMethodBilinerV2.c_str(), kMethodNearest.c_str());

    AttrValue* extrapolationValue = ctx.GetAttr("extrapolation_value");
    KERNEL_CHECK_NULLPTR(extrapolationValue, KERNEL_STATUS_PARAM_INVALID, "Get attr:[extrapolation_value] failed.");
    extrapolation_value_ = extrapolationValue->GetFloat();
    return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeMsCpuKernel::GetInputIndexX(const CpuKernelContext& ctx)
{
    // input_0: x
    Tensor* xTensor = ctx.Input(kInputIndexX);
    KERNEL_CHECK_NULLPTR(xTensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[0] failed");
    x_dtype_ = static_cast<DataType>(xTensor->GetDataType());
    std::shared_ptr<TensorShape> x_shape = xTensor->GetTensorShape();
    x_shape_ = x_shape->GetDimSizes();
    KERNEL_CHECK_FALSE((x_shape_.size() == kXShapeSize), KERNEL_STATUS_PARAM_INVALID,
                       "The shape size of input[0]:[%zu], should be [4]", x_shape_.size());

    auto image_height = x_shape_[1];
    auto image_width = x_shape_[2];
    KERNEL_CHECK_FALSE(
        (image_height > 0 && image_width > 0), KERNEL_STATUS_PARAM_INVALID,
        "The value of image_height(shape[1] of input[0]): [%ld] and image_width(shape[2] of input[0]): [%ld] "
        "should be greater than 0",
        image_height, image_width);

    inputs_.push_back(xTensor);
    return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeMsCpuKernel::GetInputBox(const CpuKernelContext& ctx)
{
    // input_1: boxes
    Tensor* boxesTensor = ctx.Input(kInputIndexBoxes);
    KERNEL_CHECK_NULLPTR(boxesTensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[1] failed");
    std::shared_ptr<TensorShape> boxes_shape = boxesTensor->GetTensorShape();
    boxes_shape_ = boxes_shape->GetDimSizes();
    static uint32_t shape_num_4 = 4;
    KERNEL_CHECK_FALSE(boxes_shape_.size() == kBoxesShapeSize, KERNEL_STATUS_PARAM_INVALID,
                       "Invalid boxes shape size: [%zu], should be [2]", boxes_shape_.size());
    KERNEL_CHECK_FALSE(boxes_shape_[1] == shape_num_4, KERNEL_STATUS_PARAM_INVALID,
                       "The boxes_shape dim[1]: [%ld] not equal to [4]", boxes_shape_[1]);

    // input_2: box_index
    Tensor* boxIndexTensor = ctx.Input(kInputIndexBoxIndex);
    KERNEL_CHECK_NULLPTR(boxIndexTensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[2] failed");
    std::shared_ptr<TensorShape> box_index_shape = boxIndexTensor->GetTensorShape();
    box_index_shape_ = box_index_shape->GetDimSizes();
    KERNEL_CHECK_FALSE(
        boxes_shape_[0] == box_index_shape_[0], KERNEL_STATUS_PARAM_INVALID,
        "Inconsistent num_boxes, boxes_shape_[0] (shape[0] of input[1]): [%ld], box_index_shape_[0] (shape[0] of "
        "input[2]): [%ld]",
        boxes_shape_[0], box_index_shape_[0]);

    inputs_.push_back(boxesTensor);
    inputs_.push_back(boxIndexTensor);
    return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeMsCpuKernel::GetInputCropSize(const CpuKernelContext& ctx)
{
    // input_3: crop_size
    Tensor* cropSizeTensor = ctx.Input(kInputIndexCropSize);
    KERNEL_CHECK_NULLPTR(cropSizeTensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[3] failed");
    KERNEL_CHECK_FALSE(cropSizeTensor->GetDataType() == DT_INT32, KERNEL_STATUS_PARAM_INVALID,
                       "Invalid crop_size dtype: [%d], should be [DT_INT32]",
                       static_cast<int>(cropSizeTensor->GetDataType()));
    std::shared_ptr<TensorShape> crop_size_shape = cropSizeTensor->GetTensorShape();
    crop_size_shape_ = crop_size_shape->GetDimSizes();
    static uint32_t shape_num_2 = 2;

    KERNEL_CHECK_FALSE(crop_size_shape_.size() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "Invalid crop_size_shape size (dim of input[3]): [%zu]", crop_size_shape_.size());
    KERNEL_CHECK_FALSE(crop_size_shape_[0] == shape_num_2, KERNEL_STATUS_PARAM_INVALID,
                       "Invalid crop_size_shape[0] (shape[0] of input[3]): [%ld]", crop_size_shape_[0]);

    inputs_.push_back(cropSizeTensor);
    return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeMsCpuKernel::GetInputAndCheck(CpuKernelContext& ctx)
{
    uint32_t ret = GetMethodAndAttr(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    ret = GetInputIndexX(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    ret = GetInputBox(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    ret = GetInputCropSize(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    // get output Tensors
    const uint32_t kNumOutput = 1;
    for (uint32_t i = 0; i < kNumOutput; ++i) {
        Tensor* tensor = ctx.Output(i);
        KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID, "Get output tensor[%d] failed", i)
        outputs_.push_back(tensor);
    }
    return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeMsCpuKernel::Compute(CpuKernelContext& ctx)
{
    uint32_t res = GetInputAndCheck(ctx);
    KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");

    std::map<int, std::function<uint32_t(std::vector<Tensor*>&, std::vector<Tensor*>&,
                                         const std::vector<int64_t>& x_shape, const std::vector<int64_t>& boxes_shape,
                                         std::string& method, float extrapolation_value, CpuKernelContext& ctx)>>
        calls;

    calls[DT_INT8] = CalCropAndResize<int8_t>;
    calls[DT_INT16] = CalCropAndResize<int16_t>;
    calls[DT_INT32] = CalCropAndResize<int32_t>;
    calls[DT_INT64] = CalCropAndResize<int64_t>;
    calls[DT_FLOAT16] = CalCropAndResize<Eigen::half>;
    calls[DT_FLOAT] = CalCropAndResize<float>;
    calls[DT_DOUBLE] = CalCropAndResize<double>;
    calls[DT_UINT8] = CalCropAndResize<uint8_t>;
    calls[DT_UINT16] = CalCropAndResize<uint16_t>;

    auto call = calls.find(x_dtype_);
    if (call == calls.end()) {
        KERNEL_LOG_ERROR("Unsupported input data type: [%d]", static_cast<int>(x_dtype_));
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return call->second(inputs_, outputs_, x_shape_, boxes_shape_, method_, extrapolation_value_, ctx);
}

OPS_CV_REGISTER_CPU_KERNELV2(kCropAndResize, CropAndResizeMsCpuKernel);
} //  namespace aicpu
