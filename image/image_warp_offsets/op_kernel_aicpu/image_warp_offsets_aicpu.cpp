/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "image_warp_offsets_aicpu.h"

#include <atomic>
#include <cmath>

#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char* const kImageWarpOffsets = "IMGWarpOffsets";
constexpr size_t kInputShapeRank = 4;
constexpr size_t kOutputShapeRank = 5;
const char* const kInputStr = "input";
const char* const kOutputStr = "output";
constexpr int64_t kPointsNum = 4;
constexpr int64_t kImageChannels = 3;
} // namespace

namespace aicpu {
const std::map<std::string, ImageWarpOffsetsCpuKernel::KernelFunction> ImageWarpOffsetsCpuKernel::kernels_ = {
    {"(DT_UINT8,DT_FLOAT,DT_UINT8)", &ImageWarpOffsetsCpuKernel::DoCompute<uint8_t, float>},
    {"(DT_FLOAT16,DT_FLOAT,DT_FLOAT16)", &ImageWarpOffsetsCpuKernel::DoCompute<Eigen::half, float>},
    {"(DT_FLOAT16,DT_INT32,DT_FLOAT16)", &ImageWarpOffsetsCpuKernel::DoCompute<Eigen::half, int32_t>},
    {"(DT_FLOAT,DT_FLOAT,DT_FLOAT)", &ImageWarpOffsetsCpuKernel::DoCompute<float, float>}};

const std::vector<std::string> ImageWarpOffsetsCpuKernel::kernels_name_ = {
    "(DT_UINT8,DT_FLOAT,DT_UINT8)", "(DT_FLOAT16,DT_FLOAT,DT_FLOAT16)", "(DT_FLOAT16,DT_INT32,DT_FLOAT16)",
    "(DT_FLOAT,DT_FLOAT,DT_FLOAT)"};

template <typename TImage, typename TIndex>
uint32_t ImageWarpOffsetsCpuKernel::DoCompute(const CpuKernelContext& ctx)
{
    auto input0_shape = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes();
    auto input1_shape = ctx.Input(kSecondInputIndex)->GetTensorShape()->GetDimSizes();
    auto output_shape = ctx.Output(kFirstOutputIndex)->GetTensorShape()->GetDimSizes();
    TImage* input0_data = reinterpret_cast<TImage*>(ctx.Input(kFirstInputIndex)->GetData());
    TIndex* input1_data = reinterpret_cast<TIndex*>(ctx.Input(kSecondInputIndex)->GetData());
    Eigen::TensorMap<Eigen::Tensor<TIndex, kInputShapeRank, Eigen::RowMajor>> input1(
        input1_data, input1_shape[0], input1_shape[1], input1_shape[2], input1_shape[3]);
    TImage* output_data = reinterpret_cast<TImage*>(ctx.Output(kFirstOutputIndex)->GetData());
    // will not overflow, shape already checking in CheckParam
    int64_t out_wxc = output_shape[3] * output_shape[4];
    int64_t out_hxwxc = output_shape[2] * out_wxc;
    int64_t in_hxwxc = input0_shape[1] * input0_shape[2] * input0_shape[3];
    // work_ret is shared by every ParallelFor worker, so it must be atomic.
    std::atomic<uint32_t> work_ret(KERNEL_STATUS_OK);
    auto work = [&output_shape, &out_hxwxc, &in_hxwxc, &work_ret, &input0_shape, &input1, input0_data, output_data,
                 input1_data](int64_t start, int64_t end) {
        int64_t input_offset = 0;
        int64_t output_offset = start * output_shape[3] * output_shape[4];
        for (int64_t i_n = 0; i_n < output_shape[0]; ++i_n) {
            for (int64_t i_i = 0; i_i < output_shape[1]; ++i_i) {
                int64_t out_index = 0;
                for (int64_t i_h = start; i_h < end; ++i_h) {
                    for (int64_t i_w = 0; i_w < output_shape[3]; ++i_w) {
                        const TIndex index_value = input1(i_n, i_i, i_h, i_w);
                        if ((index_value < 0) || (index_value > in_hxwxc - kImageChannels)) {
                            work_ret.store(KERNEL_STATUS_PARAM_INVALID, std::memory_order_relaxed);
                            string err_msg = ConcatString(
                                kImageWarpOffsets, " op input[1] value[", i_n, "][", i_i, "][", i_h, "][", i_w, "]=[",
                                index_value, "] should select 3 values within [0, ", in_hxwxc, "), image size is [",
                                input0_shape[1], " x ", input0_shape[2], " x ", input0_shape[3], "]");
                            KERNEL_LOG_ERROR("%s", err_msg.c_str());
                            return;
                        }
                        int64_t in_index = static_cast<int64_t>(index_value);
                        auto in = input0_data + input_offset + in_index;
                        auto out = output_data + output_offset + out_index;
                        out[0] = in[0];
                        out[1] = in[1];
                        out[2] = in[2];
                        out_index += 3;
                    }
                }
                output_offset += out_hxwxc;
            }
            input_offset += in_hxwxc;
        }
    };

    auto ret = CpuKernelUtils::ParallelFor(ctx, output_shape[2], 1, work);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    if (work_ret.load(std::memory_order_relaxed) != KERNEL_STATUS_OK) {
        return work_ret.load(std::memory_order_relaxed);
    }
    KERNEL_LOG_DEBUG("%s op success.", kImageWarpOffsets);
    return KERNEL_STATUS_OK;
}

uint32_t ImageWarpOffsetsCpuKernel::CheckParam(const CpuKernelContext& ctx, const string& in_or_out, uint32_t index,
                                               size_t rank) const
{
    Tensor* param = nullptr;
    if (in_or_out == kInputStr) {
        param = ctx.Input(index);
    } else if (in_or_out == kOutputStr) {
        param = ctx.Output(index);
    }
    string err_header = ConcatString(kImageWarpOffsets, " op ", in_or_out, "[", index, "]");
    KERNEL_CHECK_NULLPTR(param, KERNEL_STATUS_PARAM_INVALID, "%s tensor is nullptr.", err_header.c_str());

    auto param_shape = param->GetTensorShape();
    KERNEL_CHECK_NULLPTR(param_shape, KERNEL_STATUS_PARAM_INVALID, "%s tensor shape is nullptr.", err_header.c_str());
    auto param_dim_sizes = param_shape->GetDimSizes();
    if (param_dim_sizes.size() != rank) {
        KERNEL_LOG_ERROR("%s shape rank should be [%ld], but got shape[%s].", err_header.c_str(), rank,
                         VectorToString(param_dim_sizes).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (param->GetData() == nullptr) {
        KERNEL_CHECK_NULLPTR(param, KERNEL_STATUS_PARAM_INVALID, "%s tensor data is nullptr.", err_header.c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    DataType param_data_type = param->GetDataType();
    int64_t data_size = param->CalcDataSizeByShape();
    if (data_size < 0) {
        KERNEL_LOG_ERROR("%s shape[%s] or data type[%s] is invalid.", err_header.c_str(),
                         VectorToString(param_dim_sizes).c_str(), DTypeStr(param_data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ImageWarpOffsetsCpuKernel::CheckShapes(const CpuKernelContext& ctx) const
{
    auto input0_shape = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes();
    if (input0_shape.back() != kImageChannels) {
        KERNEL_LOG_ERROR("%s op input[0] shape last dim should be [%ld], but got "
                         "shape[%s].",
                         kImageWarpOffsets, kImageChannels, VectorToString(input0_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto input1_shape = ctx.Input(kSecondInputIndex)->GetTensorShape()->GetDimSizes();
    if (input1_shape[1] != kPointsNum) {
        KERNEL_LOG_ERROR("%s op input[1] shape second dim should be [%ld], but got "
                         "shape[%s].",
                         kImageWarpOffsets, kPointsNum, VectorToString(input1_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto output_shape = ctx.Output(kFirstOutputIndex)->GetTensorShape()->GetDimSizes();

    bool shape_check = (input0_shape[0] != input1_shape[0]) || (input0_shape[0] != output_shape[0]) ||
                       (input0_shape[3] != output_shape[4]) || (input1_shape[1] != output_shape[1]) ||
                       (input1_shape[2] != output_shape[2]) || (input1_shape[3] != output_shape[3]);

    if (shape_check) {
        KERNEL_LOG_ERROR("%s op input[0] shape[%s], input[1] shape[%s], output[0] "
                         "shape[%s] is mismatched.",
                         kImageWarpOffsets, VectorToString(input0_shape).c_str(), VectorToString(input1_shape).c_str(),
                         VectorToString(output_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ImageWarpOffsetsCpuKernel::CheckParams(const CpuKernelContext& ctx) const
{
    auto ret = CheckParam(ctx, kInputStr, kFirstInputIndex, kInputShapeRank);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    ret = CheckParam(ctx, kInputStr, kSecondInputIndex, kInputShapeRank);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    ret = CheckParam(ctx, kOutputStr, kFirstOutputIndex, kOutputShapeRank);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    ret = CheckShapes(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ImageWarpOffsetsCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_DEBUG("%s op start.", kImageWarpOffsets);
    auto input0 = ctx.Input(kFirstInputIndex);
    KERNEL_CHECK_NULLPTR(input0, KERNEL_STATUS_PARAM_INVALID, "%s input[0] tensor is nullptr.", kImageWarpOffsets);
    DataType input0_data_type = input0->GetDataType();
    KERNEL_LOG_DEBUG("%s op input[0] data type is [%s].", kImageWarpOffsets, DTypeStr(input0_data_type).c_str());
    auto input1 = ctx.Input(kSecondInputIndex);
    KERNEL_CHECK_NULLPTR(input1, KERNEL_STATUS_PARAM_INVALID, "%s input[1] tensor is nullptr.", kImageWarpOffsets);
    DataType input1_data_type = input1->GetDataType();
    KERNEL_LOG_DEBUG("%s op input[1] data type is [%s].", kImageWarpOffsets, DTypeStr(input1_data_type).c_str());
    auto output = ctx.Output(kFirstOutputIndex);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "%s output[0] tensor is nullptr.", kImageWarpOffsets);
    DataType output_data_type = output->GetDataType();
    KERNEL_LOG_DEBUG("%s op output[0] data type is [%s].", kImageWarpOffsets, DTypeStr(output_data_type).c_str());
    string kernel_name = ConcatString("(", DTypeStr(input0_data_type), ",", DTypeStr(input1_data_type), ",",
                                      DTypeStr(output_data_type), ")");
    auto it = kernels_.find(kernel_name);
    if (it != kernels_.end()) {
        auto ret = CheckParams(ctx);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
        auto kernel = it->second;
        ret = kernel(ctx);
        KERNEL_LOG_DEBUG("%s op end.", kImageWarpOffsets);
        return ret;
    }

    KERNEL_LOG_ERROR("%s op only support data type [%s], but got [%s].", kImageWarpOffsets,
                     VectorToString(kernels_name_).c_str(), kernel_name.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
}

REGISTER_CPU_KERNEL(kImageWarpOffsets, ImageWarpOffsetsCpuKernel);
} // namespace aicpu
