/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scale_and_translate_aicpu.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "sampling_kernels.h"
#include "log.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
constexpr int64_t kParallelDataNums = 1024;
const char* kScaleAndTranslate = "ScaleAndTranslate";

#define SCALEANDTRANSLATE_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
    case (DTYPE): {                                                           \
        uint32_t result = ScaleAndTranslateCompute<TYPE>(CTX);                \
        if (result != KERNEL_STATUS_OK) {                                     \
            KERNEL_LOG_ERROR("ScaleAndTranslate kernel computation failed."); \
            return result;                                                    \
        }                                                                     \
        break;                                                                \
    }

#define SWITCH_PARALLEL(SHARD, end_num, ctx)                                     \
    if ((end_num) <= kParallelDataNums) {                                        \
        for (size_t i = 0; i < size_t(end_num); i++) {                           \
            SHARD(i, i + 1);                                                     \
        }                                                                        \
    } else {                                                                     \
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, 1, SHARD), \
                            "ScaleAndTranslate #SHARD computation failed.")      \
    }

} // namespace

namespace aicpu {
uint32_t ScaleAndTranslateCpuKernel::Compute(CpuKernelContext& ctx)
{
    // check params
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                        "ScaleAndTranslate check input and output number failed.");
    KERNEL_HANDLE_ERROR(ScaleAndTranslateCheck(ctx), "ScaleAndTranslate check params failed.");
    auto data_type = ctx.Input(0)->GetDataType();
    switch (data_type) {
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_FLOAT, float, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        default:
            KERNEL_LOG_ERROR("ScaleAndTranslate kernel data type [%s] is not supported.", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ScaleAndTranslateCpuKernel::ScaleAndTranslateCheck(CpuKernelContext& ctx)
{
    auto input0_shape = ctx.Input(0)->GetTensorShape();
    auto input1_shape = ctx.Input(1)->GetTensorShape();
    auto input2_shape = ctx.Input(2)->GetTensorShape();
    auto input3_shape = ctx.Input(3)->GetTensorShape();
    // dims check
    KERNEL_CHECK_FALSE((input0_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID,
                       "The input0's dims=[%d] must be 4-dimensional", input0_shape->GetDims())
    KERNEL_CHECK_FALSE((input1_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                       "The input1's dims=[%d] must be 1-dimensional", input1_shape->GetDims())
    KERNEL_CHECK_FALSE((input1_shape->NumElements() == 2), KERNEL_STATUS_PARAM_INVALID,
                       "The number of elements of input1 must be 2, but got %ld", input1_shape->NumElements())
    KERNEL_CHECK_FALSE((input2_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                       "The input2's dims=[%d] must be 1-dimensional", input2_shape->GetDims())
    KERNEL_CHECK_FALSE((input2_shape->NumElements() == 2), KERNEL_STATUS_PARAM_INVALID,
                       "The number of elements of input2 must be 2, but got %ld", input2_shape->NumElements())
    KERNEL_CHECK_FALSE((input3_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                       "The input3's dims=[%d] must be 1-dimensional", input3_shape->GetDims())
    KERNEL_CHECK_FALSE((input3_shape->NumElements() == 2), KERNEL_STATUS_PARAM_INVALID,
                       "The number of elements of input3 must be 2, but got %ld", input3_shape->NumElements())

    DataType input1_type = ctx.Input(1)->GetDataType();
    DataType input2_type = ctx.Input(2)->GetDataType();
    DataType input3_type = ctx.Input(3)->GetDataType();

    // dtypes check
    KERNEL_CHECK_FALSE((input1_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "The input1's dtype=[%d] must be DT_INT32", DTypeStr(input1_type).c_str())
    KERNEL_CHECK_FALSE((input2_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                       "The input2's dtype=[%d] must be DT_FLOAT", DTypeStr(input2_type).c_str())
    KERNEL_CHECK_FALSE((input3_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                       "The input3's dtype=[%d] must be DT_FLOAT", DTypeStr(input3_type).c_str())

    KERNEL_LOG_INFO("ScaleAndTranslateCpuKernel[%s], input0: size[%llu], input1: size[%llu];"
                    "input2: size[%llu], input3: size[%llu], output: size[%llu].",
                    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize(),
                    ctx.Input(2)->GetDataSize(), ctx.Input(3)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_OK;
}

template <typename T>
inline const T& Clamp(const T& lower, const T& higher, const T& value)
{
    if (higher < value) {
        return higher;
    }
    if (value < lower) {
        return lower;
    }
    return value;
}

static void NormalizeSpanWeights(const std::vector<float>& temp_weights, float total_weight, int span_size, int x,
                                 Eigen::TensorMap<Eigen::Tensor<float, 1>>& weights_vec)
{
    if (std::abs(total_weight) >= 1000.0f * std::numeric_limits<float>::min()) {
        float one_over_total_weight = 1.0f / total_weight;
        int out_index = span_size * x;
        for (float weight : temp_weights) {
            weights_vec(out_index) = weight * one_over_total_weight;
            ++out_index;
        }
    }
}

template <typename Kernel>
uint32_t InitSpans(const Kernel& kernel, int64_t output_size, int64_t input_size, bool antialias, float inv_scale,
                   Spans* spans, float& kernel_scale)
{
    kernel_scale = antialias ? std::max(inv_scale, 1.0f) : 1.0f;
    const int64_t max_span_size = std::min<int64_t>(input_size, std::numeric_limits<int>::max());
    const float capped_radius = std::min(kernel.Radius() * kernel_scale, static_cast<float>(max_span_size));
    const int64_t radius_span_size = static_cast<int64_t>(std::ceil(capped_radius));
    spans->span_size = static_cast<int>(std::min(2 * radius_span_size + 1, max_span_size));

    spans->starts = new (std::nothrow) Eigen::Tensor<int32_t, 1>(output_size);
    KERNEL_CHECK_NULLPTR(spans->starts, KERNEL_STATUS_PARAM_INVALID, "New spans starts failed.")
    spans->weights = new (std::nothrow) Eigen::Tensor<float, 1>(spans->span_size * output_size);
    if (spans->weights == nullptr) {
        delete spans->starts;
        spans->starts = nullptr;
        KERNEL_LOG_ERROR("New spans weights failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

template <typename Kernel>
uint32_t ComputeSpansCore(CpuKernelContext& context, const Kernel& kernel, const int64_t output_size,
                          const int64_t input_size, const float scale, const float translate, const bool antialias,
                          Spans* spans)
{
    const float inv_scale = 1.0 / scale;
    const float inv_translate = -inv_scale * translate;
    float kernel_scale = 0.0f;
    KERNEL_HANDLE_ERROR(InitSpans(kernel, output_size, input_size, antialias, inv_scale, spans, kernel_scale),
                        "InitSpans failed.");
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
    Eigen::TensorMap<Eigen::Tensor<float, 1>> weights_vec(spans->weights->data(), spans->weights->dimensions());
    weights_vec.setZero();
    const float one_over_kernel_scale = 1.0f / kernel_scale;
    int max_span_size = 0;
    std::vector<float> temp_weights;
    uint32_t shard_ret = KERNEL_STATUS_OK;
    auto shard_x = [&](int start, int end) {
        for (auto x = start; x < end; ++x) {
            const float col_f = x + 0.5f;
            const float sample_f = col_f * inv_scale + inv_translate;
            // Don't sample when the sampling location is outside the source image.
            if (sample_f < 0 || sample_f > input_size) {
                // Add an empty span.
                starts_vec(x) = 0;
                continue;
            }
            const float support = kernel.Radius() * kernel_scale;
            const double max_span_index = static_cast<double>(
                std::min<int64_t>(input_size - 1, std::numeric_limits<int>::max() - 1));
            const double span_start_f = static_cast<double>(std::ceil(sample_f - support - 0.5f));
            const double span_end_f = static_cast<double>(std::floor(sample_f + support - 0.5f));
            int64_t span_start = static_cast<int64_t>(Clamp(0.0, max_span_index, span_start_f));
            int64_t span_end = static_cast<int64_t>(Clamp(0.0, max_span_index, span_end_f)) + 1;
            const int this_span_size = span_end - span_start;
            if (this_span_size > spans->span_size) {
                KERNEL_LOG_ERROR("Span is too large: [%d] vs [%d].", this_span_size, spans->span_size);
                shard_ret = KERNEL_STATUS_PARAM_INVALID;
                return;
            }
            float total_weight = 0.0f;
            temp_weights.clear();
            for (int source = span_start; source < span_end; ++source) {
                float kernel_pos = static_cast<float>(source) + 0.5f - sample_f;
                float weight = kernel(std::abs(kernel_pos * one_over_kernel_scale));
                total_weight += weight;
                temp_weights.push_back(weight);
            }
            max_span_size = std::max(max_span_size, this_span_size);
            NormalizeSpanWeights(temp_weights, total_weight, spans->span_size, x, weights_vec);
            starts_vec(x) = span_start;
        }
    };
    if (output_size <= kParallelDataNums) {
        for (size_t i = 0; i < size_t(output_size); i++) {
            shard_x(i, i + 1);
        }
    } else {
        uint32_t parallelRet = CpuKernelUtils::ParallelFor(context, output_size, 1, shard_x);
        if (parallelRet != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("ScaleAndTranslate shard_x computation failed.");
            delete spans->starts;
            delete spans->weights;
            spans->starts = nullptr;
            spans->weights = nullptr;
            return parallelRet;
        }
    }
    if (shard_ret != KERNEL_STATUS_OK) {
        delete spans->starts;
        delete spans->weights;
        spans->starts = nullptr;
        spans->weights = nullptr;
        return shard_ret;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ComputeSpans(CpuKernelContext& context, const SamplingKernelType kernel_type, const int64_t output_size,
                      const int64_t input_size, const float scale, const float translate, const bool antialias,
                      Spans* spans)
{
    switch (kernel_type) {
        case LANCZOS1_KERNEL: {
            return ComputeSpansCore(context, CreateLanczos1Kernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        case LANCZOS3_KERNEL: {
            return ComputeSpansCore(context, CreateLanczos3Kernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        case LANCZOS5_KERNEL: {
            return ComputeSpansCore(context, CreateLanczos5Kernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        case GAUSSIAN_KERNEL: {
            return ComputeSpansCore(context, CreateGaussianKernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        case BOX_KERNEL: {
            return ComputeSpansCore(context, CreateBoxKernel(), output_size, input_size, scale, translate, antialias,
                                    spans);
        }
        case TRIANGLE_KERNEL: {
            return ComputeSpansCore(context, CreateTriangleKernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        case KEYS_CUBIC_KERNEL: {
            return ComputeSpansCore(context, CreateKeysCubicKernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        case MITCHELL_CUBIC_KERNEL: {
            return ComputeSpansCore(context, CreateMitchellCubicKernel(), output_size, input_size, scale, translate,
                                    antialias, spans);
        }
        default:
            KERNEL_LOG_ERROR("kernel_type kernel data type [%u] is not supported.", kernel_type);
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

struct ScaleAndTranslateParams {
    SamplingKernelType kernel_type;
    bool antialias;
    int64_t batch_size;
    int64_t input_height;
    int64_t input_width;
    int64_t channels;
    int64_t output_height;
    int64_t output_width;
    float row_scale;
    float col_scale;
    float row_translation;
    float col_translation;
};

static uint32_t ParseScaleAndTranslateParams(CpuKernelContext& ctx, ScaleAndTranslateParams& p)
{
    auto input_size = reinterpret_cast<int32_t*>(ctx.Input(1)->GetData());
    auto input_scale = reinterpret_cast<float*>(ctx.Input(2)->GetData());
    auto input_translation = reinterpret_cast<float*>(ctx.Input(3)->GetData());
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("kernel_type"), KERNEL_STATUS_PARAM_INVALID, "Get attr [kernel_type] failed.");
    std::string kernel_type_str = ctx.GetAttr("kernel_type")->GetString();
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("antialias"), KERNEL_STATUS_PARAM_INVALID, "Get attr [antialias] failed.");
    p.antialias = ctx.GetAttr("antialias")->GetBool();
    p.kernel_type = SamplingKernelTypeFromString(kernel_type_str);

    auto input0_shape = ctx.Input(0)->GetTensorShape();

    p.output_height = input_size[0];
    p.output_width = input_size[1];

    p.batch_size = input0_shape->GetDimSize(0);
    p.input_height = input0_shape->GetDimSize(1);
    p.input_width = input0_shape->GetDimSize(2);
    p.channels = input0_shape->GetDimSize(3);

    KERNEL_CHECK_FALSE((p.output_height > 0 && p.output_width > 0), KERNEL_STATUS_PARAM_INVALID,
                       "output_height = [%d] and output_width = [%d] must be positive", p.output_height, p.output_width)
    KERNEL_CHECK_FALSE((p.channels > 0), KERNEL_STATUS_PARAM_INVALID, "image_channel = [%d] must be at least 1",
                       p.channels)
    KERNEL_CHECK_FALSE((p.input_height > 0 && p.input_width > 0), KERNEL_STATUS_PARAM_INVALID,
                       "input_height = [%d] and input_width = [%d] must be of non-zero size", p.input_height,
                       p.input_width)

    p.row_scale = input_scale[0];
    p.col_scale = input_scale[1];

    KERNEL_CHECK_FALSE((p.row_scale > 0 && p.col_scale > 0), KERNEL_STATUS_PARAM_INVALID,
                       "row_scale = [%d] and col_scale = [%d] must be greater than zero.", p.row_scale, p.col_scale)
    KERNEL_CHECK_FALSE((std::isfinite(1.0f / p.row_scale) && std::isfinite(1.0f / p.col_scale)),
                       KERNEL_STATUS_PARAM_INVALID, "The reciprocal of scale must be finite.")

    p.row_translation = input_translation[0];
    p.col_translation = input_translation[1];
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ScaleAndTranslateCpuKernel::ScaleAndTranslateCompute(CpuKernelContext& ctx)
{
    ScaleAndTranslateParams p;
    KERNEL_HANDLE_ERROR(ParseScaleAndTranslateParams(ctx, p), "ScaleAndTranslate parse params failed.");

    Tensor* input = ctx.Input(0);
    Tensor* output = ctx.Output(0);

    EigenTensor inputTensor(input, input->GetData());
    EigenTensor outputTensor(output, output->GetData());

    typename TTypes<T, 4>::Tensor image_data(inputTensor.tensor<T, 4>());

    typename TTypes<float, 4>::Tensor output_data(outputTensor.tensor<float, 4>());

    Spans col_spans;
    col_spans.starts = nullptr;
    col_spans.weights = nullptr;
    uint32_t col_ret = ComputeSpans(ctx, p.kernel_type, p.output_width, p.input_width, p.col_scale, p.col_translation,
                                    p.antialias, &col_spans);
    if (col_ret != KERNEL_STATUS_OK) {
        return col_ret;
    }

    Spans row_spans;
    row_spans.starts = nullptr;
    row_spans.weights = nullptr;
    uint32_t row_ret = ComputeSpans(ctx, p.kernel_type, p.output_height, p.input_height, p.row_scale, p.row_translation,
                                    p.antialias, &row_spans);
    if (row_ret != KERNEL_STATUS_OK) {
        delete col_spans.starts;
        delete col_spans.weights;
        return row_ret;
    }

    Eigen::Tensor<float, 4> intermediate_tensor_middle(p.batch_size, p.output_height, p.input_width, p.channels);
    Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_data(intermediate_tensor_middle.data(),
                                                                intermediate_tensor_middle.dimensions());
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
    Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
    Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());

    uint32_t gatherRet = GatherSpans<T>()(ctx, row_spans.span_size, row_starts, row_weights, col_spans.span_size,
                                          col_starts, col_weights, image_data, intermediate_data, output_data);

    delete col_spans.starts;
    delete col_spans.weights;
    delete row_spans.starts;
    delete row_spans.weights;

    if (gatherRet != KERNEL_STATUS_OK) {
        return gatherRet;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
inline void GatherColumnPixel(const T* inputRowStart, const int32_t* starts, const float* weights, int x, int spanSize,
                              int64_t inputWidth, int channels, float* outPixel)
{
    const T* srcPixel = inputRowStart + starts[x] * channels;
    const float* weightBase = weights + x * spanSize;
    int effectiveLen = std::min(starts[x] + spanSize, static_cast<int>(inputWidth)) - starts[x];
    const float* const weightLimit = weightBase + effectiveLen;

    std::fill_n(outPixel, channels, 0.0f);
    for (const float* wIter = weightBase; wIter != weightLimit; ++wIter) {
        float coeff = *wIter;
        for (int c = 0; c < channels; ++c) {
            outPixel[c] += coeff * static_cast<float>(srcPixel[c]);
        }
        srcPixel += channels;
    }
}

template <typename T>
uint32_t GatherColumns(CpuKernelContext& context, int spanSize, const int32_t* starts, const float* weights,
                       const T* srcData, const int64_t srcHeight, const int64_t srcWidth, const int64_t dstHeight,
                       const int64_t dstWidth, const int channelCnt, float* dstData)
{
    const int64_t srcRowStride = srcWidth * channelCnt;
    const int64_t dstRowStride = dstWidth * channelCnt;
    auto shardColumn = [&](int start, int end) {
        for (int y = start; y < end; ++y) {
            const T* rowBase = srcData + srcRowStride * y;
            float* dstPixel = dstData + dstRowStride * y;
            for (int x = 0; x < dstWidth; ++x, dstPixel += channelCnt) {
                GatherColumnPixel(rowBase, starts, weights, x, spanSize, srcWidth, channelCnt, dstPixel);
            }
        }
    };
    if (dstHeight <= kParallelDataNums) {
        for (size_t i = 0; i < size_t(dstHeight); i++) {
            shardColumn(i, i + 1);
        }
    } else {
        uint32_t ret = CpuKernelUtils::ParallelFor(context, dstHeight, 1, shardColumn);
        if (ret != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("ScaleAndTranslate shardColumn computation failed.");
            return ret;
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
inline void AddScaledVector(const T* srcVec, int vecLen, float coeff, float* dstVec)
{
    float* dstEnd = dstVec + vecLen;
    for (; dstVec != dstEnd; ++dstVec, ++srcVec) {
        *dstVec += coeff * static_cast<float>(*srcVec);
    }
}

template <typename T>
uint32_t GatherRows(CpuKernelContext& context, int spanSize, const int32_t* starts, const float* weights,
                    const T* srcData, const int64_t srcHeight, const int64_t srcWidth, const int64_t dstHeight,
                    const int64_t dstWidth, const int channelCnt, float* dstData)
{
    const int64_t srcRowStride = srcWidth * channelCnt;
    const int64_t dstRowStride = dstWidth * channelCnt;
    auto shardRows = [&](int start, int end) {
        for (int y = start; y < end; ++y) {
            float* dstRow = dstData + dstRowStride * y;
            std::fill(dstRow, dstRow + dstRowStride, 0.0f);
            int srcRowIdx = starts[y];
            const T* srcRow = srcData + srcRowStride * srcRowIdx;
            const float* weightBase = weights + y * spanSize;
            int effectiveLen = std::min(starts[y] + spanSize, static_cast<int>(srcHeight)) - starts[y];
            const float* const weightLimit = weightBase + effectiveLen;

            for (const float* wIter = weightBase; wIter != weightLimit; ++wIter) {
                AddScaledVector(srcRow, srcRowStride, *wIter, dstRow);
                srcRow += srcRowStride;
            }
        }
    };
    if (dstHeight <= kParallelDataNums) {
        for (size_t i = 0; i < size_t(dstHeight); i++) {
            shardRows(i, i + 1);
        }
    } else {
        uint32_t ret = CpuKernelUtils::ParallelFor(context, dstHeight, 1, shardRows);
        if (ret != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("ScaleAndTranslate shardRows computation failed.");
            return ret;
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GatherSpans<T>::operator()(aicpu::CpuKernelContext& context, int row_span_size,
                                    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts,
                                    Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights, int col_span_size,
                                    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts,
                                    Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights,
                                    typename TTypes<T, 4>::Tensor images,
                                    Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_buffer,
                                    typename TTypes<float, 4>::Tensor resized_images)
{
    const int batchCnt = images.dimension(0);
    const int64_t srcHeight = images.dimension(1);
    const int64_t srcWidth = images.dimension(2);
    const int channelCnt = images.dimension(3);

    const int64_t dstHeight = resized_images.dimension(1);
    const int64_t dstWidth = resized_images.dimension(2);

    const int64_t srcBatchOffset = srcWidth * srcHeight * channelCnt;
    const int64_t midBatchOffset = srcWidth * dstHeight * channelCnt;
    const int64_t dstBatchOffset = dstWidth * dstHeight * channelCnt;
    float* midData = intermediate_buffer.data();

    const T* srcData = images.data();
    float* dstData = resized_images.data();

    auto rowStartBase = row_starts.data();
    auto rowWeightBase = row_weights.data();
    for (int batchIdx = 0; batchIdx < batchCnt; ++batchIdx) {
        uint32_t rowRet = GatherRows(context, row_span_size, rowStartBase, rowWeightBase, srcData, srcHeight, srcWidth,
                                     dstHeight, srcWidth, channelCnt, midData);
        if (rowRet != KERNEL_STATUS_OK) {
            return rowRet;
        }
        uint32_t colRet = GatherColumns(context, col_span_size, col_starts.data(), col_weights.data(), midData,
                                        dstHeight, srcWidth, dstHeight, dstWidth, channelCnt, dstData);
        if (colRet != KERNEL_STATUS_OK) {
            return colRet;
        }
        srcData += srcBatchOffset;
        midData += midBatchOffset;
        dstData += dstBatchOffset;
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kScaleAndTranslate, ScaleAndTranslateCpuKernel);
} // namespace aicpu
