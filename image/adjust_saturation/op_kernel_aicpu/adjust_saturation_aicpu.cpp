/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adjust_saturation_aicpu.cpp
 * \brief AdjustSaturation aicpu kernel implementation
 */

#include "adjust_saturation_aicpu.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
constexpr std::uint32_t kAdjustSaturationInputNum{2u};
constexpr std::uint32_t kAdjustSaturationOutputNum{1u};
constexpr const char* kAdjustSaturation{"AdjustSaturation"};
constexpr std::int64_t kAdjustSaturationParallelNum{64 * 1024};
constexpr std::int64_t kAdjustSaturationEmptyImageCount{0};
constexpr std::uint32_t kAdjustSaturationMinRank{3U};
constexpr std::int64_t kAdjustSaturationRgbChannelCount{3};
constexpr std::int64_t kAdjustSaturationReservedCoreCount{2L};
constexpr std::int64_t kAdjustSaturationMinCoreCount{1L};
constexpr std::uint64_t kAdjustSaturationFactorDataSize{sizeof(std::float_t)};
} // namespace

namespace aicpu {
namespace detail {
template <typename T>
struct Rgb {
    T r;
    T g;
    T b;
} __attribute__((packed));

template <>
struct Rgb<Eigen::half> {
    Eigen::half r;
    Eigen::half g;
    Eigen::half b;
};

template <typename T>
struct Hsv {
    T h;
    T s;
    T v;
} __attribute__((packed));

template <typename T>
inline Hsv<T> RgbToHsv(Rgb<T> in)
{
    T min{in.r < in.g ? in.r : in.g};
    min = min < in.b ? min : in.b;

    T max{in.r > in.g ? in.r : in.g};
    max = max > in.b ? max : in.b;

    T delta{max - min};
    if (delta == static_cast<T>(0.0) || max == static_cast<T>(0.0)) {
        return Hsv<T>{static_cast<T>(0.0), static_cast<T>(0.0), max};
    }
    if (max < static_cast<T>(0.0)) {
        return Hsv<T>{static_cast<T>(0.0), static_cast<T>(NAN), max};
    }

    Hsv<T> out;

    if (in.r >= max) {
        out.h = (in.g - in.b) / delta;
    } else if (in.g >= max) {
        out.h = static_cast<T>(2.0) + (in.b - in.r) / delta;
    } else {
        out.h = static_cast<T>(4.0) + (in.r - in.g) / delta;
    }

    out.h /= static_cast<T>(6.0);

    if (out.h < static_cast<T>(0.0)) {
        out.h += static_cast<T>(1.0);
    }

    out.v = max;
    out.s = (delta / max);

    return out;
}

template <typename T>
inline Rgb<T> Hsv2Rgb(Hsv<T> in)
{
    if (in.s <= static_cast<T>(0.0)) {
        return Rgb<T>{in.v, in.v, in.v};
    }
    T h{in.h};
    if (h >= static_cast<T>(1.0)) {
        h = static_cast<T>(0.0);
    }
    h *= static_cast<T>(6.0);
    auto i{static_cast<long>(h)};
    auto f{static_cast<T>(h - static_cast<T>(i))};
    T p{in.v * (static_cast<T>(1.0) - in.s)};
    T q{in.v * (static_cast<T>(1.0) - (in.s * f))};
    T t{in.v * (static_cast<T>(1.0) - (in.s * (static_cast<T>(1.0) - f)))};

    switch (i) {
        case 0:
            return Rgb<T>{in.v, t, p};
        case 1:
            return Rgb<T>{q, in.v, p};
        case 2:
            return Rgb<T>{p, in.v, t};
        case 3:
            return Rgb<T>{p, q, in.v};
        case 4:
            return Rgb<T>{t, p, in.v};
        default:
            return Rgb<T>{in.v, p, q};
    }
}

template <typename T>
inline Rgb<T> ScalarAdjustSaturation(Rgb<T> image, std::float_t saturationFactor)
{
    auto hsv{RgbToHsv(image)};
    hsv.s *= static_cast<T>(saturationFactor);
    if (hsv.s > static_cast<T>(1.0)) {
        hsv.s = static_cast<T>(1.0);
    }
    return Hsv2Rgb(hsv);
}

template <>
inline Rgb<Eigen::half> ScalarAdjustSaturation(Rgb<Eigen::half> image, std::float_t saturationFactor)
{
    auto hsv{RgbToHsv(Rgb<std::float_t>{static_cast<std::float_t>(image.r), static_cast<std::float_t>(image.g),
                                        static_cast<std::float_t>(image.b)})};
    hsv.s *= static_cast<std::float_t>(saturationFactor);
    if (hsv.s > static_cast<std::float_t>(1.0)) {
        hsv.s = static_cast<std::float_t>(1.0);
    }
    auto out{Hsv2Rgb(hsv)};
    return Rgb<Eigen::half>{static_cast<Eigen::half>(out.r), static_cast<Eigen::half>(out.g),
                            static_cast<Eigen::half>(out.b)};
}

inline std::uint32_t ParallelForAdjustSaturation(const CpuKernelContext& ctx, std::int64_t total,
                                                 std::int64_t perUnitSize,
                                                 const std::function<void(std::int64_t, std::int64_t)>& work)
{
    if (total > kAdjustSaturationParallelNum) {
        return aicpu::CpuKernelUtils::ParallelFor(ctx, total, perUnitSize, work);
    }
    work(0, total);
    return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAdjustSaturationKernel(const CpuKernelContext& ctx)
{
    auto input{static_cast<Rgb<T>*>(ctx.Input(0)->GetData())};
    auto saturationFactor{static_cast<std::float_t*>(ctx.Input(1)->GetData())};
    auto output{static_cast<Rgb<T>*>(ctx.Output(0)->GetData())};
    auto adjustFunc = [&](Rgb<T> image) { return ScalarAdjustSaturation(image, saturationFactor[0]); };
    std::int64_t total{ctx.Input(0)->NumElements() / kAdjustSaturationRgbChannelCount};
    if (total == kAdjustSaturationEmptyImageCount) {
        return KERNEL_STATUS_OK;
    }
    auto cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
    std::int64_t perUnitSize{
        total / std::min(std::max(kAdjustSaturationMinCoreCount, cores - kAdjustSaturationReservedCoreCount), total)};
    return ParallelForAdjustSaturation(ctx, total, perUnitSize, [&](std::int64_t begin, std::int64_t end) {
        std::transform(input + begin, input + end, output + begin, adjustFunc);
    });
}

template <typename T>
inline std::uint32_t ComputeAdjustSaturation(const CpuKernelContext& ctx)
{
    std::uint32_t result{ComputeAdjustSaturationKernel<T>(ctx)};
    if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("AdjustSaturation computation failed.");
    }
    return result;
}

inline std::uint32_t ExtraCheckAdjustSaturation(const CpuKernelContext& ctx)
{
    auto inputShape = ctx.Input(0)->GetTensorShape();
    auto inputDims = inputShape->GetDimSizes();
    if (inputDims.size() < kAdjustSaturationMinRank || inputDims.back() != kAdjustSaturationRgbChannelCount) {
        KERNEL_LOG_ERROR("The last dimension of input images and output y shape [%s] must be %lld.",
                         VectorToString(inputDims).c_str(), kAdjustSaturationRgbChannelCount);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
        KERNEL_LOG_ERROR("The data type of the input [%s] needs to be the same as the output [%s].",
                         DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
        KERNEL_LOG_ERROR("The data size of the input [%llu] needs to be the same as the output [%llu].",
                         ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto outputDims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
    if (inputDims != outputDims) {
        KERNEL_LOG_ERROR("The shape of output y [%s] needs to be the same as input images [%s].",
                         VectorToString(outputDims).c_str(), VectorToString(inputDims).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(1)->GetDataType() != aicpu::DataType::DT_FLOAT) {
        KERNEL_LOG_ERROR("The data type of the input [%s] needs to be [%s].",
                         DTypeStr(ctx.Input(1)->GetDataType()).c_str(), DTypeStr(aicpu::DataType::DT_FLOAT).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto scaleDims = ctx.Input(1)->GetTensorShape()->GetDimSizes();
    if (scaleDims.size() != 1 || scaleDims[0] != 1) {
        KERNEL_LOG_ERROR("The shape of scale [%s] needs to be [1].", VectorToString(scaleDims).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(1)->GetDataSize() != kAdjustSaturationFactorDataSize) {
        KERNEL_LOG_ERROR("The data size of the input [%llu] needs to be [%llu].", ctx.Input(1)->GetDataSize(),
                         kAdjustSaturationFactorDataSize);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAdjustSaturation(CpuKernelContext& ctx, std::uint32_t inputsNum, std::uint32_t outputsNum)
{
    return NormalCheck(ctx, kAdjustSaturationInputNum, kAdjustSaturationOutputNum) ? KERNEL_STATUS_PARAM_INVALID :
                                                                                     ExtraCheckAdjustSaturation(ctx);
}

inline std::uint32_t ComputeAdjustSaturation(const CpuKernelContext& ctx)
{
    DataType inputType{ctx.Input(0)->GetDataType()};
    switch (inputType) {
        case DT_FLOAT16:
            return ComputeAdjustSaturation<Eigen::half>(ctx);
        case DT_FLOAT:
            return ComputeAdjustSaturation<std::float_t>(ctx);
        default:
            KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(inputType).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}
} // namespace detail

std::uint32_t AdjustSaturationCpuKernel::Compute(CpuKernelContext& ctx)
{
    return detail::CheckAdjustSaturation(ctx, kAdjustSaturationInputNum, kAdjustSaturationOutputNum) ?
               KERNEL_STATUS_PARAM_INVALID :
               detail::ComputeAdjustSaturation(ctx);
}

REGISTER_CPU_KERNEL(kAdjustSaturation, AdjustSaturationCpuKernel);
} // namespace aicpu
