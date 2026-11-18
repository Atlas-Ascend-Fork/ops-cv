/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_UTILS_SAMPLING_KERNELS_H
#define AICPU_UTILS_SAMPLING_KERNELS_H

#include <cmath>
#include <cstdio>
#include <string>
#include "cpu_context.h"
#include "utils/kernel_util.h"

namespace aicpu {
// Defines functions for different types of sampling kernels.
enum SamplingKernelType {
    // Lanczos kernel with radius 1.  Aliases but does not ring.
    LANCZOS1_KERNEL,

    /**
     * Lanczos kernel with radius 3.  High-quality practical filter but may have
     * some ringing especially on synthetic images.
     */
    LANCZOS3_KERNEL,

    /**
     * Lanczos kernel with radius 5.  Very-high-quality filter but may have
     * stronger ringing.
     */
    LANCZOS5_KERNEL,

    // Gaussian kernel with radius 3, sigma = 1.5 / 3.  Less commonly used.
    GAUSSIAN_KERNEL,

    /**
     * Rectangle function.  Equivalent to "nearest" sampling when upscaling.
     * Has value 1 in interval (-0.5, 0.5), value 0.5 on edge, and 0 elsewhere.
     */
    BOX_KERNEL,

    /**
     * Hat/tent function with radius 1.  Equivalent to "bilinear" reconstruction
     * when upsampling.
     * Has value zero at -1.0 and 1.0.
     */
    TRIANGLE_KERNEL,

    /**
     * Cubic interpolant of Keys.  Equivalent to Catmull-Rom kernel.  Reasonably
     * good quality and faster than LANCZOS3_KERNEL.
     */
    KEYS_CUBIC_KERNEL,

    /**
     * Cubic non-interpolating scheme.  For synthetic images (especially those
     * lacking proper prefiltering), less ringing than Keys cubic kernel but less
     * sharp.
     */
    MITCHELL_CUBIC_KERNEL,

    // Always insert new kernel types before this.
    SAMPLING_KERNEL_TYPE_END
};

/**
 * Converts a string into the corresponding kernel type.
 * Returns SAMPLING_KERNEL_TYPE_END if the string couldn't be converted.
 */
SamplingKernelType SamplingKernelTypeFromString(const std::string& str);

// A function object for a Lanczos kernel.
struct LanczosKernelFunc {
    // Pass 1 for Lanczos1 kernel, 3 for Lanczos3 etc.
    explicit LanczosKernelFunc(float _radius) : radius(_radius) {}
    float operator()(float x) const
    {
        constexpr float kPI = 3.14159265359f;
        x = std::abs(x);
        if (x > radius) {
            return 0.0;
        }
        // Need to special case the limit case of sin(x) / x when x is zero.
        if (x <= 1e-3) {
            return 1.0;
        }
        return radius * std::sin(kPI * x) * std::sin(kPI * x / radius) / (kPI * kPI * x * x);
    }
    float Radius() const { return radius; }
    const float radius;
};

struct GaussianKernelFunc {
    static constexpr float kRadiusMultiplier = 3.0f;
    /**
     * https://en.wikipedia.org/wiki/Gaussian_function
     * We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
     * for Common Resampling Tasks" for kernels with a support of 3 pixels:
     * www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
     * This implies a radius of 1.5,
     */
    explicit GaussianKernelFunc(float _radius = 1.5f) : radius(_radius), sigma(_radius / kRadiusMultiplier) {}
    float operator()(float x) const
    {
        x = std::abs(x);
        if (x >= radius) {
            return 0.0;
        }
        return static_cast<float>(std::exp(-x * x / (2.0 * sigma * sigma)));
    }
    float Radius() const { return radius; }
    const float radius;
    // Gaussian standard deviation
    const float sigma;
};

struct BoxKernelFunc {
    float operator()(float x) const
    {
        x = std::abs(x);
        constexpr float pointFive = 0.5f;
        constexpr float onePoint = 1.0f;
        return x < pointFive ? onePoint : x == pointFive ? pointFive : 0.0f;
    }
    float Radius() const { return 1.f; }
};

struct TriangleKernelFunc {
    // https://en.wikipedia.org/wiki/Triangle_function
    float operator()(float x) const
    {
        x = std::abs(x);
        return x < 1.0f ? 1.0f - x : 0.0f;
    }
    float Radius() const { return 1.f; }
};

struct KeysCubicKernelFunc {
    /**
     * http://ieeexplore.ieee.org/document/1163711/
     * R. G. Keys. Cubic convolution interpolation for digital image
     * processing. IEEE Transactions on Acoustics, Speech, and Signal
     * Processing, 29(6):1153-1160, 1981.
     */
    float operator()(float i) const
    {
        i = std::abs(i);
        if (i >= 2.0f) {
            return 0.0f;
        } else if (i >= 1.0f) {
            return ((-0.5f * i + 2.5f) * i - 4.0f) * i + 2.0f;
        } else {
            return ((1.5f * i - 2.5f) * i) * i + 1.0f;
        }
    }
    float Radius() const { return 2.f; }
};

struct MitchellCubicKernelFunc {
    /**
     * https://doi.org/10.1145/378456.378514
     * D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
     * graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
     * 22(4):221-228, 1988.
     */
    float operator()(float i) const
    {
        i = std::abs(i);
        if (i >= 2.0f) {
            return 0.0f;
        } else if (i >= 1.0f) {
            return (((-7.0f / 18.0f) * i + 2.0f) * i - 10.0f / 3.0f) * i + 16.0f / 9.0f;
        } else {
            return (((7.0f / 6.0f) * i - 2.0f) * i) * i + 8.0f / 9.0f;
        }
    }
    float Radius() const { return 2.f; }
};

inline LanczosKernelFunc CreateLanczos1Kernel()
{
    float i = 1.0;
    return LanczosKernelFunc(i);
}

inline LanczosKernelFunc CreateLanczos3Kernel()
{
    float i = 3.0;
    return LanczosKernelFunc(i);
}

inline LanczosKernelFunc CreateLanczos5Kernel()
{
    float i = 5.0;
    return LanczosKernelFunc(i);
}

inline GaussianKernelFunc CreateGaussianKernel()
{
    float i = 1.5;
    return GaussianKernelFunc(i);
}

inline BoxKernelFunc CreateBoxKernel()
{
    BoxKernelFunc retfunc = BoxKernelFunc();
    return retfunc;
}

inline TriangleKernelFunc CreateTriangleKernel()
{
    TriangleKernelFunc retfunc = TriangleKernelFunc();
    return retfunc;
}

inline KeysCubicKernelFunc CreateKeysCubicKernel()
{
    KeysCubicKernelFunc retfunc = KeysCubicKernelFunc();
    return retfunc;
}

inline MitchellCubicKernelFunc CreateMitchellCubicKernel()
{
    MitchellCubicKernelFunc retfunc = MitchellCubicKernelFunc();
    return retfunc;
}

} // namespace aicpu

#endif // AICPU_UTILS_SAMPLING_KERNELS_H
