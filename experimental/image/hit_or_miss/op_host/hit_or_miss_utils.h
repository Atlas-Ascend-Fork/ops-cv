/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HIT_OR_MISS_UTILS_H_
#define HIT_OR_MISS_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>

namespace hit_or_miss {
constexpr int32_t MIN_INPUT_RANK = 2;
constexpr int32_t MAX_INPUT_RANK = 8;
constexpr int64_t MAX_KERNEL_SIZE = 63;

struct CanonicalParams {
    int64_t axis0 = 0;
    int64_t axis1 = 0;
    int64_t kernelH = -1;
    int64_t kernelW = -1;
    int64_t anchorH = -1;
    int64_t anchorW = -1;
};

inline bool SafeMultiply(const int64_t lhs, const int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

// 图推导保留未知尺寸；运行时必须取得具体且合法的尺寸。
template <typename Shape>
inline const char* CanonicalizeParams(const Shape& inputShape, const Shape& kernelShape, const int64_t* axes,
                                      const size_t axesCount, const int64_t* anchor, const size_t anchorCount,
                                      const bool allowUnknown, CanonicalParams& params)
{
    if (axes == nullptr || axesCount != 2) {
        return "axes must contain two values";
    }
    if (anchor == nullptr || anchorCount != 2) {
        return "anchor must contain two values";
    }

    const bool inputUnknownRank = inputShape.GetDimNum() == 1 && inputShape.GetDim(0) == -2;
    const bool kernelUnknownRank = kernelShape.GetDimNum() == 1 && kernelShape.GetDim(0) == -2;
    if (!allowUnknown && (inputUnknownRank || kernelUnknownRank)) {
        return "runtime shapes must have known rank";
    }
    if (!inputUnknownRank && (inputShape.GetDimNum() < MIN_INPUT_RANK || inputShape.GetDimNum() > MAX_INPUT_RANK)) {
        return "input rank must be in [2, 8]";
    }
    if (!kernelUnknownRank && kernelShape.GetDimNum() != 2) {
        return "kernel rank must be 2";
    }

    if (!inputUnknownRank) {
        int64_t knownProduct = 1;
        for (size_t index = 0; index < inputShape.GetDimNum(); ++index) {
            const int64_t dim = inputShape.GetDim(index);
            if (allowUnknown && dim == -1) {
                continue;
            }
            if (dim <= 0) {
                return "input dimensions must be positive or unknown during inference";
            }
            if (!SafeMultiply(knownProduct, dim, knownProduct)) {
                return "input element count overflows int64";
            }
        }
    }

    params.kernelH = kernelUnknownRank ? -1 : kernelShape.GetDim(0);
    params.kernelW = kernelUnknownRank ? -1 : kernelShape.GetDim(1);
    const int64_t kernelDims[] = {params.kernelH, params.kernelW};
    for (const int64_t dim : kernelDims) {
        if (allowUnknown && dim == -1) {
            continue;
        }
        if (dim < 1 || dim > MAX_KERNEL_SIZE) {
            return "kernel height and width must be in [1, 63]";
        }
    }

    params.axis0 = axes[0];
    params.axis1 = axes[1];
    if (params.axis0 < -MAX_INPUT_RANK || params.axis0 >= MAX_INPUT_RANK || params.axis1 < -MAX_INPUT_RANK ||
        params.axis1 >= MAX_INPUT_RANK || params.axis0 == params.axis1) {
        return "axes must be distinct and in range";
    }
    if (!inputUnknownRank) {
        const int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
        params.axis0 = params.axis0 < 0 ? params.axis0 + rank : params.axis0;
        params.axis1 = params.axis1 < 0 ? params.axis1 + rank : params.axis1;
        if (params.axis0 < 0 || params.axis0 >= rank || params.axis1 < 0 || params.axis1 >= rank ||
            params.axis0 == params.axis1) {
            return "normalized axes must be distinct and in range";
        }
    }

    params.anchorH = anchor[0];
    params.anchorW = anchor[1];
    int64_t* canonicalAnchors[] = {&params.anchorH, &params.anchorW};
    for (size_t index = 0; index < 2; ++index) {
        int64_t& value = *canonicalAnchors[index];
        if (value < -1 || value >= MAX_KERNEL_SIZE) {
            return "anchor must be -1 or inside the kernel";
        }
        if (kernelDims[index] > 0) {
            if (value == -1) {
                value = kernelDims[index] / 2;
            }
            if (value >= kernelDims[index]) {
                return "anchor must be -1 or inside the kernel";
            }
        }
    }
    return nullptr;
}
} // namespace hit_or_miss

#endif
