/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEDIAN_FILTER3_D_VALIDATION_H
#define MEDIAN_FILTER3_D_VALIDATION_H
#include <cmath>
#include <cstring>
#include <climits>
namespace MedianFilter3DHost {
// 1024 fp32 planes * 32 lanes occupy 128 KiB; the minimum tile fits both target SoCs.
constexpr uint32_t MAX_WINDOW = 1024;
template <typename Shape>
bool ValidShape(const Shape& s, bool dynamic)
{
    if (s.GetDimNum() < 3 || s.GetDimNum() > 5) {
        return false;
    }
    int64_t product = 1;
    for (size_t i = 0; i < s.GetDimNum(); ++i) {
        int64_t dim = s.GetDim(i);
        if (dynamic && dim == -1) {
            continue;
        }
        if (dim <= 0 || dim > INT32_MAX || product > INT32_MAX / dim) {
            return false;
        }
        product *= dim;
    }
    return true;
}
template <typename Attrs>
bool Parse(const Attrs* attrs, uint32_t* k, uint32_t& mode, float& cval)
{
    if (attrs == nullptr) {
        return false;
    }
    auto* list = attrs->GetListInt(0);
    const char* str = attrs->template GetAttrPointer<char>(1);
    const float* val = attrs->GetFloat(2);
    if (list == nullptr || list->GetSize() != 3 || list->GetData() == nullptr || str == nullptr || val == nullptr) {
        return false;
    }
    uint32_t count = 1;
    for (uint32_t i = 0; i < 3; ++i) {
        auto n = list->GetData()[i];
        if (n <= 0 || n > MAX_WINDOW || count > MAX_WINDOW / n) {
            return false;
        }
        k[i] = static_cast<uint32_t>(n);
        count *= k[i];
    }
    if (std::strcmp(str, "REFLECT") == 0) {
        mode = 0;
    } else if (std::strcmp(str, "CONSTANT") == 0) {
        mode = 1;
    } else {
        return false;
    }
    cval = mode == 0 ? 0.0f : *val;
    return !std::isnan(cval);
}
inline bool Dtype(ge::DataType dt) { return dt == ge::DT_UINT8 || dt == ge::DT_FLOAT || dt == ge::DT_FLOAT16; }
} // namespace MedianFilter3DHost
#endif
