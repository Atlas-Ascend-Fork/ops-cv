/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOG_DESCRIPTOR_TEST_ATTRS_H_
#define HOG_DESCRIPTOR_TEST_ATTRS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace {
struct HogTestAttrs {
    std::vector<int64_t> ppc = {8, 8};
    std::vector<int64_t> cpb = {2, 2};
    std::vector<int64_t> stride = {1, 1};
    int64_t bins = 9;
    std::string norm = "L2-Hys";
    float threshold = 0.2f;
    bool gamma = true;
    bool signedGradient = false;
    int64_t aperture = 1;
    float eps = 1e-6f;
};

template <typename Attr>
std::vector<Attr> MakeHogAttrs(const HogTestAttrs& p = {})
{
    return {Attr("pixels_per_cell", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(p.ppc)),
            Attr("cells_per_block", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(p.cpb)),
            Attr("block_stride", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(p.stride)),
            Attr("orientations", Ops::Cv::AnyValue::CreateFrom<int64_t>(p.bins)),
            Attr("block_norm", Ops::Cv::AnyValue::CreateFrom<std::string>(p.norm)),
            Attr("l2_hys_threshold", Ops::Cv::AnyValue::CreateFrom<float>(p.threshold)),
            Attr("gamma_correction", Ops::Cv::AnyValue::CreateFrom<bool>(p.gamma)),
            Attr("signed_gradient", Ops::Cv::AnyValue::CreateFrom<bool>(p.signedGradient)),
            Attr("derivative_aperture", Ops::Cv::AnyValue::CreateFrom<int64_t>(p.aperture)),
            Attr("eps", Ops::Cv::AnyValue::CreateFrom<float>(p.eps))};
}
} // namespace

#endif // HOG_DESCRIPTOR_TEST_ATTRS_H_
