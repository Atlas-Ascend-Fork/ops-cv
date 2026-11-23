/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <limits>

#include <gtest/gtest.h>

#include "experimental/objdetect/roi_align_v2/op_kernel/roi_align_v2_tiling_data.h"
#include "tiling_case_executor.h"

namespace {

std::vector<gert::TilingContextPara::OpAttr> RoiAlignAttrs(float spatialScale, int64_t samplingRatio)
{
    return {
        gert::TilingContextPara::OpAttr("pooled_height", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
        gert::TilingContextPara::OpAttr("pooled_width", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
        gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(spatialScale)),
        gert::TilingContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(samplingRatio)),
    };
}

gert::TilingContextPara MakeParams(float spatialScale, int64_t samplingRatio)
{
    return gert::TilingContextPara(
        "RoiAlignV2",
        {{{{1, 3, 4, 4}, {1, 3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 3, 2, 2}, {1, 3, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}}, RoiAlignAttrs(spatialScale, samplingRatio));
}

} // namespace

// 对照组：合法属性必须仍然通过，确认本次收紧没有影响正常路径
TEST(RoiAlignV2AttrValidationTiling, valid_control_succeeds)
{
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(1.0F, 1), info);
    std::cout << "case=valid_control tiling_success=" << ok << " expected=true\n";
    EXPECT_TRUE(ok);
}

// spatial_scale = 0：kernel 会把所有 ROI 坐标缩放为 0，ROI 塌缩，输出静默错误，必须拒绝
TEST(RoiAlignV2AttrValidationTiling, spatial_scale_zero_is_rejected)
{
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(0.0F, 1), info);
    std::cout << "case=spatial_scale_zero tiling_success=" << ok << " expected=false\n";
    EXPECT_FALSE(ok);
}

// spatial_scale < 0：缩放后 ROI 坐标为负，采样越界，必须拒绝
TEST(RoiAlignV2AttrValidationTiling, spatial_scale_negative_is_rejected)
{
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(-1.0F, 1), info);
    std::cout << "case=spatial_scale_negative tiling_success=" << ok << " expected=false\n";
    EXPECT_FALSE(ok);
}

// spatial_scale = NaN：非有限值传播到坐标计算，必须拒绝
TEST(RoiAlignV2AttrValidationTiling, spatial_scale_nan_is_rejected)
{
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(std::numeric_limits<float>::quiet_NaN(), 1), info);
    std::cout << "case=spatial_scale_nan tiling_success=" << ok << " expected=false\n";
    EXPECT_FALSE(ok);
}

// sampling_ratio 超出 int32 可表示范围：窄化前回绕成正数，必须在窄化前拒绝
TEST(RoiAlignV2AttrValidationTiling, sampling_ratio_above_int32_max_is_rejected)
{
    const int64_t outOfRange = static_cast<int64_t>(INT32_MAX) + 1;
    TilingInfo info;
    const bool ok = ExecuteTiling(MakeParams(1.0F, outOfRange), info);
    std::cout << "case=sampling_ratio_above_int32_max tiling_success=" << ok << " expected=false\n";
    EXPECT_FALSE(ok);
}
