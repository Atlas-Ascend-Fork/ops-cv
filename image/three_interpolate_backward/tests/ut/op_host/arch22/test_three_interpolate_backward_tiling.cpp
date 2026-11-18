/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/three_interpolate_backward_tiling.h"

namespace {
constexpr int64_t kC0 = 16;

optiling::ThreeInterpolateBackwardCompileInfo g_compileInfo{48, 262144};

gert::TilingContextPara MakeContext(int64_t bs, int64_t c1, int64_t ms, int64_t ns)
{
    gert::StorageShape gradX({bs, c1, ns, 1, kC0}, {bs, c1, ns, 1, kC0});
    gert::StorageShape idx({bs, ns, 3}, {bs, ns, 3});
    gert::StorageShape weight({bs, ns, 3}, {bs, ns, 3});
    gert::StorageShape gradY({bs, c1, ms, 1, kC0}, {bs, c1, ms, 1, kC0});
    return gert::TilingContextPara("ThreeInterpolateBackward",
                                   {
                                       {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
                                       {idx, ge::DT_INT32, ge::FORMAT_ND},
                                       {weight, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
                                   },
                                   {
                                       gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(ms)),
                                   },
                                   &g_compileInfo, "Ascend910B", 48, 262144);
}
} // namespace

TEST(ThreeInterpolateBackwardTiling910B, accepts_positive_m)
{
    auto context = MakeContext(2, 1, 6, 4);
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_zero_m)
{
    auto context = MakeContext(1, 1, 0, 1);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_negative_m)
{
    auto context = MakeContext(1, 1, -1, 1);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_unsupported_idx_dtype)
{
    gert::StorageShape gradX({2, 1, 4, 1, kC0}, {2, 1, 4, 1, kC0});
    gert::StorageShape idx({2, 4, 3}, {2, 4, 3});
    gert::StorageShape weight({2, 4, 3}, {2, 4, 3});
    gert::StorageShape gradY({2, 1, 6, 1, kC0}, {2, 1, 6, 1, kC0});
    auto context = gert::TilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
            {idx, ge::DT_FLOAT, ge::FORMAT_ND},
            {weight, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
        },
        &g_compileInfo, "Ascend910B", 48, 262144);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_ub_insufficient)
{
    optiling::ThreeInterpolateBackwardCompileInfo tinyUb{48, 64};
    gert::StorageShape gradX({2, 1, 4, 1, kC0}, {2, 1, 4, 1, kC0});
    gert::StorageShape idx({2, 4, 3}, {2, 4, 3});
    gert::StorageShape weight({2, 4, 3}, {2, 4, 3});
    gert::StorageShape gradY({2, 1, 6, 1, kC0}, {2, 1, 6, 1, kC0});
    auto context = gert::TilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
            {idx, ge::DT_INT32, ge::FORMAT_ND},
            {weight, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
        },
        &tinyUb, "Ascend910B", 48, 64);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(ThreeInterpolateBackwardTiling910B, rejects_idx_n_mismatch)
{
    gert::StorageShape gradX({2, 1, 4, 1, kC0}, {2, 1, 4, 1, kC0});
    gert::StorageShape idx({2, 5, 3}, {2, 5, 3});
    gert::StorageShape weight({2, 5, 3}, {2, 5, 3});
    gert::StorageShape gradY({2, 1, 6, 1, kC0}, {2, 1, 6, 1, kC0});
    auto context = gert::TilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradX, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
            {idx, ge::DT_INT32, ge::FORMAT_ND},
            {weight, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {gradY, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(6)),
        },
        &g_compileInfo, "Ascend910B", 48, 262144);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}
