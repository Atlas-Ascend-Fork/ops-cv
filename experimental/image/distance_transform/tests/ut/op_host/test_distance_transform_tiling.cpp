/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr int64_t MAX_IMAGE_WIDTH = 1LL << 30;
struct DistanceTransformCompileInfo {};
DistanceTransformCompileInfo compileInfo;

gert::TilingContextPara MakeContext(int64_t height, int64_t width, const std::string& mode)
{
    const gert::StorageShape shape = {{height, width}, {height, width}};
    return gert::TilingContextPara("DistanceTransform", {{shape, ge::DT_UINT8, ge::FORMAT_ND}},
                                   {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {
                                       {"distance_type", Ops::Cv::AnyValue::CreateFrom<std::string>(mode)},
                                       {"mask_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)},
                                   },
                                   &compileInfo);
}
} // namespace

TEST(DistanceTransformTiling, RejectsWidthAboveLimitWithoutAllocatingImage)
{
    // The faker supplies shape metadata only; no multi-gigabyte input is allocated.
    for (const int64_t width : {MAX_IMAGE_WIDTH + 1, int64_t{1431655765}}) {
        for (const std::string mode : {"L1", "L2", "CHESSBOARD"}) {
            SCOPED_TRACE("width=" + std::to_string(width) + " mode=" + mode);
            ExecuteTestCase(MakeContext(3, width, mode), ge::GRAPH_FAILED);
        }
    }
}

TEST(DistanceTransformTiling, AcceptsWidthAtLimitWithoutAllocatingImage)
{
    for (const int64_t width : {MAX_IMAGE_WIDTH - 1, MAX_IMAGE_WIDTH}) {
        SCOPED_TRACE("width=" + std::to_string(width));
        TilingInfo info;
        EXPECT_TRUE(ExecuteTiling(MakeContext(1, width, "L2"), info));
    }
}

TEST(DistanceTransformTiling, AcceptsRegularNonAlignedShape)
{
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(MakeContext(37, 53, "L2"), info));
}
