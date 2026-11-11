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

#include <limits>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

TEST(Lut3DTiling, RejectsNegativeKnownImageDimension)
{
    struct Lut3DCompileInfo {
    } compileInfo;
    gert::TilingContextPara context("LUT3D",
                                    {
                                        {{{2, -4, 3}, {2, -4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{8, 8, 8, 3}, {8, 8, 8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, -4, 3}, {2, -4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    &compileInfo, "Ascend950", 64, 262144, 4096);

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DTiling, RejectsUnknownDimensionAtTiling)
{
    struct Lut3DCompileInfo {
    } compileInfo;
    gert::TilingContextPara context("LUT3D",
                                    {
                                        {{{2, -1, 3}, {2, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{8, 8, 8, 3}, {8, 8, 8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{2, -1, 3}, {2, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    &compileInfo, "Ascend950", 64, 262144, 4096);

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DTiling, RejectsOverflowingImageShapeSize)
{
    struct Lut3DCompileInfo {
    } compileInfo;
    // 各维均为合法非负值、末维 ==3，但维度乘积溢出 int64，GetShapeSize() 返回 INT64_MIN。
    const int64_t kHugeDim = std::numeric_limits<int64_t>::max();
    gert::TilingContextPara context("LUT3D",
                                    {
                                        {{{kHugeDim, 2, 3}, {kHugeDim, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{8, 8, 8, 3}, {8, 8, 8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{kHugeDim, 2, 3}, {kHugeDim, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    &compileInfo, "Ascend950", 64, 262144, 4096);

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
