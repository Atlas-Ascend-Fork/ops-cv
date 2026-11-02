/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "infershape_case_executor.h"

namespace {
gert::InfershapeContextPara MakeCase(int64_t height, int64_t width)
{
    return gert::InfershapeContextPara(
        "FloodFill", {{{{height, width}, {height, width}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}, {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}});
}
} // namespace

TEST(FloodFillInferShape, KnownDimensionsReceiveMaskBorder)
{
    auto context = MakeCase(128, 256);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{128, 256}, {130, 258}});
}

TEST(FloodFillInferShape, UnknownHeightIsPreserved)
{
    auto context = MakeCase(-1, 128);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 128}, {-1, 130}});
}

TEST(FloodFillInferShape, UnknownWidthIsPreserved)
{
    auto context = MakeCase(128, -1);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{128, -1}, {130, -1}});
}

TEST(FloodFillInferShape, BothUnknownDimensionsArePreserved)
{
    auto context = MakeCase(-1, -1);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, -1}, {-1, -1}});
}
