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

#include "match_template_host_test_utils.h"

using match_template_test::CheckInfer;

TEST(MatchTemplateInferShape, DerivesAllOutputShapes)
{
    CheckInfer({2, 10, 12, 3}, {1, 4, 5, 3}, ge::GRAPH_SUCCESS, {{2, 7, 8}, {2}, {2, 2}}, ge::DT_FLOAT16,
               ge::DT_FLOAT16);
}

TEST(MatchTemplateInferShape, RejectsNonRankFour) { CheckInfer({1, 8, 8}, {1, 3, 3, 1}, ge::GRAPH_FAILED); }

TEST(MatchTemplateInferShape, RejectsZeroDimension) { CheckInfer({1, 0, 8, 1}, {1, 3, 3, 1}, ge::GRAPH_FAILED); }

TEST(MatchTemplateInferShape, RejectsTemplateBatchNotOne) { CheckInfer({1, 8, 8, 1}, {2, 3, 3, 1}, ge::GRAPH_FAILED); }

TEST(MatchTemplateInferShape, RejectsChannelMismatch) { CheckInfer({1, 8, 8, 3}, {1, 3, 3, 1}, ge::GRAPH_FAILED); }

TEST(MatchTemplateInferShape, RejectsDtypeMismatch)
{
    CheckInfer({1, 8, 8, 1}, {1, 3, 3, 1}, ge::GRAPH_FAILED, {}, ge::DT_FLOAT, ge::DT_FLOAT16);
}

TEST(MatchTemplateInferShape, RejectsUnsupportedDtype)
{
    CheckInfer({1, 8, 8, 1}, {1, 3, 3, 1}, ge::GRAPH_FAILED, {}, ge::DT_INT32, ge::DT_INT32);
}

TEST(MatchTemplateInferShape, RejectsTemplateLargerThanImage)
{
    CheckInfer({1, 4, 8, 1}, {1, 5, 3, 1}, ge::GRAPH_FAILED);
    CheckInfer({1, 8, 4, 1}, {1, 3, 5, 1}, ge::GRAPH_FAILED);
}

TEST(MatchTemplateInferShape, DerivesShapesWithoutInputData)
{
    for (const ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UINT8}) {
        CheckInfer({2, 10, 12, 3}, {1, 4, 5, 3}, ge::GRAPH_SUCCESS, {{2, 7, 8}, {2}, {2, 2}}, dtype, dtype);
    }
}

TEST(MatchTemplateInferShape, PreservesUnknownDimensionsWithoutInputData)
{
    CheckInfer({-1, -1, 12, 3}, {1, 4, 5, 3}, ge::GRAPH_SUCCESS, {{-1, -1, 8}, {-1}, {-1, 2}}, ge::DT_FLOAT16,
               ge::DT_FLOAT16);
}

TEST(MatchTemplateInferShape, ValidatesDescriptorsWithoutInputData)
{
    CheckInfer({1, 8, 8, 1}, {1, 3, 3, 1}, ge::GRAPH_FAILED, {}, ge::DT_FLOAT, ge::DT_FLOAT16);
    CheckInfer({1, 8, 8, 1}, {1, 3, 3, 1}, ge::GRAPH_FAILED, {}, ge::DT_INT32, ge::DT_INT32);
}
