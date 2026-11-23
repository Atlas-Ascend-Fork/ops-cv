/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
TEST(Lut3DInferShape, SupportsUnknownInputRanks)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-2}});
}

TEST(Lut3DInferShape, PreservesUnknownImageDimensions)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, -1}, {2, 32, 32, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{-1, 17, 17, -1}, {-1, 17, 17, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 32, 32, -1}});
}

TEST(Lut3DInferShape, RejectsMismatchedImageAndLutDtypes)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, 3}, {2, 32, 32, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                            {{{17, 17, 17, 3}, {17, 17, 17, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DInferShape, PreservesZeroImageDimensionForEmptyTensor)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 0, 3}, {2, 0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{8, 8, 8, 3}, {8, 8, 8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 0, 3}});
}

TEST(Lut3DInferShape, RejectsNegativeKnownImageDimension)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, -4, 3}, {2, -4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{8, 8, 8, 3}, {8, 8, 8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DInferShape, RejectsConflictingKnownLutDimensionsAroundUnknownDimension)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, 3}, {2, 32, 32, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{17, -1, 18, 3}, {17, -1, 18, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DInferShape, RejectsKnownLutDimensionAboveMaximum)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, 3}, {2, 32, 32, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{-1, 21, 21, 3}, {-1, 21, 21, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
TEST(Lut3DInferShape, RejectsInvalidImageRank5)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, 3, 1}, {2, 32, 32, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{2, 32, 32, 3}, {2, 32, 32, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DInferShape, RejectsImageLastDimNot3)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, 4}, {2, 32, 32, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{2, 32, 32, 3}, {2, 32, 32, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(Lut3DInferShape, RejectsLutRank3AndLastDimNot3)
{
    gert::InfershapeContextPara context("LUT3D",
                                        {
                                            {{{2, 32, 32, 3}, {2, 32, 32, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{17, 17, 17}, {17, 17, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });

    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
} // namespace
