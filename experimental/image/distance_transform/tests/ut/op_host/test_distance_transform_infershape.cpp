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

#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

TEST(DistanceTransformInfershape, Rank2)
{
    gert::InfershapeContextPara context(
        "DistanceTransform",
        {
            {{{37, 53}, {37, 53}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("distance_type", Ops::Cv::AnyValue::CreateFrom<std::string>("L2")),
            gert::InfershapeContextPara::OpAttr("mask_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(5)),
        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{37, 53}});
}

TEST(DistanceTransformInfershape, Rank3)
{
    gert::InfershapeContextPara context(
        "DistanceTransform",
        {
            {{{4, 37, 53}, {4, 37, 53}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("distance_type", Ops::Cv::AnyValue::CreateFrom<std::string>("L1")),
        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{4, 37, 53}});
}

TEST(DistanceTransformInfershape, Rank4)
{
    gert::InfershapeContextPara context(
        "DistanceTransform",
        {
            {{{4, 1, 37, 53}, {4, 1, 37, 53}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("distance_type",
                                                Ops::Cv::AnyValue::CreateFrom<std::string>("CHESSBOARD")),
        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{4, 1, 37, 53}});
}
