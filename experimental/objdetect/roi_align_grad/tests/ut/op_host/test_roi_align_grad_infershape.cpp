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
#include <vector>

#include <gtest/gtest.h>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {

using OpAttr = gert::InfershapeContextPara::OpAttr;
using TensorDesc = gert::InfershapeContextPara::TensorDescription;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::InfershapeContextPara MakeContext(const std::vector<int64_t>& yDiffShape, const std::vector<int64_t>& roisShape,
                                        const std::vector<int64_t>& xdiffShape, ge::Format yDiffFormat = ge::FORMAT_ND)
{
    const std::vector<TensorDesc> inputs = {
        TensorDesc(MakeStorageShape(yDiffShape), ge::DT_FLOAT, yDiffFormat),
        TensorDesc(MakeStorageShape(roisShape), ge::DT_FLOAT, ge::FORMAT_ND),
    };
    const std::vector<TensorDesc> outputs = {
        TensorDesc(MakeStorageShape({}), ge::DT_FLOAT, yDiffFormat),
    };
    const std::vector<OpAttr> attrs = {
        OpAttr("xdiff_shape", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(xdiffShape)),
        OpAttr("pooled_width", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
        OpAttr("pooled_height", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
        OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.25F)),
        OpAttr("sample_num", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
        OpAttr("roi_end_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
    };
    return gert::InfershapeContextPara("ROIAlignGrad", inputs, outputs, attrs);
}

void ExpectFailure(const std::vector<int64_t>& yDiffShape, const std::vector<int64_t>& roisShape,
                   const std::vector<int64_t>& xdiffShape, ge::Format yDiffFormat = ge::FORMAT_ND)
{
    auto context = MakeContext(yDiffShape, roisShape, xdiffShape, yDiffFormat);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

} // namespace

class RoiAlignGradInfershapeTest : public testing::Test {};

TEST_F(RoiAlignGradInfershapeTest, accepts_nd_and_nc1hwc0_contracts)
{
    auto nd = MakeContext({2, 3, 3, 3}, {2, 5}, {1, 3, 6, 6});
    ExecuteTestCase(nd, ge::GRAPH_SUCCESS, {{1, 3, 6, 6}});

    auto nc1hwc0 = MakeContext({2, 2, 3, 3, 16}, {2, 5}, {1, 17, 6, 6}, ge::FORMAT_NC1HWC0);
    ExecuteTestCase(nc1hwc0, ge::GRAPH_SUCCESS, {{1, 2, 6, 6, 16}});
}

TEST_F(RoiAlignGradInfershapeTest, accepts_rois_with_at_least_five_columns)
{
    auto exactlyFiveColumns = MakeContext({2, 3, 3, 3}, {2, 5}, {1, 3, 6, 6});
    ExecuteTestCase(exactlyFiveColumns, ge::GRAPH_SUCCESS, {{1, 3, 6, 6}});

    auto extraRoiFields = MakeContext({2, 3, 3, 3}, {2, 6}, {1, 3, 6, 6});
    ExecuteTestCase(extraRoiFields, ge::GRAPH_SUCCESS, {{1, 3, 6, 6}});
}

TEST_F(RoiAlignGradInfershapeTest, accepts_unknown_dimensions_and_rank)
{
    auto unknownDims = MakeContext({-1, -1, -1, -1}, {-1, -1}, {1, 3, 6, 6});
    ExecuteTestCase(unknownDims, ge::GRAPH_SUCCESS, {{1, 3, 6, 6}});

    auto unknownRank = MakeContext({-2}, {-2}, {1, 3, 6, 6});
    ExecuteTestCase(unknownRank, ge::GRAPH_SUCCESS, {{-2}});
}

TEST_F(RoiAlignGradInfershapeTest, rejects_invalid_rank_and_roi_shape)
{
    ExpectFailure({2, 3, 3}, {2, 5}, {1, 3, 6, 6});
    ExpectFailure({2, 1, 3, 3, 16, 1}, {2, 5}, {1, 17, 6, 6}, ge::FORMAT_NC1HWC0);
    ExpectFailure({2, 3, 3, 3}, {2}, {1, 3, 6, 6});
    ExpectFailure({2, 3, 3, 3}, {2, 4}, {1, 3, 6, 6});
}

TEST_F(RoiAlignGradInfershapeTest, rejects_mismatched_y_diff_contract)
{
    ExpectFailure({2, 3, 3, 3}, {1, 5}, {1, 3, 6, 6});
    ExpectFailure({2, 2, 3, 3}, {2, 5}, {1, 3, 6, 6});
    ExpectFailure({2, 3, 2, 3}, {2, 5}, {1, 3, 6, 6});
    ExpectFailure({2, 1, 3, 3, 8}, {2, 5}, {1, 17, 6, 6}, ge::FORMAT_NC1HWC0);
}
