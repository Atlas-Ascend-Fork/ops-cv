/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_pooling_with_arg_max_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/roi_pooling_with_arg_max_tiling_arch35.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace ge;
using namespace std;

class TilingForRoiPoolingWithArgMax : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RoiPoolingWithArgMaxTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RoiPoolingWithArgMaxTiling TearDown" << std::endl; }
};

static gert::TilingContextPara MakeRoiPoolingWithArgMaxPara(int64_t pooledH, int64_t pooledW, float spatialH,
                                                            float spatialW, ge::DataType xDtype, int64_t n, int64_t c,
                                                            int64_t h, int64_t w, int64_t roisN, int64_t roisLast,
                                                            optiling::RoiPoolingWithArgMaxCompileInfo* compileInfo)
{
    gert::StorageShape xShape = {{n, c, h, w}, {n, c, h, w}};
    gert::StorageShape roisShape = {{roisN, roisLast}, {roisN, roisLast}};
    gert::StorageShape yShape = {{roisN, c, pooledH, pooledW}, {roisN, c, pooledH, pooledW}};
    return gert::TilingContextPara(
        "RoiPoolingWithArgMax", {{xShape, xDtype, ge::FORMAT_ND}, {roisShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{yShape, xDtype, ge::FORMAT_ND}, {yShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooledH)),
            gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooledW)),
            gert::TilingContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(spatialH)),
            gert::TilingContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(spatialW)),
            gert::TilingContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(c)),
        },
        compileInfo);
}

TEST_F(TilingForRoiPoolingWithArgMax, roi_pooling_with_arg_max_tiling_case0)
{
    optiling::RoiPoolingWithArgMaxCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "RoiPoolingWithArgMax",
        {{{{2, 16, 25, 42}, {2, 16, 25, 42}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 16, 3, 3}, {2, 16, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 16, 3, 3}, {2, 16, 3, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(16)),
        },
        &compileInfo);
    TilingInfo tilingInfo;
    bool tilingOk = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(tilingOk);
    EXPECT_EQ(tilingInfo.tilingKey, 0u);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_GT(tilingInfo.workspaceSizes[0], 0u);
}

TEST_F(TilingForRoiPoolingWithArgMax, rejects_non_positive_pooled_h)
{
    optiling::RoiPoolingWithArgMaxCompileInfo compileInfo{};
    auto para = MakeRoiPoolingWithArgMaxPara(0, 3, 1.0f, 1.0f, ge::DT_FLOAT, 2, 16, 25, 42, 2, 5, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TilingForRoiPoolingWithArgMax, rejects_non_positive_spatial_scale)
{
    optiling::RoiPoolingWithArgMaxCompileInfo compileInfo{};
    auto para = MakeRoiPoolingWithArgMaxPara(3, 3, 0.0f, 1.0f, ge::DT_FLOAT, 2, 16, 25, 42, 2, 5, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TilingForRoiPoolingWithArgMax, rejects_x_rank_not_4)
{
    optiling::RoiPoolingWithArgMaxCompileInfo compileInfo{};
    // rank-3 x: use h as last spatial dim, w unused in shape construction below
    gert::StorageShape xShape = {{2, 16, 25}, {2, 16, 25}};
    gert::StorageShape roisShape = {{2, 5}, {2, 5}};
    gert::StorageShape yShape = {{2, 16, 3, 3}, {2, 16, 3, 3}};
    gert::TilingContextPara para(
        "RoiPoolingWithArgMax", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {roisShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}, {yShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
            gert::TilingContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
            gert::TilingContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(16)),
        },
        &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TilingForRoiPoolingWithArgMax, rejects_rois_last_dim_not_5)
{
    optiling::RoiPoolingWithArgMaxCompileInfo compileInfo{};
    auto para = MakeRoiPoolingWithArgMaxPara(3, 3, 1.0f, 1.0f, ge::DT_FLOAT, 2, 16, 25, 42, 2, 4, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TilingForRoiPoolingWithArgMax, rejects_unsupported_dtype)
{
    optiling::RoiPoolingWithArgMaxCompileInfo compileInfo{};
    auto para = MakeRoiPoolingWithArgMaxPara(3, 3, 1.0f, 1.0f, ge::DT_INT32, 2, 16, 25, 42, 2, 5, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
