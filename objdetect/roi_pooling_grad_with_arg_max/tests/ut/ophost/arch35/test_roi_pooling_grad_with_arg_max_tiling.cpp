/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_pooling_grad_with_arg_max_tiling.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/roi_pooling_grad_with_arg_max_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class RoiPoolingGradWithArgMaxTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RoiPoolingGradWithArgMaxTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RoiPoolingGradWithArgMaxTiling TearDown" << std::endl; }
};

struct RoiPoolingGradWithArgMaxCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

static gert::TilingContextPara MakeTilingContext(int64_t pooledH, int64_t pooledW,
                                                 RoiPoolingGradWithArgMaxCompileInfo* compileInfo)
{
    constexpr int64_t n = 1;
    constexpr int64_t c = 32;
    constexpr int64_t inputHeight = 3;
    constexpr int64_t inputWidth = 3;
    constexpr int64_t roisN = 4;
    constexpr int64_t inputPooledH = 2;
    constexpr int64_t inputPooledW = 2;

    gert::StorageShape gradShape = {{roisN, c, inputPooledH, inputPooledW}, {roisN, c, inputPooledH, inputPooledW}};
    gert::StorageShape xShape = {{n, c, inputHeight, inputWidth}, {n, c, inputHeight, inputWidth}};
    gert::StorageShape roisShape = {{roisN, 5}, {roisN, 5}};
    gert::StorageShape roiActualNumShape = {{roisN, 5}, {roisN, 5}};
    gert::StorageShape argMaxShape = {{roisN, c, inputPooledH, inputPooledW}, {roisN, c, inputPooledH, inputPooledW}};

    return gert::TilingContextPara(
        "RoiPoolingGradWithArgMax",
        {
            {gradShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {roisShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {roiActualNumShape, ge::DT_INT32, ge::FORMAT_ND},
            {argMaxShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooledH)),
         gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(pooledW)),
         gert::TilingContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(c))},
        compileInfo);
}

TEST_F(RoiPoolingGradWithArgMaxTiling, roi_pooling_grad_with_arg_max_tiling_test_float32_case1)
{
    int n = 1;
    int c = 32;
    int poolh = 2;
    int poolw = 2;
    int height = 3;
    int width = 3;
    int rois_n = 4;

    gert::StorageShape gradShape = {{rois_n, c, poolh, poolw}, {rois_n, c, poolh, poolw}};
    gert::StorageShape xShape = {{n, c, height, width}, {n, c, height, width}};
    gert::StorageShape roisShape = {{rois_n, 5}, {rois_n, 5}};
    gert::StorageShape roiActualNumShape = {{rois_n, 5}, {rois_n, 5}};
    gert::StorageShape argMaxShape = {{rois_n, c, poolh, poolw}, {rois_n, c, poolh, poolw}};
    RoiPoolingGradWithArgMaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara(
        "RoiPoolingGradWithArgMax",
        {
            {gradShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {roisShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {roiActualNumShape, ge::DT_INT32, ge::FORMAT_ND},
            {argMaxShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(poolh)),
         gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(poolw)),
         gert::TilingContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(c))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "58 5 3 512 288 2 2 3 3 32 64 1 ";
    std::vector<size_t> expectWorkspaces = {4294968447};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RoiPoolingGradWithArgMaxTiling, roi_pooling_grad_with_arg_max_tiling_rejects_zero_batch)
{
    constexpr int64_t c = 32;
    constexpr int64_t height = 3;
    constexpr int64_t width = 3;
    constexpr int64_t roisN = 4;
    constexpr int64_t poolh = 2;
    constexpr int64_t poolw = 2;
    gert::StorageShape gradShape = {{roisN, c, poolh, poolw}, {roisN, c, poolh, poolw}};
    gert::StorageShape xShape = {{0, c, height, width}, {0, c, height, width}};
    gert::StorageShape roisShape = {{roisN, 5}, {roisN, 5}};
    gert::StorageShape roiActualNumShape = {{roisN, 5}, {roisN, 5}};
    gert::StorageShape argMaxShape = {{roisN, c, poolh, poolw}, {roisN, c, poolh, poolw}};
    RoiPoolingGradWithArgMaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara(
        "RoiPoolingGradWithArgMax",
        {
            {gradShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {roisShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {roiActualNumShape, ge::DT_INT32, ge::FORMAT_ND},
            {argMaxShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {xShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(poolh)),
         gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(poolw)),
         gert::TilingContextPara::OpAttr("spatial_scale_h", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("spatial_scale_w", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("pool_channel", Ops::Cv::AnyValue::CreateFrom<int64_t>(c))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiPoolingGradWithArgMaxTiling, roi_pooling_grad_with_arg_max_tiling_rejects_zero_pooled_h)
{
    RoiPoolingGradWithArgMaxCompileInfo compileInfo = {40, 196608};
    auto tilingContextPara = MakeTilingContext(0, 2, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiPoolingGradWithArgMaxTiling, roi_pooling_grad_with_arg_max_tiling_rejects_zero_pooled_w)
{
    RoiPoolingGradWithArgMaxCompileInfo compileInfo = {40, 196608};
    auto tilingContextPara = MakeTilingContext(2, 0, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiPoolingGradWithArgMaxTiling, roi_pooling_grad_with_arg_max_tiling_rejects_negative_pooled_h)
{
    RoiPoolingGradWithArgMaxCompileInfo compileInfo = {40, 196608};
    auto tilingContextPara = MakeTilingContext(-1, 2, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiPoolingGradWithArgMaxTiling, roi_pooling_grad_with_arg_max_tiling_rejects_negative_pooled_w)
{
    RoiPoolingGradWithArgMaxCompileInfo compileInfo = {40, 196608};
    auto tilingContextPara = MakeTilingContext(2, -1, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
