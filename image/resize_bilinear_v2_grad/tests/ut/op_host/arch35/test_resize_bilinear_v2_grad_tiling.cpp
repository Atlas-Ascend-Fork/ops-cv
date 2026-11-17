/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_resize_bilinear_v2_grad_tiling.cpp
 * \brief test_resize_bilinear_v2_grad_tiling
 */

#include "../../../../op_host/arch35/resize_bilinear_v2_grad_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class ResizeBilinearV2GradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeBilinearV2GradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeBilinearV2GradTiling TearDown" << std::endl; }
};

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case01)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 200, 2, 3}, {2, 200, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales",
                                            Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "10000 245760 0 1 2 200 5 7 2 3 2 2 3 200 6 2 2 3 200 6 4720507691772089740 0 64 218 48 "
                              "64 37 32 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case02)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({2.0, 2.0})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10002;
    string expectTilingData = "10002 245760 1 0 2 1 2 2 4 4 2 4 4 1 16 2 4 4 1 16 4515609228859845291 "
                              "4629700418014806016 0 0 0 8 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case03)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData = "10000 245760 0 1 2 1 4 4 2 2 2 2 2 1 4 2 2 2 1 4 4611686019501129728 0 32 1 0 8 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case04)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 1, 4, 4}, {2, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10002;
    string expectTilingData = "10002 245760 0 0 2 1 4 4 2 2 2 2 2 1 4 2 2 2 1 4 4611686019501129728 "
                              "4539628425446424576 0 0 0 32 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case05)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 1, 2, 200}, {2, 1, 2, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 1, 4, 200}, {2, 1, 4, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 1, 4, 200}, {2, 1, 4, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10013;
    string expectTilingData = "10013 245760 0 0 2 200 1 4 1 2 2 1 2 200 2 2 1 2 200 2 4611686019501129728 "
                              "4539628425446424576 0 0 0 64 25 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case06)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales",
                                            Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    string expectTilingData = "20000 245760 0 1 2 200 5 7 2 3 1 2 3 40 1 1 2 3 200 6 4720507691772089740 0 64 218 48 "
                              "60 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case07)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 5, 7, 200}, {2, 5, 7, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales",
                                            Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10013;
    string expectTilingData = "10013 245760 0 0 2 200 5 7 2 3 2 2 3 200 6 2 2 3 200 6 4720507691772089740 "
                              "4430100883889375413 0 0 0 64 218 48 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case08)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 200, 2, 3}, {2, 200, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{2, 200, 5, 7}, {2, 200, 5, 7}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales",
                                            Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.87025, 0.06125})),
        },
        &compileInfo);
    uint64_t expectTilingKey = 10012;
    string expectTilingData = "10012 245760 0 0 2 200 5 7 2 3 2 2 3 200 6 2 2 3 200 6 4720507691772089740 "
                              "4430100883889375413 0 0 0 64 218 48 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case09)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 6, 9, 200}, {2, 6, 9, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 6, 9, 200}, {2, 6, 9, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    string expectTilingData = "30001 245760 1 0 2 200 6 9 2 3 1 1 1 40 6 767 1 1 40 6 4647714816530579456 0 64 337 32 "
                              "60 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case10)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 2, 3, 200}, {2, 2, 3, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData = "30000 245760 1 0 2 200 2 3 2 3 2 2 3 200 6 2 2 3 30688 6 4575657222473777152 0 0 0 0 64 "
                              "37 32 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Regression coverage for the divide-by-zero issue: alignCorners=true with output H==1 && W==1
// must be routed to the PointCopy path (tiling key 30001) and tiling must succeed (no crash / no error).
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_case11_point_copy_align_corners_desHW_1)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            // grads (dest) NHWC: H==1, W==1 -> lenDesH==1 && lenDesW==1
            {{{2, 1, 1, 200}, {2, 1, 1, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            // y (src) NHWC: H==4, W==5
            {{{2, 4, 5, 200}, {2, 4, 5, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{2, 4, 5, 200}, {2, 4, 5, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    // Assert only success + PointCopy tiling key (30001); the exact tilingData bytes are not the
    // point of this regression case, so use ExecuteTiling to avoid hardcoding the full data string.
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, 30001); // TILING_KEY_POINT_COPY
}

// Regression for issue-17: origin format(ND) != storage format(NCHW).
// 图优化后张量原始format被改写为ND, 运行时(storage)format为真实排布NCHW。
// tiling 必须校验运行时(storage)format; 若回退为 GetOriginFormat 将拿到ND导致format校验失败, 本用例即失效。
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_origin_ne_storage_nchw)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({2, 1, 2, 2}, {2, 1, 2, 2}), ge::DT_FLOAT,
                                                       ge::FORMAT_ND, false, nullptr, ge::FORMAT_NCHW),
            gert::TilingContextPara::TensorDescription(gert::StorageShape({2, 1, 4, 4}, {2, 1, 4, 4}), ge::DT_FLOAT,
                                                       ge::FORMAT_ND, false, nullptr, ge::FORMAT_NCHW),
        },
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({2, 1, 4, 4}, {2, 1, 4, 4}), ge::DT_FLOAT,
                                                       ge::FORMAT_ND, false, nullptr, ge::FORMAT_NCHW),
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Regression for issue-17: origin format(ND) != storage format(NHWC). 覆盖 NHWC 运行时format分支。
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_origin_ne_storage_nhwc)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({2, 1, 2, 200}, {2, 1, 2, 200}), ge::DT_FLOAT,
                                                       ge::FORMAT_ND, false, nullptr, ge::FORMAT_NHWC),
            gert::TilingContextPara::TensorDescription(gert::StorageShape({2, 1, 4, 200}, {2, 1, 4, 200}), ge::DT_FLOAT,
                                                       ge::FORMAT_ND, false, nullptr, ge::FORMAT_NHWC),
        },
        {
            gert::TilingContextPara::TensorDescription(gert::StorageShape({2, 1, 4, 200}, {2, 1, 4, 200}), ge::DT_FLOAT,
                                                       ge::FORMAT_ND, false, nullptr, ge::FORMAT_NHWC),
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.5, 0.5})),
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Regression: GE 转排布场景下 origin format/shape(NCHW轴序) 与 storage format/shape(NHWC轴序) 同时不一致。
// GE 图优化为算子选择 NHWC 排布执行时, origin 侧保留构图语义(NCHW 轴序), storage 侧为运行时真实排布
// (NHWC 轴序)。tiling 的 format 与 shape 必须同源(均取 storage): 若 format 读 storage 而 shape 读 origin,
// CheckAxesValid 将按 NHWC 规则解析 NCHW 轴序 shape, lenC/lenSrcH/lenDesW 等维度全部取错轴。
// 场景取自 ResizeBilinearV2Grad_fp16_fuzz_add_0000 用例的 geir 复现参数。
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_transpose_nhwc_shape_sync)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            // grads: origin [12,3,8459,1](NCHW轴序) / storage [12,8459,1,3](NHWC轴序)
            gert::TilingContextPara::TensorDescription(gert::StorageShape({12, 3, 8459, 1}, {12, 8459, 1, 3}),
                                                       ge::DT_FLOAT16, ge::FORMAT_NCHW, false, nullptr,
                                                       ge::FORMAT_NHWC),
            // original_image: origin [12,3,1,1] / storage [12,1,1,3]
            gert::TilingContextPara::TensorDescription(gert::StorageShape({12, 3, 1, 1}, {12, 1, 1, 3}), ge::DT_FLOAT16,
                                                       ge::FORMAT_NCHW, false, nullptr, ge::FORMAT_NHWC),
        },
        {
            // y: origin [12,3,1,1] / storage [12,1,1,3]
            gert::TilingContextPara::TensorDescription(gert::StorageShape({12, 3, 1, 1}, {12, 1, 1, 3}), ge::DT_FLOAT16,
                                                       ge::FORMAT_NCHW, false, nullptr, ge::FORMAT_NHWC),
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr(
                "scales", Ops::Cv::AnyValue::CreateFrom<std::vector<float>>({0.000103788271925272f, 1.0f})),
        },
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 按 storage(NHWC) 轴序解析: lenN=12, lenC=3, lenSrcH=1, lenSrcW=1, lenDesH=8459, lenDesW=1
    ASSERT_EQ(tilingInfo.tilingKey, 10013); // TILING_KEY_SIMT_NHWC_DETERMINE_SCALES
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(int64_t) * 10);
    const int64_t* fields = reinterpret_cast<const int64_t*>(tilingInfo.tilingData.get());
    EXPECT_EQ(fields[0], 10013); // tilingKey
    EXPECT_EQ(fields[4], 12);    // lenN
    EXPECT_EQ(fields[5], 3);     // lenC
    EXPECT_EQ(fields[6], 1);     // lenSrcH
    EXPECT_EQ(fields[7], 1);     // lenSrcW
    EXPECT_EQ(fields[8], 8459);  // lenDesH
    EXPECT_EQ(fields[9], 1);     // lenDesW
}

// 覆盖 PointCopy UB 切分 branch2 (vol4AxisC >= cLenAlign && vol4AxisC < wcLenAlign):
// 半 UB 容纳一个对齐后的完整 C 行, 但容纳不下单个 (n,h) 行的完整 W*C, 需按 W 维切分,
// ubWFactor_ = vol4AxisC / cLenAlign。规格 fp32/NHWC (1,4,10000,200)->(1,2,5000,200),
// 无 scales 属性(2 倍整数降采样): cFactor_=200, cLenAlign=200, wFactor_=157,
// wcLenAlign=CeilAlign(157*200,8)=31400, vol4AxisC=30688, 命中 200 <= 30688 < 31400。
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_regbase_tiling_point_copy_ub_cut_w_branch)
{
    optiling::ResizeBilinearV2GradCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ResizeBilinearV2Grad",
        {
            // grads NHWC: N=1, desH=2, desW=5000, C=200
            {{{1, 2, 5000, 200}, {1, 2, 5000, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            // original_image NHWC: N=1, srcH=4, srcW=10000, C=200
            {{{1, 4, 10000, 200}, {1, 4, 10000, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            // y NHWC: N=1, srcH=4, srcW=10000, C=200
            {{{1, 4, 10000, 200}, {1, 4, 10000, 200}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("half_pixel_centers", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    // branch2: ubN=1, ubH=1, ubW=30688/200=153, ubC=200; scaleH=scaleW=2.0, inverseScale=0.5
    string expectTilingData = "30001 245760 0 0 1 200 4 10000 2 5000 1 1 157 200 10000 1 1 153 200 10000 "
                              "4611686019501129728 4539628425446424576 64 125000 0 64 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
