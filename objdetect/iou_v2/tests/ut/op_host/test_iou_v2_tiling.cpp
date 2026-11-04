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
 * \file test_iou_v2.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/iou_v2_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

class IouV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IouV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IouV2Tiling TearDown" << std::endl; }
};

struct IouV2CompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(IouV2Tiling, test_tiling_iou_aligned_f32)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1024, 1}, {1024, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iou")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "1024 1024 0 1 16 16 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IouV2Tiling, test_tiling_iou_aligned_fp16)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1024, 1}, {1024, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iou")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "1024 1024 0 1 16 16 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IouV2Tiling, test_tiling_iou_aligned_bf16)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{4, 1024}, {4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, {{{4, 1024}, {4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1024, 1}, {1024, 1}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iou")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "1024 1024 0 1 16 16 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IouV2Tiling, test_tiling_iou_aligned_true_bf16)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{4, 1024}, {4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, {{{4, 1024}, {4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1024, 1}, {1024, 1}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iou")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 6;
    string expectTilingData = "1024 1024 0 1 64 64 1065353216 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IouV2Tiling, test_tiling_iou_not_aligned_f16)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{1024, 4}, {1024, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{200, 4}, {200, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1024, 200}, {1024, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iou")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    EXPECT_EQ(info.tilingKey, 8u);
}

TEST_F(IouV2Tiling, test_tiling_iof_aligned_f32)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{4, 800}, {4, 800}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 800}, {4, 800}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{800, 1}, {800, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iof")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    EXPECT_EQ(info.tilingKey, 14u);
}

TEST_F(IouV2Tiling, test_tiling_iof_not_aligned_bf16)
{
    IouV2CompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "IouV2",
        {{{{1024, 4}, {1024, 4}}, ge::DT_BF16, ge::FORMAT_ND}, {{{200, 4}, {200, 4}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1024, 200}, {1024, 200}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<string>("iof")),
         gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    EXPECT_EQ(info.tilingKey, 19u);
}
