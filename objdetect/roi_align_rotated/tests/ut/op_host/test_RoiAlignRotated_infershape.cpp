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
 * \file test_RoiAlignRotated_proto.cpp
 * \brief
 * @version 1.0
 */
#include <gtest/gtest.h>
#include <iostream>
#include "platform/platform_info.h"
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class RoiAlignRotated : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // infershape 走 soc 分发（fe::PlatformInfoManager 单例进程内共享，three_interpolate UT 先例）：
        // 本套件为arch22 cv 契约用例，初始化为 ascend910b，确保路由到老芯片推导分支
        fe::PlatformInfoManager::Instance().InitializePlatformInfo();
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optionalInfo;
        fe::PlatformInfoManager::Instance().GetPlatformInfo("Ascend910B1", platformInfo, optionalInfo);
        optionalInfo.soc_version = "Ascend910B1";
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
        std::cout << "RoiAlignRotated Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() { std::cout << "RoiAlignRotated Proto Test TearDown" << std::endl; }
};

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{4, 8, 8, 1}, {4, 8, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 2, 2, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_invalid_x_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated", {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_invalid_rois_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{4, 8, 8, 1}, {4, 8, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_invalid_pooled_h)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{4, 8, 8, 1}, {4, 8, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_invalid_pooled_w)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{4, 8, 8, 1}, {4, 8, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(0)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_unknown_rank_is_propagated)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated", {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_unknown_dim_is_preserved)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{4, 8, 8, -1}, {4, 8, 8, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{6, 6}, {6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{6, 2, 2, -1}});
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_rois_n_3_c_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{8, 8, 8, 1}, {8, 8, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 3}, {6, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{3, 2, 2, 1}});
}

TEST_F(RoiAlignRotated, RoiAlignRotated_infershape_case_rois_n_3_c_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RoiAlignRotated",
        {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 3}, {6, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0)),
         gert::InfershapeContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{3, 2, 2, 8}});
}
