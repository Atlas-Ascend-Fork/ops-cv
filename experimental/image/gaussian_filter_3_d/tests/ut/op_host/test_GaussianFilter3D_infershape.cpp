/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_GaussianFilter3D_infershape.cpp
 * \brief GaussianFilter3D infershape UT：y 与 x 同 shape 同 dtype，rank 3/4/5
 */
#include <gtest/gtest.h>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GaussianFilter3DInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(GaussianFilter3DInfershapeTest, gaussian_filter_3_d_infershape_rank3_fp32)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("sigma", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
        gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("REFLECT")),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GaussianFilter3D", {{{{8, 16, 67}, {8, 16, 67}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 16, 67}, {8, 16, 67}}, ge::DT_FLOAT, ge::FORMAT_ND}}, attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 16, 67},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GaussianFilter3DInfershapeTest, gaussian_filter_3_d_infershape_rank4_fp16)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("sigma", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
        gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("CONSTANT")),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GaussianFilter3D", {{{{2, 8, 16, 67}, {2, 8, 16, 67}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 8, 16, 67}, {2, 8, 16, 67}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 8, 16, 67},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GaussianFilter3DInfershapeTest, gaussian_filter_3_d_infershape_rank5_bf16)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        gert::InfershapeContextPara::OpAttr("sigma", Ops::Cv::AnyValue::CreateFrom<float>(2.0f)),
        gert::InfershapeContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>("WRAP")),
    };
    gert::InfershapeContextPara infershapeContextPara(
        "GaussianFilter3D", {{{{2, 3, 8, 16, 67}, {2, 3, 8, 16, 67}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{2, 3, 8, 16, 67}, {2, 3, 8, 16, 67}}, ge::DT_BF16, ge::FORMAT_ND}}, attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 8, 16, 67},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
