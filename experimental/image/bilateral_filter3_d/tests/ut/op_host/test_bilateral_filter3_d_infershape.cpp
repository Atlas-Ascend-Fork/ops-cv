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
#include <iostream>
#include <limits>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
using Attr = gert::InfershapeContextPara::OpAttr;

// Attribute order mirrors InferShapeBilateralFilter3D: kernel_size(0) -> sigma_space(1) -> sigma_color(2).
std::vector<Attr> MakeAttrs(int64_t kernelSize, float sigmaSpace = 1.0F, float sigmaColor = 1.0F)
{
    return {Attr("kernel_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(kernelSize)),
            Attr("sigma_space", Ops::Cv::AnyValue::CreateFrom<float>(sigmaSpace)),
            Attr("sigma_color", Ops::Cv::AnyValue::CreateFrom<float>(sigmaColor))};
}
} // namespace

class BilateralFilter3DInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BilateralFilter3DInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BilateralFilter3DInfershape TearDown" << std::endl; }
};

// ---------------------------------------------------------------------------
// y always mirrors x: same shape, same dtype, ranks 3 through 5.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DInfershape, Rank3Fp32)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{37, 37, 37}, {37, 37, 37}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(5, 1.0F, 1.0F));
    std::vector<std::vector<int64_t>> expectOutputShape = {{37, 37, 37}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BilateralFilter3DInfershape, Rank4Fp16)
{
    gert::InfershapeContextPara para("BilateralFilter3D",
                                     {{{{2, 32, 50, 41}, {2, 32, 50, 41}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, MakeAttrs(7, 1.0F, 1.0F));
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 32, 50, 41}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BilateralFilter3DInfershape, Rank5Fp32)
{
    gert::InfershapeContextPara para("BilateralFilter3D",
                                     {{{{2, 3, 16, 20, 24}, {2, 3, 16, 20, 24}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(9, 1.2F, 2.0F));
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 16, 20, 24}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BilateralFilter3DInfershape, SmallestKernelOnMinimalVolume)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, 1.0F));
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---------------------------------------------------------------------------
// Rank and emptiness rejections.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DInfershape, RankTwoFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{17, 19}, {17, 19}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, RankSixFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D",
                                     {{{{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, ZeroDimFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{0, 4, 4}, {0, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// Unknown dims are placeholders, not empty tensors: they must pass through so the
// operator keeps working in graph mode, where shapes are only known at runtime.
TEST_F(BilateralFilter3DInfershape, DynamicShapePassesThrough)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, 1.0F));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BilateralFilter3DInfershape, PartiallyDynamicShapePassesThrough)
{
    gert::InfershapeContextPara para("BilateralFilter3D",
                                     {{{{2, 3, 16, -1, -1}, {2, 3, 16, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, MakeAttrs(5, 1.0F, 1.0F));
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 16, -1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---------------------------------------------------------------------------
// Attribute rejections: kernel_size in {3,5,7,9}, sigmas positive and finite.
// ---------------------------------------------------------------------------

TEST_F(BilateralFilter3DInfershape, KernelSizeEvenFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(4, 1.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, KernelSizeOneFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(1, 1.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, KernelSizeTooLargeFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(11, 1.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, SigmaSpaceZeroFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 0.0F, 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, SigmaSpaceNanFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     MakeAttrs(3, std::numeric_limits<float>::quiet_NaN(), 1.0F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, SigmaColorNegativeFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(3, 1.0F, -0.5F));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(BilateralFilter3DInfershape, SigmaColorInfiniteFails)
{
    gert::InfershapeContextPara para("BilateralFilter3D", {{{{8, 8, 8}, {8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     MakeAttrs(3, 1.0F, std::numeric_limits<float>::infinity()));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
