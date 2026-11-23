/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_host/arch35/resize_linear_grad_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;
using namespace optiling;

class ResizeLinearGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeLinearGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeLinearGradTilingTest TearDown" << std::endl; }
};

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_01)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 0 96 1 1 25088 96 96 100352 32 4575657222473777152 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_02)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape outputShape = {{1, 3, 16}, {1, 3, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 16843008;
    string expectTilingData = "48 0 1 0 1 0 0 0 16 32 4612886978239118831 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_03)
{
    gert::StorageShape inputGradsShape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape inputOriImageShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape outputShape = {{1, 3, 16}, {1, 3, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 260;
    string expectTilingData = "3 48 1 0 1 1 0 1 16 1 4575657222473777152 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_04)
{
    gert::StorageShape inputGradsShape = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape inputOriImageShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape outputShape = {{1, 3, 16}, {1, 3, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(1.0f))},
        &compileInfo);
    uint64_t expectTilingKey = 259;
    string expectTilingData = "3 48 1 0 1 1 0 1 16 1 1098907648 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 05: upsampling with integer scale, align_corners=false
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_05)
{
    gert::StorageShape inputGradsShape = {{2, 5, 32}, {2, 5, 32}};
    gert::StorageShape inputOriImageShape = {{2, 5, 16}, {2, 5, 16}};
    gert::StorageShape outputShape = {{2, 5, 16}, {2, 5, 16}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(2.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: large tensor upsampling with scales
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_06)
{
    gert::StorageShape inputGradsShape = {{200, 8, 64}, {200, 8, 64}};
    gert::StorageShape inputOriImageShape = {{200, 8, 32}, {200, 8, 32}};
    gert::StorageShape outputShape = {{200, 8, 32}, {200, 8, 32}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 07: downsampling with align_corners=true
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_07)
{
    gert::StorageShape inputGradsShape = {{1, 3, 16}, {1, 3, 16}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32}, {1, 3, 32}};
    gert::StorageShape outputShape = {{1, 3, 32}, {1, 3, 32}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 08: scaleL=0.75 on A5(Ascend950), threshold 0.8, scaleL<threshold -> determine template
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_08_a5_threshold_0p8)
{
    gert::StorageShape inputGradsShape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape inputOriImageShape = {{1, 3, 3}, {1, 3, 3}};
    gert::StorageShape outputShape = {{1, 3, 3}, {1, 3, 3}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    gert::TilingContextPara tilingContextPara(
        "ResizeLinearGrad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_ND}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo, "Ascend950");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // schId=0, isInt32=1, isCenter=0, isDetermine=1
    ASSERT_EQ(tilingInfo.tilingKey, 16777472U);
}

// DAV_9201/9202场景scaleL=0.75: 阈值0.75, 0.75不小于阈值 -> no_determine模板(key=256)
static void RunResizeLinearGradTilingWithNpuArch(const std::string& shortSoc, const std::string& npuArch,
                                                 uint64_t expectTilingKey)
{
    gert::StorageShape inputGradsShape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape inputOriImageShape = {{1, 3, 3}, {1, 3, 3}};
    gert::StorageShape outputShape = {{1, 3, 3}, {1, 3, 3}};

    ResizeLinearGradCompileInfo compileInfo = {64, 200704, 32};

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);

    std::vector<gert::Tensor*> inputTensors = {};
    std::vector<gert::Tensor*> outputTensors = {};
    std::vector<std::unique_ptr<gert::Tensor>> tensorsKeepAlive = {};
    std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(
        inputGradsShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    inputTensors.push_back(curTensor.get());
    tensorsKeepAlive.push_back(std::move(curTensor));
    curTensor = std::make_unique<gert::Tensor>(
        inputOriImageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    inputTensors.push_back(curTensor.get());
    tensorsKeepAlive.push_back(std::move(curTensor));
    curTensor = std::make_unique<gert::Tensor>(
        outputShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    outputTensors.push_back(curTensor.get());
    tensorsKeepAlive.push_back(std::move(curTensor));

    auto contextHolder = gert::TilingContextFaker()
                             .SetOpType("ResizeLinearGrad")
                             .NodeIoNum(2, 1)
                             .InputTensors(inputTensors)
                             .OutputTensors(outputTensors)
                             .Attr("align_corners", false)
                             .Attr("scale", 0.0f)
                             .CompileInfo(&compileInfo)
                             .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                             .TilingData(tilingData.get())
                             .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                             .Build();

    auto tilingContext = contextHolder.GetContext();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    std::map<std::string, std::string> socVersion = {{"Short_SoC_version", shortSoc}, {"NpuArch", npuArch}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("ResizeLinearGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    ASSERT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_SUCCESS);
    // schId=0, isInt32=1, isCenter=0, isDetermine=0
    ASSERT_EQ(tilingContext->GetTilingKey(), expectTilingKey);
}

// Test 09: scaleL=0.75 on DAV_9201, threshold 0.75 -> no_determine template
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_09_dav9201_threshold_0p75)
{
    RunResizeLinearGradTilingWithNpuArch("Ascend960DT", "9201", 256U);
}

// Test 10: scaleL=0.75 on DAV_9202, threshold 0.75 -> no_determine template
TEST_F(ResizeLinearGradTilingTest, resize_linear_grad_tiling_10_dav9202_threshold_0p75)
{
    RunResizeLinearGradTilingWithNpuArch("Ascend960PR", "9202", 256U);
}
