/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "es_cv_ops.h"
#include "log/log.h"
#include "platform/platform_info.h"

#include "../../../op_graph/fusion_pass/keep_ratio_resize_bilinear_fusion_pass.h"
#include "soc_spec_mock.h"

using namespace std;
using namespace ge;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "KeepRatioResizeBilinearFusion";
const std::string kKeepRatioType = "KeepRatioResizeBilinear";
const std::string kResizeBilinearType = "ResizeBilinearV2";
const std::string kRegbaseSoc = "Ascend950";
const std::string kRegbaseMcpSoc = "MC62CM12A";
const std::string kAscend910BSoc = "Ascend910B";
const std::string kAscend91093Soc = "Ascend910_93";
const std::string kUnsupportedSoc = "Ascend310P";
} // namespace

/*
 * Inject the platform info into the PlatformInfoManager singleton, same as the ResizeFusionPass UT.
 * The members are injected directly (instead of calling SetOptionalCompilationInfo) to avoid the
 * STL symbol conflict between the repo-generated libes_cv.so and libplatform.so.
 */

static uint32_t SocVersionToNpuArch(const std::string& socVersion)
{
    if ((socVersion == "Ascend910B") || (socVersion == "Ascend910_93")) {
        return 2201U; // DAV_2201, the A2/A3 standard arch family (verified on a real Ascend910B3)
    }
    if (socVersion == "Ascend950") {
        return 3510U; // DAV_3510
    }
    if (socVersion == "MC62CM12A") {
        return 5102U; // DAV_5102
    }
    return 2002U; // DAV_2002 (310P) stands for any unsupported arch
}

static void SetSocVersion(const std::string& socVersion)
{
    g_mockNpuArch = SocVersionToNpuArch(socVersion);
    auto& mgr = fe::PlatformInfoManager::Instance();
    fe::PlatformInfo platformInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = socVersion;
    mgr.platform_info_map_[socVersion] = platformInfo;
    mgr.opti_compilation_info_.soc_version = socVersion;
    mgr.init_flag_ = true;
}

class KeepRatioResizeBilinearFusionTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetSocVersion(kRegbaseSoc); }

    static void TearDownTestCase() {}
};

/*
 * KeepRatioResizeBilinear has no eager style API, the source graph is built with
 * CompliantNodeBuilder, as the PermuteFusionPass UT does for Permute.
 */
static std::shared_ptr<Graph> BuildKeepRatioGraph(const std::vector<int64_t>& imagesDims, Format imagesFormat,
                                                  DataType imagesDtype, const int64_t minDimension,
                                                  const int64_t maxDimension, const bool alignCorners,
                                                  const bool halfPixelCenters, const std::vector<int64_t>& yDims)
{
    auto graphBuilder = es::EsGraphBuilder("test");
    auto images = graphBuilder.CreateInput(0, "images", imagesDtype, imagesFormat, imagesDims);
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto keepRatioNode = es::CompliantNodeBuilder(graph)
                             .OpType(kKeepRatioType.c_str())
                             .Name("keep_ratio_resize_bilinear")
                             .IrDefInputs({{"images", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs({
                                 {"min_dimension", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                  es::CreateFrom(minDimension)},
                                 {"max_dimension", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                  es::CreateFrom(maxDimension)},
                                 {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                  es::CreateFromIfNotEqual(alignCorners, false)},
                                 {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                  es::CreateFromIfNotEqual(halfPixelCenters, false)},
                             })
                             .Build();

    TensorDesc imagesDesc;
    images.GetProducer()->GetOutputDesc(0, imagesDesc);
    imagesDesc.SetDataType(imagesDtype);
    imagesDesc.SetShape(Shape(imagesDims));
    imagesDesc.SetFormat(imagesFormat);
    imagesDesc.SetOriginFormat(imagesFormat);
    imagesDesc.SetOriginShape(Shape(imagesDims));
    images.GetProducer()->UpdateOutputDesc(0, imagesDesc);

    if (es::AddEdgeAndUpdatePeerDesc(*graph, *images.GetProducer(), images.GetProducerOutIndex(), keepRatioNode, 0) !=
        GRAPH_SUCCESS) {
        return nullptr;
    }
    keepRatioNode.UpdateInputDesc(0, imagesDesc);

    // The KeepRatioResizeBilinear infershape always derives y as float32 with the keep-ratio shape.
    TensorDesc yDesc;
    yDesc.SetDataType(DT_FLOAT);
    yDesc.SetShape(Shape(yDims));
    yDesc.SetFormat(imagesFormat);
    yDesc.SetOriginFormat(imagesFormat);
    yDesc.SetOriginShape(Shape(yDims));
    keepRatioNode.UpdateOutputDesc(0, yDesc);

    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(keepRatioNode, 0);
    return graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});
}

static bool HasNodeType(const std::shared_ptr<Graph>& graph, const std::string& type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == type) {
            return true;
        }
    }
    return false;
}

static bool GetNodeByType(const std::shared_ptr<Graph>& graph, const std::string& type, GNode& foundNode)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == type) {
            foundNode = node;
            return true;
        }
    }
    return false;
}

static bool GetBoolAttrFromNodeType(const std::shared_ptr<Graph>& graph, const std::string& type,
                                    const std::string& key, bool& value)
{
    GNode node;
    if (!GetNodeByType(graph, type, node)) {
        return false;
    }
    return node.GetAttr(AscendString(key.c_str()), value) == GRAPH_SUCCESS;
}

/*
 * Verify the const size input of the fused ResizeBilinearV2 node: it must be an int32 tensor of
 * the expected [newH, newW].
 */
static bool CheckTargetSizeValue(const std::shared_ptr<Graph>& graph, const std::vector<int32_t>& expectedSize)
{
    GNode targetNode;
    if (!GetNodeByType(graph, kResizeBilinearType, targetNode)) {
        return false;
    }
    Tensor sizeData;
    if (targetNode.GetInputConstData(1, sizeData) != GRAPH_SUCCESS) {
        return false;
    }
    if (sizeData.GetSize() != expectedSize.size() * sizeof(int32_t)) {
        return false;
    }
    const int32_t* sizeValues = reinterpret_cast<const int32_t*>(sizeData.GetData());
    for (size_t i = 0; i < expectedSize.size(); i++) {
        if (sizeValues[i] != expectedSize[i]) {
            return false;
        }
    }
    return true;
}

TEST_F(KeepRatioResizeBilinearFusionTest, patternTest)
{
    KeepRatioResizeBilinearFusion pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(KeepRatioResizeBilinearFusionTest, fusion950Success)
{
    // Scenario of the original canndev UT case 1: 3096x3096 NHWC with min=max=3000 -> 3000x3000.
    auto graph = BuildKeepRatioGraph({2, 3096, 3096, 3}, FORMAT_NHWC, DT_FLOAT16, 3000, 3000, false, false,
                                     {2, 3000, 3000, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));
    ASSERT_TRUE(CheckTargetSizeValue(graph, std::vector<int32_t>{3000, 3000}));

    // The output desc of the fused node must keep consistent with the original node output.
    GNode targetNode;
    ASSERT_TRUE(GetNodeByType(graph, kResizeBilinearType, targetNode));
    TensorDesc targetDesc;
    ASSERT_EQ(targetNode.GetOutputDesc(0, targetDesc), GRAPH_SUCCESS);
    EXPECT_EQ(targetDesc.GetDataType(), DT_FLOAT);
    EXPECT_EQ(targetDesc.GetShape().GetDims(), std::vector<int64_t>({2, 3000, 3000, 3}));
}

TEST_F(KeepRatioResizeBilinearFusionTest, fusion910BSuccess)
{
    // Scenario of the original canndev UT case 2: 600x1024 NHWC with min=600, max=1024 -> 600x1024.
    SetSocVersion(kAscend910BSoc);
    auto graph = BuildKeepRatioGraph({2, 600, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 600, 1024, false, false,
                                     {2, 600, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));
    ASSERT_TRUE(CheckTargetSizeValue(graph, std::vector<int32_t>{600, 1024}));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(KeepRatioResizeBilinearFusionTest, fusion91093NchwSuccess)
{
    // NCHW layout: H/W are read from dims [2]/[3].
    SetSocVersion(kAscend91093Soc);
    auto graph = BuildKeepRatioGraph({2, 3, 600, 1024}, FORMAT_NCHW, DT_FLOAT, 600, 1024, false, false,
                                     {2, 3, 600, 1024});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));
    ASSERT_TRUE(CheckTargetSizeValue(graph, std::vector<int32_t>{600, 1024}));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(KeepRatioResizeBilinearFusionTest, fusionMinScaleExceedSuccess)
{
    // 100x100 with min=60, max=50: the min-scale candidate (60x60) exceeds max_dimension, so the
    // max-scale candidate (50x50) is used.
    auto graph = BuildKeepRatioGraph({1, 100, 100, 3}, FORMAT_NHWC, DT_FLOAT, 60, 50, false, false, {1, 50, 50, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));
    ASSERT_TRUE(CheckTargetSizeValue(graph, std::vector<int32_t>{50, 50}));
}

TEST_F(KeepRatioResizeBilinearFusionTest, attrTransferSuccess)
{
    // align_corners / half_pixel_centers must be transferred to the fused ResizeBilinearV2 node.
    auto graph = BuildKeepRatioGraph({2, 600, 1024, 3}, FORMAT_NHWC, DT_FLOAT, 600, 1024, true, true,
                                     {2, 600, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));

    bool alignCorners = false;
    ASSERT_TRUE(GetBoolAttrFromNodeType(graph, kResizeBilinearType, "align_corners", alignCorners));
    EXPECT_TRUE(alignCorners);
    bool halfPixelCenters = false;
    ASSERT_TRUE(GetBoolAttrFromNodeType(graph, kResizeBilinearType, "half_pixel_centers", halfPixelCenters));
    EXPECT_TRUE(halfPixelCenters);
}

TEST_F(KeepRatioResizeBilinearFusionTest, mcpPlatformNotChangedFail)
{
    // MC62CM12A (DAV_5102) is the single arch where the registration whitelist excludes this
    // pass (the 5102 entry does not list KeepRatioResizeBilinearFusion), so the pass must not
    // change the graph there.
    SetSocVersion(kRegbaseMcpSoc);
    auto graph = BuildKeepRatioGraph({2, 600, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 600, 1024, false, false,
                                     {2, 600, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kKeepRatioType));
    EXPECT_FALSE(HasNodeType(graph, kResizeBilinearType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(KeepRatioResizeBilinearFusionTest, allChipsPlatformFusionSuccess)
{
    // Registration-model correction (2026-10-26): the whitelist only filters the regbase archs
    // it lists; the 5102 entry does not include this pass, every other arch is unconstrained,
    // so the legacy pass is active on all chips EXCEPT 5102. 310P (2002) must therefore FUSE.
    SetSocVersion(kUnsupportedSoc);
    auto graph = BuildKeepRatioGraph({2, 600, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 600, 1024, false, false,
                                     {2, 600, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(KeepRatioResizeBilinearFusionTest, minDimensionZeroFail)
{
    auto graph = BuildKeepRatioGraph({2, 600, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 0, 1024, false, false,
                                     {2, 600, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kKeepRatioType));
}

TEST_F(KeepRatioResizeBilinearFusionTest, maxDimensionZeroFail)
{
    auto graph = BuildKeepRatioGraph({2, 600, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 600, 0, false, false,
                                     {2, 600, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kKeepRatioType));
}

TEST_F(KeepRatioResizeBilinearFusionTest, inputNot4DFail)
{
    // Legacy-aligned: the original returns FAILED for a non-4-dim input shape (the whole build
    // fails during the pass); the migrated pass lets the form through MeetRequirements and
    // fails in Replacement likewise, so the run reports an error instead of NOT_CHANGED.
    auto graph = BuildKeepRatioGraph({2, 600, 1024}, FORMAT_NHWC, DT_FLOAT16, 600, 1024, false, false, {2, 600, 1024});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    printf("inputNot4D actual status=%d\n", static_cast<int>(status));
    EXPECT_NE(status, SUCCESS);
    EXPECT_NE(status, GRAPH_NOT_CHANGED);
}

TEST_F(KeepRatioResizeBilinearFusionTest, unknownShapeFail)
{
    auto graph = BuildKeepRatioGraph({2, -1, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 600, 1024, false, false,
                                     {2, -1, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kKeepRatioType));
}

TEST_F(KeepRatioResizeBilinearFusionTest, controlEdgePreservedSuccess)
{
    // A control edge into the KeepRatioResizeBilinear node must be kept on the fused target node.
    // Note the original pass creates the new node and removes the old one without re-linking the
    // control anchors, so it drops the control edge; the framework replacement keeps it.
    auto graphBuilder = es::EsGraphBuilder("test");
    auto images = graphBuilder.CreateInput(0, "images", DT_FLOAT16, FORMAT_NHWC, std::vector<int64_t>{1, 100, 100, 3});
    auto* cGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto keepRatioNode = es::CompliantNodeBuilder(cGraph)
                             .OpType(kKeepRatioType.c_str())
                             .Name("keep_ratio_ctrl")
                             .IrDefInputs({{"images", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs({
                                 {"min_dimension", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                  es::CreateFrom(int64_t(60))},
                                 {"max_dimension", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                  es::CreateFrom(int64_t(50))},
                                 {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                  es::CreateFromIfNotEqual(false, false)},
                                 {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                  es::CreateFromIfNotEqual(false, false)},
                             })
                             .Build();
    TensorDesc imagesDesc;
    images.GetProducer()->GetOutputDesc(0, imagesDesc);
    imagesDesc.SetDataType(DT_FLOAT16);
    imagesDesc.SetShape(Shape(std::vector<int64_t>{1, 100, 100, 3}));
    imagesDesc.SetOriginShape(Shape(std::vector<int64_t>{1, 100, 100, 3}));
    imagesDesc.SetFormat(FORMAT_NHWC);
    imagesDesc.SetOriginFormat(FORMAT_NHWC);
    images.GetProducer()->UpdateOutputDesc(0, imagesDesc);
    ASSERT_EQ(
        es::AddEdgeAndUpdatePeerDesc(*cGraph, *images.GetProducer(), images.GetProducerOutIndex(), keepRatioNode, 0),
        GRAPH_SUCCESS);
    keepRatioNode.UpdateInputDesc(0, imagesDesc);
    TensorDesc yDesc;
    yDesc.SetDataType(DT_FLOAT);
    yDesc.SetShape(Shape(std::vector<int64_t>{1, 50, 50, 3}));
    yDesc.SetOriginShape(Shape(std::vector<int64_t>{1, 50, 50, 3}));
    yDesc.SetFormat(FORMAT_NHWC);
    yDesc.SetOriginFormat(FORMAT_NHWC);
    keepRatioNode.UpdateOutputDesc(0, yDesc);
    (void)cGraph->AddControlEdge(*images.GetProducer(), keepRatioNode);
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(keepRatioNode, 0);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    ASSERT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));

    GNode targetNode;
    ASSERT_TRUE(GetNodeByType(graph, kResizeBilinearType, targetNode));
    // The control dependency from the images input producer must still exist in the fused graph
    // (the framework routes it through an inserted NoOp which controls the replacement nodes),
    // and the target node must be under control.
    size_t preservedCtrlEdges = 0;
    for (auto node : graph->GetAllNodes()) {
        for (auto ctrlSrc : node.GetInControlNodes()) {
            AscendString srcName;
            (void)ctrlSrc->GetName(srcName);
            if (std::string(srcName.GetString()) == "images") {
                preservedCtrlEdges++;
            }
        }
    }
    EXPECT_EQ(preservedCtrlEdges, 1);
    EXPECT_GE(targetNode.GetInControlNodes().size(), 1);
}

TEST_F(KeepRatioResizeBilinearFusionTest, zeroHeightFusionSuccess)
{
    // Legacy-aligned: the original has no zero-dim gate; H=0 walks into the same 0/0 division
    // as the original pass (and its built-in infershape) and fuses with whatever size values
    // that produces. The graph-level outcome (fused to ResizeBilinearV2) is what is asserted;
    // the size values themselves are the same UB as the original.
    auto graph = BuildKeepRatioGraph({1, 0, 1024, 3}, FORMAT_NHWC, DT_FLOAT16, 600, 1024, false, false,
                                     {1, 0, 1024, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kKeepRatioType));
}

TEST_F(KeepRatioResizeBilinearFusionTest, minDimensionAttrMissingFail)
{
    // K9 of the diff findings: without the min_dimension attr the pass must skip directly; the
    // legacy pass leaves minDims at 0 and skips at the ==0 check. Both keep the graph unchanged.
    auto graphBuilder = es::EsGraphBuilder("test");
    auto images = graphBuilder.CreateInput(0, "images", DT_FLOAT16, FORMAT_NHWC, std::vector<int64_t>{1, 100, 100, 3});
    auto* cGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto keepRatioNode = es::CompliantNodeBuilder(cGraph)
                             .OpType(kKeepRatioType.c_str())
                             .Name("keep_ratio_no_min")
                             .IrDefInputs({{"images", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs({
                                 {"max_dimension", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                  es::CreateFrom(int64_t(50))},
                                 {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                  es::CreateFromIfNotEqual(false, false)},
                                 {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                  es::CreateFromIfNotEqual(false, false)},
                             })
                             .Build();
    TensorDesc imagesDesc;
    images.GetProducer()->GetOutputDesc(0, imagesDesc);
    imagesDesc.SetDataType(DT_FLOAT16);
    imagesDesc.SetShape(Shape(std::vector<int64_t>{1, 100, 100, 3}));
    imagesDesc.SetOriginShape(Shape(std::vector<int64_t>{1, 100, 100, 3}));
    imagesDesc.SetFormat(FORMAT_NHWC);
    imagesDesc.SetOriginFormat(FORMAT_NHWC);
    images.GetProducer()->UpdateOutputDesc(0, imagesDesc);
    ASSERT_EQ(
        es::AddEdgeAndUpdatePeerDesc(*cGraph, *images.GetProducer(), images.GetProducerOutIndex(), keepRatioNode, 0),
        GRAPH_SUCCESS);
    keepRatioNode.UpdateInputDesc(0, imagesDesc);
    TensorDesc yDesc;
    yDesc.SetDataType(DT_FLOAT);
    yDesc.SetShape(Shape(std::vector<int64_t>{1, 50, 50, 3}));
    yDesc.SetOriginShape(Shape(std::vector<int64_t>{1, 50, 50, 3}));
    yDesc.SetFormat(FORMAT_NHWC);
    yDesc.SetOriginFormat(FORMAT_NHWC);
    keepRatioNode.UpdateOutputDesc(0, yDesc);
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(keepRatioNode, 0);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    KeepRatioResizeBilinearFusion pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kKeepRatioType));
    EXPECT_FALSE(HasNodeType(graph, kResizeBilinearType));
}
