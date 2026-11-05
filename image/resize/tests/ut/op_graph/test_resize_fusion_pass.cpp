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
#include "es_cv_ops.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "platform/platform_info.h"

#include "../../../op_graph/fusion_pass/resize_fusion_pass.h"
#include "soc_spec_mock.h"

using namespace std;
using namespace ge;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "ResizeFusionPass";
const std::string kResizeType = "Resize";
const std::string kResizeNearestType = "ResizeNearestNeighborV2";
const std::string kResizeBilinearType = "ResizeBilinearV2";
const std::string kResizeTrilinearType = "ResizeTrilinear";
const std::string kSliceType = "Slice";
const std::string kRegbaseSoc = "Ascend950";
const std::string kRegbaseMcpSoc = "MC62CM12A";
const std::string kAscend910BSoc = "Ascend910B";
const std::string kAscend91093Soc = "Ascend910_93";
const std::string kUnsupportedSoc = "Ascend310P";
} // namespace

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

/*
 * Inject the platform info into the PlatformInfoManager singleton. The platform gate reads the
 * NpuArch through rtGetSocSpec (mocked above); the PlatformInfoManager seeding is kept for the
 * parts of the framework that still consult it. The members are injected directly (instead of
 * calling SetOptionalCompilationInfo) to avoid the STL symbol conflict between the repo-generated
 * libes_cv.so and libplatform.so.
 */
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

class ResizeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetSocVersion(kRegbaseSoc); }

    static void TearDownTestCase() {}
};

/*
 * Build a graph with a single Resize node:
 *              x   roi  scales  sizes
 *               \    |     |    /
 *                \   |     |   /
 *                  Resize
 *                    |
 *                    y
 * The roi/scales/sizes inputs are optional in the Resize IR; the hasXxx flags build the partial
 * input forms (the original pass matches the node by type whatever the input count is).
 */
static std::shared_ptr<Graph> BuildResizeGraph(const std::string& mode, const std::string& ctmMode, DataType xDtype,
                                               Format xFormat, const std::vector<int64_t>& xDims,
                                               const std::vector<int64_t>& yDims, const bool hasRoi = true,
                                               const bool hasScales = true, const bool hasSizes = true,
                                               const bool addControlEdge = false, const bool setModeAttr = true)
{
    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", xDtype, xFormat, xDims);
    int32_t nextInputIdx = 1;
    es::EsTensorHolder roi;
    es::EsTensorHolder scales;
    es::EsTensorHolder sizes;
    if (hasRoi) {
        roi = graphBuilder.CreateInput(nextInputIdx++, "roi", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{0});
    }
    if (hasScales) {
        scales = graphBuilder.CreateInput(nextInputIdx++, "scales", DT_FLOAT, FORMAT_ND,
                                          std::vector<int64_t>{xDims.size()});
    }
    if (hasSizes) {
        sizes = graphBuilder.CreateInput(nextInputIdx++, "sizes", DT_INT32, FORMAT_ND,
                                         std::vector<int64_t>{xDims.size()});
    }
    auto y = es::Resize(x, roi, scales, sizes, ctmMode.c_str(), -0.75, 0, 0.0, mode.c_str(), "round_prefer_floor");

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(xDtype);
    xDesc.SetShape(Shape(xDims));
    xDesc.SetFormat(xFormat);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc yDesc;
    y.GetProducer()->GetOutputDesc(0, yDesc);
    yDesc.SetDataType(xDtype);
    yDesc.SetShape(Shape(yDims));
    yDesc.SetFormat(xFormat);
    y.GetProducer()->UpdateOutputDesc(0, yDesc);

    // Sync the input descs on the Resize node itself, just like what infershape does in a real
    // graph. Only the inputs actually present are updated, at their IR positions.
    GNode resizeNode = *y.GetProducer();
    resizeNode.UpdateInputDesc(0, xDesc);
    if (hasRoi) {
        TensorDesc roiDesc;
        roiDesc.SetDataType(DT_FLOAT);
        roiDesc.SetShape(Shape(std::vector<int64_t>{0}));
        roiDesc.SetFormat(FORMAT_ND);
        resizeNode.UpdateInputDesc(1, roiDesc);
    }
    if (hasScales) {
        TensorDesc scalesDesc;
        scalesDesc.SetDataType(DT_FLOAT);
        scalesDesc.SetShape(Shape(std::vector<int64_t>{static_cast<int64_t>(xDims.size())}));
        scalesDesc.SetFormat(FORMAT_ND);
        resizeNode.UpdateInputDesc(2, scalesDesc);
    }
    if (hasSizes) {
        TensorDesc sizesDesc;
        sizesDesc.SetDataType(DT_INT32);
        sizesDesc.SetShape(Shape(std::vector<int64_t>{static_cast<int64_t>(xDims.size())}));
        sizesDesc.SetFormat(FORMAT_ND);
        resizeNode.UpdateInputDesc(3, sizesDesc);
    }
    resizeNode.UpdateOutputDesc(0, yDesc);

    // The eager style API skips attrs equal to their default values (e.g. mode="nearest"),
    // while a real graph converted from onnx always carries the mode attr. Set it explicitly.
    // setModeAttr=false builds the attr-missing form (R13 of the diff findings).
    if (setModeAttr) {
        AscendString modeAttr(mode.c_str());
        resizeNode.SetAttr(AscendString("mode"), modeAttr);
    }
    AscendString ctmAttr(ctmMode.c_str());
    resizeNode.SetAttr(AscendString("coordinate_transformation_mode"), ctmAttr);

    // Optionally add a control edge into the Resize node (from the roi input producer) to verify
    // that the framework keeps the control dependencies on the fused target node.
    if (addControlEdge) {
        Graph* cGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
        if (hasRoi) {
            (void)cGraph->AddControlEdge(*roi.GetProducer(), resizeNode);
        }
        if (hasScales) {
            (void)cGraph->AddControlEdge(*scales.GetProducer(), resizeNode);
        }
    }

    return graphBuilder.BuildAndReset({y});
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

static void SetStringAttrToResizeNode(const std::shared_ptr<Graph>& graph, const std::string& key,
                                      const std::string& value)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == kResizeType) {
            AscendString attrValue(value.c_str());
            node.SetAttr(AscendString(key.c_str()), attrValue);
            return;
        }
    }
}

static bool GetBoolAttrFromNode(const std::shared_ptr<Graph>& graph, const std::string& type, const std::string& key,
                                bool& value)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == type) {
            return node.GetAttr(AscendString(key.c_str()), value) == GRAPH_SUCCESS;
        }
    }
    return false;
}

/*
 * Get the output TensorDesc of the first node with the given type. Used to verify that the
 * shape/dtype information is preserved on the fused target node.
 */
static bool GetOutputDescFromNodeType(const std::shared_ptr<Graph>& graph, const std::string& type, TensorDesc& desc)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == type) {
            return node.GetOutputDesc(0, desc) == GRAPH_SUCCESS;
        }
    }
    return false;
}

/*
 * The offsets/size inputs (index 1/2) of the fused Slice node must be explicitly DT_INT64: the
 * eager style API does not propagate const descs onto the node inputs and an unset desc stays at
 * the default dtype, which is rejected by the op kernel check (same class as the target size
 * input desc issue; verified on a real Ascend910B device with a dynamic onnx shape).
 */
static bool CheckSliceParamsDtype(const std::shared_ptr<Graph>& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == kSliceType) {
            TensorDesc offsetsDesc;
            if (node.GetInputDesc(1, offsetsDesc) != GRAPH_SUCCESS || offsetsDesc.GetDataType() != DT_INT64) {
                return false;
            }
            TensorDesc sizeDesc;
            if (node.GetInputDesc(2, sizeDesc) != GRAPH_SUCCESS || sizeDesc.GetDataType() != DT_INT64) {
                return false;
            }
            return true;
        }
    }
    return false;
}

/*
 * The size input (index 1) of the fused target node must be explicitly DT_INT32: the target IR
 * only accepts DT_INT32 and an unset desc stays at the default dtype, which is rejected by the
 * op kernel check with EZ3002 (verified on a real Ascend910B device).
 */
static bool CheckTargetSizeInputDtype(const std::shared_ptr<Graph>& graph, const std::string& type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == type) {
            TensorDesc sizeDesc;
            if (node.GetInputDesc(1, sizeDesc) != GRAPH_SUCCESS) {
                return false;
            }
            return sizeDesc.GetDataType() == DT_INT32;
        }
    }
    return false;
}

TEST_F(ResizeFusionPassTest, patternTest)
{
    ResizeFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(ResizeFusionPassTest, fusionMcpPlatformSuccess)
{
    SetSocVersion(kRegbaseMcpSoc);
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeFusionPassTest, fusion910BPlatformSuccess)
{
    SetSocVersion(kAscend910BSoc);
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeFusionPassTest, fusion91093PlatformSuccess)
{
    SetSocVersion(kAscend91093Soc);
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT16, FORMAT_NHWC, {1, 8, 8, 3}, {1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeNearestType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeFusionPassTest, allChipsPlatformFusionSuccess)
{
    // Registration-model correction (2026-10-26): the support_fusion_pass.json whitelist only
    // filters the two regbase archs it lists (3510/5102), BOTH of which include ResizeFusionPass;
    // every other arch is not governed by the whitelist, so the legacy pass is active on ALL
    // chips (a narrower {2201,3510,5102} in-code gate skipped Ascend1951/DAV_2002 and broke a
    // TF detection model that depends on the fusion). 310P (2002) must therefore FUSE.
    SetSocVersion(kUnsupportedSoc);
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeFusionPassTest, unsupportedModeCubicFail)
{
    auto graph = BuildResizeGraph("cubic", "half_pixel", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3}, {1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, unsupportedDtypeFail)
{
    // nearest mode only supports float16 and float32
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_INT32, FORMAT_NHWC, {1, 8, 8, 3}, {1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, lowDimFail)
{
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_NHWC, {1, 8, 3}, {1, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, dvppAttrFail)
{
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3}, {1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);
    SetStringAttrToResizeNode(graph, "_specified_engine_name", "DNN_VM_DVPP");

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, dynamicShapeNotOnnxFail)
{
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_NHWC, {-1, 8, 8, 3}, {-1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, nearestStaticFusionSuccess)
{
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT16, FORMAT_NHWC, {1, 8, 8, 3}, {1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeNearestType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));

    // The shape/dtype of the fused node must keep consistent with the original Resize output.
    TensorDesc targetDesc;
    ASSERT_TRUE(GetOutputDescFromNodeType(graph, kResizeNearestType, targetDesc));
    EXPECT_EQ(targetDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(targetDesc.GetShape().GetDims(), std::vector<int64_t>({1, 4, 4, 3}));
    // The size input must be explicitly int32, otherwise the kernel check rejects it with EZ3002.
    EXPECT_TRUE(CheckTargetSizeInputDtype(graph, kResizeNearestType));
}

TEST_F(ResizeFusionPassTest, linearStaticFusionSuccess)
{
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));

    // The shape/dtype of the fused node must keep consistent with the original Resize output.
    TensorDesc targetDesc;
    ASSERT_TRUE(GetOutputDescFromNodeType(graph, kResizeBilinearType, targetDesc));
    EXPECT_EQ(targetDesc.GetDataType(), DT_FLOAT);
    EXPECT_EQ(targetDesc.GetShape().GetDims(), std::vector<int64_t>({1, 3, 4, 4}));
    // The size input must be explicitly int32, otherwise the kernel check rejects it with EZ3002.
    EXPECT_TRUE(CheckTargetSizeInputDtype(graph, kResizeBilinearType));
}

TEST_F(ResizeFusionPassTest, unsupportedFormatFusionSuccess)
{
    // Strict legacy alignment (2026-10-26 review decision): the original ClcOutputSize only
    // warns on an unsupported format and fuses with a 0-element size const (CreateConstantNode);
    // reproduced with the CompliantNodeBuilder empty const. The earlier skip-gate was based on a
    // misdiagnosis of the Ascend1951 regression (real root cause: the platform gate).
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_ND, {1, 8, 8, 3}, {1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeNearestType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, linear5DStaticFusionSuccess)
{
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCDHW, {1, 3, 4, 8, 8}, {1, 3, 2, 4, 4});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeTrilinearType));
    EXPECT_TRUE(HasNodeType(graph, kSliceType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));

    // The shape/dtype of the fused node must keep consistent with the original Resize output.
    TensorDesc targetDesc;
    ASSERT_TRUE(GetOutputDescFromNodeType(graph, kResizeTrilinearType, targetDesc));
    EXPECT_EQ(targetDesc.GetDataType(), DT_FLOAT);
    EXPECT_EQ(targetDesc.GetShape().GetDims(), std::vector<int64_t>({1, 3, 2, 4, 4}));
    // The size input (from the Slice branch, dtype of the source sizes input) must be int32.
    EXPECT_TRUE(CheckTargetSizeInputDtype(graph, kResizeTrilinearType));
}

TEST_F(ResizeFusionPassTest, nearestDynamicOnnxFusionSuccess)
{
    auto graph = BuildResizeGraph("nearest", "pytorch_half_pixel", DT_FLOAT, FORMAT_NHWC, {-1, 8, 8, 3}, {-1, 4, 4, 3});
    ASSERT_NE(graph, nullptr);
    SetStringAttrToResizeNode(graph, "resize_original_type", "onnx_resize");

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeNearestType));
    EXPECT_TRUE(HasNodeType(graph, kSliceType));
    EXPECT_TRUE(CheckSliceParamsDtype(graph));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, linearDynamicOnnxFusionSuccess)
{
    auto graph = BuildResizeGraph("linear", "align_corners", DT_FLOAT, FORMAT_NCHW, {-1, 3, 8, 8}, {-1, 3, 4, 4});
    ASSERT_NE(graph, nullptr);
    SetStringAttrToResizeNode(graph, "resize_original_type", "onnx_resize");

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_TRUE(HasNodeType(graph, kSliceType));
    EXPECT_TRUE(CheckSliceParamsDtype(graph));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));

    // coordinate_transformation_mode "align_corners" should be converted to align_corners=true
    bool alignCorners = false;
    ASSERT_TRUE(GetBoolAttrFromNode(graph, kResizeBilinearType, "align_corners", alignCorners));
    EXPECT_TRUE(alignCorners);
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

TEST_F(ResizeFusionPassTest, twoInputsFusionSuccess)
{
    // [x, sizes] form (roi/scales absent): the original pass matches the node by type whatever
    // the input count is and prunes it down to [x, sizes], so this form must also fuse.
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4},
                                  /*hasRoi=*/false, /*hasScales=*/false, /*hasSizes=*/true);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
    EXPECT_TRUE(CheckTargetSizeInputDtype(graph, kResizeBilinearType));
}

TEST_F(ResizeFusionPassTest, threeInputsFusionSuccess)
{
    // [x, roi, sizes] form (scales absent): must also fuse, as the original pass does.
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_NHWC, {1, 8, 8, 3}, {1, 4, 4, 3},
                                  /*hasRoi=*/true, /*hasScales=*/false, /*hasSizes=*/true);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeNearestType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, noSizesInputFusionSuccess)
{
    // Legacy-aligned: without the sizes input the original still fuses - it keeps [x, last-input]
    // and the const branch takes the size from the static output shape (the last input, here
    // scales, is dropped exactly like roi/scales in the full form).
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4},
                                  /*hasRoi=*/true, /*hasScales=*/true, /*hasSizes=*/false);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, onlyXInputFusionSuccess)
{
    // Legacy-aligned: a [x]-only node still fuses in the const branch (the original pruning loop
    // keeps [x] and the size const comes from the static output shape).
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4},
                                  /*hasRoi=*/false, /*hasScales=*/false, /*hasSizes=*/false);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, linear6DFusionSuccess)
{
    // Legacy-aligned: the original has no rank gate for linear mode; a 6-dim static output goes
    // through the const branch into ResizeBilinearV2 with the NCDHW spatial triple as the size
    // (whether the kernel accepts it is a later, kernel-stage concern, not the pass's).
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCDHW, {1, 3, 2, 8, 8, 8},
                                  {1, 3, 2, 4, 4, 4});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, controlEdgePreservedSuccess)
{
    // A control edge into the Resize node must be kept on the fused target node: the original
    // pass re-links the in-control anchors to the new node manually, and the framework replaces
    // the subgraph keeping the control dependencies.
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4},
                                  /*hasRoi=*/true, /*hasScales=*/true, /*hasSizes=*/true,
                                  /*addControlEdge=*/true);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeBilinearType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));

    GNode targetNode;
    ASSERT_TRUE(GetNodeByType(graph, kResizeBilinearType, targetNode));
    // The framework keeps the control dependencies by routing every control edge that entered the
    // replaced node through an inserted NoOp node which then controls the replacement nodes. Both
    // original edges (from the roi and scales input producers) must still exist in the fused
    // graph, and the target node must be under control (directly or through the NoOp).
    size_t preservedCtrlEdges = 0;
    for (auto node : graph->GetAllNodes()) {
        for (auto ctrlSrc : node.GetInControlNodes()) {
            AscendString srcName;
            (void)ctrlSrc->GetName(srcName);
            const std::string src(srcName.GetString());
            if (src == "roi" || src == "scales") {
                preservedCtrlEdges++;
            }
        }
    }
    EXPECT_EQ(preservedCtrlEdges, 2);
    EXPECT_GE(targetNode.GetInControlNodes().size(), 1);
}

// The 4-dim-NCDHW form is malformed input whose legacy behavior is an out-of-bounds
// dims[4] read (unreproducible UB, documented as non-alignable item R12): the CI UT gate
// runs under ASan, where that read aborts the whole test process, so the case is
// deliberately not covered. Well-formed 5-D/6-D NCDHW coverage is kept in the other
// NCDHW cases above.

TEST_F(ResizeFusionPassTest, nearestDynamicOnnxTwoInputsFusionSuccess)
{
    // [x, sizes] form with a dynamic onnx shape: the Slice branch must work for partial inputs too.
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_NHWC, {-1, 8, 8, 3}, {-1, 4, 4, 3},
                                  /*hasRoi=*/false, /*hasScales=*/false, /*hasSizes=*/true);
    ASSERT_NE(graph, nullptr);
    SetStringAttrToResizeNode(graph, "resize_original_type", "onnx_resize");

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeNearestType));
    EXPECT_TRUE(HasNodeType(graph, kSliceType));
    EXPECT_TRUE(CheckSliceParamsDtype(graph));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
    EXPECT_TRUE(CheckTargetSizeInputDtype(graph, kResizeNearestType));
}

TEST_F(ResizeFusionPassTest, linear6DDynamicOnnxFusionSuccess)
{
    // Legacy-aligned on the branch decision: the original has no rank gate, the dynamic form
    // goes through the Slice branch. For the target type the original would SetType("") (an
    // empty unusable type - its own UB); the migrated slice branch produces ResizeTrilinear for
    // linear ranks other than 4. Unreachable in valid graphs (the infershape only allows 4/5D);
    // recorded in the diff findings as a residual dead-form difference.
    auto graph = BuildResizeGraph("linear", "half_pixel", DT_FLOAT, FORMAT_NCDHW, {-1, 3, 2, 8, 8, 8},
                                  {-1, 3, 2, 4, 4, 4});
    ASSERT_NE(graph, nullptr);
    SetStringAttrToResizeNode(graph, "resize_original_type", "onnx_resize");

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNodeType(graph, kResizeTrilinearType));
    EXPECT_TRUE(HasNodeType(graph, kSliceType));
    EXPECT_FALSE(HasNodeType(graph, kResizeType));
}

TEST_F(ResizeFusionPassTest, modeAttrMissingFail)
{
    // Legacy-aligned: the original MatchModeName fails on a missing mode attr and returns
    // PARAM_INVALID, failing the whole build DURING the pass. The migrated pass lets the node
    // through MeetRequirements and fails in Replacement likewise (the failure point, not just
    // the terminal state, is kept identical).
    auto graph = BuildResizeGraph("nearest", "half_pixel", DT_FLOAT, FORMAT_NCHW, {1, 3, 8, 8}, {1, 3, 4, 4}, true,
                                  true, true, false, /*setModeAttr=*/false);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    ResizeFusionPass pass;
    Status status = pass.Run(graph, passContext);

    printf("modeAttrMissing actual status=%d\n", static_cast<int>(status));
    EXPECT_NE(status, SUCCESS);
}
