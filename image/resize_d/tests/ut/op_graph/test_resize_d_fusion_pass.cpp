/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
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

// The platform info members are injected directly (instead of calling SetOptionalCompilationInfo)
// with the private members exposed, both ways are verified to work under the CXX11 ABI=0 build.
#define private public
#include "platform/platform_info.h"
#undef private

#include "../../../op_graph/fusion_pass/resize_d_fusion_pass.h"
#include "soc_spec_mock.h"

using namespace std;
using namespace ge;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "ResizeDFusionPass";
const std::string kResizeDType = "ResizeD";
const std::string kResizeLinearType = "ResizeLinear";
const std::string kResizeBicubicType = "ResizeBicubicV2";
const std::string kSqueezeType = "Squeeze";
const std::string kUnsqueezeType = "Unsqueeze";
const std::string kTransposeType = "Transpose";
const std::string kRegbaseSoc = "Ascend950";
const std::string kRegbaseMcpSoc = "MC62CM12A";
const std::string kAscend910BSoc = "Ascend910B";
const std::string kNonRegbaseSoc = "Ascend910_93";
} // namespace

/*
 * Inject the platform info into the PlatformInfoManager singleton. The pass reads it through
 * GetPlatformInfoWithOutSocVersion, which is only possible when the test binary is compiled with
 * the same CXX11 ABI as the CANN libraries (see run_ut.sh, -D_GLIBCXX_USE_CXX11_ABI=0).
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

class ResizeDFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetSocVersion(kRegbaseSoc); }

    static void TearDownTestCase() {}
};

/*
 * Build a graph with a single ResizeD node:
 *        x
 *        |
 *      ResizeD
 *        |
 *        y
 */
static std::shared_ptr<Graph> BuildResizeDGraph(const std::string& mode, const std::string& dataFormat,
                                                const std::string& ctmMode, DataType xDtype, Format xFormat,
                                                const std::vector<int64_t>& xDims, const std::vector<int64_t>& yDims,
                                                const std::vector<int64_t>& sizes, const std::vector<float>& scales,
                                                bool setAttrs = true, bool addControlEdge = false)
{
    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", xDtype, xFormat, xDims);
    auto y = es::ResizeD(x, sizes, scales, std::vector<int64_t>{1}, ctmMode.c_str(), -0.75, 0, 0.0, mode.c_str(),
                         "round_prefer_floor", dataFormat.c_str());

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(xDtype);
    xDesc.SetShape(Shape(xDims));
    xDesc.SetOriginShape(Shape(xDims));
    xDesc.SetFormat(xFormat);
    xDesc.SetOriginFormat(xFormat);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc yDesc;
    y.GetProducer()->GetOutputDesc(0, yDesc);
    yDesc.SetDataType(xDtype);
    yDesc.SetShape(Shape(yDims));
    yDesc.SetOriginShape(Shape(yDims));
    yDesc.SetFormat(xFormat);
    yDesc.SetOriginFormat(xFormat);
    y.GetProducer()->UpdateOutputDesc(0, yDesc);

    // Sync the descs on the ResizeD node itself, just like what infershape does in a real graph.
    GNode resizeDNode = *y.GetProducer();
    resizeDNode.UpdateInputDesc(0, xDesc);
    resizeDNode.UpdateOutputDesc(0, yDesc);

    // The eager style API skips attrs equal to their default values (e.g. data_format="NCHW"),
    // while a real graph always carries them. Set them explicitly.
    if (setAttrs) {
        AscendString modeAttr(mode.c_str());
        resizeDNode.SetAttr(AscendString("mode"), modeAttr);
        AscendString dataFormatAttr(dataFormat.c_str());
        resizeDNode.SetAttr(AscendString("data_format"), dataFormatAttr);
        AscendString ctmAttr(ctmMode.c_str());
        resizeDNode.SetAttr(AscendString("coordinate_transformation_mode"), ctmAttr);
        std::vector<int64_t> sizesValue = sizes;
        resizeDNode.SetAttr(AscendString("sizes"), sizesValue);
        std::vector<float> scalesValue = scales;
        resizeDNode.SetAttr(AscendString("scales"), scalesValue);
    }

    // Optionally add a control edge into the ResizeD node (from the x input producer) to verify
    // that the framework keeps the control dependencies on the fused target chain.
    if (addControlEdge) {
        Graph* cGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
        (void)cGraph->AddControlEdge(*x.GetProducer(), resizeDNode);
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

static bool FindNodeByType(const std::shared_ptr<Graph>& graph, const std::string& type, GNode& foundNode)
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

static bool GetBoolAttrFromNode(const std::shared_ptr<Graph>& graph, const std::string& nodeType,
                                const std::string& attrKey, bool& attrValue)
{
    GNode node;
    if (!FindNodeByType(graph, nodeType, node)) {
        return false;
    }
    return node.GetAttr(AscendString(attrKey.c_str()), attrValue) == GRAPH_SUCCESS;
}

static bool RunPass(const std::shared_ptr<Graph>& graph, Status& runStatus)
{
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    GraphPtr mutableGraph = graph;
    runStatus = pass.Run(mutableGraph, passContext);
    return runStatus == SUCCESS;
}

TEST_F(ResizeDFusionPassTest, patternTest)
{
    ResizeDFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 1);
}

TEST_F(ResizeDFusionPassTest, fusionLinearSuccess)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    Status runStatus;
    ASSERT_TRUE(RunPass(graph, runStatus));

    // Squeeze(axis=2) -> ResizeLinear -> Unsqueeze(axes=2) chain is built.
    ASSERT_TRUE(HasNodeType(graph, kSqueezeType));
    ASSERT_TRUE(HasNodeType(graph, kResizeLinearType));
    ASSERT_TRUE(HasNodeType(graph, kUnsqueezeType));
    EXPECT_FALSE(HasNodeType(graph, kResizeDType));

    GNode resizeLinearNode;
    ASSERT_TRUE(FindNodeByType(graph, kResizeLinearType, resizeLinearNode));
    EXPECT_EQ(resizeLinearNode.GetInputsSize(), 2);
    // ResizeLinear works on the 3-dim [N, C, W'] shape.
    TensorDesc linearYDesc;
    resizeLinearNode.GetOutputDesc(0, linearYDesc);
    EXPECT_EQ(linearYDesc.GetShape().GetDims(), std::vector<int64_t>({1, 1, 3}));
    EXPECT_EQ(linearYDesc.GetDataType(), DT_FLOAT);

    // Unsqueeze restores the 4-dim [N, C, 1, W'] output shape.
    GNode unsqueezeNode;
    ASSERT_TRUE(FindNodeByType(graph, kUnsqueezeType, unsqueezeNode));
    TensorDesc yDesc;
    unsqueezeNode.GetOutputDesc(0, yDesc);
    EXPECT_EQ(yDesc.GetShape().GetDims(), std::vector<int64_t>({1, 1, 1, 3}));
    EXPECT_EQ(yDesc.GetDataType(), DT_FLOAT);

    // scales attr is converted to the scale attr.
    float scale = 0.0f;
    ASSERT_TRUE(resizeLinearNode.GetAttr(AscendString("scale"), scale) == GRAPH_SUCCESS);
    EXPECT_EQ(scale, 1.5f);
    // Attr not set means the default value false when the eager style builder omits default attrs.
    bool alignCorners = true;
    if (resizeLinearNode.GetAttr(AscendString("align_corners"), alignCorners) != GRAPH_SUCCESS) {
        alignCorners = false;
    }
    EXPECT_FALSE(alignCorners);
}

TEST_F(ResizeDFusionPassTest, fusionLinearAlignCornersSuccess)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "align_corners", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{2, 4, 1, 8}, std::vector<int64_t>{2, 4, 1, 16},
                                   std::vector<int64_t>{16}, std::vector<float>{2.0f});
    Status runStatus;
    ASSERT_TRUE(RunPass(graph, runStatus));

    bool alignCorners = false;
    ASSERT_TRUE(GetBoolAttrFromNode(graph, kResizeLinearType, "align_corners", alignCorners));
    EXPECT_TRUE(alignCorners);

    GNode unsqueezeNode;
    ASSERT_TRUE(FindNodeByType(graph, kUnsqueezeType, unsqueezeNode));
    TensorDesc yDesc;
    unsqueezeNode.GetOutputDesc(0, yDesc);
    EXPECT_EQ(yDesc.GetShape().GetDims(), std::vector<int64_t>({2, 4, 1, 16}));
}

TEST_F(ResizeDFusionPassTest, fusionLinearUnknownRankSuccess)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT16, FORMAT_NCHW, std::vector<int64_t>{-2},
                                   std::vector<int64_t>{-2}, std::vector<int64_t>{3}, std::vector<float>{1.5f});
    Status runStatus;
    ASSERT_TRUE(RunPass(graph, runStatus));

    ASSERT_TRUE(HasNodeType(graph, kSqueezeType));
    ASSERT_TRUE(HasNodeType(graph, kResizeLinearType));
    ASSERT_TRUE(HasNodeType(graph, kUnsqueezeType));
    EXPECT_FALSE(HasNodeType(graph, kResizeDType));
}

TEST_F(ResizeDFusionPassTest, fusionCubicNchwSuccess)
{
    auto graph = BuildResizeDGraph("cubic", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 2, 2}, std::vector<int64_t>{1, 1, 3, 3},
                                   std::vector<int64_t>{3, 3}, std::vector<float>{1.5f, 1.5f});
    Status runStatus;
    ASSERT_TRUE(RunPass(graph, runStatus));

    GNode bicubicNode;
    ASSERT_TRUE(FindNodeByType(graph, kResizeBicubicType, bicubicNode));
    EXPECT_EQ(bicubicNode.GetInputsSize(), 2);
    // No transpose is inserted when data_format is NCHW.
    EXPECT_FALSE(HasNodeType(graph, kTransposeType));
    EXPECT_FALSE(HasNodeType(graph, kResizeDType));

    TensorDesc yDesc;
    bicubicNode.GetOutputDesc(0, yDesc);
    EXPECT_EQ(yDesc.GetShape().GetDims(), std::vector<int64_t>({1, 1, 3, 3}));
    EXPECT_EQ(yDesc.GetDataType(), DT_FLOAT);

    // The scales attr is kept for the cubic mode.
    std::vector<float> scalesAttr;
    ASSERT_TRUE(bicubicNode.GetAttr(AscendString("scales"), scalesAttr) == GRAPH_SUCCESS);
    EXPECT_EQ(scalesAttr, std::vector<float>({1.5f, 1.5f}));
}

TEST_F(ResizeDFusionPassTest, fusionCubicHwncSuccess)
{
    // x is logically [H, W, N, C] with the desc format NCHW.
    auto graph = BuildResizeDGraph("cubic", "HWNC", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 2, 2}, std::vector<int64_t>{2, 2, 3, 3},
                                   std::vector<int64_t>{3, 3}, std::vector<float>{1.5f, 1.5f});
    Status runStatus;
    ASSERT_TRUE(RunPass(graph, runStatus));

    // Transpose(perm=[2, 3, 0, 1]) is inserted before ResizeBicubicV2 (HWNC -> NCHW).
    GNode transposeNode;
    ASSERT_TRUE(FindNodeByType(graph, kTransposeType, transposeNode));
    EXPECT_EQ(transposeNode.GetInputsSize(), 2);
    TensorDesc transposeYDesc;
    transposeNode.GetOutputDesc(0, transposeYDesc);
    EXPECT_EQ(transposeYDesc.GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1}));
    EXPECT_EQ(transposeYDesc.GetDataType(), DT_FLOAT);

    GNode bicubicNode;
    ASSERT_TRUE(FindNodeByType(graph, kResizeBicubicType, bicubicNode));
    EXPECT_EQ(bicubicNode.GetInputsSize(), 2);
    TensorDesc yDesc;
    bicubicNode.GetOutputDesc(0, yDesc);
    EXPECT_EQ(yDesc.GetShape().GetDims(), std::vector<int64_t>({2, 2, 3, 3}));
    EXPECT_FALSE(HasNodeType(graph, kResizeDType));
}

TEST_F(ResizeDFusionPassTest, mcpPlatformNotChangedFail)
{
    // The original registration whitelist in canndev only enables ResizeDFusionPass on arch 3510
    // (Ascend950), so on MC62CM12A the pass must not change the graph.
    SetSocVersion(kRegbaseMcpSoc);
    auto graph = BuildResizeDGraph("cubic", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 2, 2}, std::vector<int64_t>{1, 1, 3, 3},
                                   std::vector<int64_t>{3, 3}, std::vector<float>{1.5f, 1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    ASSERT_TRUE(HasNodeType(graph, kResizeDType));
    EXPECT_FALSE(HasNodeType(graph, kResizeBicubicType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeDFusionPassTest, fusion910BNotChangedFail)
{
    // The original pass in canndev gates on IsRegBase() (fusion_common_utils.cc, DAV_3510/DAV_5102
    // only), so on the standard arch Ascend910B the pass runs but keeps the graph unchanged.
    SetSocVersion(kAscend910BSoc);
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    ASSERT_TRUE(HasNodeType(graph, kResizeDType));
    EXPECT_FALSE(HasNodeType(graph, kResizeLinearType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeDFusionPassTest, unsupportedPlatformFail)
{
    // Ascend910_93 is a standard (non-regbase) arch, same as Ascend910B the pass keeps the graph
    // unchanged, consistent with the original IsRegBase() gate.
    SetSocVersion(kNonRegbaseSoc);
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    ASSERT_TRUE(HasNodeType(graph, kResizeDType));
    SetSocVersion(kRegbaseSoc);
}

TEST_F(ResizeDFusionPassTest, unsupportedModeValueFail)
{
    auto graph = BuildResizeDGraph("nearest", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, dataFormatMissingFail)
{
    // The eager style builder omits attrs equal to their default values, so a graph built without
    // the explicit data_format attr simulates the attr missing case.
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f}, false);
    bool hasDataFormat = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == kResizeDType) {
            AscendString dataFormat;
            hasDataFormat = node.GetAttr(AscendString("data_format"), dataFormat) == GRAPH_SUCCESS;
        }
    }
    ASSERT_FALSE(hasDataFormat);
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, linearDataFormatFail)
{
    auto graph = BuildResizeDGraph("linear", "HWNC", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, cubicDataFormatFail)
{
    auto graph = BuildResizeDGraph("cubic", "NHWC", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 2, 2}, std::vector<int64_t>{1, 1, 3, 3},
                                   std::vector<int64_t>{3, 3}, std::vector<float>{1.5f, 1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, unsupportedDtypeFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_INT32, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, linearSizesNumFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3, 3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, cubicSizesNumFail)
{
    auto graph = BuildResizeDGraph("cubic", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 2, 2}, std::vector<int64_t>{1, 1, 3, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f, 1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, sizesNotPositiveFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{0}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, yDtypeMismatchFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    // Overwrite the y dtype to make it differ from x.
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && std::string(nodeType.GetString()) == kResizeDType) {
            TensorDesc yDesc;
            node.GetOutputDesc(0, yDesc);
            yDesc.SetDataType(DT_FLOAT16);
            node.UpdateOutputDesc(0, yDesc);
        }
    }
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, scalesNumFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f, 1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, xNot4DFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW, std::vector<int64_t>{1, 2, 3},
                                   std::vector<int64_t>{1, 2, 4}, std::vector<int64_t>{4}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, linearDimZeroFail)
{
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 0, 2}, std::vector<int64_t>{1, 1, 0, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, cubicXFormatFail)
{
    auto graph = BuildResizeDGraph("cubic", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NHWC,
                                   std::vector<int64_t>{1, 1, 2, 2}, std::vector<int64_t>{1, 1, 3, 3},
                                   std::vector<int64_t>{3, 3}, std::vector<float>{1.5f, 1.5f});
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
}

TEST_F(ResizeDFusionPassTest, controlEdgePreservedSuccess)
{
    // A control edge into the ResizeD node must be kept on the output node of the fused chain
    // (the Unsqueeze node): the original pass modifies the node in place (SetType) which keeps
    // the control anchors naturally, and the framework replaces the subgraph keeping them too.
    auto graph = BuildResizeDGraph("linear", "NCHW", "half_pixel", DT_FLOAT, FORMAT_NCHW,
                                   std::vector<int64_t>{1, 1, 1, 2}, std::vector<int64_t>{1, 1, 1, 3},
                                   std::vector<int64_t>{3}, std::vector<float>{1.5f}, true,
                                   /*addControlEdge=*/true);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    ResizeDFusionPass pass;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_TRUE(runStatus == SUCCESS || runStatus == GRAPH_NOT_CHANGED);
    ASSERT_TRUE(HasNodeType(graph, kResizeLinearType));

    GNode unsqueezeNode;
    ASSERT_TRUE(FindNodeByType(graph, kUnsqueezeType, unsqueezeNode));
    // The control dependency from the x input producer must still exist in the fused graph (the
    // framework routes it through an inserted NoOp which controls the replacement nodes), and the
    // output node of the chain must be under control.
    size_t preservedCtrlEdges = 0;
    for (auto node : graph->GetAllNodes()) {
        for (auto ctrlSrc : node.GetInControlNodes()) {
            AscendString srcName;
            (void)ctrlSrc->GetName(srcName);
            if (std::string(srcName.GetString()) == "x") {
                preservedCtrlEdges++;
            }
        }
    }
    EXPECT_EQ(preservedCtrlEdges, 1);
    EXPECT_GE(unsqueezeNode.GetInControlNodes().size(), 1);
}
