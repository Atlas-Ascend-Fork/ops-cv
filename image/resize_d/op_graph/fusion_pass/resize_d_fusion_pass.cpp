/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_d_fusion_pass.cpp
 * \brief resize_d fusion pass (ResizeD --> ResizeLinear / ResizeBicubicV2)
 *
 * Pattern:
 *        x
 *        |
 *      ResizeD
 *        |
 *        y
 *
 * after:
 *    mode is linear (x is 4-dim [N, C, 1, W]):
 *        x                x
 *        |                |
 *      ResizeD         Squeeze(axis=2)
 *        |                |
 *        y      ==>    ResizeLinear
 *                         |
 *                     Unsqueeze(axes=2)
 *                         |
 *                         y
 *
 *    mode is cubic and data_format is NCHW:
 *        x                x
 *        |                |
 *      ResizeD     ResizeBicubicV2
 *        |       ==>      |
 *        y                y
 *
 *    mode is cubic and data_format is HWNC:
 *        x                        x
 *        |                        |
 *      ResizeD    ==>    Transpose(perm=[2, 3, 0, 1])
 *        |                        |
 *        y                ResizeBicubicV2
 *                                 |
 *                                 y
 */

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "es_cv_ops.h"
#include "es_math_ops.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"

#include "resize_d_fusion_pass.h"

using namespace ge;
using namespace fusion;

// rtGetSocSpec lives in libruntime (declared in experiment/runtime/runtime/base.h, which the
// default open build include set does not carry); the prototype is declared verbatim here.
extern "C" int rtGetSocSpec(const char* label, const char* key, char* val, const unsigned int maxLen);

namespace ops {

namespace {
const std::string kPassName = "ResizeDFusionPass";
constexpr int64_t kCaptureIdxResizeDNode = 0;
// input index of ResizeD: x=0
constexpr int32_t kResizeDInputIdxX = 0;
constexpr size_t kDimNum3D = 3;
constexpr size_t kDimNum4D = 4;
// index of the dims in the 4-dim shape
constexpr size_t kIdxDimC = 1;
constexpr size_t kIdxDimH = 2;
constexpr size_t kIdxDimW = 3;
constexpr size_t kIdxDimN = 0;
// axis of the H dim squeezed/unsqueezed between [N, C, 1, W] and [N, C, W]
constexpr int64_t kSqueezeAxis = 2;
// The original pass sets the Squeeze/Unsqueeze axes attr to {-2} (the negative index of the H
// dim, i.e. the same dim as index 2 for the 4-dim [N, C, 1, W]). Keep the exact legacy value to
// avoid any behavioral difference in the kernel.
constexpr int64_t kLegacySqueezeAxesValue = -2;
constexpr int64_t kPermutationNum = 4;
// HWNC -> NCHW: output dim i comes from the input dim perm[i]
const std::vector<int64_t> kHwnc2NchwPermutation = {2, 3, 0, 1};
constexpr size_t kSizesNumLinear = 1;
constexpr size_t kSizesNumCubic = 2;
const std::string kModeLinear = "linear";
const std::string kModeCubic = "cubic";
const std::string kDataFormatNchw = "NCHW";
const std::string kDataFormatHwnc = "HWNC";
const std::string kCtmAlignCorners = "align_corners";
const int64_t kUnknownRankDim = -2;
} // namespace

/*
 * Replicates the original two-layer gate with the original input mechanism:
 * (a) the code gate IsSupportResizeDSoc() -> IsRegBase(): NpuArch ∈ {DAV_3510, DAV_5102}, the
 *     arch coming from rtGetSocSpec exactly as the legacy GetNpuArch does;
 * (b) the registration whitelist support_fusion_pass.json enables this pass on 3510 only (no
 *     5102 entry), so the two layers intersect to {DAV_3510} = Ascend950. The new framework has
 *     no registration-side whitelist, so the intersection is expressed in code; on 5102 the
 *     legacy pass is never even scheduled, which cannot be replicated (the plugin loads
 *     everywhere), but the functional outcome (no fusion) is identical.
 */
static NpuArch GetNpuArch()
{
    char npuArchValStr[64] = {0};
    if (rtGetSocSpec("version", "NpuArch", npuArchValStr, sizeof(npuArchValStr)) != 0) {
        OP_LOGW(kPassName.c_str(), "call rtGetSocSpec failed.");
        return NpuArch::DAV_RESV;
    }
    return static_cast<NpuArch>(std::strtoul(npuArchValStr, nullptr, 10));
}

static bool IsSupportResizeDSoc()
{
    static const std::set<NpuArch> regBaseArchList = {NpuArch::DAV_3510, NpuArch::DAV_5102};
    const NpuArch npuArch = GetNpuArch();
    if (regBaseArchList.count(npuArch) == 0U) {
        OP_LOGI(kPassName.c_str(), "only regbase arch need fusion pass");
        return false;
    }
    if (npuArch != NpuArch::DAV_3510) {
        OP_LOGI(kPassName.c_str(), "npu arch is not in the fusion pass whitelist");
        return false;
    }
    return true;
}

static bool IsUnknownRankShape(const std::vector<int64_t>& dims)
{
    return dims.size() == 1 && dims[0] == kUnknownRankDim;
}

static bool IsSupportedDtype(const DataType& dtype)
{
    return dtype == DT_FLOAT16 || dtype == DT_FLOAT || dtype == DT_BF16;
}

// The original passes data_format from the check phase to the rewrite phase through this
// file-scope static: CheckFusionSupport writes it and FusionNodeResizeDWithCubic reads it
// (canndev resize_d_fusion_pass.cc:67/188/345). Replicated here: MeetRequirements writes it,
// Replacement reads it. The framework's Run processes matches one by one with the
// Requires-then-Replace adjacency per match (GE interface reference, PatternFusionPass::Run),
// so the write->read order per node is preserved exactly as in the original's single Fusion()
// call stack.
static std::string gAttrDataFormat = "";

/*
 * Get the interpolation mode attr of ResizeD. Only linear/cubic are legal.
 */
static bool GetModeName(const GNode& resizeDNode, std::string& modeName)
{
    AscendString modeStr;
    if (resizeDNode.GetAttr("mode", modeStr) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get attr mode failed.");
        return false;
    }
    modeName = std::string(modeStr.GetString());
    if (modeName != kModeLinear && modeName != kModeCubic) {
        OP_LOGE(kPassName.c_str(), "Mode name is not linear or cubic.");
        return false;
    }
    return true;
}

static bool GetDataFormat(const GNode& resizeDNode, std::string& dataFormat)
{
    AscendString formatStr;
    if (resizeDNode.GetAttr("data_format", formatStr) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get attr data_format failed.");
        return false;
    }
    dataFormat = std::string(formatStr.GetString());
    return true;
}

static bool GetSizesAttr(const GNode& resizeDNode, std::vector<int64_t>& sizes)
{
    if (resizeDNode.GetAttr("sizes", sizes) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get attr sizes failed.");
        return false;
    }
    return true;
}

static void GetScalesAttr(const GNode& resizeDNode, const size_t defaultNum, std::vector<float>& scales)
{
    scales.assign(defaultNum, 0.0f);
    if (resizeDNode.GetAttr("scales", scales) != GRAPH_SUCCESS) {
        OP_LOGI(kPassName.c_str(), "Get attr scales failed, set to default.");
    }
}

/*
 * Checks for the linear mode, as the original CheckResizeLinearSupport:
 * x dtype in (float16, float32, bfloat16); x is 4-dim with non-zero C/H/W when rank is known;
 * sizes has 1 positive element; y dtype equals x dtype; y is 4-dim when rank is known;
 * scales has 1 element.
 */
static bool CheckLinearSupport(const GNode& resizeDNode, const TensorDesc& xDesc, const TensorDesc& yDesc)
{
    if (!IsSupportedDtype(xDesc.GetDataType())) {
        OP_LOGI(kPassName.c_str(), "Linear mode x data type is not float32, float16 or bfloat16.");
        return false;
    }
    const std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
    if (!IsUnknownRankShape(xDims)) {
        if (xDims.size() != kDimNum4D) {
            OP_LOGI(kPassName.c_str(), "Linear mode x shape is not 4D.");
            return false;
        }
        if (xDims[kIdxDimC] == 0 || xDims[kIdxDimH] == 0 || xDims[kIdxDimW] == 0) {
            OP_LOGI(kPassName.c_str(), "The second or third dim of x is 0.");
            return false;
        }
    }
    std::vector<int64_t> sizes;
    if (!GetSizesAttr(resizeDNode, sizes) || sizes.size() != kSizesNumLinear) {
        OP_LOGI(kPassName.c_str(), "The size of sizes is not 1.");
        return false;
    }
    if (sizes[0] <= 0) {
        OP_LOGI(kPassName.c_str(), "The element of sizes is not greater than 0.");
        return false;
    }
    if (yDesc.GetDataType() != xDesc.GetDataType()) {
        OP_LOGI(kPassName.c_str(), "Y data type is not same as x data type.");
        return false;
    }
    const std::vector<int64_t> yDims = yDesc.GetShape().GetDims();
    if (!IsUnknownRankShape(yDims) && yDims.size() != kDimNum4D) {
        OP_LOGI(kPassName.c_str(), "Linear mode y shape is not 4D.");
        return false;
    }
    std::vector<float> scales;
    GetScalesAttr(resizeDNode, kSizesNumLinear, scales);
    if (scales.size() != kSizesNumLinear) {
        OP_LOGI(kPassName.c_str(), "The size of scales is not 1.");
        return false;
    }
    return true;
}

/*
 * Checks for the cubic mode, as the original CheckResizeBicubicV2Support:
 * x dtype in (float16, float32, bfloat16); x format is NCHW; x is 4-dim when rank is known;
 * sizes has 2 positive elements; y dtype/format equal x dtype/format; y is 4-dim when rank
 * is known; scales has 2 elements.
 */
static bool CheckCubicSupport(const GNode& resizeDNode, const TensorDesc& xDesc, const TensorDesc& yDesc)
{
    if (!IsSupportedDtype(xDesc.GetDataType())) {
        OP_LOGI(kPassName.c_str(), "Cubic mode x data type is not float32, float16 or bfloat16.");
        return false;
    }
    if (xDesc.GetFormat() != FORMAT_NCHW) {
        OP_LOGI(kPassName.c_str(), "Cubic mode x format is not NCHW.");
        return false;
    }
    const std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
    if (!IsUnknownRankShape(xDims) && xDims.size() != kDimNum4D) {
        OP_LOGI(kPassName.c_str(), "Cubic mode x shape is not 4D.");
        return false;
    }
    std::vector<int64_t> sizes;
    if (!GetSizesAttr(resizeDNode, sizes) || sizes.size() != kSizesNumCubic) {
        OP_LOGI(kPassName.c_str(), "The size of sizes is not 2.");
        return false;
    }
    if (sizes[0] <= 0 || sizes[1] <= 0) {
        OP_LOGI(kPassName.c_str(), "The element of sizes is not greater than 0.");
        return false;
    }
    if (yDesc.GetDataType() != xDesc.GetDataType()) {
        OP_LOGI(kPassName.c_str(), "Y data type is not same as x data type.");
        return false;
    }
    if (yDesc.GetFormat() != xDesc.GetFormat()) {
        OP_LOGI(kPassName.c_str(), "Y format is not same as x format.");
        return false;
    }
    const std::vector<int64_t> yDims = yDesc.GetShape().GetDims();
    if (!IsUnknownRankShape(yDims) && yDims.size() != kDimNum4D) {
        OP_LOGI(kPassName.c_str(), "Cubic mode y shape is not 4D.");
        return false;
    }
    std::vector<float> scales;
    GetScalesAttr(resizeDNode, kSizesNumCubic, scales);
    if (scales.size() != kSizesNumCubic) {
        OP_LOGI(kPassName.c_str(), "The size of scales is not 2.");
        return false;
    }
    return true;
}

static TensorDesc MakeSizeDesc(const int64_t elementNum, const DataType& dataType)
{
    TensorDesc sizeDesc(Shape(std::vector<int64_t>{elementNum}), FORMAT_ND, dataType);
    // The original InsertSizeTensor sets the origin format/shape explicitly, keep the same.
    sizeDesc.SetOriginFormat(FORMAT_ND);
    sizeDesc.SetOriginShape(Shape(std::vector<int64_t>{elementNum}));
    return sizeDesc;
}

std::vector<PatternUniqPtr> ResizeDFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for ResizeDFusionPass.");
    std::vector<PatternUniqPtr> patternGraphs;

    // ResizeD has no eager style API (not in the CANN-installed es_cv package), the pattern
    // node is built with CompliantNodeBuilder (as the keep_ratio pass does).
    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graphBuilder.CreateInput(0);
    auto* patternGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto resizeDNode = es::CompliantNodeBuilder(patternGraph)
                           .OpType("ResizeD")
                           .Name("pattern_resize_d")
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"sizes", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                         es::CreateFrom(std::vector<int64_t>{1})}})
                           .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*patternGraph, *x.GetProducer(), x.GetProducerOutIndex(), resizeDNode, 0) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge in pattern for ResizeD.");
        return patternGraphs;
    }
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(resizeDNode, 0);
    auto graph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    NodeIo nodeIo = {y->GetProducer(), 0};
    pattern->CaptureTensor(nodeIo);

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool ResizeDFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for ResizeDFusionPass.");

    // 1. The original soc gate (regbase arch ∩ registration whitelist), first thing in the
    //    original Fusion as well.
    if (!IsSupportResizeDSoc()) {
        return false;
    }

    // 2. Get the matched ResizeD node.
    NodeIo resizeDNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxResizeDNode, resizeDNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured resize_d node.");
        return false;
    }
    const GNode resizeDNode = resizeDNodeIo.node;

    // 3. Check the interpolation mode, only linear/cubic are supported.
    std::string modeName;
    if (!GetModeName(resizeDNode, modeName)) {
        return false;
    }

    // 4. Check data_format: linear only supports NCHW, cubic supports NCHW or HWNC. The value
    //    is written to the file-scope gAttrDataFormat here, exactly as the original
    //    CheckFusionSupport does, for the rewrite phase to read (FusionNodeResizeDWithCubic).
    std::string dataFormat;
    if (!GetDataFormat(resizeDNode, dataFormat)) {
        return false;
    }
    gAttrDataFormat = dataFormat;
    if (modeName == kModeLinear && dataFormat != kDataFormatNchw) {
        OP_LOGI(kPassName.c_str(), "Linear mode data_format is only support NCHW.");
        return false;
    }
    if (modeName == kModeCubic && dataFormat != kDataFormatNchw && dataFormat != kDataFormatHwnc) {
        OP_LOGI(kPassName.c_str(), "Cubic mode data_format is only support NCHW or HWNC.");
        return false;
    }

    TensorDesc xDesc;
    if (resizeDNode.GetInputDesc(kResizeDInputIdxX, xDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc x of resize_d node.");
        return false;
    }
    TensorDesc yDesc;
    if (resizeDNode.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get output desc of resize_d node.");
        return false;
    }

    // 5. Check dtype/shape/format/sizes/scales support according to the mode.
    const bool supportResult = (modeName == kModeLinear) ? CheckLinearSupport(resizeDNode, xDesc, yDesc) :
                                                           CheckCubicSupport(resizeDNode, xDesc, yDesc);
    if (!supportResult) {
        OP_LOGI(kPassName.c_str(), "Check fusion support not success, not fusion.");
        return false;
    }
    return true;
}

GraphUniqPtr ResizeDFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for ResizeDFusionPass.");

    // 1. Get the matched ResizeD node and its input/output descs.
    NodeIo resizeDNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxResizeDNode, resizeDNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured resize_d node.");
        return nullptr;
    }
    const GNode resizeDNode = resizeDNodeIo.node;

    std::string modeName;
    if (!GetModeName(resizeDNode, modeName)) {
        OP_LOGE(kPassName.c_str(), "Failed to get mode name in replacement.");
        return nullptr;
    }
    // The data_format comes from the check phase through the file-scope gAttrDataFormat, as the
    // original FusionNodeResizeDWithCubic reads AttrDataFormat instead of re-reading the attr.
    const std::string& dataFormat = gAttrDataFormat;
    std::vector<int64_t> sizes;
    if (!GetSizesAttr(resizeDNode, sizes)) {
        OP_LOGE(kPassName.c_str(), "Failed to get sizes attr in replacement.");
        return nullptr;
    }

    TensorDesc xDesc;
    if (resizeDNode.GetInputDesc(kResizeDInputIdxX, xDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc x of resize_d node.");
        return nullptr;
    }
    TensorDesc yDesc;
    if (resizeDNode.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get output desc of resize_d node.");
        return nullptr;
    }

    // 2. Convert coordinate_transformation_mode to align_corners, and get the scales attr.
    bool alignCorners = false;
    AscendString ctmStr;
    if (resizeDNode.GetAttr("coordinate_transformation_mode", ctmStr) == GRAPH_SUCCESS) {
        alignCorners = (std::string(ctmStr.GetString()) == kCtmAlignCorners);
    } else {
        OP_LOGI(kPassName.c_str(), "Get attr coordinate_transformation_mode failed, set to default.");
    }
    const size_t scalesNum = (modeName == kModeLinear) ? kSizesNumLinear : kSizesNumCubic;
    std::vector<float> scales;
    GetScalesAttr(resizeDNode, scalesNum, scales);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rX = replaceGraphBuilder.CreateInput(0, "x", xDesc.GetDataType(), xDesc.GetFormat(),
                                              xDesc.GetShape().GetDims());
    // Build the const size node from the attr sizes, whose dtype must be int32.
    std::vector<int32_t> sizeValue(sizes.begin(), sizes.end());
    auto sizeInput = replaceGraphBuilder.CreateConst(sizeValue, {static_cast<int64_t>(sizeValue.size())});

    es::EsTensorHolder y;
    if (modeName == kModeLinear) {
        // 3a. Linear: Squeeze(axis=2) -> ResizeLinear -> Unsqueeze(axes=2), because the target op
        //     works on the 3-dim [N, C, W] while the source op has the 4-dim [N, C, 1, W].
        //     The original has an explicit xShapeSize==3 shortcut here (FusionNodeResizeDWithLinear:
        //     "xShapeSize is 3, no need to change dims") that skips the Squeeze/Unsqueeze insertion
        //     and lets the already-retyped ResizeLinear take x directly. The branch is unreachable
        //     through any real pipeline - CheckResizeLinearSupport rejects a known-rank non-4-dim x
        //     on both sides, and a [-2] unknown rank is a 1-element vector, not 3 - but it is kept
        //     line by line for faithful migration.
        const bool xUnknownRank = IsUnknownRankShape(xDesc.GetShape().GetDims());
        const bool yUnknownRank = IsUnknownRankShape(yDesc.GetShape().GetDims());
        if (!xUnknownRank && xDesc.GetShape().GetDims().size() == kDimNum3D) {
            OP_LOGI(kPassName.c_str(), "xShapeSize is 3, no need to change dims");
            auto resizeLinearOut = es::ResizeLinear(rX, sizeInput, alignCorners, scales[0]);
            GNode resizeLinearNode = *resizeLinearOut.GetProducer();
            resizeLinearNode.UpdateInputDesc(0, xDesc);
            resizeLinearNode.UpdateInputDesc(1, MakeSizeDesc(kSizesNumLinear, DT_INT32));
            resizeLinearNode.UpdateOutputDesc(0, yDesc);
            y = resizeLinearOut;
        } else {
            //     The original sets the attr name "axes" on BOTH Squeeze and Unsqueeze - a name that
            //     does not even match the Squeeze IR declaration ("axis"); the old built-in pass
            //     framework tolerates it, but the new PatternFusionPass framework validates attr
            //     names against the IR definition during replacement recovery (RecoverIrAttrNames)
            //     and rejects the node outright, failing the whole graph build (950 A/B evidenced:
            //     aclgrphBuildModel -1 vs legacy EXEC_OK). The IR name "axis" is therefore required;
            //     the attr value keeps the legacy {-2} sentinel. Intentional divergence, see the
            //     diff findings list.
            auto* cGraph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
            auto squeezeNode = es::CompliantNodeBuilder(cGraph)
                                   .OpType("Squeeze")
                                   .Name("replacement_squeeze")
                                   .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                   .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                   .IrDefAttrs({{"axis", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                                 es::CreateFrom(std::vector<int64_t>{kLegacySqueezeAxesValue})}})
                                   .Build();
            if (es::AddEdgeAndUpdatePeerDesc(*cGraph, *rX.GetProducer(), rX.GetProducerOutIndex(), squeezeNode, 0) !=
                GRAPH_SUCCESS) {
                OP_LOGE(kPassName.c_str(), "Failed to add edge for the squeeze node.");
                return nullptr;
            }
            auto squeezeOut = es::EsTensorHolder(
                replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(squeezeNode, 0));
            squeezeNode.UpdateInputDesc(0, xDesc);
            TensorDesc squeezeDesc = xDesc;
            squeezeDesc.SetFormat(FORMAT_ND);
            squeezeDesc.SetOriginFormat(FORMAT_ND);
            if (!xUnknownRank) {
                std::vector<int64_t> squeezeDims = xDesc.GetShape().GetDims();
                squeezeDims.erase(squeezeDims.begin() + kSqueezeAxis);
                squeezeDesc.SetShape(Shape(squeezeDims));
                squeezeDesc.SetOriginShape(Shape(squeezeDims));
            }
            squeezeNode.UpdateOutputDesc(0, squeezeDesc);

            auto resizeLinearOut = es::ResizeLinear(squeezeOut, sizeInput, alignCorners, scales[0]);
            GNode resizeLinearNode = *resizeLinearOut.GetProducer();
            resizeLinearNode.UpdateInputDesc(0, squeezeDesc);
            resizeLinearNode.UpdateInputDesc(1, MakeSizeDesc(kSizesNumLinear, DT_INT32));
            TensorDesc linearYDesc = squeezeDesc;
            if (!yUnknownRank) {
                std::vector<int64_t> linearDims = yDesc.GetShape().GetDims();
                linearDims.erase(linearDims.begin() + kSqueezeAxis);
                linearYDesc.SetShape(Shape(linearDims));
                linearYDesc.SetOriginShape(Shape(linearDims));
            }
            resizeLinearNode.UpdateOutputDesc(0, linearYDesc);

            auto unsqueezeOut = es::Unsqueeze(resizeLinearOut, std::vector<int64_t>{kLegacySqueezeAxesValue});
            GNode unsqueezeNode = *unsqueezeOut.GetProducer();
            unsqueezeNode.UpdateInputDesc(0, linearYDesc);
            // The original InsertUnSqueezeNode4Output hardcodes FORMAT_NCHW on the Unsqueeze output
            // desc (format and origin format alike) instead of reusing the source y desc format;
            // the shape is the re-inserted [N, C, 1, W] (or the [-2] sentinel for unknown rank).
            TensorDesc unsqueezeOutDesc(Shape(yDesc.GetShape().GetDims()), FORMAT_NCHW, yDesc.GetDataType());
            unsqueezeOutDesc.SetOriginFormat(FORMAT_NCHW);
            unsqueezeOutDesc.SetOriginShape(yDesc.GetShape());
            unsqueezeNode.UpdateOutputDesc(0, unsqueezeOutDesc);
            y = unsqueezeOut;
        }
    } else {
        // 3b. Cubic: ResizeBicubicV2 directly for NCHW, or with a Transpose(perm=[2, 3, 0, 1])
        //     before it for HWNC (HWNC -> NCHW).
        es::EsTensorHolder targetInput = rX;
        TensorDesc targetXDesc = xDesc;
        if (dataFormat == kDataFormatHwnc) {
            auto permutationConst = replaceGraphBuilder.CreateConst(kHwnc2NchwPermutation,
                                                                    std::vector<int64_t>{kPermutationNum});
            auto transposeOut = es::Transpose(rX, permutationConst);
            GNode transposeNode = *transposeOut.GetProducer();
            transposeNode.UpdateInputDesc(0, xDesc);
            transposeNode.UpdateInputDesc(1, MakeSizeDesc(kPermutationNum, DT_INT64));
            TensorDesc transposeDesc = xDesc;
            if (!IsUnknownRankShape(xDesc.GetShape().GetDims())) {
                const std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
                std::vector<int64_t> transposeDims;
                for (const int64_t permIdx : kHwnc2NchwPermutation) {
                    transposeDims.push_back(xDims[permIdx]);
                }
                transposeDesc.SetShape(Shape(transposeDims));
                transposeDesc.SetOriginShape(Shape(transposeDims));
            }
            transposeNode.UpdateOutputDesc(0, transposeDesc);
            targetInput = transposeOut;
            targetXDesc = transposeDesc;
        }

        auto bicubicOut = es::ResizeBicubicV2(targetInput, sizeInput, alignCorners, scales);
        GNode bicubicNode = *bicubicOut.GetProducer();
        bicubicNode.UpdateInputDesc(0, targetXDesc);
        bicubicNode.UpdateInputDesc(1, MakeSizeDesc(kSizesNumCubic, DT_INT32));
        bicubicNode.UpdateOutputDesc(0, yDesc);
        y = bicubicOut;
    }

    return replaceGraphBuilder.BuildAndReset({y});
}

REG_FUSION_PASS(ResizeDFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops
