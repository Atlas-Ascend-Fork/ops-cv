/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file keep_ratio_resize_bilinear_fusion_pass.cpp
 * \brief keep_ratio_resize_bilinear fusion pass (KeepRatioResizeBilinear --> ResizeBilinearV2)
 *
 * Pattern:
 *        images
 *           |
 *   KeepRatioResizeBilinear
 *           |
 *           y
 *
 * after:
 *        images                images
 *           |                     |
 *   KeepRatioResizeBilinear    ResizeBilinearV2
 *           |        ==>          |
 *           y               Const size [newH, newW]
 *                              |
 *                              y
 *
 * The target size is computed with the same keep-ratio scaling as the original pass: scale the
 * shorter side up to min_dimension and the longer side up to max_dimension, then pick the
 * min-scale candidate unless its longer side exceeds max_dimension.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "es_cv_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"

#include "keep_ratio_resize_bilinear_fusion_pass.h"

using namespace ge;
using namespace fusion;

// rtGetSocSpec lives in libruntime (declared in experiment/runtime/runtime/base.h, which the
// default open build include set does not carry); the prototype is declared verbatim here.
extern "C" int rtGetSocSpec(const char* label, const char* key, char* val, const unsigned int maxLen);

namespace ops {

namespace {
const std::string kPassName = "KeepRatioResizeBilinearFusion";
constexpr int64_t kCaptureIdxKeepRatioNode = 0;
// input index of KeepRatioResizeBilinear: images=0
constexpr int32_t kKeepRatioInputIdxImages = 0;
constexpr size_t kDimNum4D = 4;
// index of H/W in the 4-dim images shape: NHWC -> {1, 2}, otherwise (NCHW) -> {2, 3}
constexpr size_t kIdxDimH = 1;
constexpr size_t kIdxDimW = 2;
constexpr size_t kNchwIdxDimH = 2;
constexpr size_t kNchwIdxDimW = 3;
// element num of the ResizeBilinearV2 size input
constexpr int64_t kSizeElementNum = 2;
const char* kKeepRatioType = "KeepRatioResizeBilinear";
const char* kAttrMinDimension = "min_dimension";
const char* kAttrMaxDimension = "max_dimension";
const char* kAttrAlignCorners = "align_corners";
const char* kAttrHalfPixelCenters = "half_pixel_centers";
} // namespace

/*
 * Mirrors the original registration mechanism: the REG_PASS registration is unconditional and
 * the support_fusion_pass.json whitelist only FILTERS the two regbase archs it lists. The 3510
 * entry includes this pass and the 5102 entry does NOT, so the legacy pass is active on all
 * chips EXCEPT 5102 (every other arch - 2201, 2002, ... - is not governed by the whitelist at
 * all). Expressed in code with the same NpuArch input the whitelist uses - rtGetSocSpec,
 * exactly as the legacy GetNpuArch does.
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

static bool IsSupportedPlatform()
{
    const NpuArch npuArch = GetNpuArch();
    const bool isSupported = npuArch != NpuArch::DAV_5102;
    OP_LOGD(kPassName.c_str(), "NpuArch %u, is_supported: %d.", static_cast<uint32_t>(npuArch), isSupported ? 1 : 0);
    return isSupported;
}

static bool GetDimensionAttrs(const GNode& keepRatioNode, int64_t& minDimension, int64_t& maxDimension)
{
    if (keepRatioNode.GetAttr(kAttrMinDimension, minDimension) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get attr min_dimension failed.");
        return false;
    }
    if (keepRatioNode.GetAttr(kAttrMaxDimension, maxDimension) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get attr max_dimension failed.");
        return false;
    }
    return true;
}

/*
 * The align_corners / half_pixel_centers attrs are optional with default value false, as the
 * original pass which leaves them untouched when the attrs are missing.
 */
static void GetBoolAttrOrDefault(const GNode& keepRatioNode, const char* attrName, bool& value)
{
    value = false;
    if (keepRatioNode.GetAttr(attrName, value) != GRAPH_SUCCESS) {
        OP_LOGI(kPassName.c_str(), "Get attr %s failed, set to default false.", attrName);
        value = false;
    }
}

/*
 * Get the H/W dims from the 4-dim images shape according to the format: NHWC -> {dims[1], dims[2]},
 * any other format (NCHW) -> {dims[2], dims[3]}, as the original pass.
 */
static void GetHeightAndWidthDims(const TensorDesc& imagesDesc, int64_t& heightDims, int64_t& widthDims)
{
    const std::vector<int64_t> imagesDims = imagesDesc.GetShape().GetDims();
    if (imagesDesc.GetFormat() == FORMAT_NHWC) {
        heightDims = imagesDims[kIdxDimH];
        widthDims = imagesDims[kIdxDimW];
    } else {
        heightDims = imagesDims[kNchwIdxDimH];
        widthDims = imagesDims[kNchwIdxDimW];
    }
}

/*
 * Compute the target [newH, newW] with the same keep-ratio scaling as the original pass:
 *  1. min scale = min(H, W) / min_dimension, candidate = {floor(H / scale + 0.5), floor(W / scale + 0.5)};
 *  2. max scale = max(H, W) / max_dimension, candidate likewise;
 *  3. if the longer side of the min-scale candidate exceeds max_dimension, use the max-scale one.
 */
static std::vector<int32_t> CalcKeepRatioSize(const int64_t heightDims, const int64_t widthDims,
                                              const int64_t minDimension, const int64_t maxDimension)
{
    const float minDimsFloat = static_cast<float>(minDimension);
    const float maxDimsFloat = static_cast<float>(maxDimension);
    const int64_t minShapeDims = std::min(heightDims, widthDims);
    const int64_t maxShapeDims = std::max(heightDims, widthDims);

    float resizeScale = static_cast<float>(minShapeDims) / minDimsFloat;
    const float minNewShapeH = static_cast<float>(std::floor((heightDims / resizeScale) + 0.5));
    const float minNewShapeW = static_cast<float>(std::floor((widthDims / resizeScale) + 0.5));
    const float minNewShapeMaxDim = std::max(minNewShapeH, minNewShapeW);

    resizeScale = static_cast<float>(maxShapeDims) / maxDimsFloat;
    const float maxNewShapeH = static_cast<float>(std::floor((heightDims / resizeScale) + 0.5));
    const float maxNewShapeW = static_cast<float>(std::floor((widthDims / resizeScale) + 0.5));

    std::vector<int32_t> sizeVec;
    if (minNewShapeMaxDim > maxDimsFloat) {
        sizeVec.push_back(static_cast<int32_t>(maxNewShapeH));
        sizeVec.push_back(static_cast<int32_t>(maxNewShapeW));
    } else {
        sizeVec.push_back(static_cast<int32_t>(minNewShapeH));
        sizeVec.push_back(static_cast<int32_t>(minNewShapeW));
    }
    return sizeVec;
}

std::vector<PatternUniqPtr> KeepRatioResizeBilinearFusion::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for KeepRatioResizeBilinearFusion.");
    std::vector<PatternUniqPtr> patternGraphs;

    // KeepRatioResizeBilinear has no eager style API, build the pattern node with
    // CompliantNodeBuilder (as the PermuteFusionPass does for Permute).
    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto images = graphBuilder.CreateInput(0);
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto keepRatioNode = es::CompliantNodeBuilder(graph)
                             .OpType(kKeepRatioType)
                             .IrDefInputs({{"images", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *images.GetProducer(), images.GetProducerOutIndex(), keepRatioNode,
                                     kKeepRatioInputIdxImages) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge in pattern for KeepRatioResizeBilinear.");
        return patternGraphs;
    }

    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(keepRatioNode, 0);
    auto builtGraph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});

    auto pattern = std::make_unique<Pattern>(std::move(*builtGraph));
    NodeIo nodeIo = {y->GetProducer(), 0};
    pattern->CaptureTensor(nodeIo); // Capture the KeepRatioResizeBilinear node
    patternGraphs.emplace_back(std::move(pattern));

    return patternGraphs;
}

bool KeepRatioResizeBilinearFusion::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for KeepRatioResizeBilinearFusion.");

    // 1. Only run on the supported platforms (standard archs by default registration, plus
    //    Ascend950 through the original registration whitelist).
    if (!IsSupportedPlatform()) {
        OP_LOGD(kPassName.c_str(), "Current platform is not supported, no need to change.");
        return false;
    }

    // 2. Get the matched KeepRatioResizeBilinear node.
    NodeIo keepRatioNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxKeepRatioNode, keepRatioNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured keep_ratio_resize_bilinear node.");
        return false;
    }
    const GNode keepRatioNode = keepRatioNodeIo.node;

    // 3. The images input shape: an EMPTY shape or a non-4-dim shape returns FAILED in the
    //    original (the whole build fails during the pass); those forms are let through here
    //    and fail inside Replacement to keep the failure point identical. (Both forms are
    //    unreachable in valid graphs: the built-in infershape rejects them first.) The
    //    original check order is empty-shape first, then the 4-dim requirement.
    TensorDesc imagesDesc;
    if (keepRatioNode.GetInputDesc(kKeepRatioInputIdxImages, imagesDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc images of keep_ratio_resize_bilinear node.");
        return false;
    }
    const std::vector<int64_t> imagesDims = imagesDesc.GetShape().GetDims();
    if (imagesDims.empty()) {
        OP_LOGD(kPassName.c_str(), "Input shape is empty, let through to fail as the original pass.");
        return true;
    }
    if (imagesDims.size() != kDimNum4D) {
        OP_LOGD(kPassName.c_str(), "Input shape is not equal 4, let through to fail as the original pass.");
        return true;
    }

    // 4. The H/W dims must be known, otherwise the const size node cannot be built. Exactly the
    //    original check: PatternFusionUtil::IsUnknownShape matches only -1 (unknown dim) and -2
    //    (unknown rank); any other value (including 0 or negatives below -2) proceeds as the
    //    original does, reproducing its behavior line by line (0-dim H/W walks into the same
    //    0/0 division as the original pass and the built-in infershape).
    int64_t heightDims = 0;
    int64_t widthDims = 0;
    GetHeightAndWidthDims(imagesDesc, heightDims, widthDims);
    constexpr int64_t kUnknownDim = -1;
    constexpr int64_t kUnknownRankDim = -2;
    const bool heightUnknown = (heightDims == kUnknownDim) || (heightDims == kUnknownRankDim);
    const bool widthUnknown = (widthDims == kUnknownDim) || (widthDims == kUnknownRankDim);
    if (heightUnknown || widthUnknown) {
        OP_LOGI(kPassName.c_str(), "KeepRatioResizeBilinearFusion cannot be applied for unknown shape.");
        return false;
    }

    // 5. The min/max dimension attrs: a MISSING attr keeps the initial 0 and the ==0 check
    //    below skips the fusion - exactly the original path, which does not check the GetInt
    //    return value. Zero min/max skips with NOT_CHANGED as the original (this check sits
    //    after the unknown-shape check in the original, between the float conversions and the
    //    division).
    int64_t minDimension = 0;
    int64_t maxDimension = 0;
    (void)keepRatioNode.GetAttr(kAttrMinDimension, minDimension);
    (void)keepRatioNode.GetAttr(kAttrMaxDimension, maxDimension);
    if (minDimension == 0 || maxDimension == 0) {
        OP_LOGI(kPassName.c_str(), "Min dimension or max dimension can not be zero.");
        return false;
    }
    return true;
}

/*
 * Replicates the legacy V2D const2attr attempt: the original produces the V2 first, then builds
 * a ResizeBilinearV2D desc (the const size converted to an attr) and asks the op stores whether
 * it is supported (the fe-internal CheckOpSupported, unavailable in the open build); on a
 * supported platform ConstToAttrWithNode converts the product to the V2D form (size as an attr,
 * no const size input), otherwise the V2 + const form stays. The probe is rebuilt with the open
 * GeUtils::CheckNodeSupportOnAicore; V2D is registered in the AI Core store only. On all
 * current target platforms V2D has no kernel, so the probe fails and the product stays the
 * V2 + const form (identical to the legacy outcome, A/B bit-exact).
 */
static bool IsResizeBilinearV2DSupported(const TensorDesc& imagesDesc, const TensorDesc& yDesc,
                                         const std::vector<int32_t>& sizeVec, const bool alignCorners,
                                         const bool halfPixelCenters)
{
    es::EsGraphBuilder probeBuilder("v2d_probe");
    auto probeX = probeBuilder.CreateInput(0, "x", imagesDesc.GetDataType(), imagesDesc.GetFormat(),
                                           imagesDesc.GetShape().GetDims());
    auto* probeGraph = probeBuilder.GetCGraphBuilder()->GetGraph();
    const std::vector<int64_t> sizeAttr(sizeVec.begin(), sizeVec.end());
    auto probeNode = es::CompliantNodeBuilder(probeGraph)
                         .OpType("ResizeBilinearV2D")
                         .Name("v2d_probe")
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs(
                             {{"size", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt", es::CreateFrom(sizeAttr)},
                              {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                               es::CreateFrom(alignCorners)},
                              {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                               es::CreateFrom(halfPixelCenters)}})
                         .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*probeGraph, *probeX.GetProducer(), probeX.GetProducerOutIndex(), probeNode, 0) !=
        GRAPH_SUCCESS) {
        return false;
    }
    probeNode.UpdateInputDesc(0, imagesDesc);
    probeNode.UpdateOutputDesc(0, yDesc);
    bool isSupported = false;
    AscendString unsupportedReason;
    if (ge::GeUtils::CheckNodeSupportOnAicore(probeNode, isSupported, unsupportedReason) != SUCCESS) {
        OP_LOGD(kPassName.c_str(), "V2D probe check failed, treat as unsupported.");
        return false;
    }
    OP_LOGD(kPassName.c_str(), "V2D probe supported: %d, reason: %s.", isSupported ? 1 : 0,
            unsupportedReason.GetString());
    return isSupported;
}

GraphUniqPtr KeepRatioResizeBilinearFusion::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for KeepRatioResizeBilinearFusion.");

    // 1. Get the matched node and its input/output descs.
    NodeIo keepRatioNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxKeepRatioNode, keepRatioNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured keep_ratio_resize_bilinear node.");
        return nullptr;
    }
    const GNode keepRatioNode = keepRatioNodeIo.node;

    int64_t minDimension = 0;
    int64_t maxDimension = 0;
    if (!GetDimensionAttrs(keepRatioNode, minDimension, maxDimension)) {
        OP_LOGE(kPassName.c_str(), "Failed to get dimension attrs in replacement.");
        return nullptr;
    }
    bool alignCorners = false;
    bool halfPixelCenters = false;
    GetBoolAttrOrDefault(keepRatioNode, kAttrAlignCorners, alignCorners);
    GetBoolAttrOrDefault(keepRatioNode, kAttrHalfPixelCenters, halfPixelCenters);

    TensorDesc imagesDesc;
    if (keepRatioNode.GetInputDesc(kKeepRatioInputIdxImages, imagesDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc images of keep_ratio_resize_bilinear node.");
        return nullptr;
    }
    // Mirror the original FAILED returns: an empty or non-4-dim input shape fails the whole
    // build during the pass (those forms are let through MeetRequirements for exactly this).
    const std::vector<int64_t> imagesDims = imagesDesc.GetShape().GetDims();
    if (imagesDims.empty() || imagesDims.size() != kDimNum4D) {
        OP_LOGE(kPassName.c_str(), "Node input shape is empty or not equal 4, fusion failed.");
        return nullptr;
    }
    TensorDesc yDesc;
    if (keepRatioNode.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get output desc of keep_ratio_resize_bilinear node.");
        return nullptr;
    }

    // 2. Compute the target size with the keep-ratio scaling, as the original pass.
    int64_t heightDims = 0;
    int64_t widthDims = 0;
    GetHeightAndWidthDims(imagesDesc, heightDims, widthDims);
    const std::vector<int32_t> sizeVec = CalcKeepRatioSize(heightDims, widthDims, minDimension, maxDimension);

    // 3. Build the replacement graph: ResizeBilinearV2(images, Const size).
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rImages = replaceGraphBuilder.CreateInput(0, "images", imagesDesc.GetDataType(), imagesDesc.GetFormat(),
                                                   imagesDesc.GetShape().GetDims());

    // Legacy-aligned V2D const2attr attempt: on a supported platform the product is the V2D
    // form - the size as an attr on the target node, no const size input - exactly what the
    // original's ConstToAttrWithNode leaves behind when its CheckOpSupported succeeds.
    if (IsResizeBilinearV2DSupported(imagesDesc, yDesc, sizeVec, alignCorners, halfPixelCenters)) {
        auto* cGraph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
        const std::vector<int64_t> sizeAttr(sizeVec.begin(), sizeVec.end());
        auto v2dNode = es::CompliantNodeBuilder(cGraph)
                           .OpType("ResizeBilinearV2D")
                           .Name("replacement_v2d")
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .IrDefAttrs({{"size", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                         es::CreateFrom(sizeAttr)},
                                        {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                         es::CreateFrom(alignCorners)},
                                        {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                         es::CreateFrom(halfPixelCenters)}})
                           .Build();
        if (es::AddEdgeAndUpdatePeerDesc(*cGraph, *rImages.GetProducer(), rImages.GetProducerOutIndex(), v2dNode, 0) !=
            GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to add edge for the v2d node.");
            return nullptr;
        }
        v2dNode.UpdateInputDesc(kKeepRatioInputIdxImages, imagesDesc);
        v2dNode.UpdateOutputDesc(0, yDesc);
        auto v2dY = es::EsTensorHolder(replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(v2dNode, 0));
        OP_LOGD(kPassName.c_str(), "KeepRatioResizeBilinear to ResizeBilinearV2D (supported) success.");
        return replaceGraphBuilder.BuildAndReset({v2dY});
    }

    auto sizeInput = replaceGraphBuilder.CreateConst(sizeVec, std::vector<int64_t>{kSizeElementNum});
    auto y = es::ResizeBilinearV2(rImages, sizeInput, alignCorners, halfPixelCenters);
    GNode targetNode = *y.GetProducer();
    targetNode.UpdateInputDesc(kKeepRatioInputIdxImages, imagesDesc);
    TensorDesc sizeDesc(Shape(std::vector<int64_t>{kSizeElementNum}), FORMAT_ND, DT_INT32);
    sizeDesc.SetOriginFormat(FORMAT_ND);
    sizeDesc.SetOriginShape(Shape(std::vector<int64_t>{kSizeElementNum}));
    targetNode.UpdateInputDesc(1, sizeDesc);
    targetNode.UpdateOutputDesc(0, yDesc);

    OP_LOGD(kPassName.c_str(), "KeepRatioResizeBilinear to ResizeBilinearV2 success.");
    return replaceGraphBuilder.BuildAndReset({y});
}

REG_FUSION_PASS(KeepRatioResizeBilinearFusion).Stage(CustomPassStage::kAfterBuiltinFusionPass);

} // namespace ops
