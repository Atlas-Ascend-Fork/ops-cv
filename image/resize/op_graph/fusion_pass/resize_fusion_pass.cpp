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
 * \file resize_fusion_pass.cpp
 * \brief resize fusion pass (Resize --> ResizeNearestNeighborV2/ResizeBilinearV2/ResizeTrilinear)
 *
 * Pattern:
 *        x    roi   scales  sizes
 *         \     |      |     /
 *          \    |      |    /
 *              Resize
 *                |
 *               y
 *
 * after:
 *    output shape is const (size is static):
 *        x                                   x
 *        |                                   |
 *        Resize              ==>   ResizeNearestNeighborV2 / ResizeBilinearV2
 *        |                                   |        (size comes from a const node
 *        y                                   y         extracted from output shape)
 *
 *    output shape is not const (dynamic shape from onnx) or mode is linear with 5-dim input:
 *        x     roi  scales  sizes       x                          sizes
 *         \     |     |     /            |                           |
 *          \    |     |    /      ==>    |     Slice(offsets=2)     /
 *            Resize                      ResizeNearestNeighborV2 / ResizeBilinearV2 / ResizeTrilinear
 *              |                          |
 *              y                          y
 */

#include <algorithm>
#include <string>
#include <vector>

#include "es_cv_ops.h"
#include "es_math_ops.h"
#include "ge/ge_utils.h"
#include "log/log.h"

#include "resize_fusion_pass.h"

using namespace ge;
using namespace fusion;

namespace ops {

namespace {
const std::string kPassName = "ResizeFusionPass";
constexpr int64_t kCaptureIdxResizeNode = 0;
// input index of Resize: x=0, roi=1, scales=2, sizes=3
constexpr int32_t kResizeInputIdxX = 0;
constexpr int32_t kResizeInputIdxRoi = 1;
constexpr int32_t kResizeInputIdxScales = 2;
constexpr int32_t kResizeInputIdxSizes = 3;
// Resize IR: 1 required input (x) + 3 optional inputs (roi/scales/sizes)
constexpr int32_t kResizeRequiredInputNum = 1;
constexpr int32_t kResizeOptionalInputNum = 3;
constexpr int32_t kResizeTotalInputNum = kResizeRequiredInputNum + kResizeOptionalInputNum;
constexpr size_t kMinSupportedDimNum = 4;
constexpr size_t kDimNum4D = 4;
constexpr size_t kDimNum5D = 5;
// index of spatial dims in output shape
constexpr size_t kIdxDim1 = 1;
constexpr size_t kIdxDim2 = 2;
constexpr size_t kIdxDim3 = 3;
constexpr size_t kIdxDim4 = 4;
constexpr int64_t kSliceOffsetValue = 2;
constexpr int64_t kSliceSize4D = 2;
constexpr int64_t kSliceSize5D = 3;
const std::string kModeNearest = "nearest";
const std::string kModeLinear = "linear";
const std::string kModeCubic = "cubic";
const std::string kDvppEngineName = "DNN_VM_DVPP";
const std::string kDvppKernelLibName = "dvpp_ops_kernel";
const std::string kOnnxResizeType = "onnx_resize";
} // namespace

/*
 * No platform gate at all, mirroring the original registration mechanism: the REG_PASS
 * registration is unconditional and the support_fusion_pass.json whitelist only FILTERS the
 * two regbase archs it lists (3510/5102), both of which include this pass. Every other arch
 * (910B/910_93=2201, 310P=2002, ...) is not governed by the whitelist at all, so the legacy
 * pass runs on ALL chips. A narrower in-code gate caused a real regression: on Ascend1951
 * (DAV_2002) the baseline legacy pass fused a TF detection model's Resize (whose degenerate
 * [0]-shape sizes input makes the source op uncompilable - the model depends on the fusion),
 * while the narrower {2201, 3510, 5102} gate skipped the node and the build failed with
 * "invalid shape input, shape:[0]".
 */

/*
 * Check whether the Resize node is specified to run on DVPP, which should not be changed by this pass.
 */
static bool IsDvppSpecified(const GNode& resizeNode)
{
    AscendString engineName;
    if (resizeNode.GetAttr("_specified_engine_name", engineName) == GRAPH_SUCCESS) {
        if (std::string(engineName.GetString()) == kDvppEngineName) {
            OP_LOGD(kPassName.c_str(), "Op is specified with engine DNN_VM_DVPP, no need to change.");
            return true;
        }
    }
    AscendString kernelLibName;
    if (resizeNode.GetAttr("_specified_kernel_lib_name", kernelLibName) == GRAPH_SUCCESS) {
        if (std::string(kernelLibName.GetString()) == kDvppKernelLibName) {
            OP_LOGD(kPassName.c_str(), "Op is specified with kernel lib dvpp_ops_kernel, no need to change.");
            return true;
        }
    }
    return false;
}

/*
 * Get the interpolation mode attr of Resize. Only nearest/linear/cubic are legal.
 */
static bool GetModeName(const GNode& resizeNode, std::string& modeName)
{
    AscendString modeStr;
    if (resizeNode.GetAttr("mode", modeStr) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get attr mode failed.");
        return false;
    }
    modeName = std::string(modeStr.GetString());
    if (modeName != kModeNearest && modeName != kModeLinear && modeName != kModeCubic) {
        OP_LOGE(kPassName.c_str(), "Mode name is not nearest, linear or cubic.");
        return false;
    }
    return true;
}

static bool IsOnnxResize(const GNode& resizeNode)
{
    AscendString originalType;
    if (resizeNode.GetAttr("resize_original_type", originalType) != GRAPH_SUCCESS) {
        return false;
    }
    return std::string(originalType.GetString()) == kOnnxResizeType;
}

/*
 * Extract spatial dims from the static output shape according to the input format:
 *   NHWC -> {dims[1], dims[2]}, NCHW -> {dims[2], dims[3]}, NCDHW -> {dims[2], dims[3], dims[4]}.
 * Any other format only draws a warning and leaves the size empty, exactly as the original
 * ClcOutputSize (the empty size then flows into the const node as-is).
 */
static void CalcOutputSize(const Format& inputFormat, const std::vector<int64_t>& yDims,
                           std::vector<int64_t>& outputSize)
{
    if (inputFormat == FORMAT_NHWC) {
        outputSize.push_back(yDims[kIdxDim1]);
        outputSize.push_back(yDims[kIdxDim2]);
    } else if (inputFormat == FORMAT_NCHW) {
        outputSize.push_back(yDims[kIdxDim2]);
        outputSize.push_back(yDims[kIdxDim3]);
    } else if (inputFormat == FORMAT_NCDHW) {
        outputSize.push_back(yDims[kIdxDim2]);
        outputSize.push_back(yDims[kIdxDim3]);
        outputSize.push_back(yDims[kIdxDim4]);
    } else {
        OP_LOGW(kPassName.c_str(), "Not supported this format %d, output tensor will be wrong.",
                static_cast<int32_t>(inputFormat));
    }
}

std::vector<PatternUniqPtr> ResizeFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for ResizeFusionPass.");
    std::vector<PatternUniqPtr> patternGraphs;

    // The IR of Resize is 1 required input (x) plus 3 optional inputs (roi/scales/sizes) and the
    // original pattern matches the node by type regardless of the input count. Build one pattern
    // per possible input count: the framework matches by the number of non-null inputs and does
    // not distinguish which optional input is absent (existence is checked in MeetRequirements).
    for (int nullOptionalNum = 0; nullOptionalNum <= kResizeOptionalInputNum; ++nullOptionalNum) {
        auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
        auto inputs = graphBuilder.CreateInputs<kResizeTotalInputNum>();
        for (int i = kResizeRequiredInputNum; i < kResizeRequiredInputNum + nullOptionalNum; ++i) {
            inputs[i] = nullptr;
        }
        auto y = es::Resize(inputs[kResizeInputIdxX], inputs[kResizeInputIdxRoi], inputs[kResizeInputIdxScales],
                            inputs[kResizeInputIdxSizes]);
        auto graph = graphBuilder.BuildAndReset({y});

        auto pattern = std::make_unique<Pattern>(std::move(*graph));
        pattern->CaptureTensor({*y.GetProducer(), 0});

        patternGraphs.emplace_back(std::move(pattern));
    }
    return patternGraphs;
}

bool ResizeFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for ResizeFusionPass.");

    // 1. Get the matched Resize node.
    NodeIo resizeNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxResizeNode, resizeNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured resize node.");
        return false;
    }
    const GNode resizeNode = resizeNodeIo.node;

    // 2. Skip the op specified to run on DVPP.
    if (IsDvppSpecified(resizeNode)) {
        return false;
    }

    // 3. Check the interpolation mode. Cubic mode is not supported by this pass (NOT_CHANGED,
    //    as the original). An INVALID OR MISSING mode is deliberately let through here and fails
    //    inside Replacement (returning nullptr): the original pass returns PARAM_INVALID from
    //    MatchModeName at this point and the whole build fails during the pass, so the failure
    //    point is kept identical (check order: DVPP -> mode, as the original - no platform
    //    check, the legacy pass is active on all chips).
    std::string modeName;
    if (!GetModeName(resizeNode, modeName)) {
        OP_LOGD(kPassName.c_str(), "Mode name is invalid, let through to fail as the original pass.");
        return true;
    }
    // 5. The dim num of output y must be not less than 4 (as the original: NOT_CHANGED). The
    //    original checks the dim BEFORE the cubic skip (Fusion order: mode -> dims -> cubic),
    //    so with both conditions present the dim is reported first; keep that order.
    TensorDesc yDesc;
    if (resizeNode.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get output desc of resize node.");
        return false;
    }
    const std::vector<int64_t> yDims = yDesc.GetShape().GetDims();
    if (yDims.size() < kMinSupportedDimNum) {
        OP_LOGD(kPassName.c_str(), "Output[y]'s dim must be >= 4, but got %zu.", yDims.size());
        return false;
    }

    if (modeName == kModeCubic) {
        OP_LOGD(kPassName.c_str(), "Cubic mode, no need to change.");
        return false;
    }

    // 6. Dynamic shape is only allowed for onnx resize (as the original: NOT_CHANGED).
    const bool isDynamicShape = std::find(yDims.begin(), yDims.end(), UNKNOWN_DIM) != yDims.end();
    if (isDynamicShape && !IsOnnxResize(resizeNode)) {
        OP_LOGD(kPassName.c_str(), "Output shape is dynamic and not onnx resize, no need to change.");
        return false;
    }

    TensorDesc xDesc;
    if (resizeNode.GetInputDesc(kResizeInputIdxX, xDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc x of resize node.");
        return false;
    }

    // 7. ResizeNearestNeighborV2 only supports float16 and float32 (as the original:
    //    NOT_CHANGED for other dtypes in nearest mode).
    if (modeName == kModeNearest) {
        const DataType xDtype = xDesc.GetDataType();
        if (xDtype != DT_FLOAT16 && xDtype != DT_FLOAT) {
            OP_LOGD(kPassName.c_str(), "ResizeNearestNeighborV2 only supports float16 and float32.");
            return false;
        }
    }

    // No sizes-existence, linear-rank or input-format gate here: the original pass has none of
    // them. It prunes the node down to [x, last-input] whatever the inputs are (a missing sizes
    // input is handled by the const branch reading the output shape), and for an unsupported
    // input format ClcOutputSize only warns and leaves the size EMPTY - the original then fuses
    // with a 0-element size const (CreateConstantNode), reproduced line by line in Replacement.
    // The NCDHW+4-dim form walks into the same out-of-bounds dims[4] read as the original.
    // STRICT legacy alignment per review decision (2026-10-26); the earlier format gate was
    // based on a misdiagnosis of the Ascend1951 regression, whose real root cause was the
    // too-narrow platform gate (see the diff findings "1951 回归根因修正").
    return true;
}

/*
 * Replicates the legacy V2D probe of ChangeOpTypeWithD: the original retypes the node to
 * ResizeNearestNeighborV2D first and asks the op stores whether it is supported (the
 * fe-internal CheckOpSupported, unavailable in the open build); on a supported platform the
 * product STAYS a V2D with the size carried as an attr and no const size input, otherwise the
 * fallback is the V2 + const form. The probe is rebuilt with the open
 * GeUtils::CheckNodeSupportOnAicore on a throwaway probe graph carrying the same descs/attrs.
 * V2D is registered in the AI Core store only, so the AI Core-scope check reaches the same
 * store the legacy query could accept for this op. On all current target platforms V2D has no
 * kernel, so the probe fails and the product stays the V2 + const form (identical to the
 * legacy fallback product, E2E-verified).
 */
static bool IsResizeNearestV2DSupported(const TensorDesc& xDesc, const TensorDesc& yDesc,
                                        const std::vector<int64_t>& outputSize, const bool alignCorners,
                                        const bool halfPixelCenters)
{
    es::EsGraphBuilder probeBuilder("v2d_probe");
    auto probeX = probeBuilder.CreateInput(0, "x", xDesc.GetDataType(), xDesc.GetFormat(), xDesc.GetShape().GetDims());
    auto* probeGraph = probeBuilder.GetCGraphBuilder()->GetGraph();
    auto probeNode = es::CompliantNodeBuilder(probeGraph)
                         .OpType("ResizeNearestNeighborV2D")
                         .Name("v2d_probe")
                         .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .IrDefAttrs({{"size", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                       es::CreateFrom(outputSize)},
                                      {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                       es::CreateFrom(alignCorners)},
                                      {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                       es::CreateFrom(halfPixelCenters)}})
                         .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*probeGraph, *probeX.GetProducer(), probeX.GetProducerOutIndex(), probeNode, 0) !=
        GRAPH_SUCCESS) {
        return false;
    }
    probeNode.UpdateInputDesc(0, xDesc);
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

GraphUniqPtr ResizeFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for ResizeFusionPass.");

    // 1. Get the matched Resize node and its input/output descs.
    NodeIo resizeNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxResizeNode, resizeNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured resize node.");
        return nullptr;
    }
    const GNode resizeNode = resizeNodeIo.node;

    std::string modeName;
    if (!GetModeName(resizeNode, modeName)) {
        OP_LOGE(kPassName.c_str(), "Failed to get mode name in replacement.");
        return nullptr;
    }

    TensorDesc xDesc;
    if (resizeNode.GetInputDesc(kResizeInputIdxX, xDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc x of resize node.");
        return nullptr;
    }
    TensorDesc yDesc;
    if (resizeNode.GetOutputDesc(0, yDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get output desc of resize node.");
        return nullptr;
    }
    const std::vector<int64_t> yDims = yDesc.GetShape().GetDims();
    const size_t dimNum = yDims.size();

    // 2. Convert coordinate_transformation_mode to half_pixel_centers / align_corners.
    bool alignCorners = false;
    bool halfPixelCenters = false;
    AscendString ctmStr;
    if (resizeNode.GetAttr("coordinate_transformation_mode", ctmStr) == GRAPH_SUCCESS) {
        const std::string ctmValue(ctmStr.GetString());
        if (ctmValue == "pytorch_half_pixel") {
            halfPixelCenters = true;
        } else if (ctmValue == "align_corners") {
            alignCorners = true;
        }
    } else {
        OP_LOGI(kPassName.c_str(), "Get attr coordinate_transformation_mode failed, set to default.");
    }

    // 3. Decide which branch to take: with Slice (dynamic size from onnx / linear 5-dim) or with const size.
    const bool isDynamicShape = std::find(yDims.begin(), yDims.end(), UNKNOWN_DIM) != yDims.end();
    const bool needSlice = isDynamicShape || (modeName == kModeLinear && dimNum == kDimNum5D);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    // The replacement graph keeps the boundary inputs in the same order as the matched subgraph:
    // only the inputs actually present on the source node are created. The original pass prunes
    // the node down to [x, LAST-input] whatever the inputs are: the last connected input is
    // treated as the sizes source (sliced in the dynamic branch, dropped in the const branch
    // where the size comes from the static output shape). Mirror that: the roi/scales inputs are
    // unconsumed boundary inputs (the original deletes them; consuming them via Identity is
    // forbidden - GE may assign those Identity nodes to the host_cpu engine and the resulting
    // host-side tasks break AICPU models, observed on a real Ascend910B).
    TensorDesc roiDesc;
    TensorDesc scalesDesc;
    TensorDesc sizesDesc;
    const bool hasRoi = resizeNode.GetInputDesc(kResizeInputIdxRoi, roiDesc) == GRAPH_SUCCESS;
    const bool hasScales = resizeNode.GetInputDesc(kResizeInputIdxScales, scalesDesc) == GRAPH_SUCCESS;
    const bool hasSizes = resizeNode.GetInputDesc(kResizeInputIdxSizes, sizesDesc) == GRAPH_SUCCESS;
    int32_t boundaryInputIdx = 0;
    auto rX = replaceGraphBuilder.CreateInput(boundaryInputIdx++, "x", xDesc.GetDataType(), xDesc.GetFormat(),
                                              xDesc.GetShape().GetDims());
    es::EsTensorHolder rRoi;
    es::EsTensorHolder rScales;
    es::EsTensorHolder rSizes;
    if (hasRoi) {
        rRoi = replaceGraphBuilder.CreateInput(boundaryInputIdx++, "roi", roiDesc.GetDataType(), roiDesc.GetFormat(),
                                               roiDesc.GetShape().GetDims());
    }
    if (hasScales) {
        rScales = replaceGraphBuilder.CreateInput(boundaryInputIdx++, "scales", scalesDesc.GetDataType(),
                                                  scalesDesc.GetFormat(), scalesDesc.GetShape().GetDims());
    }
    if (hasSizes) {
        rSizes = replaceGraphBuilder.CreateInput(boundaryInputIdx++, "sizes", sizesDesc.GetDataType(),
                                                 sizesDesc.GetFormat(), sizesDesc.GetShape().GetDims());
    }
    // The last connected input, as the original keeps [x, last-input]: sizes when present,
    // otherwise scales, otherwise roi.
    es::EsTensorHolder rLastInput;
    TensorDesc lastInputDesc;
    if (hasSizes) {
        rLastInput = rSizes;
        lastInputDesc = sizesDesc;
    } else if (hasScales) {
        rLastInput = rScales;
        lastInputDesc = scalesDesc;
    } else if (hasRoi) {
        rLastInput = rRoi;
        lastInputDesc = roiDesc;
    }

    es::EsTensorHolder sizeInput;
    TensorDesc targetSizeDesc;
    if (needSlice) {
        // Slice(last-input, offsets={2}, size={2 or 3}): the original InsertSliceNode works on
        // the node's input anchor 1 after pruning, i.e. on the last connected input, whatever it
        // is (for a [x, scales] dynamic form it slices the scales data and the downstream kernel
        // then rejects the float size - same terminal failure as the original).
        if (rLastInput.GetCTensorHolder() == nullptr) {
            // No optional input at all ([x] only): the original gets a null InDataAnchor(1) in
            // InsertSliceNode and returns PARAM_INVALID; fail here likewise.
            OP_LOGE(kPassName.c_str(), "No last input to slice, fusion failed.");
            return nullptr;
        }
        const int64_t sliceLen = (dimNum == kDimNum5D) ? kSliceSize5D : kSliceSize4D;
        auto offsetsConst = replaceGraphBuilder.CreateConst(std::vector<int64_t>{kSliceOffsetValue}, {1});
        auto sliceSizeConst = replaceGraphBuilder.CreateConst(std::vector<int64_t>{sliceLen}, {1});
        sizeInput = es::Slice(rLastInput, offsetsConst, sliceSizeConst);
        GNode sliceNode = *sizeInput.GetProducer();
        // All Slice input/output descs are set explicitly (the eager style API does not
        // propagate descs onto the node inputs and an unset desc stays at a default dtype):
        // an unset input 0 desc makes FE insert a dtype-converting Cast between the last input
        // and the Slice, which GE assigns to the host_cpu engine and which broke models whose
        // remaining ops run on AICPU (ResizeTrilinear execution failure, observed on a real
        // Ascend910B). The desc VALUES mirror the original CreateSliceNode line by line:
        // the input desc carries the FIXED shape {4} ({5} for 5-dim) whatever the actual
        // last-input desc is, and both descs take the peer's format/origin format and dtype
        // (not a hardcoded ND); origin shapes carry the same fixed values as the shapes.
        const int64_t sliceInLen = (dimNum == kDimNum5D) ? kDimNum5D : kDimNum4D;
        const std::vector<int64_t> sliceInShape = {sliceInLen};
        TensorDesc sliceInDesc(Shape(sliceInShape), lastInputDesc.GetFormat(), lastInputDesc.GetDataType());
        sliceInDesc.SetOriginFormat(lastInputDesc.GetOriginFormat());
        sliceInDesc.SetOriginShape(Shape(sliceInShape));
        TensorDesc sliceParamDesc(Shape(std::vector<int64_t>{1}), FORMAT_ND, DT_INT64);
        sliceParamDesc.SetOriginFormat(FORMAT_ND);
        sliceParamDesc.SetOriginShape(Shape(std::vector<int64_t>{1}));
        (void)sliceNode.UpdateInputDesc(0, sliceInDesc);
        (void)sliceNode.UpdateInputDesc(1, sliceParamDesc);
        (void)sliceNode.UpdateInputDesc(2, sliceParamDesc);
        TensorDesc sliceOutDesc(Shape(std::vector<int64_t>{sliceLen}), lastInputDesc.GetFormat(),
                                lastInputDesc.GetDataType());
        sliceOutDesc.SetOriginFormat(lastInputDesc.GetOriginFormat());
        sliceOutDesc.SetOriginShape(Shape(std::vector<int64_t>{sliceLen}));
        sliceNode.UpdateOutputDesc(0, sliceOutDesc);
        targetSizeDesc = sliceOutDesc;
    } else {
        // Extract the spatial dims from the static output shape and feed them as a const size
        // node (works whatever the last input is, as the original const branch).
        std::vector<int64_t> outputSize;
        CalcOutputSize(xDesc.GetFormat(), yDims, outputSize);
        // Legacy-aligned V2D probe (ChangeOpTypeWithD, nearest only): on a supported platform
        // the product is the in-place V2D form - size as an attr, x as the only input, no const
        // size node - exactly what the original leaves behind when CheckOpSupported succeeds.
        if (modeName == kModeNearest &&
            IsResizeNearestV2DSupported(xDesc, yDesc, outputSize, alignCorners, halfPixelCenters)) {
            auto* cGraph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
            auto v2dNode = es::CompliantNodeBuilder(cGraph)
                               .OpType("ResizeNearestNeighborV2D")
                               .Name("replacement_v2d")
                               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                               .IrDefAttrs({{"size", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                             es::CreateFrom(outputSize)},
                                            {"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                             es::CreateFrom(alignCorners)},
                                            {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                             es::CreateFrom(halfPixelCenters)}})
                               .Build();
            if (es::AddEdgeAndUpdatePeerDesc(*cGraph, *rX.GetProducer(), rX.GetProducerOutIndex(), v2dNode, 0) !=
                GRAPH_SUCCESS) {
                OP_LOGE(kPassName.c_str(), "Failed to add edge for the v2d node.");
                return nullptr;
            }
            v2dNode.UpdateInputDesc(0, xDesc);
            v2dNode.UpdateOutputDesc(0, yDesc);
            auto v2dY = es::EsTensorHolder(replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(v2dNode, 0));
            OP_LOGD(kPassName.c_str(), "Resize to ResizeNearestNeighborV2D (supported) success.");
            return replaceGraphBuilder.BuildAndReset({v2dY});
        }
        std::vector<int32_t> sizeValue(outputSize.begin(), outputSize.end());
        if (sizeValue.empty()) {
            // The original ClcOutputSize leaves the size EMPTY for an unsupported input format
            // (warning only) and CreateConstantNode then builds a 0-element Constant node. The
            // eager style CreateConst cannot express a 0-element const (it asserts on the null
            // data pointer), so build the empty const node with CompliantNodeBuilder exactly
            // like the original CreateConstantNode does. Strict legacy alignment: the product is
            // what the original produces, kernel rejection included.
            TensorDesc emptyDesc(Shape(std::vector<int64_t>{0}), FORMAT_ND, DT_INT32);
            emptyDesc.SetOriginFormat(FORMAT_ND);
            emptyDesc.SetOriginShape(Shape(std::vector<int64_t>{0}));
            auto emptyTensor = std::make_unique<Tensor>(emptyDesc);
            ge::AttrValue tensorAttr;
            (void)tensorAttr.SetAttrValue(*emptyTensor);
            auto* cGraph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
            auto constNode = es::CompliantNodeBuilder(cGraph)
                                 .OpType("Const")
                                 .Name("replacement_empty_size")
                                 .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .IrDefAttrs(
                                     {{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", tensorAttr}})
                                 .InstanceOutputShape("y", std::vector<int64_t>{0})
                                 .InstanceOutputDataType("y", DT_INT32)
                                 .InstanceOutputFormat("y", FORMAT_ND)
                                 .Build();
            sizeInput = es::EsTensorHolder(
                replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(constNode, 0));
        } else {
            sizeInput = replaceGraphBuilder.CreateConst(sizeValue, {static_cast<int64_t>(sizeValue.size())});
        }
        targetSizeDesc = TensorDesc(Shape(std::vector<int64_t>{static_cast<int64_t>(sizeValue.size())}), FORMAT_ND,
                                    DT_INT32);
        targetSizeDesc.SetOriginFormat(FORMAT_ND);
        targetSizeDesc.SetOriginShape(Shape(std::vector<int64_t>{static_cast<int64_t>(sizeValue.size())}));
    }

    // 4. Create the target op, mirroring the original branch-dependent selection:
    //    - const branch (ChangeOpTypeWithD): nearest -> ResizeNearestNeighborV2, linear ->
    //      ResizeBilinearV2 whatever the output rank;
    //    - Slice branch (ChangeOpType): nearest -> ResizeNearestNeighborV2, linear 4-dim ->
    //      ResizeBilinearV2, linear 5-dim -> ResizeTrilinear (the original leaves an empty type
    //      for other linear ranks in this branch - unbuildable here, the 5-dim Trilinear is the
    //      closest constructible form and those ranks are unreachable in valid graphs).
    //    ResizeTrilinear has no eager style API (not in the CANN-installed es_cv package), it is
    //    built with CompliantNodeBuilder (as the keep_ratio pass does for KeepRatioResizeBilinear).
    es::EsTensorHolder y;
    if (modeName == kModeNearest) {
        y = es::ResizeNearestNeighborV2(rX, sizeInput, alignCorners, halfPixelCenters);
    } else if (!needSlice || dimNum == kDimNum4D) {
        y = es::ResizeBilinearV2(rX, sizeInput, alignCorners, halfPixelCenters);
    } else {
        auto* targetGraph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
        auto trilinearNode = es::CompliantNodeBuilder(targetGraph)
                                 .OpType("ResizeTrilinear")
                                 .Name("replacement_trilinear")
                                 .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                               {"size", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .IrDefAttrs({{"align_corners", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                               es::CreateFrom(alignCorners)},
                                              {"half_pixel_centers", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                               es::CreateFrom(halfPixelCenters)}})
                                 .Build();
        if (es::AddEdgeAndUpdatePeerDesc(*targetGraph, *rX.GetProducer(), rX.GetProducerOutIndex(), trilinearNode, 0) !=
                GRAPH_SUCCESS ||
            es::AddEdgeAndUpdatePeerDesc(*targetGraph, *sizeInput.GetProducer(), sizeInput.GetProducerOutIndex(),
                                         trilinearNode, 1) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to add edges for the trilinear node.");
            return nullptr;
        }
        trilinearNode.UpdateInputDesc(0, xDesc);
        trilinearNode.UpdateInputDesc(1, targetSizeDesc);
        trilinearNode.UpdateOutputDesc(0, yDesc);
        auto trilinearY = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(trilinearNode, 0);
        OP_LOGD(kPassName.c_str(), "Resize to ResizeTrilinear success.");
        return replaceGraphBuilder.BuildAndReset({es::EsTensorHolder(trilinearY)});
    }
    GNode targetNode = *y.GetProducer();
    targetNode.UpdateInputDesc(0, xDesc);
    // The size input desc must be set explicitly: the target IR only accepts DT_INT32 for the
    // size input, and an unset desc stays at the default dtype and is rejected by the op kernel
    // check with EZ3002 (size dtype not supported).
    targetNode.UpdateInputDesc(1, targetSizeDesc);
    targetNode.UpdateOutputDesc(0, yDesc);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({y});
    OP_LOGD(kPassName.c_str(), "Resize to target op in mode[%s] success.", modeName.c_str());
    return replaceGraph;
}

REG_FUSION_PASS(ResizeFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);

} // namespace ops
