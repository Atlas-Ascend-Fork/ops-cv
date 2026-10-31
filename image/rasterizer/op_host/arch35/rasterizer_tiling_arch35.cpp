/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/rasterizer_tiling_struct.h"
#include "register/op_def_registry.h"
#include "rasterizer_tiling_arch35.h"

namespace optiling {
namespace {
constexpr uint64_t kMaxDimension = 4096U;
constexpr uint64_t kFindicesBytesPerPixel = sizeof(int32_t);
constexpr uint64_t kBarycentricBytesPerPixel = 3U * sizeof(float);
constexpr uint64_t kWorkspacePlaneCount = 5U;
// Task 26/28 freeze Rasterizer's reviewed Ascend 950 scheduling model at 20
// AIV workers. Keep the platform query as the hardware upper bound, then apply
// the operator-specific budget so Host tiling and the workspace ABI stay stable.
constexpr uint32_t kReviewedAivCoreBudget = 20U;

const char* RasterizerPublicTilingStatusName(RasterizerPublicTilingStatus status)
{
    switch (status) {
        case RasterizerPublicTilingStatus::kSuccess:
            return "success";
        case RasterizerPublicTilingStatus::kNullInput:
            return "null_input";
        case RasterizerPublicTilingStatus::kDtypeNotSupported:
            return "dtype_not_supported";
        case RasterizerPublicTilingStatus::kShapeMismatch:
            return "shape_mismatch";
        case RasterizerPublicTilingStatus::kAttributeValueOutOfRange:
            return "attribute_value_out_of_range";
        case RasterizerPublicTilingStatus::kPlatformInvalid:
            return "platform_invalid";
        case RasterizerPublicTilingStatus::kOutOfMemory:
            return "out_of_memory";
        case RasterizerPublicTilingStatus::kArithmeticOverflow:
            return "arithmetic_overflow";
        case RasterizerPublicTilingStatus::kNotImplemented:
            return "not_implemented";
        default:
            return "unknown";
    }
}

void SetFailure(RasterizerPublicTilingResult& result, RasterizerPublicTilingStatus status)
{
    result = RasterizerPublicTilingResult{};
    result.status = status;
    result.kernelDispatchAllowed = false;
    result.tilingKey = 0;
    result.totalPixels = 0;
    result.alignedPixels = 0;
    result.pixelAlignment = 0;
    result.vectorAlignment = 0;
    result.serializedBytes = 0;
    result.totalWorkspaceBytes = 0;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t& out)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    out = lhs + rhs;
    return true;
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& out)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>((value % divisor) != 0U);
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& out)
{
    if (alignment == 0U) {
        return false;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0U ? (out = value, true) : CheckedAdd(value, alignment - remainder, out);
}

bool IsRank(const RasterizerPublicTensorDesc& tensor, uint32_t rank) { return tensor.rank == rank; }

RasterizerPublicDataType ToPublicDataType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return RasterizerPublicDataType::kFloat32;
        case ge::DT_INT32:
            return RasterizerPublicDataType::kInt32;
        case ge::DT_FLOAT16:
            return RasterizerPublicDataType::kFloat16;
        default:
            return RasterizerPublicDataType::kUnknown;
    }
}

bool FillTensorDesc(const gert::Tensor* tensor, const gert::StorageShape* storageShape, RasterizerPublicTensorDesc& out)
{
    out = RasterizerPublicTensorDesc{};
    if (tensor == nullptr) {
        return false;
    }
    // Preserve "present" and dtype even when the shape metadata is malformed so
    // required tensors are classified as shape_mismatch rather than null_input.
    out.present = true;
    out.dtype = ToPublicDataType(tensor->GetDataType());
    out.ndContiguous = tensor->GetStorageFormat() == ge::FORMAT_ND && tensor->GetOriginFormat() == ge::FORMAT_ND;
    if (storageShape == nullptr) {
        out.rank = UINT32_MAX;
        return false;
    }
    const gert::Shape& shape = storageShape->GetStorageShape();
    const std::size_t rank = shape.GetDimNum();
    if (rank > 3U) {
        out.rank = UINT32_MAX;
        return false;
    }
    out.rank = static_cast<uint32_t>(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            out.dims[i] = UINT64_MAX;
            return false;
        }
        out.dims[i] = static_cast<uint64_t>(dim);
    }
    return true;
}

bool FillOutputDesc(const gert::StorageShape* storageShape, const gert::CompileTimeTensorDesc* tensorDesc,
                    RasterizerPublicTensorDesc& out)
{
    out = RasterizerPublicTensorDesc{};
    if (storageShape == nullptr || tensorDesc == nullptr || !tensorDesc->IsExist()) {
        return false;
    }
    out.present = true;
    out.dtype = ToPublicDataType(tensorDesc->GetDataType());
    out.ndContiguous = tensorDesc->GetStorageFormat() == ge::FORMAT_ND &&
                       tensorDesc->GetOriginFormat() == ge::FORMAT_ND;
    const gert::Shape& shape = storageShape->GetStorageShape();
    const std::size_t rank = shape.GetDimNum();
    if (rank > 3U) {
        out.rank = UINT32_MAX;
        return false;
    }
    out.rank = static_cast<uint32_t>(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            out.dims[i] = UINT64_MAX;
            return false;
        }
        out.dims[i] = static_cast<uint64_t>(dim);
    }
    return true;
}

bool ParsePositiveUint64(const std::string& value, uint64_t& out)
{
    if (value.empty() || value.front() == '-' || value.front() == '+') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || end == nullptr || *end != '\0' || parsed == 0U) {
        return false;
    }
    out = static_cast<uint64_t>(parsed);
    return true;
}
} // namespace

void ComputeRasterizerPublicTiling(const RasterizerPublicTilingInputs& in, RasterizerPublicTilingResult& result)
{
    SetFailure(result, RasterizerPublicTilingStatus::kNotImplemented);

    if (!in.v.present || !in.f.present) {
        SetFailure(result, RasterizerPublicTilingStatus::kNullInput);
        return;
    }
    if (in.v.dtype != RasterizerPublicDataType::kFloat32 || in.f.dtype != RasterizerPublicDataType::kInt32 ||
        (in.d.present && in.d.dtype != RasterizerPublicDataType::kFloat32)) {
        SetFailure(result, RasterizerPublicTilingStatus::kDtypeNotSupported);
        return;
    }
    if (!IsRank(in.v, 2U) || in.v.dims[0] == 0U || in.v.dims[1] != 4U || !IsRank(in.f, 2U) || in.f.dims[0] == 0U ||
        in.f.dims[0] > kRasterizerMaxFaces || in.f.dims[1] != 3U || (in.d.present && !IsRank(in.d, 2U)) ||
        !in.v.ndContiguous || !in.f.ndContiguous || (in.d.present && !in.d.ndContiguous) || !in.findices.present ||
        in.findices.dtype != RasterizerPublicDataType::kInt32 || !IsRank(in.findices, 2U) ||
        !in.findices.ndContiguous || !in.barycentric.present ||
        in.barycentric.dtype != RasterizerPublicDataType::kFloat32 || !IsRank(in.barycentric, 3U) ||
        !in.barycentric.ndContiguous) {
        SetFailure(result, RasterizerPublicTilingStatus::kShapeMismatch);
        return;
    }
    if (in.width < 1 || in.width > static_cast<int64_t>(kMaxDimension) || in.height < 1 ||
        in.height > static_cast<int64_t>(kMaxDimension) || !std::isfinite(in.occlusionTruncation) ||
        in.useDepthPrior != 0) {
        SetFailure(result, RasterizerPublicTilingStatus::kAttributeValueOutOfRange);
        return;
    }

    const uint64_t width = static_cast<uint64_t>(in.width);
    const uint64_t height = static_cast<uint64_t>(in.height);
    if (in.d.present && in.d.dims[0] != 0U && in.d.dims[1] != 0U && (in.d.dims[0] != height || in.d.dims[1] != width)) {
        SetFailure(result, RasterizerPublicTilingStatus::kShapeMismatch);
        return;
    }
    if (in.findices.dims[0] != height || in.findices.dims[1] != width || in.barycentric.dims[0] != height ||
        in.barycentric.dims[1] != width || in.barycentric.dims[2] != 3U) {
        SetFailure(result, RasterizerPublicTilingStatus::kShapeMismatch);
        return;
    }
    if (!in.outputAllocationSucceeded) {
        SetFailure(result, RasterizerPublicTilingStatus::kOutOfMemory);
        return;
    }
    if (!in.aivCoreQuerySucceeded || !in.ubQuerySucceeded || !in.blockBytesQuerySucceeded ||
        !in.vectorBytesQuerySucceeded || in.availableAivCores == 0U || in.ubBytes == 0U || in.blockBytes == 0U ||
        in.vectorBytes == 0U || (in.vectorBytes % sizeof(float)) != 0U) {
        SetFailure(result, RasterizerPublicTilingStatus::kPlatformInvalid);
        return;
    }
    if (!in.systemWorkspaceQuerySucceeded || in.systemWorkspaceBytes == 0U) {
        SetFailure(result, RasterizerPublicTilingStatus::kOutOfMemory);
        return;
    }

    uint64_t pixels = 0;
    if (!CheckedMul(height, width, pixels)) {
        SetFailure(result, RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    const uint64_t alignFindices = in.blockBytes / std::gcd(in.blockBytes, kFindicesBytesPerPixel);
    const uint64_t alignBarycentric = in.blockBytes / std::gcd(in.blockBytes, kBarycentricBytesPerPixel);
    const uint64_t alignGcd = std::gcd(alignFindices, alignBarycentric);
    uint64_t pixelAlignment = 0;
    if (!CheckedMul(alignFindices / alignGcd, alignBarycentric, pixelAlignment) || pixelAlignment == 0U) {
        SetFailure(result, RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    uint64_t alignedPixels = 0;
    if (!AlignUp(pixels, pixelAlignment, alignedPixels)) {
        SetFailure(result, RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    const uint64_t vectorAlignment = in.vectorBytes / sizeof(float);
    if (vectorAlignment == 0U || vectorAlignment > std::numeric_limits<uint32_t>::max()) {
        SetFailure(result, RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    // Ascend 950 currently dispatches only PixelDirect. Keeping FaceGroup unreachable
    // avoids a toolchain/runtime VEC timeout caused merely by referencing that branch.
    const uint32_t tilingKey = 0U;
    if (tilingKey == 0U) {
        RasterizerBranch0TilingInputs branchInputs{};
        branchInputs.numFaces = in.f.dims[0];
        branchInputs.height = height;
        branchInputs.width = width;
        branchInputs.forcePixelDirect = true;
        branchInputs.aivCoreQuerySucceeded = in.aivCoreQuerySucceeded;
        branchInputs.ubQuerySucceeded = in.ubQuerySucceeded;
        branchInputs.blockBytesQuerySucceeded = in.blockBytesQuerySucceeded;
        branchInputs.vectorBytesQuerySucceeded = in.vectorBytesQuerySucceeded;
        branchInputs.systemWorkspaceQuerySucceeded = in.systemWorkspaceQuerySucceeded;
        branchInputs.availableAivCores = in.availableAivCores;
        branchInputs.ubBytes = in.ubBytes;
        branchInputs.blockBytes = in.blockBytes;
        branchInputs.vectorBytes = in.vectorBytes;
        branchInputs.systemWorkspaceBytes = in.systemWorkspaceBytes;
        branchInputs.outputAllocationSucceeded = in.outputAllocationSucceeded;
        branchInputs.workspaceAllocationSucceeded = in.workspaceAllocationSucceeded;
        branchInputs.runtimeSizeMax = in.runtimeSizeMax;

        RasterizerBranch0TilingResult branchResult{};
        ComputeRasterizerBranch0Tiling(branchInputs, branchResult);
        if (branchResult.status != RasterizerPublicTilingStatus::kSuccess || !branchResult.branchSelected ||
            !branchResult.kernelDispatchAllowed || branchResult.tilingKey != 0U) {
            SetFailure(result, branchResult.status);
            return;
        }

        RasterizerPublicTilingResult computed{};
        computed.status = branchResult.status;
        computed.kernelDispatchAllowed = branchResult.kernelDispatchAllowed;
        computed.tilingKey = branchResult.tilingKey;
        computed.totalPixels = branchResult.totalPixels;
        computed.alignedPixels = branchResult.alignedPixels;
        computed.pixelAlignment = branchResult.pixelAlignment;
        computed.vectorAlignment = branchResult.vectorAlignment;
        computed.serializedBytes = branchResult.serializedBytes;
        computed.totalWorkspaceBytes = branchResult.totalWorkspaceBytes;
        computed.tilingData = branchResult.tilingData;
        result = computed;
        return;
    }

    RasterizerBranch1TilingInputs branchInputs{};
    branchInputs.numFaces = in.f.dims[0];
    branchInputs.height = height;
    branchInputs.width = width;
    branchInputs.aivCoreQuerySucceeded = in.aivCoreQuerySucceeded;
    branchInputs.ubQuerySucceeded = in.ubQuerySucceeded;
    branchInputs.blockBytesQuerySucceeded = in.blockBytesQuerySucceeded;
    branchInputs.vectorBytesQuerySucceeded = in.vectorBytesQuerySucceeded;
    branchInputs.systemWorkspaceQuerySucceeded = in.systemWorkspaceQuerySucceeded;
    branchInputs.availableAivCores = in.availableAivCores;
    branchInputs.ubBytes = in.ubBytes;
    branchInputs.blockBytes = in.blockBytes;
    branchInputs.vectorBytes = in.vectorBytes;
    branchInputs.systemWorkspaceBytes = in.systemWorkspaceBytes;
    branchInputs.outputAllocationSucceeded = in.outputAllocationSucceeded;
    branchInputs.workspaceAllocationSucceeded = in.workspaceAllocationSucceeded;
    // The context glue performs the real SetScheduleMode(1) call after this
    // pure computation succeeds. Direct Branch-1 UT can inject its failure.
    branchInputs.scheduleModeSetSucceeded = true;
    branchInputs.runtimeSizeMax = in.runtimeSizeMax;

    RasterizerBranch1TilingResult branchResult{};
    ComputeRasterizerBranch1Tiling(branchInputs, branchResult);
    if (branchResult.status != RasterizerPublicTilingStatus::kSuccess || !branchResult.branchSelected ||
        !branchResult.kernelDispatchAllowed || branchResult.tilingKey != 1U || branchResult.scheduleMode != 1U) {
        SetFailure(result, branchResult.status);
        return;
    }

    RasterizerPublicTilingResult computed{};
    computed.status = branchResult.status;
    computed.kernelDispatchAllowed = branchResult.kernelDispatchAllowed;
    computed.tilingKey = branchResult.tilingKey;
    computed.totalPixels = branchResult.totalPixels;
    computed.alignedPixels = branchResult.alignedPixels;
    computed.pixelAlignment = branchResult.pixelAlignment;
    computed.vectorAlignment = branchResult.vectorAlignment;
    computed.serializedBytes = branchResult.serializedBytes;
    computed.totalWorkspaceBytes = branchResult.totalWorkspaceBytes;
    computed.tilingData = branchResult.tilingData;
    result = computed;
}

void ComputeRasterizerBranch0Tiling(const RasterizerBranch0TilingInputs& inputs, RasterizerBranch0TilingResult& result)
{
    auto setFailure = [&result](RasterizerPublicTilingStatus status, bool branchSelected = false,
                                uint32_t tilingKey = UINT32_MAX) {
        result = RasterizerBranch0TilingResult{};
        result.status = status;
        result.branchSelected = branchSelected;
        result.kernelDispatchAllowed = false;
        result.tilingKey = tilingKey;
    };

    setFailure(RasterizerPublicTilingStatus::kNotImplemented);
    if (inputs.height == 0U || inputs.width == 0U || inputs.numFaces == 0U) {
        setFailure(RasterizerPublicTilingStatus::kShapeMismatch);
        return;
    }
    if (!inputs.outputAllocationSucceeded) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory);
        return;
    }
    if (!inputs.aivCoreQuerySucceeded || !inputs.ubQuerySucceeded || !inputs.blockBytesQuerySucceeded ||
        !inputs.vectorBytesQuerySucceeded || inputs.availableAivCores == 0U || inputs.ubBytes == 0U ||
        inputs.blockBytes == 0U || inputs.vectorBytes == 0U || (inputs.vectorBytes % sizeof(float)) != 0U) {
        setFailure(RasterizerPublicTilingStatus::kPlatformInvalid);
        return;
    }
    if (!inputs.systemWorkspaceQuerySucceeded || inputs.systemWorkspaceBytes == 0U) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory);
        return;
    }

    uint64_t pixels = 0U;
    if (!CheckedMul(inputs.height, inputs.width, pixels)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    const uint64_t findicesAlignment = inputs.blockBytes / std::gcd(inputs.blockBytes, kFindicesBytesPerPixel);
    const uint64_t barycentricAlignment = inputs.blockBytes / std::gcd(inputs.blockBytes, kBarycentricBytesPerPixel);
    const uint64_t alignmentGcd = std::gcd(findicesAlignment, barycentricAlignment);
    uint64_t pixelAlignment = 0U;
    if (!CheckedMul(findicesAlignment / alignmentGcd, barycentricAlignment, pixelAlignment) || pixelAlignment == 0U) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    const uint64_t vectorAlignment = inputs.vectorBytes / sizeof(float);
    if (vectorAlignment == 0U || vectorAlignment > std::numeric_limits<uint32_t>::max()) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    uint64_t alignedPixels = 0U;
    if (!AlignUp(pixels, pixelAlignment, alignedPixels)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    // DESIGN §5.5: key 0 is the exact complement of
    // (P < C_hw && F > P), including both equality boundaries.
    const bool branchSelected = inputs.forcePixelDirect || pixels >= inputs.availableAivCores ||
                                inputs.numFaces <= pixels;
    if (!branchSelected) {
        setFailure(RasterizerPublicTilingStatus::kSuccess, false, 1U);
        result.totalPixels = pixels;
        result.alignedPixels = alignedPixels;
        result.pixelAlignment = pixelAlignment;
        result.vectorAlignment = static_cast<uint32_t>(vectorAlignment);
        return;
    }

    uint64_t facePacketBytes = 0U;
    uint64_t planeQuantumBytes = 0U;
    uint64_t minimumWinnerBytes = 0U;
    uint64_t minimumUbBytes = 0U;
    if (!AlignUp(60U, inputs.vectorBytes, facePacketBytes) ||
        !CheckedMul(sizeof(float), vectorAlignment, planeQuantumBytes) ||
        !CheckedMul(kWorkspacePlaneCount, planeQuantumBytes, minimumWinnerBytes) ||
        !CheckedAdd(facePacketBytes, minimumWinnerBytes, minimumUbBytes)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 0U);
        return;
    }
    if (inputs.ubBytes < minimumUbBytes) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory, true, 0U);
        return;
    }

    // DESIGN-BRANCH-0 §2.2: C_t, B_p and C_p describe disjoint contiguous
    // pixel ranges [b*B_p, b*B_p+valid_b), with blockDim=C_p.
    const uint64_t targetCoreNum = std::min(inputs.availableAivCores, CeilDiv(pixels, pixelAlignment));
    uint64_t pixelBlockLen = 0U;
    if (targetCoreNum == 0U || !AlignUp(CeilDiv(pixels, targetCoreNum), pixelAlignment, pixelBlockLen)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 0U);
        return;
    }
    const uint64_t activeCoreNum = CeilDiv(pixels, pixelBlockLen);
    if (activeCoreNum == 0U || activeCoreNum > std::numeric_limits<uint32_t>::max()) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 0U);
        return;
    }

    // DESIGN-BRANCH-0 §2.3: reverse the five-plane UB inequality and align
    // T_0 down to A_v. The minimum-UB check above guarantees at least one A_v.
    uint64_t tileDenominator = 0U;
    if (!CheckedMul(kWorkspacePlaneCount, planeQuantumBytes, tileDenominator)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 0U);
        return;
    }
    const uint64_t tileQuanta = (inputs.ubBytes - facePacketBytes) / tileDenominator;
    uint64_t maxTilePixels = 0U;
    if (!CheckedMul(vectorAlignment, tileQuanta, maxTilePixels)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 0U);
        return;
    }
    const uint64_t boundedTilePixels = std::min(alignedPixels, maxTilePixels);
    uint64_t pixelTileLen = (boundedTilePixels / vectorAlignment) * vectorAlignment;
    pixelTileLen = std::max(pixelTileLen, vectorAlignment);
    if (pixelTileLen > std::numeric_limits<uint32_t>::max()) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 0U);
        return;
    }

    if (sizeof(RasterizerTilingData) > inputs.runtimeSizeMax || inputs.systemWorkspaceBytes > inputs.runtimeSizeMax ||
        !inputs.workspaceAllocationSucceeded) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory, true, 0U);
        return;
    }

    RasterizerBranch0TilingResult computed{};
    computed.status = RasterizerPublicTilingStatus::kSuccess;
    computed.branchSelected = true;
    computed.kernelDispatchAllowed = true;
    computed.tilingKey = 0U;
    computed.totalPixels = pixels;
    computed.alignedPixels = alignedPixels;
    computed.pixelAlignment = pixelAlignment;
    computed.vectorAlignment = static_cast<uint32_t>(vectorAlignment);
    computed.targetCoreNum = targetCoreNum;
    computed.blockDim = static_cast<uint32_t>(activeCoreNum);
    computed.serializedBytes = sizeof(RasterizerTilingData);
    computed.totalWorkspaceBytes = inputs.systemWorkspaceBytes;

    RasterizerTilingData& data = computed.tilingData;
    data.numFaces = inputs.numFaces;
    data.totalPixels = pixels;
    data.height = static_cast<uint32_t>(inputs.height);
    data.width = static_cast<uint32_t>(inputs.width);
    data.activeCoreNum = static_cast<uint32_t>(activeCoreNum);
    data.mergeCoreNum = 0U;
    data.pixelBlockLen = pixelBlockLen;
    data.pixelTileLen = static_cast<uint32_t>(pixelTileLen);
    data.workspacePixelStride = 0U;
    data.workspacePlaneStrideBytes = 0U;
    data.workspaceCoreStrideBytes = 0U;
    data.userWorkspaceBytes = 0U;
    data.systemWorkspaceOffsetBytes = 0U;
    data.systemWorkspaceBytes = inputs.systemWorkspaceBytes;
    result = computed;
}

void ComputeRasterizerBranch1Tiling(const RasterizerBranch1TilingInputs& inputs, RasterizerBranch1TilingResult& result)
{
    auto setFailure = [&result](RasterizerPublicTilingStatus status, bool branchSelected = false,
                                uint32_t tilingKey = UINT32_MAX) {
        result = RasterizerBranch1TilingResult{};
        result.status = status;
        result.branchSelected = branchSelected;
        result.kernelDispatchAllowed = false;
        result.tilingKey = tilingKey;
    };

    setFailure(RasterizerPublicTilingStatus::kNotImplemented);
    if (inputs.height == 0U || inputs.width == 0U || inputs.numFaces == 0U) {
        setFailure(RasterizerPublicTilingStatus::kShapeMismatch);
        return;
    }
    if (!inputs.outputAllocationSucceeded) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory);
        return;
    }
    if (!inputs.aivCoreQuerySucceeded || !inputs.ubQuerySucceeded || !inputs.blockBytesQuerySucceeded ||
        !inputs.vectorBytesQuerySucceeded || inputs.availableAivCores == 0U || inputs.ubBytes == 0U ||
        inputs.blockBytes == 0U || inputs.vectorBytes == 0U || (inputs.vectorBytes % sizeof(float)) != 0U) {
        setFailure(RasterizerPublicTilingStatus::kPlatformInvalid);
        return;
    }
    if (!inputs.systemWorkspaceQuerySucceeded || inputs.systemWorkspaceBytes == 0U) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory);
        return;
    }

    uint64_t pixels = 0U;
    if (!CheckedMul(inputs.height, inputs.width, pixels)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    const uint64_t findicesAlignment = inputs.blockBytes / std::gcd(inputs.blockBytes, kFindicesBytesPerPixel);
    const uint64_t barycentricAlignment = inputs.blockBytes / std::gcd(inputs.blockBytes, kBarycentricBytesPerPixel);
    const uint64_t alignmentGcd = std::gcd(findicesAlignment, barycentricAlignment);
    uint64_t pixelAlignment = 0U;
    if (!CheckedMul(findicesAlignment / alignmentGcd, barycentricAlignment, pixelAlignment) || pixelAlignment == 0U) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    const uint64_t vectorAlignment = inputs.vectorBytes / sizeof(float);
    if (vectorAlignment == 0U || vectorAlignment > std::numeric_limits<uint32_t>::max()) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    uint64_t alignedPixels = 0U;
    if (!AlignUp(pixels, pixelAlignment, alignedPixels)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow);
        return;
    }

    // DESIGN §5.5: FACE_GROUP is selected only by the strict conjunction.
    // Its complement (including either equality boundary) belongs to key 0.
    const bool branchSelected = pixels < inputs.availableAivCores && inputs.numFaces > pixels;
    if (!branchSelected) {
        setFailure(RasterizerPublicTilingStatus::kSuccess, false, 0U);
        result.totalPixels = pixels;
        result.alignedPixels = alignedPixels;
        result.pixelAlignment = pixelAlignment;
        result.vectorAlignment = static_cast<uint32_t>(vectorAlignment);
        return;
    }

    // DESIGN-BRANCH-1 §2.4: one aligned 60-byte face packet plus ten
    // simultaneously-live, vector-aligned float32 planes.
    uint64_t ubFaceBytes = 0U;
    uint64_t planeQuantumBytes = 0U;
    uint64_t tenPlaneQuantumBytes = 0U;
    uint64_t minimumUbBytes = 0U;
    if (!AlignUp(60U, inputs.vectorBytes, ubFaceBytes) ||
        !CheckedMul(sizeof(float), vectorAlignment, planeQuantumBytes) ||
        !CheckedMul(10U, planeQuantumBytes, tenPlaneQuantumBytes) ||
        !CheckedAdd(ubFaceBytes, tenPlaneQuantumBytes, minimumUbBytes)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 1U);
        return;
    }
    if (inputs.ubBytes < minimumUbBytes) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory, true, 1U);
        return;
    }

    // Phase 1 uses quotient/remainder face slicing. These global values let
    // every kernel derive [b*Q_f+min(b,R_f), ... ) without overlap or gaps.
    const uint64_t faceCoreNum = std::min(inputs.availableAivCores, inputs.numFaces);
    const uint64_t faceQuotient = inputs.numFaces / faceCoreNum;
    const uint64_t faceRemainder = inputs.numFaces % faceCoreNum;

    // Phase 2 partitions [0,P) into A_p-aligned contiguous pixel blocks.
    const uint64_t mergeCoreNum = std::min(faceCoreNum, CeilDiv(pixels, pixelAlignment));
    uint64_t pixelBlockLen = 0U;
    if (mergeCoreNum == 0U || !AlignUp(CeilDiv(pixels, mergeCoreNum), pixelAlignment, pixelBlockLen)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 1U);
        return;
    }

    const uint64_t tileQuanta = (inputs.ubBytes - ubFaceBytes) / tenPlaneQuantumBytes;
    uint64_t maxTilePixels = 0U;
    if (!CheckedMul(vectorAlignment, tileQuanta, maxTilePixels)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 1U);
        return;
    }
    const uint64_t boundedTilePixels = std::min(alignedPixels, maxTilePixels);
    uint64_t pixelTileLen = (boundedTilePixels / vectorAlignment) * vectorAlignment;
    pixelTileLen = std::max(pixelTileLen, vectorAlignment);
    if (faceCoreNum > std::numeric_limits<uint32_t>::max() || mergeCoreNum > std::numeric_limits<uint32_t>::max() ||
        pixelTileLen > std::numeric_limits<uint32_t>::max()) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 1U);
        return;
    }

    // DESIGN-BRANCH-1 §2.5/§8: five core-major planes followed immediately
    // by the framework system workspace. Check every multiplication/addition.
    uint64_t workspacePlaneStrideBytes = 0U;
    uint64_t workspaceCoreStrideBytes = 0U;
    uint64_t userWorkspaceBytes = 0U;
    uint64_t totalWorkspaceBytes = 0U;
    if (!CheckedMul(sizeof(float), alignedPixels, workspacePlaneStrideBytes) ||
        !CheckedMul(kWorkspacePlaneCount, workspacePlaneStrideBytes, workspaceCoreStrideBytes) ||
        !CheckedMul(faceCoreNum, workspaceCoreStrideBytes, userWorkspaceBytes) ||
        !CheckedAdd(userWorkspaceBytes, inputs.systemWorkspaceBytes, totalWorkspaceBytes)) {
        setFailure(RasterizerPublicTilingStatus::kArithmeticOverflow, true, 1U);
        return;
    }
    if (sizeof(RasterizerTilingData) > inputs.runtimeSizeMax || totalWorkspaceBytes > inputs.runtimeSizeMax ||
        !inputs.workspaceAllocationSucceeded) {
        setFailure(RasterizerPublicTilingStatus::kOutOfMemory, true, 1U);
        return;
    }
    if (!inputs.scheduleModeSetSucceeded) {
        setFailure(RasterizerPublicTilingStatus::kPlatformInvalid, true, 1U);
        return;
    }

    RasterizerBranch1TilingResult computed{};
    computed.status = RasterizerPublicTilingStatus::kSuccess;
    computed.branchSelected = true;
    computed.kernelDispatchAllowed = true;
    computed.tilingKey = 1U;
    computed.totalPixels = pixels;
    computed.alignedPixels = alignedPixels;
    computed.pixelAlignment = pixelAlignment;
    computed.vectorAlignment = static_cast<uint32_t>(vectorAlignment);
    computed.faceCoreNum = faceCoreNum;
    computed.faceQuotient = faceQuotient;
    computed.faceRemainder = faceRemainder;
    computed.mergeCoreNum = mergeCoreNum;
    computed.blockDim = static_cast<uint32_t>(faceCoreNum);
    computed.scheduleMode = 1U;
    computed.ubFaceBytes = ubFaceBytes;
    computed.maxTilePixels = maxTilePixels;
    computed.serializedBytes = sizeof(RasterizerTilingData);
    computed.totalWorkspaceBytes = totalWorkspaceBytes;

    RasterizerTilingData& data = computed.tilingData;
    data.numFaces = inputs.numFaces;
    data.totalPixels = pixels;
    data.height = static_cast<uint32_t>(inputs.height);
    data.width = static_cast<uint32_t>(inputs.width);
    data.activeCoreNum = static_cast<uint32_t>(faceCoreNum);
    data.mergeCoreNum = static_cast<uint32_t>(mergeCoreNum);
    data.pixelBlockLen = pixelBlockLen;
    data.pixelTileLen = static_cast<uint32_t>(pixelTileLen);
    data.workspacePixelStride = alignedPixels;
    data.workspacePlaneStrideBytes = workspacePlaneStrideBytes;
    data.workspaceCoreStrideBytes = workspaceCoreStrideBytes;
    data.userWorkspaceBytes = userWorkspaceBytes;
    data.systemWorkspaceOffsetBytes = userWorkspaceBytes;
    data.systemWorkspaceBytes = inputs.systemWorkspaceBytes;
    result = computed;
}

static ge::graphStatus RasterizerTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE(context, "Rasterizer: tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    RasterizerPublicTilingInputs inputs{};
    const gert::Tensor* vTensor = context->GetRequiredInputTensor(0U);
    const gert::StorageShape* vShape = context->GetRequiredInputShape(0U);
    const gert::Tensor* fTensor = context->GetRequiredInputTensor(1U);
    const gert::StorageShape* fShape = context->GetRequiredInputShape(1U);
    (void)FillTensorDesc(vTensor, vShape, inputs.v);
    (void)FillTensorDesc(fTensor, fShape, inputs.f);

    const gert::Tensor* dTensor = context->GetOptionalInputTensor(2U);
    const gert::StorageShape* dShape = context->GetOptionalInputShape(2U);
    if ((dTensor == nullptr) != (dShape == nullptr)) {
        OP_LOGE(context, "Rasterizer: optional depth tensor and shape must both exist or both be null");
        return ge::GRAPH_FAILED;
    }
    if (dTensor != nullptr && !FillTensorDesc(dTensor, dShape, inputs.d)) {
        OP_LOGE(context, "Rasterizer: depth tensor desc is invalid, present=%d, rank=%u",
                static_cast<int>(inputs.d.present), inputs.d.rank);
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* findicesShape = context->GetOutputShape(0U);
    const gert::StorageShape* barycentricShape = context->GetOutputShape(1U);
    const bool findicesDescReady = FillOutputDesc(findicesShape, context->GetOutputDesc(0U), inputs.findices);
    const bool barycentricDescReady = FillOutputDesc(barycentricShape, context->GetOutputDesc(1U), inputs.barycentric);
    inputs.outputAllocationSucceeded = findicesShape != nullptr && barycentricShape != nullptr;
    if (!findicesDescReady || !barycentricDescReady) {
        OP_LOGE(context, "Rasterizer: output desc is invalid, findicesReady=%d, barycentricReady=%d",
                static_cast<int>(findicesDescReady), static_cast<int>(barycentricDescReady));
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context, "Rasterizer: runtime attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const int64_t* width = attrs->GetInt(0U);
    const int64_t* height = attrs->GetInt(1U);
    const float* occlusionTruncation = attrs->GetFloat(2U);
    const int64_t* useDepthPrior = attrs->GetInt(3U);
    if (width == nullptr || height == nullptr || occlusionTruncation == nullptr || useDepthPrior == nullptr) {
        OP_LOGE(context, "Rasterizer: attr width/height/occlusion_truncation/use_depth_prior is missing");
        return ge::GRAPH_FAILED;
    }
    inputs.width = *width;
    inputs.height = *height;
    inputs.occlusionTruncation = static_cast<double>(*occlusionTruncation);
    inputs.useDepthPrior = *useDepthPrior;

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        inputs.availableAivCores = std::min(platform.GetCoreNumAiv(), kReviewedAivCoreBudget);
        inputs.aivCoreQuerySucceeded = true;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, inputs.ubBytes);
        inputs.ubQuerySucceeded = true;
        inputs.vectorBytes = platform.GetVecRegLen();
        inputs.vectorBytesQuerySucceeded = true;
        inputs.systemWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
        inputs.systemWorkspaceQuerySucceeded = true;

        std::string blockBytes;
        uint64_t parsedBlockBytes = 0;
        if (platformInfo->GetPlatformRes("VectorCoreSpec", "ubblock_size", blockBytes) &&
            ParsePositiveUint64(blockBytes, parsedBlockBytes)) {
            inputs.blockBytes = parsedBlockBytes;
            inputs.blockBytesQuerySucceeded = true;
        }
    }

    inputs.workspaceAllocationSucceeded = true;
    inputs.runtimeSizeMax = static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());

    RasterizerPublicTilingResult result{};
    ComputeRasterizerPublicTiling(inputs, result);
    if (result.status != RasterizerPublicTilingStatus::kSuccess || !result.kernelDispatchAllowed) {
        OP_LOGE(context,
                "Rasterizer: public tiling failed, status=%d(%s), tilingKey=%u, width=%lld, height=%lld, "
                "availableAivCores=%llu, ubBytes=%llu, vectorBytes=%llu",
                static_cast<int32_t>(result.status), RasterizerPublicTilingStatusName(result.status), result.tilingKey,
                static_cast<long long>(inputs.width), static_cast<long long>(inputs.height),
                static_cast<unsigned long long>(inputs.availableAivCores),
                static_cast<unsigned long long>(inputs.ubBytes), static_cast<unsigned long long>(inputs.vectorBytes));
        return ge::GRAPH_FAILED;
    }

    size_t* workspaceSizes = context->GetWorkspaceSizes(1U);
    gert::TilingData* rawTilingData = context->GetRawTilingData();
    if (workspaceSizes == nullptr || rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
        rawTilingData->GetCapacity() < sizeof(RasterizerTilingData)) {
        OP_LOGE(context, "Rasterizer: tiling data buffer is invalid, capacity=%zu, required=%zu",
                rawTilingData == nullptr ? std::size_t{0U} : rawTilingData->GetCapacity(),
                sizeof(RasterizerTilingData));
        return ge::GRAPH_FAILED;
    }

    const uint64_t oldTilingKey = context->GetTilingKey();
    const uint32_t oldBlockDim = context->GetBlockDim();
    const uint32_t oldScheduleMode = context->GetScheduleMode();
    const uint32_t scheduleMode = result.tilingKey == 1U ? 1U : 0U;
    if (context->SetTilingKey(result.tilingKey) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Rasterizer: SetTilingKey failed, tilingKey=%u", result.tilingKey);
        return ge::GRAPH_FAILED;
    }
    if (context->SetBlockDim(result.tilingData.activeCoreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Rasterizer: SetBlockDim failed, blockDim=%u", result.tilingData.activeCoreNum);
        (void)context->SetTilingKey(oldTilingKey);
        return ge::GRAPH_FAILED;
    }
    if (context->SetScheduleMode(scheduleMode) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Rasterizer: SetScheduleMode failed, scheduleMode=%u", scheduleMode);
        (void)context->SetBlockDim(oldBlockDim);
        (void)context->SetTilingKey(oldTilingKey);
        (void)context->SetScheduleMode(oldScheduleMode);
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context, "Rasterizer: tiling succeeded, tilingKey=%u, scheduleMode=%u, blockDim=%u, workspaceBytes=%llu",
            result.tilingKey, scheduleMode, result.tilingData.activeCoreNum,
            static_cast<unsigned long long>(result.totalWorkspaceBytes));

    std::memcpy(rawTilingData->GetData(), &result.tilingData, sizeof(result.tilingData));
    rawTilingData->SetDataSize(sizeof(result.tilingData));
    workspaceSizes[0] = static_cast<size_t>(result.totalWorkspaceBytes);
    return ge::GRAPH_SUCCESS;
}
struct RasterizerCompileInfo {};

static ge::graphStatus TilingParseForRasterizer([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Rasterizer).Tiling(RasterizerTilingFunc).TilingParse<RasterizerCompileInfo>(TilingParseForRasterizer);

} // namespace optiling
