/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RASTERIZER_TILING_STRUCT_H
#define RASTERIZER_TILING_STRUCT_H

#include <cstddef>
#include <cstdint>

// Host and Kernel share this exact wire-format declaration. All pixel counts and
// lengths are measured in elements/pixels unless a field name explicitly ends in
// Bytes. The tiling key is carried by the framework and is intentionally absent.
struct alignas(8) RasterizerTilingData {
    uint64_t numFaces;                   // faces, F >= 1
    uint64_t totalPixels;                // pixels, height * width
    uint32_t height;                     // pixels
    uint32_t width;                      // pixels
    uint32_t activeCoreNum;              // active vector-core count; equals blockDim
    uint32_t mergeCoreNum;               // cores in key1 merge phase; key0 is 0
    uint64_t pixelBlockLen;              // pixels assigned per active/merge core
    uint32_t pixelTileLen;               // pixels per UB tile
    uint64_t workspacePixelStride;       // pixels/elements per workspace plane
    uint64_t workspacePlaneStrideBytes;  // bytes between workspace planes
    uint64_t workspaceCoreStrideBytes;   // bytes between source-core workspaces
    uint64_t userWorkspaceBytes;         // bytes in the operator-owned workspace
    uint64_t systemWorkspaceOffsetBytes; // byte offset of framework system workspace
    uint64_t systemWorkspaceBytes;       // bytes in framework system workspace
};

constexpr std::size_t kRasterizerTilingDataSerializedSize = 96U;

// findices (int32) stores the 1-based winning face index, and the kernel keeps the
// 0-based face index inside int32 with INT32_MAX reserved as the "no winner"
// sentinel, so the interface can only serve numFaces <= INT32_MAX. Larger inputs
// must be rejected by Host tiling before dispatch.
constexpr uint64_t kRasterizerMaxFaces = 2147483647U;

constexpr bool RasterizerTilingDataHasValidCommonFields(const RasterizerTilingData& data) noexcept
{
    return data.numFaces >= 1U && data.numFaces <= kRasterizerMaxFaces && data.height >= 1U && data.height <= 4096U &&
           data.width >= 1U && data.width <= 4096U &&
           data.totalPixels == static_cast<uint64_t>(data.height) * data.width && data.totalPixels >= 1U &&
           data.totalPixels <= UINT64_C(16777216) && data.activeCoreNum >= 1U && data.pixelBlockLen >= 1U &&
           data.pixelTileLen >= 1U;
}

// DESIGN-BRANCH-0 §1: direct pixel ownership has no merge phase or user workspace.
constexpr bool RasterizerTilingDataIsKey0Consistent(const RasterizerTilingData& data) noexcept
{
    return RasterizerTilingDataHasValidCommonFields(data) && data.mergeCoreNum == 0U &&
           data.workspacePixelStride == 0U && data.workspacePlaneStrideBytes == 0U &&
           data.workspaceCoreStrideBytes == 0U && data.userWorkspaceBytes == 0U &&
           data.systemWorkspaceOffsetBytes == 0U;
}

// DESIGN-BRANCH-1 §1: five uint32 workspace planes per source core.
constexpr bool RasterizerTilingDataIsKey1Consistent(const RasterizerTilingData& data) noexcept
{
    constexpr uint64_t kUint64Max = ~UINT64_C(0);
    if (!RasterizerTilingDataHasValidCommonFields(data) || data.mergeCoreNum < 1U ||
        data.mergeCoreNum > data.activeCoreNum || data.workspacePixelStride < data.totalPixels ||
        data.workspacePixelStride > kUint64Max / 4U) {
        return false;
    }
    const uint64_t expectedPlaneStride = data.workspacePixelStride * 4U;
    if (data.workspacePlaneStrideBytes != expectedPlaneStride || expectedPlaneStride > kUint64Max / 5U) {
        return false;
    }
    const uint64_t expectedCoreStride = expectedPlaneStride * 5U;
    if (data.workspaceCoreStrideBytes != expectedCoreStride || expectedCoreStride > kUint64Max / data.activeCoreNum) {
        return false;
    }
    const uint64_t expectedUserWorkspace = expectedCoreStride * static_cast<uint64_t>(data.activeCoreNum);
    return data.userWorkspaceBytes == expectedUserWorkspace && data.systemWorkspaceOffsetBytes == expectedUserWorkspace;
}

static_assert(sizeof(uint32_t) == 4U, "Rasterizer tiling requires 32-bit uint32_t");
static_assert(sizeof(uint64_t) == 8U, "Rasterizer tiling requires 64-bit uint64_t");
static_assert(alignof(RasterizerTilingData) == 8U, "RasterizerTilingData must be 8-byte aligned");
static_assert(sizeof(RasterizerTilingData) == kRasterizerTilingDataSerializedSize,
              "RasterizerTilingData serialized length changed");
static_assert(sizeof(RasterizerTilingData) % 8U == 0U, "RasterizerTilingData serialized length must be 8-byte aligned");
static_assert(offsetof(RasterizerTilingData, numFaces) == 0U, "numFaces offset changed");
static_assert(offsetof(RasterizerTilingData, totalPixels) == 8U, "totalPixels offset changed");
static_assert(offsetof(RasterizerTilingData, height) == 16U, "height offset changed");
static_assert(offsetof(RasterizerTilingData, width) == 20U, "width offset changed");
static_assert(offsetof(RasterizerTilingData, activeCoreNum) == 24U, "activeCoreNum offset changed");
static_assert(offsetof(RasterizerTilingData, mergeCoreNum) == 28U, "mergeCoreNum offset changed");
static_assert(offsetof(RasterizerTilingData, pixelBlockLen) == 32U, "pixelBlockLen offset changed");
static_assert(offsetof(RasterizerTilingData, pixelTileLen) == 40U, "pixelTileLen offset changed");
static_assert(offsetof(RasterizerTilingData, workspacePixelStride) == 48U, "workspacePixelStride offset changed");
static_assert(offsetof(RasterizerTilingData, workspacePlaneStrideBytes) == 56U,
              "workspacePlaneStrideBytes offset changed");
static_assert(offsetof(RasterizerTilingData, workspaceCoreStrideBytes) == 64U,
              "workspaceCoreStrideBytes offset changed");
static_assert(offsetof(RasterizerTilingData, userWorkspaceBytes) == 72U, "userWorkspaceBytes offset changed");
static_assert(offsetof(RasterizerTilingData, systemWorkspaceOffsetBytes) == 80U,
              "systemWorkspaceOffsetBytes offset changed");
static_assert(offsetof(RasterizerTilingData, systemWorkspaceBytes) == 88U, "systemWorkspaceBytes offset changed");

#endif // RASTERIZER_TILING_STRUCT_H
