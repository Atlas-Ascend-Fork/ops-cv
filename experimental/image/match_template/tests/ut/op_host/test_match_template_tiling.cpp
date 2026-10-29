/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "../../../op_kernel/match_template_tiling_data.h"
#include "../../../op_kernel/match_template_tiling_key.h"
#include "tiling_case_executor.h"

namespace {

constexpr int64_t kDataBlockBytes = 32;
constexpr int64_t kSystemWorkspaceBytes = 16 * 1024 * 1024;
uint8_t g_compileInfo = 0;

int64_t AlignUp(int64_t value, int64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

gert::TilingContextPara MakeContext(int64_t batch, int64_t imageH, int64_t imageW, int64_t channels, int64_t templateH,
                                    int64_t templateW, int64_t method, ge::DataType imageType = ge::DT_FLOAT,
                                    ge::DataType templateType = ge::DT_FLOAT, uint64_t coreNum = 48,
                                    uint64_t ubSize = 262144)
{
    const int64_t outputH = imageH - templateH + 1;
    const int64_t outputW = imageW - templateW + 1;
    gert::StorageShape imageShape{{batch, imageH, imageW, channels}, {batch, imageH, imageW, channels}};
    gert::StorageShape templateShape{{1, templateH, templateW, channels}, {1, templateH, templateW, channels}};
    gert::StorageShape scoreShape{{batch, outputH, outputW}, {batch, outputH, outputW}};
    gert::StorageShape bestScoreShape{{batch}, {batch}};
    gert::StorageShape bestLocationShape{{batch, 2}, {batch, 2}};
    return gert::TilingContextPara(
        "MatchTemplate", {{imageShape, imageType, ge::FORMAT_ND}, {templateShape, templateType, ge::FORMAT_ND}},
        {{scoreShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {bestScoreShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {bestLocationShape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("method", Ops::Cv::AnyValue::CreateFrom<int64_t>(method))}, &g_compileInfo,
        "Ascend910B", coreNum, ubSize, sizeof(MatchTemplateTilingData));
}

const MatchTemplateTilingData* GetTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(MatchTemplateTilingData));
    return reinterpret_cast<const MatchTemplateTilingData*>(info.tilingData.get());
}

void ExpectCompactWorkspaceLayout(const TilingInfo& info, const MatchTemplateTilingData& data)
{
    const int64_t entries = data.batch * data.usedCores;
    const int64_t scoreBytes = AlignUp(entries * static_cast<int64_t>(sizeof(float)), kDataBlockBytes);
    const int64_t indexBytes = AlignUp(entries * static_cast<int64_t>(sizeof(int64_t)), kDataBlockBytes);
    EXPECT_EQ(data.workspaceIndexOffset, scoreBytes);
    EXPECT_EQ(data.workspacePackedImageOffset, scoreBytes + indexBytes);
    EXPECT_EQ(data.workspaceIndexOffset % kDataBlockBytes, 0);
    EXPECT_EQ(data.workspacePackedImageOffset % kDataBlockBytes, 0);
    EXPECT_EQ(data.packedImageBytes % kDataBlockBytes, 0);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GE(info.workspaceSizes[0],
              static_cast<size_t>(kSystemWorkspaceBytes + scoreBytes + indexBytes + data.packedImageBytes));
}

// Account for the buffers actually allocated by KernelMatchTemplate::Init,
// independently of the Host's candidate-selection calculations.
uint64_t KernelUbBytes(const MatchTemplateTilingData& data, ge::DataType dtype)
{
    uint64_t bytes = data.templateBufferBytes + data.haloBufferBytes + data.templateFloatBufferBytes +
                     data.haloFloatBufferBytes + data.templateStatsBufferBytes;
    const bool compact = data.useC3VectorTile && dtype == ge::DT_UINT8 && data.packedImageBytes > 0 &&
                         (data.method == 2 || data.method == 5);
    if (data.useHaloTile) {
        bytes += 9 * 512 * sizeof(float);
        if (data.channels == 3) {
            bytes += 3 * 576 * sizeof(float);
        }
        if (data.useC3VectorTile) {
            const int64_t planes = data.method == 2 ? 1 : 5;
            bytes += planes * data.tileH * 512 * sizeof(float);
        }
    }
    bytes += !data.useC3VectorTile || (data.method == 2 && !compact) ?
                 data.tileH * AlignUp(data.tileW * sizeof(float), kDataBlockBytes) :
                 kDataBlockBytes;
    return bytes + 2 * kDataBlockBytes;
}

TEST(MatchTemplateTiling, SelectsHaloTileAndConsistentWorkspace)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext(1, 64, 64, 1, 8, 8, 0), info));
    const auto* data = GetTilingData(info);

    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(MATCH_TEMPLATE_SCH_MODE_HALO_TILE, 0));
    EXPECT_EQ(data->batch, 1);
    EXPECT_EQ(data->outputH, 57);
    EXPECT_EQ(data->outputW, 57);
    EXPECT_EQ(data->useHaloTile, 1);
    EXPECT_GT(data->tileH, 0);
    EXPECT_GT(data->tileW, 0);
    EXPECT_EQ(data->tileRows, (data->outputH + data->tileH - 1) / data->tileH);
    EXPECT_EQ(data->tileCols, (data->outputW + data->tileW - 1) / data->tileW);
    EXPECT_EQ(data->tilesPerBatch, data->tileRows * data->tileCols);
    EXPECT_EQ(data->totalTiles, data->batch * data->tilesPerBatch);
    EXPECT_EQ(info.blockNum, static_cast<size_t>(data->usedCores));

    ExpectCompactWorkspaceLayout(info, *data);
}

TEST(MatchTemplateTiling, WorkspaceGrowthOnlyIncludesAlignedBuffers)
{
    size_t systemWorkspaceBytes = 0;
    for (const bool usePrepack : {false, true}) {
        for (const int64_t batch : {1, 3}) {
            for (const uint64_t coreNum : {1U, 3U, 48U}) {
                TilingInfo info;
                const auto context = usePrepack ?
                                         MakeContext(batch, 512, 512, 3, 16, 16, 2, ge::DT_UINT8, ge::DT_UINT8,
                                                     coreNum) :
                                         MakeContext(batch, 64, 64, 1, 8, 8, 0, ge::DT_FLOAT, ge::DT_FLOAT, coreNum);
                ASSERT_TRUE(ExecuteTiling(context, info));
                const auto* data = GetTilingData(info);
                const int64_t expectedCores = data->totalTiles < static_cast<int64_t>(coreNum) ?
                                                  data->totalTiles :
                                                  static_cast<int64_t>(coreNum);
                EXPECT_EQ(data->usedCores, expectedCores);
                EXPECT_EQ(data->packedImageBytes, usePrepack ? batch * 512 * 512 * 3 * 2 : 0);
                ExpectCompactWorkspaceLayout(info, *data);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
                const auto userWorkspaceBytes = static_cast<size_t>(data->workspacePackedImageOffset +
                                                                    data->packedImageBytes);
                ASSERT_GE(info.workspaceSizes[0], userWorkspaceBytes);
                if (systemWorkspaceBytes == 0) {
                    systemWorkspaceBytes = info.workspaceSizes[0] - userWorkspaceBytes;
                }
                EXPECT_EQ(info.workspaceSizes[0], systemWorkspaceBytes + userWorkspaceBytes);
            }
        }
    }
}

TEST(MatchTemplateTiling, ProducesMethodSpecializedTilingKeys)
{
    for (int64_t method = 0; method < 6; ++method) {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(MakeContext(1, 32, 40, 3, 5, 7, method), info));
        EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(MATCH_TEMPLATE_SCH_MODE_HALO_TILE, static_cast<uint32_t>(method)));
        EXPECT_EQ(GetTilingData(info)->method, method);
    }
}

TEST(MatchTemplateTiling, CoversNonDivisibleTailTiles)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext(2, 73, 79, 3, 7, 11, 5), info));
    const auto* data = GetTilingData(info);

    EXPECT_EQ(data->outputH, 67);
    EXPECT_EQ(data->outputW, 69);
    EXPECT_GE(data->tileRows * data->tileH, data->outputH);
    EXPECT_LT((data->tileRows - 1) * data->tileH, data->outputH);
    EXPECT_GE(data->tileCols * data->tileW, data->outputW);
    EXPECT_LT((data->tileCols - 1) * data->tileW, data->outputW);
    EXPECT_EQ(data->totalTiles, data->batch * data->tileRows * data->tileCols);
    ExpectCompactWorkspaceLayout(info, *data);
}

TEST(MatchTemplateTiling, SelectsGmFallbackWhenTemplateExceedsUbBudget)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext(1, 250, 250, 1, 249, 249, 5), info));
    const auto* data = GetTilingData(info);

    EXPECT_EQ(data->useHaloTile, 0);
    EXPECT_EQ(data->templateBufferBytes, 0);
    EXPECT_EQ(data->haloBufferBytes, 0);
    EXPECT_EQ(data->templateFloatBufferBytes, 0);
    EXPECT_EQ(data->haloFloatBufferBytes, 0);
    EXPECT_EQ(data->templateElements, 249 * 249);
    ExpectCompactWorkspaceLayout(info, *data);
}

TEST(MatchTemplateTiling, SelectsC3VectorTileForTargetCases)
{
    for (const ge::DataType dtype : {ge::DT_FLOAT16, ge::DT_UINT8}) {
        for (int64_t method = 2; method <= 5; ++method) {
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(MakeContext(1, 512, 512, 3, 16, 16, method, dtype, dtype), info));
            const auto* data = GetTilingData(info);

            EXPECT_EQ(info.tilingKey,
                      GET_TPL_TILING_KEY(MATCH_TEMPLATE_SCH_MODE_HALO_TILE, static_cast<uint32_t>(method)));
            EXPECT_EQ(data->useHaloTile, 1);
            EXPECT_EQ(data->useC3VectorTile, 1);
            const bool compactUint8C3 = dtype == ge::DT_UINT8 && (method == 2 || method == 5);
            const int64_t expectedTileH = method == 2 ? 16 : 4;
            EXPECT_EQ(data->tileH, expectedTileH);
            const int64_t expectedTileW = dtype == ge::DT_UINT8 && !compactUint8C3 ? 256 : 512;
            const int64_t expectedTileCols = dtype == ge::DT_UINT8 && !compactUint8C3 ? 2 : 1;
            EXPECT_EQ(data->tileW, expectedTileW);
            const int64_t expectedTileRows = method == 2 ? 32 : 125;
            EXPECT_EQ(data->tileRows, expectedTileRows);
            EXPECT_EQ(data->tileCols, expectedTileCols);
            EXPECT_EQ(data->tilesPerBatch, expectedTileRows * expectedTileCols);
            EXPECT_EQ(data->totalTiles, expectedTileRows * expectedTileCols);
            if (compactUint8C3) {
                const int64_t expectedFp16HaloBytes = (16 + expectedTileH - 1) *
                                                      AlignUp((expectedTileW + 16 - 1) * 3 * 2, kDataBlockBytes);
                EXPECT_EQ(data->haloBufferBytes, 16 * 1024);
                EXPECT_EQ(data->haloFloatBufferBytes, expectedFp16HaloBytes);
                EXPECT_GT(data->packedImageBytes, 0);
                EXPECT_EQ(data->prepackScratchBytes, 16 * 1024);
            } else if (dtype == ge::DT_UINT8) {
                EXPECT_GT(data->haloFloatBufferBytes, 0);
            }
            const int64_t expectedTiles = expectedTileRows * expectedTileCols;
            const int64_t expectedUsedCores = expectedTiles < 48 ? expectedTiles : 48;
            EXPECT_EQ(data->usedCores, expectedUsedCores);

            ExpectCompactWorkspaceLayout(info, *data);
        }
    }

    TilingInfo batchInfo;
    ASSERT_TRUE(ExecuteTiling(MakeContext(2, 512, 512, 3, 16, 16, 5, ge::DT_FLOAT16, ge::DT_FLOAT16), batchInfo));
    const auto* batchData = GetTilingData(batchInfo);
    EXPECT_EQ(batchData->useC3VectorTile, 1);
    EXPECT_EQ(batchData->totalTiles, 250);
    EXPECT_EQ(batchData->usedCores, 48);
    ExpectCompactWorkspaceLayout(batchInfo, *batchData);
}

TEST(MatchTemplateTiling, BudgetsFullScoreBufferForHalfCcorrTail)
{
    constexpr uint64_t reserve = 16 * 1024;
    // 1536 template + 98208 halo + 18432 vector scratch + 6912 planar
    // + 32768 C3 accumulators + 32768 generic scores + 64 local results.
    constexpr uint64_t required = 190688;
    TilingInfo fits;
    ASSERT_TRUE(
        ExecuteTiling(MakeContext(1, 512, 512, 3, 16, 16, 2, ge::DT_FLOAT16, ge::DT_FLOAT16, 48, 262144), fits));
    const auto* full = GetTilingData(fits);
    EXPECT_EQ(full->useC3VectorTile, 1);
    EXPECT_EQ(full->tileH, 16);
    EXPECT_EQ(full->tileW, 512);
    EXPECT_EQ(full->outputH % full->tileH, 1);
    EXPECT_EQ(KernelUbBytes(*full, ge::DT_FLOAT16), required);

    TilingInfo limited;
    ASSERT_TRUE(ExecuteTiling(MakeContext(1, 512, 512, 3, 16, 16, 2, ge::DT_FLOAT16, ge::DT_FLOAT16, 48,
                                          required + reserve - kDataBlockBytes),
                              limited));
    const auto* fallback = GetTilingData(limited);
    EXPECT_EQ(fallback->useC3VectorTile, 0);
    EXPECT_LE(KernelUbBytes(*fallback, ge::DT_FLOAT16), required - kDataBlockBytes);
}

TEST(MatchTemplateTiling, AllSelectedBuffersFitUbBudget)
{
    constexpr uint64_t reserve = 16 * 1024;
    for (const ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UINT8}) {
        for (int64_t method = 0; method < 6; ++method) {
            for (const uint64_t ubBytes : {65536U, 131072U, 262144U}) {
                TilingInfo info;
                ASSERT_TRUE(
                    ExecuteTiling(MakeContext(2, 512, 512, 3, 16, 16, method, dtype, dtype, 48, ubBytes), info));
                EXPECT_LE(KernelUbBytes(*GetTilingData(info), dtype), ubBytes - reserve)
                    << "dtype=" << dtype << " method=" << method << " UB=" << ubBytes;
            }
        }
    }
}

TEST(MatchTemplateTiling, RejectsUbTooSmallForGmFallbackBuffers)
{
    constexpr uint64_t reserve = 16 * 1024;
    // CCOEFF stats (32B), a 4x8-aligned FP32 score tile (128B), two 32B slots.
    constexpr uint64_t required = 224;
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(
        MakeContext(1, 250, 250, 1, 249, 249, 5, ge::DT_FLOAT, ge::DT_FLOAT, 48, reserve + required - kDataBlockBytes),
        info));
    ASSERT_TRUE(ExecuteTiling(MakeContext(1, 250, 250, 1, 249, 249, 5, ge::DT_FLOAT, ge::DT_FLOAT, 48, 65536), info));
    const auto* data = GetTilingData(info);
    EXPECT_EQ(data->useHaloTile, 0);
    EXPECT_EQ(KernelUbBytes(*data, ge::DT_FLOAT), required);
}

TEST(MatchTemplateTiling, RejectsInvalidMethods)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext(1, 16, 16, 1, 3, 3, -1), info));
    EXPECT_FALSE(ExecuteTiling(MakeContext(1, 16, 16, 1, 3, 3, 6), info));
}

TEST(MatchTemplateTiling, RejectsMismatchedDtypes)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext(1, 16, 16, 1, 3, 3, 0, ge::DT_FLOAT, ge::DT_FLOAT16), info));
}

} // namespace
