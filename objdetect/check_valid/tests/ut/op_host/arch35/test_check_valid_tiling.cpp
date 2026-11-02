/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iostream>
#include <cstring>
#include <limits>
#include <utility>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/check_valid_tiling_data.h"

using namespace std;
using namespace ge;

class CheckValidTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckValidTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckValidTiling TearDown" << std::endl; }
};

static std::map<std::string, std::string> soc_versions_infos = {{"Short_SoC_version", "Ascend950"}};

constexpr size_t CV_SYS_WORKSPACE_SIZE = 16777216;

TEST_F(CheckValidTiling, cv_fp32_normal)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "8 5440 8 1 1 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_fp16_normal)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "8 10880 8 1 1 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_fp32_empty)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 1}, {0, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "0 0 0 0 0 0 0 0 -4647714812233515008 ",
                    {0});
}

TEST_F(CheckValidTiling, cv_fp16_empty)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{0, 4}, {0, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 1}, {0, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "0 0 0 0 0 0 0 0 -4647714812233515008 ",
                    {0});
}

TEST_F(CheckValidTiling, cv_runtime_rows_reject_unresolved_or_invalid_values)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t rows : {-3, -2, -1}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", runtime N=" << rows);
            gert::TilingContextPara para(
                "CheckValid", {{{{rows, 4}, {rows, 4}}, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                {{{{rows, 1}, {rows, 1}}, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", 64, 262144,
                4096);
            TilingInfo info;
            EXPECT_FALSE(ExecuteTiling(para, info));
        }
    }
}

TEST_F(CheckValidTiling, cv_dynamic_graph_shape_uses_concrete_runtime_rows)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t rows : {0, 8}) {
            for (bool unknownRank : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << "dtype=" << dtype << ", runtime N=" << rows << ", unknownRank=" << unknownRank);
                gert::StorageShape bbox = unknownRank ? gert::StorageShape({-2}, {rows, 4}) :
                                                        gert::StorageShape({-1, 4}, {rows, 4});
                gert::StorageShape output = unknownRank ? gert::StorageShape({-2}, {rows, 1}) :
                                                          gert::StorageShape({-1, 1}, {rows, 1});
                gert::TilingContextPara para(
                    "CheckValid", {{bbox, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                    {{output, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", 64, 262144, 4096);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(para, info));
                ASSERT_EQ(info.tilingDataSize, sizeof(CheckValidTilingData));
                CheckValidTilingData td{};
                std::memcpy(&td, info.tilingData.get(), sizeof(td));
                EXPECT_EQ(td.N, rows);
            }
        }
    }
}

TEST_F(CheckValidTiling, cv_fp32_large_multicore)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{100000, 4}, {100000, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{100000, 1}, {100000, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "100000 5440 2080 19 19 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_fp16_nonaligned)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{33, 4}, {33, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{33, 1}, {33, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey,
                    "33 10880 33 1 1 1 0 87360 -4647714812233515008 ", {0});
}

TEST_F(CheckValidTiling, cv_vector_load_stays_within_input_buffer)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const int64_t elementBytes = (dtype == ge::DT_FLOAT16) ? 2 : 4;
        for (int64_t ubBytes : {253952, 262144}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", UB=" << ubBytes);
            gert::TilingContextPara para(
                "CheckValid", {{{{100000, 4}, {100000, 4}}, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                {{{{100000, 1}, {100000, 1}}, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", 64, ubBytes,
                4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            ASSERT_EQ(info.tilingDataSize, sizeof(CheckValidTilingData));
            CheckValidTilingData td{};
            std::memcpy(&td, info.tilingData.get(), sizeof(td));
            EXPECT_LE(3 * td.per_buf_bytes, ubBytes);
            for (int64_t rows : {td.tile_n, td.tile_n_tail}) {
                // The load has no mask: even a partial final vector reads all 64 lanes.
                const int64_t loadedRows = ((rows + 63) / 64) * 64;
                EXPECT_LE((3 * td.tile_n + loadedRows) * elementBytes, td.per_buf_bytes);
            }
        }
    }
}

TEST_F(CheckValidTiling, cv_ub_must_fit_one_vector_per_column)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const int64_t elementBytes = (dtype == ge::DT_FLOAT16) ? 2 : 4;
        const int64_t minimumUb = 3 * 4 * 64 * elementBytes;
        for (int64_t ubBytes : {minimumUb - 1, minimumUb}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", UB=" << ubBytes);
            gert::TilingContextPara para(
                "CheckValid", {{{{65, 4}, {65, 4}}, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                {{{{65, 1}, {65, 1}}, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", 64, ubBytes, 4096);
            TilingInfo info;
            EXPECT_EQ(ExecuteTiling(para, info), ubBytes >= minimumUb);
        }
    }
}

TEST_F(CheckValidTiling, cv_int64_row_boundaries)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    const int64_t maxRows = std::numeric_limits<int64_t>::max();
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const int64_t elementBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
        for (int64_t ubBytes : {253952, 262144}) {
            const int64_t tileRows = (((ubBytes / 3) & ~31) / elementBytes / 4 / 64) * 64;
            for (int64_t rows : {int64_t{1}, tileRows - 1, tileRows, tileRows + 1, 2 * tileRows - 1, 2 * tileRows,
                                 2 * tileRows + 1, 64 * tileRows - 1, 64 * tileRows, 64 * tileRows + 1,
                                 maxRows - tileRows, maxRows - tileRows + 1, maxRows - 1, maxRows}) {
                for (int64_t cores : {1, 64}) {
                    SCOPED_TRACE(testing::Message()
                                 << "dtype=" << dtype << ", UB=" << ubBytes << ", N=" << rows << ", cores=" << cores);
                    // Shapes are metadata only; no large input/output allocation or device execution.
                    gert::TilingContextPara para(
                        "CheckValid",
                        {{{{rows, 4}, {rows, 4}}, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                        {{{{rows, 1}, {rows, 1}}, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", cores,
                        ubBytes, 4096);
                    TilingInfo info;
                    const __int128 elementCount = static_cast<__int128>(rows) * 4;
                    const bool capacityFits = elementCount <= std::numeric_limits<int64_t>::max() &&
                                              elementCount * elementBytes <= std::numeric_limits<uint64_t>::max();
                    const bool accepted = ExecuteTiling(para, info);
                    EXPECT_EQ(accepted, capacityFits);
                    if (!capacityFits || !accepted) {
                        continue;
                    }
                    ASSERT_EQ(info.tilingDataSize, sizeof(CheckValidTilingData));
                    CheckValidTilingData td{};
                    std::memcpy(&td, info.tilingData.get(), sizeof(td));
                    // A wider integer oracle checks the exact mathematical ceiling independently.
                    const __int128 wideRows = rows;
                    const int64_t expectedTiles = static_cast<int64_t>((wideRows + tileRows - 1) / tileRows);
                    EXPECT_EQ(td.N, rows);
                    EXPECT_EQ(td.tile_n, tileRows);
                    EXPECT_EQ(td.num_tiles, expectedTiles);
                    EXPECT_EQ(td.tile_n_tail,
                              static_cast<int64_t>(wideRows - static_cast<__int128>(expectedTiles - 1) * tileRows));
                    EXPECT_EQ(td.num_cores, std::min(cores, expectedTiles));
                    EXPECT_EQ(static_cast<__int128>(td.tiles_main) * td.num_cores + td.cores_tail,
                              static_cast<__int128>(expectedTiles));
                    EXPECT_GE(td.cores_tail, 0);
                    EXPECT_LT(td.cores_tail, td.num_cores);
                }
            }
        }
    }
}

TEST_F(CheckValidTiling, cv_bbox_address_capacity)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        const int64_t elementBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
        const int64_t maxRows = static_cast<int64_t>(
            std::min(static_cast<__int128>(std::numeric_limits<int64_t>::max()) / 4,
                     static_cast<__int128>(std::numeric_limits<uint64_t>::max()) / (4 * elementBytes)));
        for (int64_t ubBytes : {253952, 262144}) {
            for (int64_t rows : {maxRows - 1, maxRows, maxRows + 1, std::numeric_limits<int64_t>::max()}) {
                SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", UB=" << ubBytes << ", N=" << rows);
                gert::TilingContextPara para(
                    "CheckValid", {{{{rows, 4}, {rows, 4}}, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                    {{{{rows, 1}, {rows, 1}}, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", 64, ubBytes,
                    4096);
                TilingInfo info;
                const bool accepted = ExecuteTiling(para, info);
                EXPECT_EQ(accepted, rows <= maxRows);
                if (rows > maxRows || !accepted) {
                    continue;
                }
                ASSERT_EQ(info.tilingDataSize, sizeof(CheckValidTilingData));
                CheckValidTilingData td{};
                std::memcpy(&td, info.tilingData.get(), sizeof(td));
                EXPECT_EQ(td.N, rows);
                EXPECT_EQ(static_cast<__int128>(td.num_tiles - 1) * td.tile_n + td.tile_n_tail,
                          static_cast<__int128>(rows));
            }
        }
    }
}

TEST_F(CheckValidTiling, cv_does_not_recheck_ge_managed_formats)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    const std::pair<ge::Format, ge::Format> formatCases[] = {
        {ge::FORMAT_ND, ge::FORMAT_ND},         {ge::FORMAT_NCHW, ge::FORMAT_NCHW},
        {ge::FORMAT_NHWC, ge::FORMAT_NHWC},     {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ},
        {ge::FORMAT_ND, ge::FORMAT_NCHW},       {ge::FORMAT_ND, ge::FORMAT_NHWC},
        {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ},
    };
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t rows : {0, 8}) {
            for (const auto& formatCase : formatCases) {
                const auto originFormat = formatCase.first;
                const auto storageFormat = formatCase.second;
                SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", N=" << rows << ", origin=" << originFormat
                                                << ", storage=" << storageFormat);
                gert::TilingContextPara para(
                    "CheckValid",
                    {gert::TilingContextPara::TensorDescription(gert::StorageShape({rows, 4}, {rows, 4}), dtype,
                                                                originFormat, false, nullptr, storageFormat),
                     gert::TilingContextPara::TensorDescription(gert::StorageShape({3}, {3}), dtype, originFormat,
                                                                false, nullptr, storageFormat)},
                    {gert::TilingContextPara::TensorDescription(gert::StorageShape({rows, 1}, {rows, 1}), ge::DT_INT8,
                                                                originFormat, false, nullptr, storageFormat)},
                    {}, &compileInfo, "Ascend950", 64, 262144, 4096);
                TilingInfo info;
                EXPECT_TRUE(ExecuteTiling(para, info));
            }
        }
    }
}

TEST_F(CheckValidTiling, cv_invalid_rank1)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_last_dim)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_dtype_int64)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_dtype_mismatch)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_invalid_img_metas_short)
{
    struct CheckValidCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("CheckValid",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 1}, {8, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 64, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidTiling, cv_img_metas_shape_contract)
{
    struct MetaCase {
        const char* name;
        gert::StorageShape shape;
        bool valid;
    };
    const MetaCase metaCases[] = {
        {"length3", {{3}, {3}}, true},
        {"length4", {{4}, {4}}, true},
        {"length16", {{16}, {16}}, true},
        {"length17", {{17}, {17}}, true},
        {"empty", {{0}, {0}}, false},
        {"length1", {{1}, {1}}, false},
        {"length2", {{2}, {2}}, false},
        {"scalar", {{}, {}}, false},
        {"rank2_row", {{1, 3}, {1, 3}}, false},
        {"rank2_column", {{3, 1}, {3, 1}}, false},
        {"rank2_length16", {{1, 16}, {1, 16}}, false},
        {"rank2_matrix", {{3, 2}, {3, 2}}, false},
        {"rank3_last_axis", {{1, 1, 3}, {1, 1, 3}}, false},
        {"rank3_first_axis", {{3, 1, 1}, {3, 1, 1}}, false},
        {"rank4", {{1, 1, 1, 3}, {1, 1, 1, 3}}, false},
    };
    struct CheckValidCompileInfo {
    } compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t rows : {0, 8}) {
            for (const auto& meta : metaCases) {
                SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", N=" << rows << ", img_metas=" << meta.name);
                gert::TilingContextPara para(
                    "CheckValid", {{{{rows, 4}, {rows, 4}}, dtype, ge::FORMAT_ND}, {meta.shape, dtype, ge::FORMAT_ND}},
                    {{{{rows, 1}, {rows, 1}}, ge::DT_INT8, ge::FORMAT_ND}}, {}, &compileInfo, "Ascend950", 64, 262144,
                    4096);
                TilingInfo info;
                EXPECT_EQ(ExecuteTiling(para, info), meta.valid);
            }
        }
    }
}
