/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../op_host/ssim_tiling.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {

// Mirror of the serialized SsimTilingData layout (packed, all fields are 4 bytes).
#pragma pack(1)
struct SsimTilingDataView {
    uint32_t totalOutRows;
    uint32_t perImageOutRows;
    uint32_t perImageInRows;
    uint32_t outW;
    uint32_t inW;
    uint32_t inH;
    uint32_t channels;
    uint32_t batch;
    uint32_t blockRows;
    uint32_t lastBlockRows;
    uint32_t colTile;
    uint32_t totalPixels;
    uint32_t blockDim;
    float invTotalPixels;
    float c1;
    float c2;
    uint32_t isUint8;
};
#pragma pack()

SsimCompileInfo MakeCompileInfo(uint32_t coreNum = 64, uint64_t ubSize = 262144)
{
    SsimCompileInfo compileInfo;
    compileInfo.totalCoreNum = coreNum;
    compileInfo.totalUbSize = ubSize;
    return compileInfo;
}

gert::TilingContextPara MakeTilingPara(std::initializer_list<int64_t> shape, ge::DataType dtype,
                                       SsimCompileInfo* compileInfo)
{
    gert::StorageShape tensorShape(shape, shape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {{tensorShape, dtype, ge::FORMAT_ND},
                                                                      {tensorShape, dtype, ge::FORMAT_ND}};
    gert::StorageShape outShape({1}, {1});
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    return gert::TilingContextPara("Ssim", inputs, outputs, compileInfo, "Ascend910B", 64, 262144, 4096);
}

} // namespace

class SsimTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(SsimTilingTest, ssim_tiling_rank2_fp32_success)
{
    SsimCompileInfo compileInfo = MakeCompileInfo();
    auto para = MakeTilingPara({64, 64}, ge::DT_FLOAT, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for rank2 float32";
    if (!ok) {
        return;
    }

    EXPECT_EQ(info.tilingKey, 2) << "float32 should dispatch tiling key 2";
    EXPECT_EQ(info.blockNum, 54) << "blockDim = min(totalOutRows=54, cores=64)";
    EXPECT_EQ(info.tilingDataSize, 72u) << "17 x 4-byte fields + trailing padding";
    ASSERT_EQ(info.workspaceSizes.size(), 1);
    EXPECT_GT(info.workspaceSizes[0], 0u);

    SsimTilingDataView td;
    std::memcpy(&td, info.tilingData.get(), sizeof(td));
    EXPECT_EQ(td.totalOutRows, 54);
    EXPECT_EQ(td.channels, 1);
    EXPECT_EQ(td.batch, 1);
    EXPECT_EQ(td.colTile, 256);
    EXPECT_EQ(td.totalPixels, 2916);
    EXPECT_EQ(td.blockDim, 54);
    EXPECT_FLOAT_EQ(td.invTotalPixels, 1.0f / 2916.0f);
    EXPECT_FLOAT_EQ(td.c1, 0.01f * 0.01f);
    EXPECT_FLOAT_EQ(td.c2, 0.03f * 0.03f);
    EXPECT_EQ(td.isUint8, 0);
}

TEST_F(SsimTilingTest, ssim_tiling_hw_too_small_fail)
{
    SsimCompileInfo compileInfo = MakeCompileInfo();
    auto para = MakeTilingPara({10, 64}, ge::DT_FLOAT, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail when H/W < 11";
}

TEST_F(SsimTilingTest, ssim_tiling_uint8_uses_l255_constants)
{
    SsimCompileInfo compileInfo = MakeCompileInfo();
    auto para = MakeTilingPara({64, 64}, ge::DT_UINT8, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    ASSERT_TRUE(ok) << "Tiling should succeed for rank2 uint8";
    EXPECT_EQ(info.tilingKey, 1) << "uint8 should dispatch tiling key 1";

    SsimTilingDataView td;
    std::memcpy(&td, info.tilingData.get(), sizeof(td));
    EXPECT_EQ(td.isUint8, 1);
    EXPECT_FLOAT_EQ(td.c1, (0.01f * 255.0f) * (0.01f * 255.0f));
    EXPECT_FLOAT_EQ(td.c2, (0.03f * 255.0f) * (0.03f * 255.0f));
}

TEST_F(SsimTilingTest, ssim_tiling_batch_overflow_fail)
{
    // Leading dims [65536, 65536] give batch = 2^32, which no longer fits the 32-bit
    // tiling fields; tiling must reject it instead of wrapping around.
    SsimCompileInfo compileInfo = MakeCompileInfo();
    auto para = MakeTilingPara({65536, 65536, 64, 64, 1}, ge::DT_FLOAT, &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_FALSE(ok) << "Tiling should fail when the batch product overflows uint32";
}
