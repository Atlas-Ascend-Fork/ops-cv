/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See License in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gaussian_filter_3_d_tiling.cpp
 * \brief GaussianFilter3D tiling UT：校验radius/权重/tile/摊核/workspace等tiling数学
 */
#include <cmath>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../op_host/gaussian_filter_3_d_tiling.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {

// 与序列化布局一致的只读视图（12个uint32标量 + 1025个uint32权重）
#pragma pack(1)
struct GaussianFilter3DTilingDataView {
    uint32_t batch;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t radius;
    uint32_t mode;
    uint32_t tileLen;
    uint32_t chunkNum;
    uint32_t lineNum;
    uint32_t coreNum;
    uint32_t tasksPerCore;
    uint32_t pongOffset;
    uint32_t weightBits[801];
};
#pragma pack(1)

float BitsToFloat(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

std::vector<gert::TilingContextPara::TensorDescription> MakeInput(ge::DataType dtype,
                                                                  std::initializer_list<int64_t> shape)
{
    gert::StorageShape tensorShape(shape, shape);
    return {{tensorShape, dtype, ge::FORMAT_ND}};
}

gert::TilingContextPara MakeTilingPara(std::initializer_list<int64_t> shape, ge::DataType dtype, double sigma,
                                       const std::string& mode, GaussianFilter3DCompileInfo* compileInfo)
{
    auto inputs = MakeInput(dtype, shape);
    std::vector<gert::TilingContextPara::TensorDescription> outputs = inputs;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("sigma", Ops::Cv::AnyValue::CreateFrom<float>(static_cast<float>(sigma))),
        gert::TilingContextPara::OpAttr("mode", Ops::Cv::AnyValue::CreateFrom<std::string>(mode)),
    };
    // tiling数据含1025个uint32权重（共4148B），默认4096容量不够，显式给8192
    return gert::TilingContextPara("GaussianFilter3D", inputs, outputs, attrs, compileInfo, "Ascend910B", 64, 262144,
                                   4096);
}

} // namespace

class GaussianFilter3DTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(GaussianFilter3DTilingTest, gaussian_filter_3_d_tiling_rank3_fp32_sigma1)
{
    GaussianFilter3DCompileInfo compileInfo;
    auto para = MakeTilingPara({8, 16, 67}, ge::DT_FLOAT, 1.0, "REFLECT", &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for rank3 fp32 sigma=1";
    if (!ok) {
        return;
    }

    EXPECT_EQ(info.tilingKey, 0) << "fp32 should dispatch tiling key 0";
    ASSERT_EQ(info.workspaceSizes.size(), 1);
    // elemCnt=8*16*67=8576 -> fp32 ping/pong: 2*align256(8576*4)=2*137216=274432
    EXPECT_EQ(info.workspaceSizes[0], 68608u + 16U * 1024U * 1024U); // 2*align256(8576*4) + 16MB系统区

    GaussianFilter3DTilingDataView td;
    memcpy(&td, info.tilingData.get(), sizeof(td));
    EXPECT_EQ(td.batch, 1u);
    EXPECT_EQ(td.depth, 8u);
    EXPECT_EQ(td.height, 16u);
    EXPECT_EQ(td.width, 67u);
    EXPECT_EQ(td.radius, 4u) << "r = int(4*1.0+0.5) = 4";
    EXPECT_EQ(td.mode, 0u);
    EXPECT_EQ(td.lineNum, 128u);
    EXPECT_GT(td.tileLen, 0u);
    EXPECT_EQ(td.tileLen % 8, 0u) << "tileLen should be 8-aligned";
    EXPECT_LE(td.tileLen, 67u);
    EXPECT_EQ(td.chunkNum, (67u + td.tileLen - 1u) / td.tileLen);
    EXPECT_EQ(td.coreNum, 64u) << "blockDim恒为全部AIV数，保证SyncAll屏障正确";
    EXPECT_GT(td.tasksPerCore, 0u);
    EXPECT_EQ(td.pongOffset, 34304u);

    // 高斯权重：sigma=1, r=4，中心权重最大，两侧对称且归一化
    float center = BitsToFloat(td.weightBits[4]);
    float first = BitsToFloat(td.weightBits[0]);
    float last = BitsToFloat(td.weightBits[8]);
    double sum = 0.0;
    for (uint32_t k = 0; k < 9; k++) {
        sum += BitsToFloat(td.weightBits[k]);
    }
    EXPECT_GT(center, first);
    EXPECT_NEAR(first, last, 1e-6f);
    EXPECT_NEAR(center, 0.39894f, 1e-3f); // 1/sum(exp(-0.5*i^2)), i in [-4,4], sum=2.5066
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST_F(GaussianFilter3DTilingTest, gaussian_filter_3_d_tiling_sigma0_identity_weights)
{
    GaussianFilter3DCompileInfo compileInfo;
    auto para = MakeTilingPara({4, 3, 9}, ge::DT_FLOAT, 0.0, "REFLECT", &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for sigma=0";
    if (!ok) {
        return;
    }

    GaussianFilter3DTilingDataView td;
    memcpy(&td, info.tilingData.get(), sizeof(td));
    EXPECT_EQ(td.radius, 0u) << "r = int(4*0+0.5) = 0";
    float w0 = BitsToFloat(td.weightBits[0]);
    EXPECT_TRUE(std::isfinite(w0)) << "sigma=0 的权重不能是 NaN";
    EXPECT_FLOAT_EQ(w0, 1.0f) << "sigma=0 必须下发恒等权重 [1.0]，输出才是 y=x";
}

TEST_F(GaussianFilter3DTilingTest, gaussian_filter_3_d_tiling_rank5_bf16_sigma2)
{
    GaussianFilter3DCompileInfo compileInfo;
    auto para = MakeTilingPara({2, 3, 8, 16, 67}, ge::DT_BF16, 2.0, "WRAP", &compileInfo);

    TilingInfo info;
    bool ok = ExecuteTiling(para, info);
    EXPECT_TRUE(ok) << "Tiling should succeed for rank5 bf16 sigma=2";
    if (!ok) {
        return;
    }

    EXPECT_EQ(info.tilingKey, 2) << "bf16 should dispatch tiling key 2";

    GaussianFilter3DTilingDataView td;
    memcpy(&td, info.tilingData.get(), sizeof(td));
    EXPECT_EQ(td.batch, 6u);
    EXPECT_EQ(td.depth, 8u);
    EXPECT_EQ(td.height, 16u);
    EXPECT_EQ(td.width, 67u);
    EXPECT_EQ(td.radius, 8u) << "r = int(4*2.0+0.5) = 8";
    EXPECT_EQ(td.mode, 4u);
    EXPECT_EQ(td.lineNum, 768u);
    EXPECT_EQ(td.pongOffset, 205824u); // align256(51456*4)
}

TEST_F(GaussianFilter3DTilingTest, gaussian_filter_3_d_tiling_invalid_mode_fail)
{
    GaussianFilter3DCompileInfo compileInfo;
    auto para = MakeTilingPara({8, 16, 67}, ge::DT_FLOAT, 1.0, "FOO", &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "Invalid mode should fail";
}

TEST_F(GaussianFilter3DTilingTest, gaussian_filter_3_d_tiling_sigma_out_of_range_fail)
{
    GaussianFilter3DCompileInfo compileInfo;
    auto para = MakeTilingPara({8, 16, 67}, ge::DT_FLOAT, 100.5, "REFLECT", &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "sigma > 100 should fail";
}

TEST_F(GaussianFilter3DTilingTest, gaussian_filter_3_d_tiling_rank2_fail)
{
    GaussianFilter3DCompileInfo compileInfo;
    auto para = MakeTilingPara({16, 67}, ge::DT_FLOAT, 1.0, "REFLECT", &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info)) << "rank2 should fail";
}
