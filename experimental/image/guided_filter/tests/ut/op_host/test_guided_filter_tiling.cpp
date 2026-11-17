/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstring>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/guided_filter_tiling_data.h"

using namespace std;
using namespace ge;

class GuidedFilterTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GuidedFilterTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GuidedFilterTiling TearDown" << std::endl; }
};

namespace {
struct GuidedFilterCompileInfo {};

// 构造 4 个属性（按 def.cpp 的顺序：radius / eps / dDepth / scale）
std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(int64_t radius, float eps, int64_t dDepth, float scale)
{
    return {
        gert::TilingContextPara::OpAttr("radius", Ops::Cv::AnyValue::CreateFrom<int64_t>(radius)),
        gert::TilingContextPara::OpAttr("eps", Ops::Cv::AnyValue::CreateFrom<float>(eps)),
        gert::TilingContextPara::OpAttr("dDepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(dDepth)),
        gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(scale)),
    };
}

const GuidedFilterTilingData* AsTiling(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(GuidedFilterTilingData));
    return reinterpret_cast<const GuidedFilterTilingData*>(info.tilingData.get());
}
} // namespace

// rank3 f32 9x16x3 -> 3：检查几何、原子图数量、workspace
TEST_F(GuidedFilterTiling, guided_filter_tiling_f32_hwc_3x3)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const GuidedFilterTilingData* t = AsTiling(info);
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_GT(info.blockNum, 0U);
    EXPECT_EQ(t->batch, 1U);
    EXPECT_EQ(t->height, 9U);
    EXPECT_EQ(t->width, 16U);
    EXPECT_EQ(t->guideCh, 3U);
    EXPECT_EQ(t->srcCh, 3U);
    EXPECT_EQ(t->radius, 2U);
    EXPECT_EQ(t->atomNum, 21U); // C + C(C+1)/2 + Cs + C*Cs = 3 + 6 + 3 + 9
    EXPECT_EQ(t->coefNum, 12U); // C*Cs + Cs = 9 + 3
    EXPECT_EQ(t->planeElems, 144U);
    EXPECT_EQ(t->outDtype, 0U); // float32
    EXPECT_NEAR(t->invWin, 1.0F / 25.0F, 1e-7F);
    EXPECT_NEAR(t->eps, 0.01F, 1e-7F);
    EXPECT_GE(t->haloW, static_cast<uint32_t>(GfGeo::Align8(static_cast<int32_t>(t->tileW) + 2 * 2)));
    EXPECT_EQ(t->nTx, static_cast<uint32_t>(GfGeo::CeilDivI(16, static_cast<int32_t>(t->tileW))));

    // workspace = libSize + (C*Cs + Cs) * N * planeElems * 4
    ASSERT_FALSE(info.workspaceSizes.empty());
    const uint64_t dataBytes = 12ULL * 1ULL * t->planeElems * sizeof(float);
    EXPECT_GE(info.workspaceSizes[0], dataBytes);
    EXPECT_EQ(t->alphaOff, 0ULL);
    EXPECT_EQ(t->betaOff, 9ULL * t->planeElems * sizeof(float));
}

// rank2 [H,W]：guide/src 通道数都按 1 处理
TEST_F(GuidedFilterTiling, guided_filter_tiling_f32_hw_gray)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{17, 23}, {17, 23}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{17, 23}, {17, 23}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{17, 23}, {17, 23}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(4, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const GuidedFilterTilingData* t = AsTiling(info);
    EXPECT_EQ(t->guideCh, 1U);
    EXPECT_EQ(t->srcCh, 1U);
    EXPECT_EQ(t->atomNum, 4U); // 1 + 1 + 1 + 1
    EXPECT_EQ(t->coefNum, 2U); // 1 + 1
    EXPECT_EQ(t->planeElems, static_cast<uint32_t>(GfGeo::Align8(17 * 23)));
}

// u8 rank4 [N,H,W,C]，dDepth=0 -> outDtype=2
TEST_F(GuidedFilterTiling, guided_filter_tiling_u8_nhwc)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{2, 9, 13, 3}, {2, 9, 13, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     {{{2, 9, 13, 4}, {2, 9, 13, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 9, 13, 4}, {2, 9, 13, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.04F, 0, 1.0F), &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const GuidedFilterTilingData* t = AsTiling(info);
    EXPECT_EQ(t->batch, 2U);
    EXPECT_EQ(t->guideCh, 3U);
    EXPECT_EQ(t->srcCh, 4U);
    EXPECT_EQ(t->outDtype, 2U);
    EXPECT_EQ(t->atomNum, static_cast<uint32_t>(GfGeo::AtomNum(3, 4)));
    EXPECT_EQ(t->coefNum, static_cast<uint32_t>(GfGeo::CoefNum(3, 4)));
}

// f16 输入 + dDepth=7 -> outDtype=1
TEST_F(GuidedFilterTiling, guided_filter_tiling_f16_out_f16)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(1, 0.01F, 7, 1.0F), &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(AsTiling(info)->outDtype, 1U);
}

// ---------------- 非法输入 ----------------

TEST_F(GuidedFilterTiling, guided_filter_tiling_rank_mismatch)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16}, {9, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16}, {9, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(GuidedFilterTiling, guided_filter_tiling_spatial_mismatch)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 17, 1}, {9, 17, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 17, 1}, {9, 17, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// rank4 的 batch 不一致：tiling 里 batch 只取 guide 的 N，src.N 与之不同必须拒绝，
// 否则 src.N 偏小会越界读 src、偏大则 dst 尾部 batch 永不写出
TEST_F(GuidedFilterTiling, guided_filter_tiling_batch_mismatch)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{2, 9, 16, 1}, {2, 9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 9, 16, 1}, {1, 9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 9, 16, 1}, {1, 9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 单维超过 INT32_MAX：DeriveGeo 以 int32_t 接收会静默截断，必须在相乘/窄化前拒掉
TEST_F(GuidedFilterTiling, guided_filter_tiling_dim_too_large)
{
    constexpr int64_t kHuge = 3000000000LL; // > INT32_MAX，窄化到 int32 会截断
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{1, kHuge, kHuge, 1}, {1, kHuge, kHuge, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, kHuge, kHuge, 1}, {1, kHuge, kHuge, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, kHuge, kHuge, 1}, {1, kHuge, kHuge, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// guide 通道数超过 3
TEST_F(GuidedFilterTiling, guided_filter_tiling_guide_ch_too_large)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 4}, {9, 16, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// src 通道数超过 64
TEST_F(GuidedFilterTiling, guided_filter_tiling_src_ch_too_large)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{4, 8, 1}, {4, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{4, 8, 65}, {4, 8, 65}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{4, 8, 65}, {4, 8, 65}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(1, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(GuidedFilterTiling, guided_filter_tiling_radius_zero)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(0, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(GuidedFilterTiling, guided_filter_tiling_eps_negative)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, -1.0F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// scale 非 1.0：首期显式拒绝
TEST_F(GuidedFilterTiling, guided_filter_tiling_scale_not_one)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 0.5F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(GuidedFilterTiling, guided_filter_tiling_ddepth_invalid)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 1}, {9, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, 3, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 单行字节数超过 65535（DataCopyPad blockLen 上限）
TEST_F(GuidedFilterTiling, guided_filter_tiling_row_bytes_too_large)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{8, 20000}, {8, 20000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{8, 20000}, {8, 20000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 20000}, {8, 20000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(2, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// 巨大半径 + 3 通道：UB 放不下，tiling 阶段拒绝
TEST_F(GuidedFilterTiling, guided_filter_tiling_radius_too_large_for_ub)
{
    GuidedFilterCompileInfo compileInfo;
    gert::TilingContextPara para("GuidedFilter",
                                 {
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{9, 16, 3}, {9, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 MakeAttrs(512, 0.01F, -1, 1.0F), &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}
