/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_pixel_shuffle_tiling.cpp
 * \brief PixelShuffle Tiling UT：校验 tiling 参数推导、多核切分与非法输入报错
 */

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include "../../../op_kernel/pixel_shuffle_tiling_data.h"

using namespace std;
using namespace ge;

class PixelShuffleTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PixelShuffleTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PixelShuffleTiling TearDown" << std::endl; }
};

struct PixelShuffleCompileInfoStub {
    int64_t coreNum = 48;
    uint64_t ubSizePlatForm = 196608;
    bool regBase = false;
};

static PixelShuffleCompileInfoStub compileInfo;

// 4D fp32 r=2: [2, 16, 8, 8] -> tiling 参数推导
TEST_F(PixelShuffleTiling, tiling_4d_r2_fp32)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 16, 16}, {2, 4, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto* td = reinterpret_cast<const PixelShuffleTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->N, 2);
    EXPECT_EQ(td->C_in, 16);
    EXPECT_EQ(td->D, 1);
    EXPECT_EQ(td->H, 8);
    EXPECT_EQ(td->W, 8);
    EXPECT_EQ(td->r, 2);
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->totalLength, 2048);
    EXPECT_GT(td->tileNum, 0);
    EXPECT_GT(td->tileDataNum, 0U);
    EXPECT_GT(tilingInfo.blockNum, 0U);
}

// 5D fp16 r=2: [2, 32, 3, 4, 5] -> tiling 参数推导
TEST_F(PixelShuffleTiling, tiling_5d_r2_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{2, 32, 3, 4, 5}, {2, 32, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 4, 6, 8, 10}, {2, 4, 6, 8, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto* td = reinterpret_cast<const PixelShuffleTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->N, 2);
    EXPECT_EQ(td->C_in, 32);
    EXPECT_EQ(td->D, 3);
    EXPECT_EQ(td->H, 4);
    EXPECT_EQ(td->W, 5);
    EXPECT_EQ(td->r, 2);
    EXPECT_EQ(td->rank, 5);
    EXPECT_EQ(td->totalLength, 3840);
    EXPECT_GT(tilingInfo.blockNum, 0U);
}

// 4D fp16 r=3: [1, 18, 5, 7] -> tiling 参数推导
TEST_F(PixelShuffleTiling, tiling_4d_r3_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{1, 18, 5, 7}, {1, 18, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 2, 15, 21}, {1, 2, 15, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(3))}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto* td = reinterpret_cast<const PixelShuffleTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->N, 1);
    EXPECT_EQ(td->C_in, 18);
    EXPECT_EQ(td->H, 5);
    EXPECT_EQ(td->W, 7);
    EXPECT_EQ(td->r, 3);
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->totalLength, 630);
    EXPECT_GT(tilingInfo.blockNum, 0U);
}

// 非法 rank=3 -> GRAPH_FAILED
TEST_F(PixelShuffleTiling, tiling_invalid_rank)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 16}, {2, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 非法 r=1 -> GRAPH_FAILED
TEST_F(PixelShuffleTiling, tiling_invalid_scale)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 非法通道不可被 r^2 整除 (C=15, r=2) -> GRAPH_FAILED
TEST_F(PixelShuffleTiling, tiling_channel_not_divisible)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{2, 15, 8, 8}, {2, 15, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 3, 16, 16}, {2, 3, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 非法畸形 shape：dim < -1。C=-4、H=-3 相乘为正，可穿透 totalLength>0 与通道整除校验
TEST_F(PixelShuffleTiling, tiling_invalid_negative_dim)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{1, -4, -3, 2}, {1, -4, -3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 6, 4}, {1, 1, 6, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 非法 scale_factor=0 -> GRAPH_FAILED（下界校验）
TEST_F(PixelShuffleTiling, tiling_invalid_scale_zero)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 16, 8, 8}, {2, 16, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 非法 scale_factor 超上界 -> GRAPH_FAILED（上界校验）
// r=2^22 走 5D 时 r^3 = 2^66，修复前 chanPerGroup 累乘溢出为 0，第 131-137 行 "C_in % chanPerGroup" 除零崩溃
TEST_F(PixelShuffleTiling, tiling_scale_overflow_bound)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{1, 32, 2, 4, 4}, {1, 32, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 2, 4, 4}, {1, 1, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(4194304LL))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 大宽度 4K fp32: [1, 8, 1080, 3840] r=2 -> W 方向分段，tiling 应成功
// 修复前 r^2*W = 15360 元素/块，UB 需求约 300KB 超出 192KB，kernel InitBuffer 失败
TEST_F(PixelShuffleTiling, tiling_large_width_4k_fp32)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{1, 8, 1080, 3840}, {1, 8, 1080, 3840}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2, 2160, 7680}, {1, 2, 2160, 7680}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto* td = reinterpret_cast<const PixelShuffleTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->H, 1080);
    EXPECT_EQ(td->W, 3840);
    EXPECT_EQ(td->r, 2);
    EXPECT_GT(tilingInfo.blockNum, 0U);
}

// 大宽度 8K fp16: [1, 8, 720, 7680] r=2 -> W 方向分段，tiling 应成功
TEST_F(PixelShuffleTiling, tiling_large_width_8k_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "PixelShuffle", {{{{1, 8, 720, 7680}, {1, 8, 720, 7680}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 2, 1440, 15360}, {1, 2, 1440, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale_factor", Ops::Cv::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto* td = reinterpret_cast<const PixelShuffleTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->W, 7680);
    EXPECT_EQ(td->r, 2);
    EXPECT_GT(tilingInfo.blockNum, 0U);
}
