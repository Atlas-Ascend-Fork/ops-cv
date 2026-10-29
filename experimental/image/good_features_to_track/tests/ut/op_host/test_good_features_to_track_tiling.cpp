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
 * \file test_good_features_to_track_tiling.cpp
 * \brief GoodFeaturesToTrack Host Tiling / 参数校验 UT
 *
 * 覆盖：
 *  1) 合法 shape / dtype / mask / 属性组合 ⇒ Tiling 成功；
 *  2) 非法输入（rank、dtype、属性、mask 尺寸或 dtype 不匹配）⇒ Tiling 返回失败。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/good_features_to_track_tiling_data.h"

using namespace std;
using namespace ge;

namespace {
// 对齐 op_host/good_features_to_track_def.cpp 的属性顺序：
// max_corners / quality_level / min_distance / block_size / use_harris_detector / k
#define GFT_ATTRS(mc, ql, md, bs, uh, kk)                                                                \
    {                                                                                                    \
        gert::TilingContextPara::OpAttr("max_corners", Ops::Cv::AnyValue::CreateFrom<int64_t>(mc)),      \
        gert::TilingContextPara::OpAttr("quality_level", Ops::Cv::AnyValue::CreateFrom<float>(ql)),      \
        gert::TilingContextPara::OpAttr("min_distance", Ops::Cv::AnyValue::CreateFrom<float>(md)),       \
        gert::TilingContextPara::OpAttr("block_size", Ops::Cv::AnyValue::CreateFrom<int64_t>(bs)),       \
        gert::TilingContextPara::OpAttr("use_harris_detector", Ops::Cv::AnyValue::CreateFrom<bool>(uh)), \
        gert::TilingContextPara::OpAttr("k", Ops::Cv::AnyValue::CreateFrom<float>(kk)),                  \
    }

// 满 rank=3 输入的 type 简写
struct CompileInfo {};

constexpr int64_t kMaxCorners = 1000;
constexpr int64_t kBlockSize = 3;
constexpr float kQuality = 0.01F;
constexpr float kMinDistance = 1.0F;
constexpr float kK = 0.04F;
} // namespace

class GoodFeaturesToTrackTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GoodFeaturesToTrackTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GoodFeaturesToTrackTiling TearDown" << std::endl; }
};

// ============================ 合法用例：Tiling 成功 ============================

namespace {
// 校验 tiling 成功，并用输入 shape 反查 tiling data 关键字段。
// 不用 ExecuteTestCase：它对空 expectWorkspaces 会越界读
// （tiling_case_executor.cpp 里 ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i])），且字节串比对脆弱。
void ExpectTilingOk(const gert::TilingContextPara& para)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling 执行失败";
    ASSERT_EQ(info.tilingDataSize, sizeof(GoodFeaturesToTrackTilingData));
    const auto* td = reinterpret_cast<const GoodFeaturesToTrackTilingData*>(info.tilingData.get());

    const gert::Shape& shape = para.inputTensorDesc_[0].shape_.GetStorageShape();
    ASSERT_EQ(shape.GetDimNum(), 3U);
    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t height = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t width = static_cast<uint32_t>(shape.GetDim(2));

    EXPECT_EQ(td->batch, batch);
    EXPECT_EQ(td->height, height);
    EXPECT_EQ(td->width, width);
    EXPECT_EQ(td->imageElements, height * width);
    EXPECT_GT(td->coreNum, 0U);
    EXPECT_LE(td->coreNum, 64U);
    EXPECT_EQ(td->coreNum, info.blockNum);
    EXPECT_GT(td->outputCapacity, 0U);
    EXPECT_EQ(td->hasMask, para.inputTensorDesc_.size() > 1U ? 1U : 0U);
    EXPECT_EQ(info.workspaceSizes.size(), 1U);
    if (!info.workspaceSizes.empty()) {
        // 至少要容得下 batch*H*W 个 float 的 response 缓冲
        EXPECT_GE(info.workspaceSizes[0], static_cast<size_t>(batch) * height * width * sizeof(float));
    }
}
} // namespace

TEST_F(GoodFeaturesToTrackTiling, fp32_default_64x64)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, fp16_default_64x64)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, fp32_non_square_128x64)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 128, 64}, {1, 128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, fp32_batch2)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{2, 512, 512}, {2, 512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 1000, 2}, {2, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

// 宽画布（2K 宽度）：覆盖 w 较大时的 Tiling 分支
TEST_F(GoodFeaturesToTrackTiling, fp32_wide_2048)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 512, 2048}, {1, 512, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

// 超宽画布（4K 宽度）：kernel 侧列分块路径由功能用例覆盖，这里只校验 Tiling 通过
TEST_F(GoodFeaturesToTrackTiling, fp32_ultra_wide_4096)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 512, 4096}, {1, 512, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, fp32_1080p)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 1080, 1920}, {1, 1080, 1920}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

// 带 mask：第二个输入为 UINT8 且与 image 同 shape
TEST_F(GoodFeaturesToTrackTiling, fp32_with_mask)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, fp32_harris)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, true, 0.05F), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

// max_corners <= 0：输出容量退化为 H*W
TEST_F(GoodFeaturesToTrackTiling, fp32_max_corners_zero_capacity)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 4096, 2}, {1, 4096, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(0, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

// 自定义属性组合：block_size=5、max_corners=10、min_distance=10、quality_level=0.1
TEST_F(GoodFeaturesToTrackTiling, fp32_custom_attrs)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 10, 2}, {1, 10, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(10, 0.1F, 10.0F, 5, false, kK), &compileInfo, "Ascend910B", 64, 262144,
                                 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, fp16_custom_attrs_block7)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 256, 256}, {1, 256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 5000, 2}, {1, 5000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(5000, 0.5F, 0.0F, 7, true, 0.0F), &compileInfo, "Ascend910B", 64, 262144,
                                 4096);
    ExpectTilingOk(para);
}

// ============================ 非法用例：Tiling 失败 ============================

// 输入 rank 必须为 3
// 超宽图：4*W = 65536 > 65535，blockLen(uint16) 截断为 0（整行不搬运）=> 必须拒绝
TEST_F(GoodFeaturesToTrackTiling, invalid_width_over_row_copy)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 8, 16384}, {1, 8, 16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 上一条的边界正例：4*W = 65532 <= 65535，应正常出 tiling（避免过度拒绝）
TEST_F(GoodFeaturesToTrackTiling, fp32_width_16383_row_copy_limit)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 8, 16383}, {1, 8, 16383}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

// H*W 超过 2^24（fp32 直方图精确计数上限）：4097*4096 = 16,781,312 > 16,777,216 => 必须拒绝
TEST_F(GoodFeaturesToTrackTiling, invalid_area_over_hist_exact_count)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 4097, 4096}, {1, 4097, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 边界正例：4096*4096 = 16,777,216 == 2^24，应正常出 tiling（不能过度拒绝）
TEST_F(GoodFeaturesToTrackTiling, fp32_area_eq_2_pow24_ok)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 4096, 4096}, {1, 4096, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExpectTilingOk(para);
}

TEST_F(GoodFeaturesToTrackTiling, invalid_rank4)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// image dtype 仅支持 FLOAT16 / FLOAT32
TEST_F(GoodFeaturesToTrackTiling, invalid_dtype_uint8)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GoodFeaturesToTrackTiling, invalid_dtype_int32)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// block_size 必须为正奇数
TEST_F(GoodFeaturesToTrackTiling, invalid_block_size_even)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, 4, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GoodFeaturesToTrackTiling, invalid_block_size_zero)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, 0, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// quality_level 必须落在 (0, 1]
TEST_F(GoodFeaturesToTrackTiling, invalid_quality_zero)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, 0.0F, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(GoodFeaturesToTrackTiling, invalid_quality_gt_one)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, 1.5F, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// min_distance 必须非负
TEST_F(GoodFeaturesToTrackTiling, invalid_min_distance_negative)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, -1.0F, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// k 必须非负
TEST_F(GoodFeaturesToTrackTiling, invalid_harris_k_negative)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, true, -0.1F), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// mask 必须与 image 同 shape
TEST_F(GoodFeaturesToTrackTiling, invalid_mask_shape_mismatch)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 32, 32}, {1, 32, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// mask dtype 必须为 UINT8
TEST_F(GoodFeaturesToTrackTiling, invalid_mask_dtype)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("GoodFeaturesToTrack",
                                 {
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 64, 64}, {1, 64, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1000, 2}, {1, 1000, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 GFT_ATTRS(kMaxCorners, kQuality, kMinDistance, kBlockSize, false, kK), &compileInfo,
                                 "Ascend910B", 64, 262144, 4096);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
