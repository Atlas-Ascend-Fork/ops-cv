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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_host/arch35/resize_bicubic_v2_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeBicubicV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeBicubicV2TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeBicubicV2TilingTest TearDown" << std::endl; }
};

TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_01)
{
    gert::StorageShape inputXShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    int size_value[2] = {32, 32};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 65541;
    string expectTilingData = "64 48 0 32 32 32 32 3 1 4575657222473777152 1 32 32 3 1 32 32 25056 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_02)
{
    gert::StorageShape inputXShape = {{1, 225, 225, 128}, {1, 225, 225, 128}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 113, 113, 128}, {1, 113, 113, 128}};
    int size_value[2] = {113, 113};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 6;
    string expectTilingData = "57 25538 0 225 225 113 113 128 1 4611686019501129728 1 2 113 128 1 1 113 128 ";
    std::vector<size_t> expectWorkspaces = {0};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 03: NCHW upsampling integer scale (PointCopy, DIM_6)
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_03)
{
    gert::StorageShape inputXShape = {{1, 3, 16, 16}, {1, 3, 16, 16}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    int size_value[2] = {32, 32};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 04: NCHW upsampling align_corners=true (DIM_4)
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_04)
{
    gert::StorageShape inputXShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 3, 64, 64}, {1, 3, 64, 64}};
    int size_value[2] = {64, 64};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 05: NHWC upsampling with scales (DIM_0 fallback)
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_05)
{
    gert::StorageShape inputXShape = {{1, 16, 16, 64}, {1, 16, 16, 64}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 32, 32, 64}, {1, 32, 32, 64}};
    int size_value[2] = {32, 32};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: NCHW downsample non-integer scale (DIM_0 fallback)
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_06)
{
    gert::StorageShape inputXShape = {{1, 3, 64, 64}, {1, 3, 64, 64}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 3, 20, 20}, {1, 3, 20, 20}};
    int size_value[2] = {20, 20};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({3.0f, 3.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// tilingKey 位布局(见 resize_bicubic_v2_tiling_key.h 的 ASCENDC_TPL_UINT_DECL 顺序,各 8bit):
//   byte0=schId, byte1=isInt32, byte2=isHalfPixel, byte3=isNchw
static inline int64_t GetIsNchwFromTilingKey(int64_t tilingKey) { return (tilingKey >> 24) & 0xFF; }
static inline int64_t GetIsInt32FromTilingKey(int64_t tilingKey) { return (tilingKey >> 8) & 0xFF; }

// Test 07 [回归]: NHWC 单空间维变化(仅 H 变、W 不变)必须走 NHWC 布局核(isNchw=0)。
// 缺陷(修复前 tiling arch35 L297 用 &&):NHWC 仅当 H、W 都变才置 isNchw_=0,单维变时 isNchw_ 停在 1,
// 用 NCHW 布局核处理 channel-last 数据 → 真机 ~99.8% 元素错(见 op-quality-review DEFECT 报告)。
// 修复(&& -> ||):NHWC 任一空间维变化即置 isNchw_=0。此用例严格断言 isNchw 字节=0,修复前必失败、修复后通过。
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_07_nhwc_single_dim_h)
{
    gert::StorageShape inputXShape = {{1, 16, 16, 64}, {1, 16, 16, 64}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 32, 16, 64}, {1, 32, 16, 64}}; // H:16->32, W:16 不变
    int size_value[2] = {32, 16};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(GetIsNchwFromTilingKey(tilingInfo.tilingKey), 0); // NHWC 必走 NHWC 布局核
}

// Test 08 [回归]: NHWC 单空间维变化(仅 W 变、H 不变)同样必须 isNchw=0。
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_08_nhwc_single_dim_w)
{
    gert::StorageShape inputXShape = {{1, 16, 16, 64}, {1, 16, 16, 64}};
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 16, 32, 64}, {1, 16, 32, 64}}; // W:16->32, H:16 不变
    int size_value[2] = {16, 32};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(GetIsNchwFromTilingKey(tilingInfo.tilingKey), 0);
}

// Test 09 [回归]: 输入输出的H和W轴超 INT32_MAX 必须走 int64 索引核(isInt32=0)。
// 缺陷(修复前用 `> UINT32_MAX`):isInt32_==1 时 SIMT 核绑定 <T_IDX=uint32_t, T_IDX2=int32_t>(apt.cpp),
// T_IDX2 是**有符号** int32,承接空间坐标(topY/leftX、lenSrcH1=lenSrcH-1),上限 INT32_MAX(2^31-1)。
// 旧 UINT32_MAX 阈值只护无符号 T_IDX,输入输出的H和W轴(INT32_MAX, UINT32_MAX] 区间被误判 isInt32=1 → int32 有符号
// 溢出为负 → GetSrc 钳位错乱 → GM 偏移错乱静默数据错。本用例lenSrcH = 2147483648 > INT32_MAX
TEST_F(ResizeBicubicV2TilingTest, resize_bicubic_v2_tiling_09_int32_overflow_coarse)
{
    gert::StorageShape inputXShape = {{1, 1, 2147483648, 1}, {1, 1, 2147483648, 1}}; // lenSrcH = 2147483648 > INT32_MAX
    gert::StorageShape inputSizeShape = {{2}, {2}};
    gert::StorageShape outputShape = {{1, 1, 70, 83}, {1, 1, 70, 83}};
    int size_value[2] = {70, 83};

    ResizeBicubicV2CompileInfo compileInfo = {64, 200704};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2",
        {{inputXShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputSizeShape, ge::DT_INT32, ge::FORMAT_ND, true, size_value}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(GetIsInt32FromTilingKey(tilingInfo.tilingKey), 0); // lenSrcH超 INT32_MAX 必走 int64
}
