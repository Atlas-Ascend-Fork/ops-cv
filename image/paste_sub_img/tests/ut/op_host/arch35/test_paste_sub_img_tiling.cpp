/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/paste_sub_img_tiling_data.h"

class PasteSubImgTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PasteSubImgTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PasteSubImgTiling TearDown" << std::endl; }
};

struct PasteSubImgCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

using TTD = gert::TilingContextPara::TensorDescription;

static TilingInfo RunTiling(ge::DataType dtype, int64_t pH, int64_t pW, int64_t pC, int64_t cH, int64_t cW, int64_t cC,
                            const int32_t* patchCoord, const int32_t* coreAreaCoord, float scale)
{
    PasteSubImgCompileInfo compileInfo;
    gert::StorageShape patchS({pH, pW, pC}, {pH, pW, pC});
    gert::StorageShape coordS({4}, {4});
    gert::StorageShape combineS({cH, cW, cC}, {cH, cW, cC});
    gert::TilingContextPara tilingContextPara(
        "PasteSubImg",
        {TTD(patchS, dtype, ge::FORMAT_ND),
         TTD(coordS, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<int32_t*>(patchCoord)),
         TTD(coordS, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<int32_t*>(coreAreaCoord)),
         TTD(combineS, dtype, ge::FORMAT_ND)},
        {TTD(combineS, dtype, ge::FORMAT_ND)},
        {gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(scale))}, &compileInfo,
        "Ascend950", 64, 262144, 4096);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    return tilingInfo;
}

static bool RunTilingExpectReject(ge::DataType dtype, int64_t pH, int64_t pW, int64_t pC, int64_t cH, int64_t cW,
                                  int64_t cC, const int32_t* patchCoord, const int32_t* coreAreaCoord, float scale)
{
    PasteSubImgCompileInfo compileInfo;
    gert::StorageShape patchS({pH, pW, pC}, {pH, pW, pC});
    gert::StorageShape coordS({4}, {4});
    gert::StorageShape combineS({cH, cW, cC}, {cH, cW, cC});
    gert::TilingContextPara tilingContextPara(
        "PasteSubImg",
        {TTD(patchS, dtype, ge::FORMAT_ND),
         TTD(coordS, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<int32_t*>(patchCoord)),
         TTD(coordS, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<int32_t*>(coreAreaCoord)),
         TTD(combineS, dtype, ge::FORMAT_ND)},
        {TTD(combineS, dtype, ge::FORMAT_ND)},
        {gert::TilingContextPara::OpAttr("scale", Ops::Cv::AnyValue::CreateFrom<float>(scale))}, &compileInfo,
        "Ascend950", 64, 262144, 4096);

    TilingInfo tilingInfo;
    return ExecuteTiling(tilingContextPara, tilingInfo);
}

// KEY=2 (UBAXIS_H): split H axis, GetTilingKey -> index 1 -> tilingKey=1
TEST_F(PasteSubImgTiling, paste_sub_img_tiling_key_h_split)
{
    int32_t patchCoord[] = {0, 0, 0, 0};
    int32_t coreAreaCoord[] = {0, 0, 4, 4};
    auto info = RunTiling(ge::DT_UINT8, 4, 4, 3, 4, 4, 3, patchCoord, coreAreaCoord, 1.0f);
    EXPECT_EQ(info.tilingKey, 1);
    auto* td = reinterpret_cast<const PasteSubImgTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->activeH, 4);
    EXPECT_EQ(td->activeW, 4);
    EXPECT_EQ(td->activeC, 3);
}

// KEY=1 (UBAXIS_WC): split WC axis, GetTilingKey -> index 0 -> tilingKey=0
TEST_F(PasteSubImgTiling, paste_sub_img_tiling_key_wc_split)
{
    int32_t patchCoord[] = {0, 0, 0, 0};
    int32_t coreAreaCoord[] = {0, 0, 16, 1};
    auto info = RunTiling(ge::DT_UINT8, 16, 16, 3, 32, 32, 3, patchCoord, coreAreaCoord, 1.0f);
    EXPECT_EQ(info.tilingKey, 0);
    auto* td = reinterpret_cast<const PasteSubImgTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->activeH, 1);
    EXPECT_EQ(td->activeW, 16);
    EXPECT_EQ(td->activeC, 3);
}

// WC 元素数接近 int32 边界但仍在范围内（715827840 * 3 = 2147483520 <= INT32_MAX），tiling 应成功。
// 取 715827840 而非 floor(INT32_MAX/3)=715827882，是因为坐标经 float scale 转换后，
// 715827882 会被舍入为 715827904（> INT32_MAX/3），触发越界检查。
TEST_F(PasteSubImgTiling, paste_sub_img_tiling_wc_count_within_int32_range)
{
    int32_t patchCoord[] = {0, 0, 0, 0};
    int32_t coreAreaCoord[] = {0, 0, 715827840, 1};
    auto info = RunTiling(ge::DT_UINT8, 1, 715827840, 3, 1, 715827840, 3, patchCoord, coreAreaCoord, 1.0f);
    auto* td = reinterpret_cast<const PasteSubImgTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->activeH, 1);
    EXPECT_EQ(td->activeW, 715827840);
    EXPECT_EQ(td->activeC, 3);
}

// WC 元素数超过 int32 范围（activeW * activeC > INT32_MAX），tiling 应失败。
TEST_F(PasteSubImgTiling, paste_sub_img_tiling_wc_count_exceeds_int32_range_should_fail)
{
    int32_t patchCoord[] = {0, 0, 0, 0};
    int32_t coreAreaCoord[] = {0, 0, 1000000000, 1};
    EXPECT_FALSE(
        RunTilingExpectReject(ge::DT_UINT8, 1, 1000000000, 3, 1, 1000000000, 3, patchCoord, coreAreaCoord, 1.0f));
}
