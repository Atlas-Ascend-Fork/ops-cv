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
 * \file test_three_interpolate_backward_arch35_tiling.cpp
 * \brief arch35 tiling boundary tests for ops-cv issues #1098 / #1126:
 *        - #1098: attr m must be validated (0, INT32_MAX] BEFORE narrowing
 *          (int64 attr -> int32_t; m = 2^32+1 would wrap to 1 and bypass m<=0)
 *        - #1126: b*c > 2^32 is supported via MODE_LARGE (64-bit bcIdx decode)
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/three_interpolate_backward_tiling_data.h"

using namespace std;
using namespace ge;

class ThreeInterpolateBackwardTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThreeInterpolateBackwardTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ThreeInterpolateBackwardTilingArch35 TearDown" << std::endl; }
};

namespace {
bool RunTilingWithShape(int64_t b, int64_t c, int64_t n, int64_t mValue, TilingInfo& tilingInfo)
{
    struct ThreeInterpolateBackwardCompileInfo {
    } compileInfo;
    gert::StorageShape gradXShape = {{b, c, n}, {b, c, n}};
    gert::StorageShape idxShape = {{b, n, 3}, {b, n, 3}};
    gert::StorageShape weightShape = {{b, n, 3}, {b, n, 3}};
    gert::StorageShape gradYShape = {{b, c, mValue}, {b, c, mValue}};

    gert::TilingContextPara tilingContextPara(
        "ThreeInterpolateBackward",
        {
            {gradXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {idxShape, ge::DT_INT32, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {gradYShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("m", Ops::Cv::AnyValue::CreateFrom<int64_t>(mValue)),
        },
        &compileInfo, "Ascend950",
        64,     // coreNum
        262144, // ubSize
        4096);  // tilingDataSize
    return ExecuteTiling(tilingContextPara, tilingInfo);
}

bool RunTilingWithM(int64_t mValue, TilingInfo& tilingInfo) { return RunTilingWithShape(2, 3, 4, mValue, tilingInfo); }
} // namespace

// issue #1098：m = 2^32 + 1 窄化为 int32 后回绕为 1（仍为正），必须在窄化前拒绝
TEST_F(ThreeInterpolateBackwardTilingArch35, m_positive_wrap_rejected)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(RunTilingWithM(4294967297LL, tilingInfo));
}

// issue #1098：m = INT32_MAX + 1 超出 int32 承载，拒绝
TEST_F(ThreeInterpolateBackwardTilingArch35, m_int32_max_plus_1_rejected)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(RunTilingWithM(2147483648LL, tilingInfo));
}

// issue #1098 边界：m = INT32_MAX 合法，应接受且字段不截断
TEST_F(ThreeInterpolateBackwardTilingArch35, m_int32_max_accepted)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunTilingWithM(2147483647LL, tilingInfo));
    const auto* tilingData = reinterpret_cast<const ThreeInterpolateBackwardTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->m, 2147483647);
}

// 对照：m = -1，被拒绝
TEST_F(ThreeInterpolateBackwardTilingArch35, m_negative_rejected)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(RunTilingWithM(-1LL, tilingInfo));
}

// 对照：正常小 m 走 MODE_DEFAULT，字段一致
TEST_F(ThreeInterpolateBackwardTilingArch35, m_normal_accepted)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunTilingWithM(8LL, tilingInfo));
    const auto* tilingData = reinterpret_cast<const ThreeInterpolateBackwardTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->m, 8);
    EXPECT_EQ(tilingData->b, 2);
    EXPECT_EQ(tilingData->c, 3);
    EXPECT_EQ(tilingData->n, 4);
    EXPECT_EQ(tilingData->totalElements, 6);
}

// 边界：b*c = 2^32（> 2^31 安全上界），应接受并由 MODE_LARGE（64 位解码）承接，totalElements 不截断
TEST_F(ThreeInterpolateBackwardTilingArch35, bc_max_uint32_accepted)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunTilingWithShape(65536LL, 65536LL, 1LL, 1LL, tilingInfo));
    const auto* tilingData = reinterpret_cast<const ThreeInterpolateBackwardTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalElements, 4294967296LL);
}

// b*c > 2^32 由 MODE_LARGE（64 位解码）承接，应接受且不截断
TEST_F(ThreeInterpolateBackwardTilingArch35, bc_over_uint32_accepted_large_mode)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunTilingWithShape(65537LL, 65537LL, 1LL, 1LL, tilingInfo));
    const auto* tilingData = reinterpret_cast<const ThreeInterpolateBackwardTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalElements, 4295098369LL);
}

// LR-02 门限：32 位 UintDiv magic 快除的安全上界是 2^31（非 2^32）。
// b*c == 2^31 仍走 MODE_DEFAULT，b*c == 2^31+2^16 必须切到 MODE_LARGE（tilingKey 不同）。
TEST_F(ThreeInterpolateBackwardTilingArch35, bc_mode_switch_at_2p31_boundary)
{
    TilingInfo atBound;
    TilingInfo overBound;
    ASSERT_TRUE(RunTilingWithShape(65536LL, 32768LL, 1LL, 1LL, atBound));   // b*c == 2^31
    ASSERT_TRUE(RunTilingWithShape(65536LL, 32769LL, 1LL, 1LL, overBound)); // b*c == 2^31 + 2^16
    const auto* atData = reinterpret_cast<const ThreeInterpolateBackwardTilingData*>(atBound.tilingData.get());
    const auto* overData = reinterpret_cast<const ThreeInterpolateBackwardTilingData*>(overBound.tilingData.get());
    EXPECT_EQ(atData->totalElements, 2147483648LL);
    EXPECT_EQ(overData->totalElements, 2147549184LL);
    EXPECT_NE(atBound.tilingKey, overBound.tilingKey);
}
