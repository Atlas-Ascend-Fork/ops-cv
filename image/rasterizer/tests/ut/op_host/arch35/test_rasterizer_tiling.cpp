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
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/rasterizer_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class RasterizerTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RasterizerTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RasterizerTilingArch35 TearDown" << std::endl; }
};

struct RasterizerCompileInfo {};

TEST_F(RasterizerTilingArch35, rasterizer_tiling_arch35_001)
{
    gert::StorageShape v_shape = {{3, 4}, {3, 4}};
    gert::StorageShape f_shape = {{1, 3}, {1, 3}};
    gert::StorageShape d_shape = {{10, 10}, {10, 10}};
    gert::StorageShape findices_shape = {{10, 10}, {10, 10}};
    gert::StorageShape bary_shape = {{10, 10, 3}, {10, 10, 3}};
    RasterizerCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Rasterizer",
        {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {f_shape, ge::DT_INT32, ge::FORMAT_ND},
         {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{findices_shape, ge::DT_INT32, ge::FORMAT_ND}, {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
         gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo, "Ascend950", 28, 253952);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.tilingDataSize, 96U);
    EXPECT_EQ(info.workspaceSizes.size(), 1U);

    const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
    EXPECT_EQ(data[0], 1);                                // numFaces
    EXPECT_EQ(data[1], 100);                              // totalPixels = height * width
    EXPECT_EQ(static_cast<uint32_t>(data[2]), 10U);       // height
    EXPECT_EQ(static_cast<uint32_t>(data[2] >> 32), 10U); // width
    EXPECT_GT(info.workspaceSizes[0], 0U);
}
