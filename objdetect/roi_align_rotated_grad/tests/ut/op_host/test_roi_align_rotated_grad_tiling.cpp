/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_align_rotated_grad_tiling.cpp
 * \brief
 */
#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "../../../op_host/arch22/roi_align_rotated_grad_tiling_arch22.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

namespace {
struct RoiAlignRotatedGradTilingDataPod {
    uint32_t coreRoisNums;
    uint32_t coreRoisTail;
    uint32_t boxSize;
    int32_t pooledHeight;
    int32_t pooledWidth;
    uint32_t batchSize;
    uint32_t channelNum;
    uint32_t width;
    uint32_t height;
    bool aligned;
    bool clockwise;
    uint8_t samplingRatioPH[2];
    int32_t samplingRatio;
    float spatialScale;
    uint32_t coreNum;
};
} // namespace

class TilingForRoiAlignRotatedGrad : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TilingForRoiAlignRotatedGrad SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TilingForRoiAlignRotatedGrad TearDown" << std::endl; }
};

TEST_F(TilingForRoiAlignRotatedGrad, roi_align_rotated_grad_tiling_0)
{
    optiling::RoiAlignRotatedGradCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "RoiAlignRotatedGrad",
        {{{{8, 8, 8, 8}, {8, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 2, 2, 8}, {8, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("y_grad_shape", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({8, 2, 2, 8})),
         gert::TilingContextPara::OpAttr("pooled_h", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("pooled_w", Ops::Cv::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("spatial_scale", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
         gert::TilingContextPara::OpAttr("sampling_ratio", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("aligned", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("clockwise", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);

    std::vector<size_t> expectWorkspaces = {16777216};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    uint8_t* tilingDataPtr = tilingInfo.tilingData.get();
    ASSERT_NE(tilingDataPtr, nullptr);
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaces.size());
    for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
        EXPECT_EQ(tilingInfo.workspaceSizes[i], expectWorkspaces[i]);
    }

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(RoiAlignRotatedGradTilingDataPod));
    const auto* td = reinterpret_cast<const RoiAlignRotatedGradTilingDataPod*>(tilingDataPtr);
    EXPECT_EQ(td->coreRoisNums, 0u);
    EXPECT_EQ(td->coreRoisTail, 8u);
    EXPECT_EQ(td->boxSize, 8u);
    EXPECT_EQ(td->pooledHeight, 2);
    EXPECT_EQ(td->pooledWidth, 2);
    EXPECT_EQ(td->batchSize, 8u);
    EXPECT_EQ(td->channelNum, 8u);
    EXPECT_EQ(td->width, 2u);
    EXPECT_EQ(td->height, 2u);
    EXPECT_EQ(td->aligned, false);
    EXPECT_EQ(td->clockwise, false);
    EXPECT_EQ(td->samplingRatio, 1);
    EXPECT_FLOAT_EQ(td->spatialScale, 0.5f);
    EXPECT_EQ(td->coreNum, 64u);
}
