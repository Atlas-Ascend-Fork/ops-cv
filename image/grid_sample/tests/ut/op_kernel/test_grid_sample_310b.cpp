/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_host/grid_sample_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#define __CCE_AICORE__ 300
using namespace std;
using namespace optiling;

extern "C" void grid_sample(uint8_t* x, uint8_t* grid, uint8_t* y, uint8_t* workspace, uint8_t* tiling);

class grid_sample_310b_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "grid_sample_310b_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "grid_sample_310b_test TearDown\n" << endl; }
};

TEST_F(grid_sample_310b_test, test_case_fp16_test1)
{
    int64_t N = 2;
    int64_t x_h = 16;
    int64_t x_w = 16;
    int64_t C = 64;
    int64_t grid_h = 8;
    int64_t grid_w = 64;
    int64_t dim = 2;
    size_t inputByteSize = N * x_h * x_w * C * sizeof(half);
    size_t gridByteSize = N * grid_h * grid_w * dim * sizeof(half);
    size_t outputByteSize = N * grid_h * grid_w * C * sizeof(half);
    GridSampleCompileInfo compileInfo = {48, 0, false};
    string socVersion = "Ascend310b";
    gert::TilingContextPara tilingContextPara(
        "GridSample",
        {{{{2, 64, 16, 16}, {2, 64, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 8, 64, 2}, {2, 8, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("interpolation_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("nearest")),
         gert::TilingContextPara::OpAttr("padding_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("border")),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("channel_last", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scheduler_mode", Ops::Cv::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo, socVersion);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* grid = (uint8_t*)AscendC::GmAlloc(gridByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint32_t numBlocks = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grid_sample, numBlocks, x, grid, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grid);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
