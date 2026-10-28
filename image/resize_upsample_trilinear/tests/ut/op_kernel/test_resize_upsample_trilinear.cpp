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
#include "../../../op_host/resize_upsample_trilinear_tiling.h"
#include "data_utils.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

extern "C" __global__ __aicore__ void resize_upsample_trilinear(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                                GM_ADDR tiling);

class ResizeUpsampleTrilinearTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ResizeUpsampleTrilinearTest SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "ResizeUpsampleTrilinearTest TearDown\n" << endl; }
};

TEST_F(ResizeUpsampleTrilinearTest, test_case_float32)
{
    system("cp -rf "
           "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
           "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    struct ResizeUpsampleTrilinearCompileInfo {
        uint16_t totalCoreNum = 48;
        uint16_t socVersionType = 220;
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{{{1, 2, 2, 4, 4}, {1, 2, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2, 8, 8, 16}, {1, 2, 8, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8, 16})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float32'");

    size_t inputByteSize = 2 * 2 * 4 * 4 * sizeof(float);
    size_t outputByteSize = 2 * 8 * 8 * 16 * sizeof(float);
    uint32_t numBlocks = tilingInfo.blockNum;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    std::string fileName = "./resize_upsample_trilinear_data/float32_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    ICPU_RUN_KF(resize_upsample_trilinear, numBlocks, x, y, workspace, tiling);
    fileName = "./resize_upsample_trilinear_data/float32_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float32'");
}

TEST_F(ResizeUpsampleTrilinearTest, test_case_float16)
{
    system("cp -rf "
           "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
           "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    struct ResizeUpsampleTrilinearCompileInfo {
        uint16_t totalCoreNum = 48;
        uint16_t socVersionType = 220;
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{{{1, 2, 2, 4, 4}, {1, 2, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 2, 8, 8, 16}, {1, 2, 8, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8, 16})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float16'");

    size_t inputByteSize = 2 * 2 * 4 * 4 * sizeof(half);
    size_t outputByteSize = 2 * 8 * 8 * 16 * sizeof(half);
    uint32_t numBlocks = tilingInfo.blockNum;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    std::string fileName = "./resize_upsample_trilinear_data/float16_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    ICPU_RUN_KF(resize_upsample_trilinear, numBlocks, x, y, workspace, tiling);
    fileName = "./resize_upsample_trilinear_data/float16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float16'");
}

TEST_F(ResizeUpsampleTrilinearTest, test_case_bfloat16)
{
    system("cp -rf "
           "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
           "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    struct ResizeUpsampleTrilinearCompileInfo {
        uint16_t totalCoreNum = 20;
        uint16_t socVersionType = 220;
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ResizeUpsampleTrilinear", {{{{1, 2, 2, 4, 4}, {1, 2, 2, 4, 4}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 2, 8, 8, 16}, {1, 2, 8, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8, 16})),
         gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'bfloat16'");

    size_t inputByteSize = 2 * 2 * 4 * 4 * sizeof(half);
    size_t outputByteSize = 2 * 8 * 8 * 16 * sizeof(half);
    uint32_t numBlocks = tilingInfo.blockNum;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    std::string fileName = "./resize_upsample_trilinear_data/bfloat16_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    ICPU_RUN_KF(resize_upsample_trilinear, numBlocks, x, y, workspace, tiling);
    fileName = "./resize_upsample_trilinear_data/bfloat16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float16'");
}
